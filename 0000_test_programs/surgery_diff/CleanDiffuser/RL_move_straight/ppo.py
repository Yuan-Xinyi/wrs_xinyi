"""
CleanRL-compatible PPO for FK-only XArm reaching task
- No physics (FK-only)
- Stable continuous action PPO
- Delta-q limited to +/- 0.02
- Joint limit clamp
- Fully shaped reward: -dist + progress + smoothness
- Small goal region (curriculum-ready)

This file is ready to run.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import wandb   # ★★★ WandB 追加

import wrs.neuro.xarm_lite6_neuro as xarm6_gpu


# ---------------------------------------------------------------------
# 1. CleanRL Args
# ---------------------------------------------------------------------

@dataclass
class Args:
    exp_name: str = "ppo_xarm"
    seed: int = 1
    cuda: bool = True
    torch_deterministic: bool = True

    total_timesteps: int = 2_000_000
    learning_rate: float = 3e-4
    num_envs: int = 1024
    num_steps: int = 128
    anneal_lr: bool = True

    gamma: float = 0.99
    gae_lambda: float = 0.95

    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    target_kl: float = None

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# ---------------------------------------------------------------------
# 2. Utilities
# ---------------------------------------------------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ---------------------------------------------------------------------
# 3. Agent (Gaussian policy)
# ---------------------------------------------------------------------

class Agent(nn.Module):
    """
    Observations: 15 dims
    Actions: 6D Delta-Q
    """
    def __init__(self, envs):
        super().__init__()

        obs_dim = envs.single_observation_space.shape[0]
        act_dim = envs.single_action_space.shape[0]

        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

        # Actor mean
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, act_dim), std=0.01),
        )

        # Action log std
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -3.0)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(x)

        return action, logprob, entropy, value


# ---------------------------------------------------------------------
# 4. RobotsEnv (FK-only)
# ---------------------------------------------------------------------

class Robots:
    def __init__(self, batch_size):
        self.robot = xarm6_gpu.XArmLite6GPU()
        self.jnts = self.robot.robot.rand_conf_batch(batch_size=batch_size)

    def fk(self, q):
        pos, rotmat = self.robot.robot.fk_batch(jnt_values=q)
        return pos, rotmat


class RobotsEnv:
    """FK-only env: reward = forward distance + straightness"""

    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device

        self.robot = Robots(batch_size=num_envs)

        self.current_qpos = None
        self.current_qvel = torch.zeros(num_envs, 6).to(device)
        self.start_pos = None        # ★ initial TCP pos
        self.tcp_pos = None
        self.prev_tcp_pos = None

        self._single_action_space = type("AS", (), {"shape": (6,)})()
        self._single_observation_space = type("OS", (), {"shape": (15,)})()

        self.max_steps = 100
        self.episode_steps = torch.zeros(num_envs).to(device)

        self.qmin = self.robot.robot.robot.jnt_ranges[:,0].to(device)
        self.qmax = self.robot.robot.robot.jnt_ranges[:,1].to(device)

    @property
    def single_action_space(self):
        return self._single_action_space

    @property
    def single_observation_space(self):
        return self._single_observation_space

    # -------------------------
    # reset
    # -------------------------
    def reset(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        # q = self.robot.robot.robot.rand_conf_batch(batch_size=self.num_envs)
        # self.current_qpos = q0.unsqueeze(0).repeat(self.num_envs, 1).to(self.device)
        q0 = torch.tensor([0., 0.173311, 0.555015, 0., 0.381703, 0.], dtype=torch.float32)
        self.current_qpos = q0.unsqueeze(0).repeat(self.num_envs, 1).to(self.device)
        self.current_qvel = torch.zeros_like(self.current_qpos)

        # compute TCP
        pos, _ = self.robot.robot.robot.fk_batch(jnt_values=self.current_qpos)
        self.tcp_pos = pos.to(self.device)
        self.prev_tcp_pos = self.tcp_pos.clone()

        # ★ fixed start pos
        self.start_pos = self.tcp_pos.clone()

        self.episode_steps = torch.zeros(self.num_envs).to(self.device)

        return self._get_obs().cpu().numpy(), {}

    # -------------------------
    # obs
    # -------------------------
    def _get_obs(self):
        return torch.cat([
            self.current_qpos,       # 6
            self.tcp_pos,            # 3
            self.start_pos,          # 3
            self.tcp_pos - self.start_pos  # 3 direction vector
        ], dim=1)

    # -------------------------
    # reward components
    # -------------------------
    def _straight_reward(self):
        v1 = self.prev_tcp_pos - self.start_pos
        v2 = self.tcp_pos - self.start_pos

        norm1 = torch.norm(v1, dim=1) + 1e-6
        norm2 = torch.norm(v2, dim=1) + 1e-6

        cos = torch.sum(v1 * v2, dim=1) / (norm1 * norm2)
        return cos  # [-1, 1]

    def _compute_reward(self):
        dist_from_start = torch.norm(self.tcp_pos - self.start_pos, dim=1)
        straight = self._straight_reward()

        reward = 1.0 * dist_from_start + 10.0 * straight
        return reward

    # -------------------------
    # step
    # -------------------------
    def step(self, action_np):
        action = torch.tensor(action_np, dtype=torch.float32).to(self.device)

        dq_limit = 0.02
        action = torch.clamp(action, -dq_limit, dq_limit)

        self.prev_tcp_pos = self.tcp_pos.clone()

        new_qpos = torch.clamp(self.current_qpos + action, self.qmin, self.qmax)

        self.current_qvel = new_qpos - self.current_qpos
        self.current_qpos = new_qpos

        # FK
        pos, _ = self.robot.robot.robot.fk_batch(jnt_values=new_qpos)
        self.tcp_pos = torch.tensor(pos, dtype=torch.float32).to(self.device)

        # reward
        reward = self._compute_reward()

        # done when max steps reached
        self.episode_steps += 1
        done = (self.episode_steps >= self.max_steps)

        info = {"final_info":[None]*self.num_envs}

        for i in range(self.num_envs):
            if done[i]:
                info["final_info"][i] = {"episode":{
                    "r": reward[i].item(),
                    "l": self.episode_steps[i].item()
                }}

                # reset env i
                q = self.robot.robot.robot.rand_conf_batch(batch_size=1)
                self.current_qpos[i] = torch.tensor(q[0], dtype=torch.float32).to(self.device)
                self.current_qvel[i] = 0.0

                pos, _ = self.robot.robot.robot.fk_batch(jnt_values=self.current_qpos[i:i+1])
                self.tcp_pos[i] = torch.tensor(pos[0], dtype=torch.float32).to(self.device)

                self.start_pos[i] = self.tcp_pos[i].clone()
                self.prev_tcp_pos[i] = self.tcp_pos[i].clone()
                self.episode_steps[i] = 0

        return (
            self._get_obs().cpu().numpy(),
            reward.cpu().numpy(),
            done.cpu().numpy(),
            done.cpu().numpy(),
            info
        )



# ---------------------------------------------------------------------
# 5. Main: CleanRL PPO Loop
# ---------------------------------------------------------------------

if __name__ == "__main__":
    args = tyro.cli(Args)

    # WandB Init ★★★
    wandb.init(
        project="wrs_xarm_ppo",
        name=f"{args.exp_name}_seed{args.seed}",
        config=vars(args),
        save_code=True,
    )

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    writer = SummaryWriter(f"runs/{args.exp_name}_{int(time.time())}")

    envs = RobotsEnv(args.num_envs, device)
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs, 15), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, 6), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations+1):

        if args.anneal_lr:
            frac = 1 - (iteration-1)/args.num_iterations
            optimizer.param_groups[0]['lr'] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs

            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(next_obs)
                values[step] = value.view(-1)

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, rew, term, trunc, info = envs.step(action.cpu().numpy())
            term = torch.tensor(term, dtype=torch.bool, device=device)
            trunc = torch.tensor(trunc, dtype=torch.bool, device=device)
            next_done = (term | trunc).float()

            rewards[step] = torch.tensor(rew, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32).to(device)

            # episodic stats
            if "final_info" in info:
                for f in info["final_info"]:
                    if f is not None:
                        ep_r = f["episode"]["r"]
                        ep_l = f["episode"]["l"]

                        writer.add_scalar("episodic_return", ep_r, global_step)
                        writer.add_scalar("episodic_length", ep_l, global_step)

                        wandb.log({          # ★★★ WandB 追加
                            "episodic_return": ep_r,
                            "episodic_length": ep_l,
                        }, step=global_step)

                        print(f"[step {global_step}] return={ep_r}")

        # ------------------- GAE -------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs)

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1 - next_done
                nextvalues = next_value.view(-1)
            else:
                nextnonterminal = 1 - dones[t+1]
                nextvalues = values[t+1].view(-1)
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

        returns = values + advantages

        # flatten
        b_obs = obs.reshape(-1, 15)
        b_actions = actions.reshape(-1, 6)
        b_logprobs = logprobs.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ------------------- PPO update -------------------
        inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb = inds[start:start+args.minibatch_size]

                _, newlogprob, entropy, newvalues = agent.get_action_and_value(
                    b_obs[mb], b_actions[mb]
                )
                newvalues = newvalues.view(-1)

                logratio = newlogprob - b_logprobs[mb]
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1).abs() > args.clip_coef).float().mean().item())

                adv = b_adv[mb]
                if args.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio,
                                         1-args.clip_coef,
                                         1+args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                v_unclipped = (newvalues - b_returns[mb])**2
                v_clipped = b_values[mb] + torch.clamp(
                    newvalues - b_values[mb],
                    -args.clip_coef, args.clip_coef
                )
                v_clipped = (v_clipped - b_returns[mb])**2
                v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                ent = entropy.mean()
                loss = pg_loss - args.ent_coef * ent + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and kl > args.target_kl:
                break

        writer.add_scalar("loss/policy", pg_loss.item(), global_step)
        writer.add_scalar("loss/value", v_loss.item(), global_step)
        writer.add_scalar("loss/entropy", ent.item(), global_step)
        writer.add_scalar("loss/clipfrac", np.mean(clipfracs), global_step)

        wandb.log({                     # ★★★ WandB 追加
            "loss/policy": pg_loss.item(),
            "loss/value": v_loss.item(),
            "loss/entropy": ent.item(),
            "loss/clipfrac": np.mean(clipfracs),
        }, step=global_step)

        sps = global_step/(time.time()-start_time)
        print(f"SPS: {sps:.1f}")

        wandb.log({"SPS": sps}, step=global_step)   # ★★★ WandB 追加

        # -- Save model checkpoint --
        if iteration % 20 == 0 or iteration == args.num_iterations:
            save_dir = f"runs/{args.exp_name}_{start_time}/checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            save_path = f"{save_dir}/agent_iter{iteration}_step{global_step}.pth"
            torch.save(agent.state_dict(), save_path)
            print(f"[Checkpoint] Model saved to {save_path}")

            wandb.save(save_path)      # ★★★ WandB 追加


    writer.close()
    wandb.finish()   # ★★★ WandB 追加
