# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal # For continuous action space (Gaussian Policy)

from wrs import wd, rm, mcm
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
import wrs.modeling.geometric_model as mgm


@dataclass
class Args:
    # ... (Args class remains unchanged) ...
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Adapted Agent class for Continuous Action Space (Delta Q, Gaussian Policy).
    Observation space: 15D (qpos(6) + tcp_pos(3) + goal_pos(3) + tcp_to_goal(3))
    Action space: 6D (Delta Q)
    """
    def __init__(self, envs):
        super().__init__()
        # Use single_observation_space and single_action_space from the custom env
        obs_dim = envs.single_observation_space.shape[0]
        action_dim = envs.single_action_space.shape[0] # Should be 6
        
        # Value Network (Critic)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Policy Mean Network (Actor)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        # Log standard deviation parameter (learned)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        # Continuous action space logic (Gaussian)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample() # Sample Delta Q
        
        # Calculate log_prob: sum across action dimension (6D)
        log_prob = probs.log_prob(action).sum(1)
        
        # Calculate entropy: sum across action dimension (6D)
        entropy = probs.entropy().sum(1)
        
        return action, log_prob, entropy, self.critic(x)


class Robots:
    """
    The original Robots class, now acts as a batched controller/simulator.
    NOTE: The 'jnts' variable must be treated as the current batched joint positions.
    We will manage this state in RobotsEnv.
    """
    def __init__(self, batch_size):
        self.robot = xarm6_gpu.XArmLite6GPU()
        # Initialize jnts for the batch. This will be updated externally.
        self.jnts = self.robot.robot.rand_conf_batch(batch_size=batch_size) 
        self.tcps = None
    
    def fk(self, jnt_values_np: np.ndarray):
        """
        Modified FK function to take joint values as input and return pos/rotmat.
        Assumes jnt_values_np is of shape (batch_size, 6).
        """
        pos, rotmat = self.robot.robot.fk_batch(jnt_values=jnt_values_np)
        return pos, rotmat

    # The original fk() logic is now redundant as we use fk(jnt_values_np)
    # The original class is left untouched, but its usage is updated in RobotsEnv.

# ----------------------------------------------------------------------
# RobotsEnv: Custom Vector Environment Wrapper for Point-to-Goal Task
# ----------------------------------------------------------------------

class RobotsEnv:
    """
    Simulated vector environment class for XArm Lite6 point-to-goal task.
    This mimics the gym.vector.AsyncVectorEnv interface.
    """
    def __init__(self, num_envs, device, batch_size):
        self.num_envs = num_envs
        self.device = device
        
        # Initialize the ORIGINAL Robots controller class
        self.robot_controller = Robots(batch_size=num_envs)
        
        # State variables (Batched tensors on device)
        # 6 joint positions
        self.current_qpos = None 
        # 6 joint velocities (used for static reward)
        self.current_qvel = torch.zeros(num_envs, 6).to(device) 
        self.goal_pos = None     # Target TCP position (3D)
        self.tcp_pos = None      # Current TCP position (3D)
        self.prev_tcp_pos = None # Previous TCP position (for progress reward)

        # Define Observation and Action Space shapes
        # Action: Delta Q (6DOF)
        self._single_action_space = type('ActionSpace', (object,), {'shape': (6,), 'n': 6})() 
        # Obs: qpos(6) + tcp_pos(3) + goal_pos(3) + tcp_to_goal(3) = 15
        self._single_observation_space = type('ObsSpace', (object,), {'shape': (15,)})()
        
        # Task parameters
        self.table_center = torch.tensor([0.6, 0.0, 0.05]).to(device) 
        self.table_half_size = 0.5 # Define reachable area on the table
        self.goal_thresh = 0.025
        self.max_episode_steps = 50

        # State flags
        self.episode_steps = torch.zeros(num_envs).to(device)


    @property
    def single_observation_space(self):
        return self._single_observation_space

    @property
    def single_action_space(self):
        return self._single_action_space

    def _generate_random_pos(self, center, half_size):
        """Generates random [x, y] positions on the table and sets z above the table."""
        xy = (torch.rand(self.num_envs, 2).to(self.device) * half_size * 2) - half_size
        xy[:, 0] += center[0]
        xy[:, 1] += center[1]
        z = center[2] + 0.1 
        pos = torch.cat([xy, torch.full((self.num_envs, 1), z).to(self.device)], dim=1)
        return pos

    def _get_obs(self):
        """Constructs the observation vector."""
        tcp_to_goal_pos = self.goal_pos - self.tcp_pos
        
        obs = torch.cat([
            self.current_qpos, 
            self.tcp_pos, 
            self.goal_pos, 
            tcp_to_goal_pos
        ], dim=1)
        return obs

    def _update_tcp(self):
        """Updates TCP position based on current_qpos using the original Robots class."""
        # Use the extended fk method from the original Robots class
        pos_np, _ = self.robot_controller.fk(self.current_qpos)
        self.tcp_pos = torch.tensor(pos_np, dtype=torch.float32).to(self.device)


    def reset(self, seed=None):
        """Resets the environment and returns the initial observation."""
        if seed is not None:
            torch.manual_seed(seed)
        
        # 1. Initialize joint positions (qpos) randomly
        qpos_np = self.robot_controller.robot.robot.rand_conf_batch(batch_size=self.num_envs)
        self.current_qpos = torch.tensor(qpos_np, dtype=torch.float32).to(self.device)
        self.current_qvel = torch.zeros_like(self.current_qpos).to(self.device)

        # 2. Randomly initialize TCP Goal position
        self.goal_pos = self._generate_random_pos(self.table_center, self.table_half_size)
        
        # 3. Calculate initial TCP position
        self._update_tcp()
        self.prev_tcp_pos = self.tcp_pos.clone() 

        # Reset episode step counter
        self.episode_steps = torch.zeros(self.num_envs).to(self.device)

        return self._get_obs().cpu().numpy(), {"info": "reset"}
    
    def step(self, action: np.ndarray):
        """
        Applies action (delta_q) and advances the simulation/state.
        action: numpy array of shape (num_envs, 6)
        """
        action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device)
        
        # Save previous TCP pos before update
        self.prev_tcp_pos = self.tcp_pos.clone() 
        
        # 1. Apply Delta Q action
        dq_limit = 0.05 # Max change per step
        action_tensor = torch.clamp(action_tensor, -dq_limit, dq_limit)
        
        new_qpos = self.current_qpos + action_tensor
        
        # 2. Update Q velocity and Q position
        self.current_qvel = new_qpos - self.current_qpos 
        self.current_qpos = new_qpos

        # 3. Calculate new TCP position (Forward Kinematics)
        self._update_tcp()

        # 4. Check termination/truncation
        self.episode_steps += 1
        
        # Task Termination: Goal Reached
        tcp_to_goal_dist = torch.linalg.norm(self.goal_pos - self.tcp_pos, dim=1)
        terminations = (tcp_to_goal_dist <= self.goal_thresh).float()
        
        # Truncation: Max steps reached
        truncations = (self.episode_steps >= self.max_episode_steps).float()

        done = torch.logical_or(terminations.bool(), truncations.bool()).float()

        # 5. Compute Reward
        reward = self._compute_reward(
            tcp_pos=self.tcp_pos,
            prev_tcp_pos=self.prev_tcp_pos, 
            goal_pos=self.goal_pos,
            is_done=terminations
        )

        # 6. Prepare next observation
        next_obs = self._get_obs()
        
        # 7. Info structure (mimics gym step info)
        infos = {"final_info": [None] * self.num_envs}
        
        # Handle environment reset after termination/truncation
        for i in range(self.num_envs):
            if done[i].item() == 1.0:
                # Record episodic stats
                infos["final_info"][i] = {
                    "episode": {
                        "r": reward[i].item(), 
                        "l": self.episode_steps[i].item(),
                    }
                }
                # Reset state variables for the done env
                self._reset_single_env(i)
                # Recalculate next_obs for the just-reset environment
                next_obs[i] = self._get_obs()[i]

        return next_obs.cpu().numpy(), reward.cpu().numpy(), terminations.bool().cpu().numpy(), truncations.bool().cpu().numpy(), infos
    
    def _reset_single_env(self, idx):
        """Resets the state of a single environment."""
        # 1. Reset qpos and qvel
        self.current_qpos[idx] = torch.tensor(self.robot_controller.robot.robot.rand_conf_batch(batch_size=1), \
                                              dtype=torch.float32).to(self.device)[0]
        self.current_qvel[idx] = 0.0
        
        # 2. Reset Goal pos
        self.goal_pos[idx] = self._generate_random_pos(self.table_center, self.table_half_size)[0]
        
        # 3. Recalculate initial TCP pos
        self._update_tcp() # Update all TCPs based on current_qpos
        self.prev_tcp_pos[idx] = self.tcp_pos[idx].clone()

        # 4. Reset step counter
        self.episode_steps[idx] = 0
    
    def _compute_reward(self, tcp_pos, prev_tcp_pos, goal_pos, is_done):
        """
        Custom Dense Reward for Point-to-Goal Task.
        """
        
        # ----- 1. Approach: Shaped reward -----
        dist_to_goal = torch.linalg.norm(goal_pos - tcp_pos, dim=1)
        approach_reward = 2 * (1 - torch.tanh(5 * dist_to_goal))
        reward = approach_reward
        
        # ----- 2. Distance Reduction Reward (Progress Reward) -----
        dist_to_goal_prev = torch.linalg.norm(goal_pos - prev_tcp_pos, dim=1)
        dist_reduction = dist_to_goal_prev - dist_to_goal
        
        dist_reduction_reward = torch.clamp(dist_reduction, min=0.0) * 10.0
        reward += dist_reduction_reward
        
        # ----- 3. Forward Movement in Desired Direction -----
        move = tcp_pos - prev_tcp_pos
        dir_goal = (goal_pos - tcp_pos)
        
        dir_goal = dir_goal / (torch.norm(dir_goal, dim=1, keepdim=True) + 1e-6)

        proj = torch.sum(move * dir_goal, dim=1)
        
        forward_reward = torch.clamp(proj, min=0.0)
        reward += forward_reward * 20.0 

        # ----- 4. Near Goal Bonus -----
        close_bonus = (dist_to_goal < 0.05).float() * 3.0
        reward += close_bonus

        # ----- 5. Success Bonus (on termination) -----
        reward[is_done.bool()] = 8.0 
        
        return reward.float()
    
    def close(self):
        """Placeholder for closing resources."""
        pass


# --- 2. Main Training Loop Adaptation ---

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # ... (Batch size calculation, run name, logging setup remains unchanged) ...
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup (Visualization elements remain)
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    # table_size = np.array([1.5, 1.5, 0.05])
    # table_pos  = np.array([0.6, 0, -0.025])
    # table = mcm.gen_box(xyz_lengths=table_size, pos=table_pos, rgb=np.array([0.6, 0.4, 0.2]), alpha=1)
    # table.attach_to(base)

    # paper_size = np.array([1.0, 1.0, 0.002])
    # paper_pos = table_pos.copy()
    # paper_pos[2] = table_pos[2] + table_size[2]/2 + paper_size[2]/2
    # paper = mcm.gen_box(xyz_lengths=paper_size, pos=paper_pos, rgb=np.array([1, 1, 1]), alpha=1)
    # paper.attach_to(base)

    # --- ENVIRONMENT REPLACEMENT ---
    envs = RobotsEnv(args.num_envs, device, args.batch_size) 
    
    # Agent initialization uses the custom environment's space shapes
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup (Adapt action tensor size)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    # Action tensor shape adapted to (..., 6) for Delta Q
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device) 
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # next_obs is now a numpy array from RobotsEnv.reset()
    next_obs_np, _ = envs.reset(seed=args.seed) 
    next_obs = torch.Tensor(next_obs_np).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # action is now a continuous tensor of Delta Q
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # action.cpu().numpy() is the Delta Q to pass to the environment
            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            
            next_done = np.logical_or(terminations, truncations)
            
            # Convert results back to tensors
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs_np).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        # Action tensor is continuous (6D)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape) 
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # b_actions is continuous, pass the tensor directly (not .long())
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds]) 
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()