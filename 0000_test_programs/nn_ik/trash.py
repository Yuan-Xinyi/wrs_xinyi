import os

'''walk these ways'''
import mujoco_py
import glob
import pickle as pkl
from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
import torch.nn.functional as F

import d4rl
import gym
import numpy as np
import torch
import h5py
from argparse import Namespace
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from copy import deepcopy
import hydra
import wandb
from datetime import datetime
from hydra import compose, initialize
from omegaconf import OmegaConf

'''cleandiffuser imports'''
from cleandiffuser.diffusion.diffusionsde import BaseDiffusionSDE, DiscreteDiffusionSDE, BaseDiffusionSDE
from cleandiffuser.nn_condition import PearceObsCondition
from cleandiffuser.nn_diffusion import PearceMlp, JannerUNet1d
from cleandiffuser.utils import report_parameters
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from pipelines.utils import set_seed
from go1.datasets.sequence import SequenceDataset, PrefScriptedSequenceDataset
from go1.utils.arrays import batch_to_device

gait_type = "pacing"  # "pronking", "trotting", "bounding", "pacing"
TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
rootpath = f'{TimeCode}_{gait_type}'
wandb.init(project="locomotion_icra")


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'cumulative_rewards': []
            }

def append_data(data, s, a, r, done, cum_reward):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)
    data['cumulative_rewards'].append(cum_reward),

def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(headless=False):
    dirs = ["/home/xinyi/loco_diff_isaac/walk-these-ways/runs/gait-conditioned-agility/pretrain-v0/train/025417.456545"]
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type =  "actuator_net"  # "actuator_net" P

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=True, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy_walk = load_policy(logdir)

    return env, policy_walk

# ---------------------- Prepare env ----------------------
env, _ = load_env(headless=False)
num_eval_steps = 500
gaits = {"pronking": [0, 0, 0],
        "trotting": [0.5, 0, 0],
        "bounding": [0, 0.5, 0],
        "pacing": [0, 0, 0.5]}

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.0, 0.0, 0.0
body_height_cmd = 0.0
step_frequency_cmd = 3.0
gait = torch.tensor(gaits[gait_type])
footswing_height_cmd = 0.08
pitch_cmd = 0.0
roll_cmd = 0.0
stance_width_cmd = 0.25

env.commands[:, 0] = x_vel_cmd
env.commands[:, 1] = y_vel_cmd
env.commands[:, 2] = yaw_vel_cmd
env.commands[:, 3] = body_height_cmd
env.commands[:, 4] = step_frequency_cmd
env.commands[:, 5:8] = gait
env.commands[:, 8] = 0.5
env.commands[:, 9] = footswing_height_cmd
env.commands[:, 10] = pitch_cmd
env.commands[:, 11] = roll_cmd
env.commands[:, 12] = stance_width_cmd

obs = env.reset()

@hydra.main(config_path="/home/xinyi/CleanDiffuser/configs/diffuser/go1", config_name="organized_pipeline", version_base=None)
def train_diffuser(args):

    # ---------------------- preparation ----------------------
    args.gait = gait_type
    set_seed(args.seed)
    save_path = f'results/{rootpath}/models/diffuser/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    dataset_name = f'go1_walk_{args.gait}_ep2500'

    # ---------------------- Create Dataset ----------------------
    dataset = SequenceDataset(env = dataset_name, horizon = args.horizon, noise_level = args.noise_level,
        normalizer = args.normalizer, max_path_length = args.max_path_length)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim


    # --------------- Network Architecture -----------------
    nn_diffusion = JannerUNet1d(
        observation_dim + action_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.dim_mult,
        timestep_emb_type="positional", attention=False, kernel_size=5)
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)


    # ----------------- Masking -------------------
    fix_mask = torch.zeros((args.horizon, observation_dim + action_dim))
    fix_mask[0, action_dim:] = 1.
    loss_weight = torch.ones((args.horizon, observation_dim + action_dim))
    loss_weight[0, :action_dim] = args.action_loss_weight


    # --------------- Diffusion Model --------------------
    agent = DiscreteDiffusionSDE(nn_diffusion, 
                                 None, 
                                 dataset = dataset,
                                 fix_mask=fix_mask, 
                                 loss_weight=loss_weight, 
                                 ema_rate=args.ema_rate,
                                 device=args.device, 
                                 diffusion_steps=args.diffusion_steps, 
                                 predict_noise=args.predict_noise)
    

    # ---------------------- Training ----------------------
    diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.diffusion_gradient_steps)
    agent.train()
    n_gradient_step = 0
    log = {"avg_loss_diffusion": 0.}

    for batch in loop_dataloader(dataloader):

        x = batch.trajectories.to(args.device)
        
        # ----------- Gradient Step ------------
        current_loss = agent.update(x)['loss']  # domain_rand loss
        log["avg_loss_diffusion"] += current_loss  # BaseDiffusionSDE.update
        # print(f'[t={n_gradient_step + 1}] diffusion loss = {current_loss}')
        diffusion_lr_scheduler.step()

        # ----------- Logging ------------
        if (n_gradient_step + 1) % args.log_interval == 0:
            log["gradient_steps"] = n_gradient_step + 1
            log["avg_loss_diffusion"] /= args.log_interval
            wandb.log({"avg_training_loss": log["avg_loss_diffusion"]}, commit = True)
            print(log)
            log = {"avg_loss_diffusion": 0.}

        # ----------- Saving ------------
        if (n_gradient_step + 1) % args.save_interval == 0:
            agent.save(save_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
            agent.save(save_path + f"diffusion_ckpt_latest.pt")

        n_gradient_step += 1
        if n_gradient_step >= args.diffusion_gradient_steps:
            break

    # ---------------------- prepare dirs ----------------------
    model_path = f'results/{rootpath}/models/diffuser/diffusion_ckpt_latest.pt'
    file_path = f'results/{rootpath}/rollouts/'
    fname = f'{args.gait}_diffuser.hdf5'
    

    # ---------------------- Load Diffusion ----------------------
    agent.load(model_path)
    agent.eval()

    prior = torch.zeros((1, args.horizon, observation_dim + action_dim), device=args.device)

    data = reset_data()

    for episode in range(args.num_episodes):
        t = 0
        cum_rew = 0
        obs = env.reset()
        while t < int(num_eval_steps):
            dof = torch.cat([obs['obs'][:, 18:42], obs['obs'][:, 66:70]], dim=1)
            prior[:, 0, action_dim:] = dof
            trajectory, log = agent.sample(
                prior,
                solver=args.solver,
                n_samples = 1,
                sample_steps=args.sampling_steps,
                use_ema=args.use_ema, w_cg=args.w_cg, temperature=args.temperature)

            # sample actions and unnorm
            actions = trajectory[0, :, :action_dim] * 0.8 # (64,12)

            # step all actions
            for j in range(actions.shape[0]):
                obs, rew, done, info = env.step(actions[j].unsqueeze(0))
                action = torch.squeeze(actions[j]).detach().cpu().numpy()
                s = torch.squeeze(obs['obs']).detach().cpu().numpy()
                reward = rew.detach().cpu().numpy()
                reward = reward[0]
                done = done.detach().cpu().numpy()
                done = done[0]
                cum_rew += reward

                if t == num_eval_steps-1 or env.reset_buf:
                    done = True
                    append_data(data, s, action, reward, done, cum_rew)
                    
                    wandb.log({"cumulative_reward": cum_rew}, commit = True)
                    wandb.log({"episode average reward": cum_rew/num_eval_steps}, commit = True)
                    wandb.log({"real time x_vels": env.base_lin_vel[0, 0]}, commit = True)
                    print('episode:', episode, 't: ',t, 'done. cumulative reward: ', cum_rew)
                    
                    break
                else:
                    append_data(data, s, action, reward, done, cum_rew)
                    t += 1
            else:
                continue  # continue to the next episode if the inner loop wasn't broken
            break  # break the outer loop if the inner loop was broken
    
    # save the h5py file
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)
    dataset = h5py.File(file_path + fname, 'w')

    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')   
        

@hydra.main(config_path="/home/xinyi/CleanDiffuser/configs/diffuser/go1", config_name="organized_pipeline", version_base=None)
def finetune_diffuser(args):
    args.gait = gait_type
    for finetune_loop in range(args.finetune_loops):

        # ---------------------- prepare parameters ----------------------
        set_seed(args.seed)
        dataset_name = f'go1_walk_{args.gait}_ep2500'
        optimal_dataset_dir = f'/home/xinyi/CleanDiffuser/go1/hdf5_files/go1_walk_{args.gait}_ep2500.hdf5' # remain the same
        
        if finetune_loop == 0:
            key = 'diffuser'
            diffuser_save_dir = f'results/{rootpath}/models/{key}/diffusion_ckpt_latest.pt'
        else:
            key = f'finetune{finetune_loop}'
            diffuser_save_dir = f'results/{rootpath}/models/{key}/finetune_ckpt_latest.pt'

        sub_optimal_dataset_dir = f'results/{rootpath}/rollouts/{args.gait}_{key}.hdf5'
        finetune_save_path = f'results/{rootpath}/models/finetune{finetune_loop + 1}/'
        
        print(f'diffuser_save_dir: {diffuser_save_dir}')
        print(f'optimal_dataset_dir: {optimal_dataset_dir}')
        print(f'sub_optimal_dataset_dir: {sub_optimal_dataset_dir}')
        print(f'finetune_save_path: {finetune_save_path}')

        if os.path.exists(finetune_save_path) is False:
            os.makedirs(finetune_save_path)
        
        # ---------------------- Create Dataset ----------------------
        dataset = PrefScriptedSequenceDataset(
            env = dataset_name, 
            horizon = args.horizon, 
            query_len = args.query_len,
            query_number = args.query_number,
            optimal_dataset_dir = optimal_dataset_dir,
            sub_optimal_dataset_dir = sub_optimal_dataset_dir,
            termination_penalty = None,
            noise_level = args.noise_level,
            normalizer = args.normalizer, 
            max_path_length = args.max_path_length)

        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        
        observation_dim = dataset.observation_dim
        action_dim = dataset.action_dim 


        # --------------- Network Architecture -----------------
        nn_diffusion = JannerUNet1d(
            observation_dim + action_dim, model_dim=args.model_dim, emb_dim=args.model_dim, dim_mult=args.dim_mult,
            timestep_emb_type="positional", attention=False, kernel_size=5)
        print(f"======================= Parameter Report of Diffusion Model =======================")
        report_parameters(nn_diffusion)

        # ----------------- Masking -------------------
        fix_mask = torch.zeros((args.horizon, observation_dim + action_dim))
        fix_mask[0, action_dim:] = 1.
        loss_weight = torch.ones((args.horizon, observation_dim + action_dim))
        # loss_weight[:, :action_dim] = args.action_loss_weight


        # --------------- Diffusion Model --------------------
        agent = DiscreteDiffusionSDE(nn_diffusion, 
                                    None, 
                                    dataset = dataset,
                                    fix_mask=fix_mask, 
                                    loss_weight=loss_weight, 
                                    ema_rate=args.ema_rate,
                                    device=args.device, 
                                    diffusion_steps=args.diffusion_steps, 
                                    predict_noise=args.predict_noise,
                                    bc_coef = args.bc_coef,
                                    dpo_beta = args.dpo_beta,
                                    regulizer_lambda = args.regulizer_lambda)
    

        # ---------------------- Finetune ----------------------
        # preparation
        finetune_lr_scheduler = CosineAnnealingLR(agent.optimizer, args.finetune_gradient_steps)
        agent.load(diffuser_save_dir)
        
        agent.train()
        n_gradient_step = 0
        log = {"avg_finetune_loss": 0.}  

        for batch in loop_dataloader(dataloader):
            
            # ----------- Prepare the batch ------------
            batch = batch_to_device(batch, device = args.device)
            traj_w, traj_l = batch

            # ----------- Gradient Step ------------
            current_loss = agent.update_finetune(traj_w, traj_l)['finetune loss']  # agent:DiscreteDiffusionSDE
            log["avg_finetune_loss"] += current_loss  # BaseDiffusionSDE.update
            finetune_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.finetune_log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["avg_finetune_loss"] /= args.finetune_log_interval
                wandb.log({"avg_finetune_loss": log["avg_finetune_loss"]}, commit = True)
                print(log)
                log = {"avg_finetune_loss": 0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.finetune_save_interval == 0:
                agent.save(finetune_save_path + f"finetune_ckpt_{n_gradient_step + 1}.pt")
                agent.save(finetune_save_path + f"finetune_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.finetune_gradient_steps:
                break
        
        print(f'# ----------------- Rollouts Finetune {finetune_loop}-------------------#')
        

        # ---------------------- prepare dirs ----------------------
        model_path = f'results/{rootpath}/models/finetune{finetune_loop + 1}/finetune_ckpt_latest.pt'
        file_path = f'results/{rootpath}/rollouts/'
        fname = f'{args.gait}_finetune{finetune_loop + 1}.hdf5'

        # ---------------------- Load Diffusion ----------------------
        agent.load(model_path)
        agent.eval()

        prior = torch.zeros((1, args.horizon, observation_dim + action_dim), device=args.device)

        data = reset_data()

        for episode in range(args.num_episodes):
            t = 0
            cum_rew = 0
            obs = env.reset()
            while t < int(num_eval_steps):
                dof = torch.cat([obs['obs'][:, 18:42], obs['obs'][:, 66:70]], dim=1)
                prior[:, 0, action_dim:] = dof
                trajectory, log = agent.sample(
                    prior,
                    solver=args.solver,
                    n_samples = 1,
                    sample_steps=args.sampling_steps,
                    use_ema=args.use_ema, w_cg=args.w_cg, temperature=args.temperature)

                # sample actions and unnorm
                actions = trajectory[0, :, :action_dim] * 1.0 # (64,12)

                # step all actions
                for j in range(actions.shape[0]):
                    obs, rew, done, info = env.step(actions[j].unsqueeze(0))
                    action = torch.squeeze(actions[j]).detach().cpu().numpy()
                    s = torch.squeeze(obs['obs']).detach().cpu().numpy()
                    reward = rew.detach().cpu().numpy()
                    reward = reward[0]
                    done = done.detach().cpu().numpy()
                    done = done[0]
                    cum_rew += reward

                    if t == num_eval_steps-1 or env.reset_buf:
                        done = True
                        append_data(data, s, action, reward, done, cum_rew)
                        
                        wandb.log({"cumulative_reward": cum_rew}, commit = True)
                        wandb.log({"episode average reward": cum_rew/num_eval_steps}, commit = True)
                        wandb.log({"real time x_vels": env.base_lin_vel[0, 0]}, commit = True)
                        print('episode:', episode, 't: ',t, 'done. cumulative reward: ', cum_rew)
                        
                        break
                    else:
                        append_data(data, s, action, reward, done, cum_rew)
                        t += 1
                else:
                    continue  # continue to the next episode if the inner loop wasn't broken
                break  # break the outer loop if the inner loop was broken
        
        # save the h5py file
        if os.path.exists(file_path) is False:
            os.makedirs(file_path)
        dataset = h5py.File(file_path + fname, 'w')

        npify(data)
        for k in data:
            dataset.create_dataset(k, data=data[k], compression='gzip')   



if __name__ == "__main__":
    train_diffuser()
    # finetune_diffuser()      
