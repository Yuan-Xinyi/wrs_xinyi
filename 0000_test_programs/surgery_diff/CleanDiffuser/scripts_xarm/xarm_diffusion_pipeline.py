import os
import sys
import warnings
warnings.filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import zarr
import wandb
import gym
import pathlib
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from datetime import datetime
import random

from torch.optim.lr_scheduler import CosineAnnealingLR
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters
from drillhole_image_dataset import DrillHoleImageDataset

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

'''load the config file'''
current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))
config_file = os.path.join(current_file_dir, 'gendata_config.yaml')
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

'''dataset loading'''
dataset_path = os.path.join(parent_dir, 'datasets/surgery.zarr')
# store = zarr.DirectoryStore(dataset_path)
# root = zarr.open(store, mode="r")

# agent_positions = root["agent_pos"][:]  # (N, 7)
# images = root["images"][:]  # (N, 96, 192, 3)

# print("Dataset Agent Pos Shape:", agent_positions.shape)
# print("Dataset Images Shape:", images.shape)

dataset = DrillHoleImageDataset(dataset_path, horizon=config['horizon'], obs_keys=config['obs_keys'], 
                                pad_before=config['obs_steps']-1, pad_after=config['action_steps']-1, abs_action=config['abs_action'])
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config["batch_size"],
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True
)
    
# --------------- Create Diffusion Model -----------------
set_seed(config['seed'])
assert config["nn"] == "chi_unet"
assert config['diffusion'] == "ddpm"

from cleandiffuser.nn_condition import MultiImageObsCondition
from cleandiffuser.nn_diffusion import ChiUNet1d
from cleandiffuser.diffusion.ddpm import DDPM

nn_condition = MultiImageObsCondition(
    shape_meta=config['shape_meta'], emb_dim=256, rgb_model_name=config['rgb_model'], resize_shape=config['resize_shape'],
    crop_shape=config['crop_shape'], random_crop=config['random_crop'], 
    use_group_norm=config['use_group_norm'], use_seq=config['use_seq']).to(config['device'])
nn_diffusion = ChiUNet1d(
    config['action_dim'], 256, config['obs_steps'], model_dim=256, emb_dim=256, dim_mult=[1, 2, 2],
    obs_as_global_cond=True, timestep_emb_type="positional").to(config['device'])
    

print(f"======================= Parameter Report of Diffusion Model =======================")
report_parameters(nn_diffusion)
print(f"==============================================================================")

x_max = torch.ones((1, config['horizon'], config['action_dim']), device=config['device']) * +1.0
x_min = torch.ones((1, config['horizon'], config['action_dim']), device=config['device']) * -1.0
agent = DDPM(
    nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=config['device'],
    diffusion_steps=config['sample_steps'], x_max=x_max, x_min=x_min,
    optim_params={"lr": config['lr']})

lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=config['gradient_steps'])

if config['mode'] == "train":
    # --------------- Data Loading -----------------
    '''prepare the save path'''
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f"{TimeCode}_h{config['horizon']}"
    current_file_dir = os.path.dirname(__file__)
    save_path = os.path.join(current_file_dir, 'results', rootpath)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project="surgery_diff_traj", name=rootpath)

    # ----------------- Training ----------------------
    agent.train()
    n_gradient_step = 0
    diffusion_loss_list = []
    log = {'avg_loss_diffusion': 0.}
    start_time = time.time()
    
    for batch in loop_dataloader(dataloader):
        # get condition
        nobs = batch['obs']
        condition = {}
        for k in nobs.keys():
            condition[k] = nobs[k][:, :config['obs_steps'], :].to(torch.float32).to(config['device'])

        naction = batch['action'].to(torch.float32).to(config['device'])

        # update diffusion
        diffusion_loss = agent.update(naction, condition)['loss']
        log["avg_loss_diffusion"] += diffusion_loss
        lr_scheduler.step()
        diffusion_loss_list.append(diffusion_loss)

        if n_gradient_step % config['log_freq'] == 0:
            log['gradient_steps'] = n_gradient_step
            log["avg_loss_diffusion"] /= config['log_freq']
            diffusion_loss_list = []
            wandb.log(
                {'step': log['gradient_steps'],
                'avg_training_loss': log['avg_loss_diffusion'],
                'total_time': time.time() - start_time}, commit = True)
            print(log)
            log = {"avg_loss_diffusion": 0.}
        
        if n_gradient_step % config['save_freq'] == 0:
            agent.save(save_path + f"/diffusion_ckpt_{n_gradient_step + 1}.pt")
            agent.save(save_path + f"/diffusion_ckpt_latest.pt")
        
        n_gradient_step += 1
        if n_gradient_step >= config['gradient_steps']:
            break
    wandb.finish()

# elif args.mode == "inference":
#     # ----------------- Inference ----------------------
#     if args.model_path:
#         agent.load(args.model_path)
#     else:
#         raise ValueError("Empty model for inference")
#     agent.model.eval()
#     agent.model_ema.eval()

#     metrics = {'step': 0}
#     metrics.update(inference(args, envs, dataset, agent, logger))
#     logger.log(metrics, category='inference')
    
else:
    raise ValueError("Illegal mode")