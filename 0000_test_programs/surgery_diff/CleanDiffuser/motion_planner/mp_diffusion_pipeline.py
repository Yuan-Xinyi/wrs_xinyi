import zarr
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
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters
from motion_planner_dataset import MotionPlanningDataset

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

'''load the config file'''
current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))
config_file = os.path.join(current_file_dir, 'config.yaml')
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

'''dataset loading'''
dataset_path = os.path.join(parent_dir, 'datasets/xarm_toppra_mp.zarr')

dataset = MotionPlanningDataset(dataset_path, horizon=config['horizon'], obs_keys=config['obs_keys'], 
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

from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import ChiUNet1d, JannerUNet1d
from cleandiffuser.diffusion.ddpm import DDPM

nn_condition = IdentityCondition(dropout=0.0).to(config['device'])
nn_diffusion = ChiUNet1d(
    config['action_dim'], config['obs_dim'], config['obs_steps'], model_dim=256, emb_dim=256, dim_mult=[1, 2, 2],
    obs_as_global_cond=True, timestep_emb_type="positional").to(config['device'])

print(f"======================= Parameter Report of Diffusion Model =======================")
report_parameters(nn_diffusion)
print(f"==============================================================================")

import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm_s
from wrs import wd, rm, mcm
robot_s = xarm_s.XArmLite6(enable_cc=True)

'''define the robot joint limits'''
jnt_v_max = rm.np.asarray([rm.pi * 2 / 3] * robot_s.n_dof)
jnt_config_range = robot_s.jnt_ranges
x_max = torch.zeros((1, config['horizon']-1, config['action_dim']), device=config['device'])
x_min = torch.zeros((1, config['horizon']-1, config['action_dim']), device=config['device'])

x_max[:, :, :6] = torch.tensor(jnt_config_range[:, 1], device=config['device']) 
x_min[:, :, :6] = torch.tensor(jnt_config_range[:, 0], device=config['device'])
x_max[:, :, -6:] = torch.tensor(jnt_v_max, device=config['device'])
x_min[:, :, -6:] = torch.tensor(-jnt_v_max, device=config['device'])

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

    wandb.init(project="mp_diff", name=rootpath)

    # ----------------- Training ----------------------
    agent.train()
    n_gradient_step = 0
    diffusion_loss_list = []
    log = {'avg_loss_diffusion': 0.}
    start_time = time.time()
    
    for batch in loop_dataloader(dataloader):
        # get condition
        nobs = batch['obs'].to(config['device'])
        naction = batch['action'].to(config['device'])
        
        condition = nobs[:, :config['obs_steps'], :]  # (B, obs_horizon, obs_dim)
        if config['nn'] == 'chi_unet' or config['nn'] == 'dit':
            condition = condition.flatten(start_dim=1)  # (B, obs_horizon*obs_dim)
        else:
            pass  # (B, obs_horizon, obs_dim)

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

elif config['mode'] == "inference":
    # ----------------- Inference ----------------------
    # model_path = '0000_test_programs/surgery_diff/CleanDiffuser/scripts_xarm/results/0312_1418_h16/diffusion_ckpt_latest.pt'
    # model_path = '0000_test_programs/surgery_diff/CleanDiffuser/scripts_xarm/results/0313_1012_h64/diffusion_ckpt_latest.pt'
    model_path = '0000_test_programs/surgery_diff/CleanDiffuser/scripts_xarm/results/0313_1142_h128/diffusion_ckpt_latest.pt'
    agent.load(model_path)
    agent.model.eval()
    agent.model_ema.eval()

    '''capture the image'''
    sys.path.append('/home/lqin/wrs_xinyi/wrs')
    import wrs.robot_con.xarm_lite6.xarm_lite6_x as xarm_x
    import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm_s
    import wrs.visualization.panda.world as wd
    from wrs import wd, rm, mcm
    import wrs.modeling.geometric_model as mgm
    import cv2
    
    # init
    robot_x = xarm_x.XArmLite6X(ip = '192.168.1.190')
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot_s = xarm_s.XArmLite6(enable_cc=True)
    rgb_camera = []
    cam_idx = config['camera_idx']
    rgb_camera.append(cv2.VideoCapture(cam_idx[0]))
    rgb_camera.append(cv2.VideoCapture(cam_idx[1]))

    # move to the starting point
    robot_x.move_j(config['start_jnt'])

    img = np.zeros((2, 96, 96, 3), dtype=np.uint8)

    for id, camera in enumerate(rgb_camera):
        ret, frame = camera.read()

        '''crop the image'''
        crop_size = config['crop_size']
        h, w, _ = frame.shape
        crop_center = config['crop_center']
        center_x, center_y = crop_center[id][0], crop_center[id][1]
        x1 = max(center_x - crop_size // 2, 0)
        x2 = min(center_x + crop_size // 2, w)
        y1 = max(center_y - crop_size // 2, 0)
        y2 = min(center_y + crop_size // 2, h)

        if id == 0:
            img[id] = frame[y1:y2, x1:x2]
        else:
            cropped = frame[y1:y2, x1:x2]
            img[id] = np.rot90(cropped, 2)
            
    concatenated_image = np.hstack(img)

    # inference
    solver = 'ddim'

    for step in range(132-8):
        print("Step:", step)
        obs_dict = {}
        pos, rot_mat = robot_s.fk(robot_x.get_jnt_values())
        rot_quat = rm.quaternion_from_rotmat(rot_mat)
        agent_pos = np.concatenate((pos, rot_quat))
        
        obs_dict['image'], obs_dict['agent_pos'] = np.moveaxis(concatenated_image, -1, 0) / 255, agent_pos
        for k in obs_dict.keys():
            obs_seq = obs_dict[k].astype(np.float32)
            nobs = dataset.normalizer['obs'][k].normalize(obs_seq)
            obs_dict[k] = nobs = torch.tensor(np.expand_dims(np.expand_dims(nobs, axis=0), axis=0), 
                                            device=config['device'], dtype=torch.float32)

        with torch.no_grad():
            condition = obs_dict
            prior = torch.zeros((1, config['horizon'], config['action_dim']), device=config['device'])
            naction, _ = agent.sample(prior=prior, n_samples=1, sample_steps=config['sample_steps'],
                                    solver=solver, condition_cfg=condition, w_cfg=1.0, use_ema=True)
        # unnormalize prediction
        naction = naction.detach().to('cpu').numpy()  # (num_envs, horizon, action_dim)
        action_pred = dataset.normalizer['action'].unnormalize(naction)  

        # get action
        start = config['obs_steps'] - 1
        end = start + config['action_steps']
        action = action_pred[:, start:end, :]  # (1,16,7)
        path = []

        for idx in range(action.shape[1]):
            tgt_pos = action[0, idx, :3]
            tgt_rotmat = rm.rotmat_from_quaternion(action[0, idx, 3:])
            jnt_values = robot_s.ik(tgt_pos, tgt_rotmat)
            path.append(jnt_values)
            
            # plot
            # print(jnt_values.shape)
            robot_s.goto_given_conf(jnt_values=jnt_values)
            arm_mesh = robot_s.gen_meshmodel(rgb=rm.const.steel_blue, alpha=0.02)
            arm_mesh.attach_to(base)


        for jnt_values in tqdm(path):
            # robot_x.move_j(jnt_values, speed=0.1)
            robot_x.move_j(jnt_values)



    
else:
    raise ValueError("Illegal mode")