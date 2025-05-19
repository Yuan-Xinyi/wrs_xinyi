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
import matplotlib.pyplot as plt
import json

from torch.optim.lr_scheduler import CosineAnnealingLR
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters
from ruckig_dataset import PolynomialDataset
from torch.utils.data import random_split

def pad_to_power_of_two(x):
    """
    将输入张量在第二个维度填充到最近的 2^n
    """
    num_dim = x.shape[1]
    if num_dim & (num_dim - 1) == 0:
        # 已经是 2^n，直接返回
        return x
    else:
        # 计算下一个 2^n
        next_power_of_two = 2 ** (num_dim - 1).bit_length()
        padding_size = next_power_of_two - num_dim
        
        # 在第二个维度进行 padding
        x_padded = torch.nn.functional.pad(x, (0, 0, 0, padding_size), mode='constant', value=0)
        return x_padded

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

'''load the config file'''
current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))
config_file = os.path.join(current_file_dir,'config', 'ruckig_polynomial_config.yaml')
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

'''dataset loading'''
dataset_path = os.path.join('/home/lqin', 'zarr_datasets', config['dataset_name'])

dataset = PolynomialDataset(dataset_path, horizon=config['horizon'], obs_keys=config['obs_keys'],
                            poly_coef_range=config['poly_coef_range'], normalize=config['normalize'], abs_action=config['abs_action'])
print('dataset loaded in:', dataset_path)
if config['mode'] == "train":
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

# --------------- Create Diffusion Model -----------------
if config['mode'] == "train":
    set_seed(config['seed'])
assert config['diffusion'] == "ddpm"

from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import ChiUNet1d, ChiTransformer
from cleandiffuser.diffusion.ddpm import DDPM

# --------------- Network Architecture -----------------
if config['nn'] == "chi_unet":
    from cleandiffuser.nn_diffusion import ChiUNet1d
    nn_diffusion = ChiUNet1d(
        config['action_dim'], config['obs_dim'], config['obs_steps'], model_dim=256, emb_dim=256, dim_mult=config['dim_mult'],
        obs_as_global_cond=True, timestep_emb_type="positional").to(config['device'])
elif config['nn'] == "chi_transformer":
    from cleandiffuser.nn_diffusion import ChiTransformer
    '''for small model'''
    # nn_diffusion = ChiTransformer(
    #         config['action_dim'],config['obs_dim'], config['horizon'], config['obs_steps'], d_model=256, nhead=4, num_layers=4,
    #         timestep_emb_type="positional").to(config['device'])
    '''for large model'''
    nn_diffusion = ChiTransformer(
            config['action_dim'],config['obs_dim'], config['horizon'], config['obs_steps'], d_model=320, nhead=10, num_layers=8,
            timestep_emb_type="positional").to(config['device'])
elif config['nn'] == "dit":
    from cleandiffuser.nn_diffusion import DiT1d
    nn_diffusion = DiT1d(
                config['action_dim'], emb_dim=128, d_model=320, n_heads=10, depth=2, timestep_emb_type="fourier").to(config['device'])

if config['condition'] == "identity":
    nn_condition = IdentityCondition(dropout=0.0).to(config['device'])
    print("Using Identity Condition")
else:
    nn_condition = None
    print("Using No Condition")


print(f"======================= Parameter Report of Diffusion Model =======================")
report_parameters(nn_diffusion)


import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from wrs import wd, rm, mcm
robot_s = franka.FrankaResearch3(enable_cc=True)

agent = DDPM(
    nn_diffusion=nn_diffusion, nn_condition=nn_condition,
    device=config['device'], diffusion_steps=config['diffusion_steps'],
    optim_params={"lr": config['lr']}, predict_noise=config['predict_noise'])

lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=config['gradient_steps'])

if config['mode'] == "train":
    # --------------- Data Loading -----------------
    '''prepare the save path'''
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f"{TimeCode}_{config['horizon']}h_norm{config['normalize']}_poly_{config['nn']}"
    save_path = os.path.join(current_file_dir, 'results', rootpath)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project="ruckig_poly", name=rootpath)

    # ----------------- Training ----------------------
    agent.train()
    n_gradient_step = 0
    diffusion_loss_list = []
    log = {'avg_loss_diffusion': 0.}
    start_time = time.time()
    
    for batch in loop_dataloader(train_loader):
        # get condition
        if config['condition'] == "identity":
            condition = torch.stack([batch['start_conf'], batch['goal_conf']], dim=1)
            if config['nn'] == 'chi_unet' or config['nn'] == 'dit':
                condition = condition.flatten(start_dim=1)
        else:
            condition = None
        condition = condition.to(config['device'])
        action = batch['poly_coef'].to(config['device']) # (batch,horizon,7)
        
        # update diffusion
        diffusion_loss = agent.update(action, condition)['loss']
        log["avg_loss_diffusion"] += diffusion_loss
        lr_scheduler.step()
        diffusion_loss_list.append(diffusion_loss)

        if n_gradient_step != 0 and n_gradient_step % config['log_freq'] == 0:
            log['gradient_steps'] = n_gradient_step
            log["avg_loss_diffusion"] /= config['log_freq']
            diffusion_loss_list = []
            wandb.log(
                {'step': log['gradient_steps'],
                'avg_training_loss': log['avg_loss_diffusion'],
                'total_time': time.time() - start_time}, commit = True)
            print(log)
            log = {"avg_loss_diffusion": 0.}
        
        if n_gradient_step != 0 and n_gradient_step % config['save_freq'] == 0:
            agent.save(save_path + f"/diffusion_ckpt_{n_gradient_step}.pt")
            agent.save(save_path + f"/diffusion_ckpt_latest.pt")
        
        n_gradient_step += 1
        if n_gradient_step >= config['gradient_steps']:
            break
    wandb.finish()

if config['mode'] == "inference":
    raise NotImplementedError("Inference mode is not implemented yet")
    from scipy.interpolate import make_lsq_spline, BSpline

    model_path = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/results/' \
    '0512_2022_64h_64b_normFalse/diffusion_ckpt_latest.pt'
    agent.load(model_path)
    agent.model.eval()
    agent.model_ema.eval()

    '''capture the image'''
    sys.path.append('/home/lqin/wrs_xinyi/wrs')
    import wrs.visualization.panda.world as wd
    from wrs import wd, rm, mcm
    import wrs.modeling.geometric_model as mgm
    import mp_datagen_ruckig_obstacle as mp_helper
    import copy
    
    # init
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    # init the b-spline parameter
    degree = 4
    num_ctrl_pts = 64
    ctrl_points = np.linspace(0, 1, num_ctrl_pts)
    knots = np.linspace(0, 1, num_ctrl_pts - degree + 1)
    knots = np.concatenate(([0] * degree, knots, [1] * degree))

    # inference
    solver = config['inference_solver']
    inference_steps = config['inference_steps']
    visualization = config['visualization']
    
    success_num = 0
    n_samples = 1

    root = zarr.open(dataset_path, mode='r')
    start_list = []
    goal_list = []

    '''if get the start and goal from the dataset'''
    # for id in tqdm(range(root['meta']['episode_ends'].shape[0]-1)):
    #     traj_end = int(root['meta']['episode_ends'][id])
    #     control_points = root['data']['control_points'][traj_end-64:traj_end]
    #     # print(f"current traj start: {traj_start}, end: {traj_end}")
    #     start_list.append(control_points[0])
    #     goal_list.append(control_points[-1])

    '''single traj test'''
    root = zarr.open('/home/lqin/zarr_datasets/franka_ruckig_100hz_bspline.zarr', mode='r')
    traj_id = 0
    traj_end = int(root['meta']['episode_ends'][traj_id])
    control_points = root['data']['control_points'][traj_end-64:traj_end]
    
    # --------------- Inference -----------------
    # for traj_id in tqdm(range(len(start_list))):
    for traj_id in range(1):
        '''prepare the start and goal config'''
        # start_conf, goal_conf = start_list[traj_id], goal_list[traj_id]
        start_conf = control_points[0]
        goal_conf = control_points[-1]
        # start_conf = robot_s.rand_conf()
        # goal_conf = robot_s.rand_conf()
        # start_conf = np.array([ 1.00625858,  0.15694599, -2.51097176, -1.60473395,  0.40952758,
        # 3.05204021,  2.61735282])
        # goal_conf = np.array([ 0.61532624,  0.1395668 ,  2.00860207, -2.93296071,  1.4677425 ,
        # 1.3641184 , -2.99075278])
        print('**'*100)
        print(f"Start Conf: {start_conf}, Goal Conf: {goal_conf}")
        tgt_pos, tgt_rotmat = robot_s.fk(jnt_values=goal_conf)
        
        '''simulate the robot with the start and goal config'''
        robot_s.goto_given_conf(jnt_values=goal_conf)
        robot_s.gen_meshmodel(alpha=0.2, rgb=[0,1,0]).attach_to(base)
        robot_s.goto_given_conf(jnt_values=start_conf)
        robot_s.gen_meshmodel(alpha=0.2, rgb=[0,0,1]).attach_to(base)

        '''inference the trajectory'''
        condition = None
        prior = torch.zeros((1, config['horizon'], config['action_dim']), device=config['device'])
        prior[:, 0, :] = torch.tensor(start_conf).to(config['device'])
        prior[:, -1, :] = torch.tensor(goal_conf).to(config['device'])
        with torch.no_grad():
            action, _ = agent.sample(prior=prior, n_samples=n_samples, sample_steps=config['sample_steps'], temperature=0.1,
                                    solver=solver, condition_cfg=condition, use_ema=True)
        

        # '''recons b-spline'''
        # # spline = BSpline(knots, control_points, degree)
        # spline = BSpline(knots, action[0].cpu().numpy().copy(), degree)
        # print(repr(action[0].cpu().numpy().copy()))
        T_total_list = [5]
        # results = []

        # for T_total_new in T_total_list:
        #     print(f"\nTesting with T_total = {T_total_new}s")
        #     result = (T_total_new, *update_bspline_with_new_time(spline, T_total_new))
        #     results.append(result)

        # plot_bspline(results, robot_s.n_dof, overlay=True)
        # plot_control_points_comparison(control_points, action[0].cpu().numpy().copy(), robot_s.n_dof)
        
        '''compare the action with the control points'''
        generated_c = action[0].cpu().numpy().copy()
        generated_c[:4] = control_points[:4]
        generated_c[-4:] = control_points[-4:]
        compare_bspline_two_methods(knots, control_points, generated_c, 
                                    degree, T_total_list, robot_s.n_dof)
else:
    raise ValueError("Illegal mode")