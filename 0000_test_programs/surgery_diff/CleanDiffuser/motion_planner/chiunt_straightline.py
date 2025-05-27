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
from ruckig_dataset import StraightLineDataset
from torch.utils.data import random_split

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def initjnt2jntpath(robot, base, jnt_seed, pos_s, pos,g, waypoint_interval=0.1, max_try_time=5,
                   check_collision=True, obstacle_list=None, visualize=False):
    """
    Initialize a joint path from jnt_seed to jnt_goal.
    """
    jnt_path = []
    jnt_path.append(jnt_seed)

    start_time = time.time()
    while jnt is None and time.time() - start_time < max_try_time:
        try:
            rotmat = rm.rotmat_from_euler(np.pi/2,0,0)
            j = robot.ik(tgt_pos=pos, tgt_rotmat=rotmat, seed_jnt_values = jnt_list[-1] if jnt_list else None)
            if j is None:
                continue
            robot.goto_given_conf(j)
            if check_collision and robot.cc.is_collided(obstacle_list=obstacle_list):
                continue
            jnt = j
            success_count += 1
            if visualize:
                mcm.mgm.gen_frame(pos=pos, rotmat=rotmat).attach_to(base)
                robot.gen_meshmodel(alpha=.2).attach_to(base)
            break
        except:
            break

    return jnt_path

'''load the config file'''
current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))
config_file = os.path.join(current_file_dir,'config', 'straightline_config.yaml')
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

'''dataset loading'''
dataset_path = os.path.join('/home/lqin', 'zarr_datasets', config['dataset_name'])

dataset = StraightLineDataset(dataset_path, horizon=config['horizon'], obs_keys=config['obs_keys'], 
                         normalize=config['normalize'], abs_action=config['abs_action'])
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
assert config["nn"] == "chi_unet"
assert config['diffusion'] == "ddpm"

from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import ChiUNet1d
from cleandiffuser.diffusion.ddpm import DDPM

# --------------- Network Architecture -----------------
nn_diffusion = ChiUNet1d(
    config['action_dim'], config['obs_dim'], config['obs_steps'], model_dim=256, emb_dim=256, dim_mult=config['dim_mult'],
    obs_as_global_cond=True, timestep_emb_type="positional").to(config['device'])

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

'''define the robot joint limits'''
jnt_config_range = torch.tensor(robot_s.jnt_ranges, device=config['device'])

x_max = (jnt_config_range[:,1]).repeat(1, config['horizon'], 1)
x_min = (jnt_config_range[:,0]).repeat(1, config['horizon'], 1)

loss_weight = torch.ones((config['horizon'], config['action_dim']))
loss_weight[0, :] = config['action_loss_weight']

agent = DDPM(
    nn_diffusion=nn_diffusion, nn_condition=nn_condition, loss_weight=loss_weight,
    device=config['device'], diffusion_steps=config['diffusion_steps'], x_max=x_max, x_min=x_min,
    optim_params={"lr": config['lr']}, predict_noise=config['predict_noise'])

lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=config['gradient_steps'])

if config['mode'] == "train":
    # --------------- Data Loading -----------------
    '''prepare the save path'''
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f"{TimeCode}gostraight_h{config['horizon']}_b{config['batch_size']}_norm{config['normalize']}"
    save_path = os.path.join(current_file_dir, 'results', rootpath)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project="straight_line", name=rootpath)

    # ----------------- Training ----------------------
    agent.load('0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/results/' \
    '0525_1950gostraight_h16_b64_normTrue/diffusion_ckpt_latest.pt')
    agent.train()
    n_gradient_step = 0
    diffusion_loss_list = []
    log = {'avg_loss_diffusion': 0.}
    start_time = time.time()
    
    for batch in loop_dataloader(train_loader):
        # get condition
        if config['condition'] == "identity":
            condition = batch['condition'].to(config['device'])
            condition = condition.flatten(start_dim=1) # (batch,14)
        else:
            condition = None
        action = batch['jnt_pos'].to(config['device']) # (batch,horizon,7)
        
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
    from scipy.interpolate import make_lsq_spline, BSpline

    model_path = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/results/' \
    '0525_1950gostraight_h16_b64_normTrue/diffusion_ckpt_latest.pt'
    agent.load(model_path)
    agent.model.eval()
    agent.model_ema.eval()

    '''capture the image'''
    sys.path.append('/home/lqin/wrs_xinyi/wrs')
    import wrs.visualization.panda.world as wd
    from wrs import wd, rm, mcm
    import wrs.modeling.geometric_model as mgm
    import copy
    
    # init
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    # inference
    solver = config['inference_solver']
    inference_steps = config['inference_steps']
    n_samples = 1

    '''single traj test from the dataset'''
    # root = zarr.open(dataset_path, mode='r')
    # start_list = []
    # goal_list = []
    # traj_id = 22
    # traj_end = int(root['meta']['episode_ends'][traj_id])
    # workspc_pos = root['data']['position'][:traj_end]
    # jnt_pos = root['data']['jnt_pos'][:traj_end]
    # jnt_seed = jnt_pos[0]
    # pos_start = workspc_pos[0]
    # pos_goal = workspc_pos[-1]
    
    # --------------- Inference -----------------
    '''fixed sample used in the mid-term presentation'''
    # for traj_id in tqdm(range(len(start_list))):
    for traj_id in range(1):
        '''prepare the start and goal config'''
        condition = torch.tensor([-0.2569, -0.1540,  0.7276, -0.1069, -0.1540,  0.7276], device=config['device']).unsqueeze(0).float()  # (1, 6)
        robot_s.goto_given_conf([-1.7201, -1.4743, -0.4602, -2.5136, -0.5191, 2.5270, 0.3964])
        robot_s.gen_meshmodel(rgb = [0,1,0], alpha=0.3).attach_to(base)
        
        pos, rot = robot_s.fk(jnt_values=[-1.7201, -1.4743, -0.4602, -2.5136, -0.5191, 2.5270, 0.3964], toggle_jacobian=True, update=True)
        mgm.gen_arrow(spos=np.array([-0.2569, -0.1540,  0.7276]), 
                      epos=np.array([-0.1069, -0.1540,  0.7276]), stick_radius=.005, rgb=[1,0,0]).attach_to(base)

        '''inference the trajectory'''
        prior = torch.zeros((n_samples, config['horizon'], config['action_dim']), device=config['device'])
        if n_samples != 1:
            condition = condition.repeat(n_samples, 1)
        with torch.no_grad():
            action, _ = agent.sample(prior=prior, n_samples=n_samples, sample_steps=config['sample_steps'], temperature=1.0,
                                    solver=solver, condition_cfg=condition, w_cfg = 1.0, use_ema=True)
        for i in range(n_samples):
            pred_jnt_seed = action[i,0,:].cpu().numpy()
            # pred_jnt_seed = [-1.7201, -1.4743, -0.4602, -2.5136, -0.5191, 2.5270, 0.3964]
            print(f"Predicted joint seed: {repr(pred_jnt_seed)}")
            robot_s.goto_given_conf(pred_jnt_seed)
            if n_samples != 1:
                robot_s.gen_meshmodel(rgb = None, alpha=0.2).attach_to(base)
            else:
                jnt_list = [pred_jnt_seed]
                pos = [-0.2569, -0.1540,  0.7276]
                _, rot = robot_s.fk(jnt_values=pred_jnt_seed, toggle_jacobian=True, update=True)
                for _ in range(200):
                    pos[0] += 0.01
                    res = robot_s.ik(tgt_pos=pos, tgt_rotmat=rot, seed_jnt_values=jnt_list[-1])
                    if res is None:
                        print(f"IK failed at sample {_},...")
                        break
                    else:
                        jnt_list.append(res)
                for idx in [1,-1]:
                    robot_s.goto_given_conf(jnt_list[idx])
                    robot_s.gen_meshmodel(rgb = [0,0,1], alpha=0.2).attach_to(base)
                
                pos,_ = robot_s.fk(jnt_values=jnt_list[-1])
                print(f'Predicted joint length: {len(jnt_list)}, last position: {pos}')
                print(f'distance ')
        base.run()

    '''random traj test'''
    # # generate the random condition
    # gth_jnt_seed = robot_s.rand_conf()
    # pos, rot = robot_s.fk(jnt_values=gth_jnt_seed)
    # pos_start = copy.deepcopy(pos)  # start position
    # jnt_list = [gth_jnt_seed]
    
    # axis = random.choice(['x', 'y', 'z'])
    # axis_map = {'x': 0, 'y': 1, 'z': 2}
    # axis_idx = axis_map[axis]
    
    # for _ in range(200):
    #     pos[axis_idx] += 0.01
    #     jnt = robot_s.ik(tgt_pos=pos, tgt_rotmat=rot, seed_jnt_values=jnt_list[-1])
    #     if jnt is not None:
    #         jnt_list.append(jnt)
    #     else:
    #         print('-' * 40)
    #         print(f"IK failed at sample {_},...")
    #         break
    # print(f"Generated {len(jnt_list)} joint configurations.")
    # pos_goal = copy.deepcopy(robot_s.fk(jnt_values=jnt_list[-1])[0])

    # '''report the parameters'''
    # print('=' * 80)
    # print(f"Start position: {pos_start}, Goal position: {pos_goal}")
    # print(f"Move axis: {axis}, Move distance: {pos[axis_idx] - pos_start[axis_idx]}m")
    # print(f"Total {len(jnt_list)} joint configurations sampled.")
    # print('=' * 80)


    # '''prepare the start and goal config'''
    # condition = np.concatenate([pos_start, pos_goal], axis=0)  # (6,)
    # condition = torch.tensor(condition, device=config['device']).unsqueeze(0).float()  # (1, 6)
    # robot_s.goto_given_conf(gth_jnt_seed)
    # robot_s.gen_meshmodel(rgb = [0,1,0], alpha=0.3).attach_to(base)

    # if axis == 'x':
    #     rgb = [1, 0, 0]
    # elif axis == 'y':
    #     rgb = [0, 1, 0]
    # elif axis == 'z':
    #     rgb = [0, 0, 1]
    # else:
    #     raise ValueError("Illegal axis, should be one of ['x', 'y', 'z']")
    # mgm.gen_arrow(spos=np.array(pos_start), 
    #                 epos=np.array(pos_goal), stick_radius=.005, rgb=rgb).attach_to(base)

    # '''inference the trajectory'''
    # prior = torch.zeros((n_samples, config['horizon'], config['action_dim']), device=config['device'])
    # if n_samples != 1:
    #     condition = condition.repeat(n_samples, 1)
    # with torch.no_grad():
    #     action, _ = agent.sample(prior=prior, n_samples=n_samples, sample_steps=config['sample_steps'], temperature=1.0,
    #                             solver=solver, condition_cfg=condition, w_cfg = 1.0, use_ema=True)
    # for i in range(n_samples):
    #     pred_jnt_seed = action[i,0,:].cpu().numpy()
    #     print(f"Predicted joint seed: {repr(pred_jnt_seed)}")
    #     robot_s.goto_given_conf(pred_jnt_seed)
    #     if n_samples != 1:
    #         robot_s.gen_meshmodel(rgb = None, alpha=0.2).attach_to(base)
    #     else:
    #         jnt_list = [pred_jnt_seed]
    #         robot_s.goto_given_conf(pred_jnt_seed)
    #         robot_s.gen_meshmodel(rgb = [1,0,0], alpha=0.1).attach_to(base)
    #         pos = copy.deepcopy(pos_start)
    #         _, rot = robot_s.fk(jnt_values=pred_jnt_seed, toggle_jacobian=True, update=True)
    #         for _ in range(200):
    #             pos[axis_idx] += 0.01
    #             res = robot_s.ik(tgt_pos=pos, tgt_rotmat=rot, seed_jnt_values=jnt_list[-1])
    #             if res is None:
    #                 print(f"IK failed at sample {_},...")
    #                 break
    #             else:
    #                 jnt_list.append(res)
    #         for idx in [1,-1]:
    #             robot_s.goto_given_conf(jnt_list[idx])
    #             robot_s.gen_meshmodel(rgb = [0,0,1], alpha=0.2).attach_to(base)

    #         pos,_ = robot_s.fk(jnt_values=jnt_list[-1])
    #         print(f'Predicted joint length: {len(jnt_list)}, last position: {pos}')
    # base.run()
else:
    raise ValueError("Illegal mode")