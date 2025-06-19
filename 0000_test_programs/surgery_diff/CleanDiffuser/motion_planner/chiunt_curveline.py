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
from ruckig_dataset import CurveLineDataset
from torch.utils.data import random_split
from generate_complex_trajectory import ComplexTrajectoryGenerator

def AttachTraj2base(path, radius=0.005):
    for i in range(path.shape[0] - 1):
        spos = path[i]
        epos = path[i + 1]
        stick_sgm = mgm.gen_stick(spos=spos, epos=epos, radius=radius)
        stick_sgm.attach_to(base)


'''if want to test the directory'''
# import os
# print(os.getcwd())
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

'''load the config file'''
current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))
config_file = os.path.join(current_file_dir,'config', 'curveline_config.yaml')
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

'''dataset loading'''
dataset_path = os.path.join('/home/lqin', 'zarr_datasets', config['dataset_name'])

dataset = CurveLineDataset(dataset_path, horizon=config['horizon'], obs_keys=config['obs_keys'], 
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

from cleandiffuser.nn_condition import IdentityCondition, MLPCondition
from cleandiffuser.nn_diffusion import ChiUNet1d
from cleandiffuser.diffusion.ddpm import DDPM

# --------------- Network Architecture -----------------
nn_diffusion = ChiUNet1d(
    config['action_dim'], config['obs_dim'], config['obs_steps'], model_dim=256, emb_dim=256, dim_mult=config['dim_mult'],
    obs_as_global_cond=True, timestep_emb_type="positional").to(config['device'])

if config['condition'] == "identity":
    nn_condition = IdentityCondition(dropout=0.0).to(config['device'])
    print("Using Identity Condition")
elif config['condition'] == "mlp":
    nn_condition = MLPCondition(
        in_dim=16*3, out_dim=64, hidden_dims=256, dropout=0.0).to(config['device'])
    print("Using MLP Condition")
else:
    nn_condition = None
    print("Using No Condition")


print(f"======================= Parameter Report of Diffusion Model =======================")
report_parameters(nn_diffusion)


import wrs.robot_sim.robots.xarmlite6_wg.x6wg2 as xarm_s
from wrs import wd, rm, mcm
robot_s = xarm_s.XArmLite6WG2(enable_cc=True)

'''define the robot joint limits'''
if config['normalize']:
    x_max = torch.ones((1, config['horizon'], config['action_dim']), device=config['device']) * +1.0
    x_min = torch.ones((1, config['horizon'], config['action_dim']), device=config['device']) * -1.0
    print('*'*100)
    print("Using Normalized Action Space. the action space is normalized to [-1, 1]")
    print('*'*100)
else:
    jnt_config_range = torch.tensor(robot_s.jnt_ranges, device=config['device'])
    x_max = (jnt_config_range[:,1]).repeat(1, config['horizon'], 1)
    x_min = (jnt_config_range[:,0]).repeat(1, config['horizon'], 1)
    print('*'*50)
    print("Using Absolute Action Space. the action space is absolute joint configuration")
    print('*'*50)

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

    wandb.init(project="curve_line", name=rootpath)

    # ----------------- Training ----------------------
    agent.train()
    n_gradient_step = 0
    diffusion_loss_list = []
    log = {'avg_loss_diffusion': 0.}
    start_time = time.time()
    
    for batch in loop_dataloader(train_loader):
        # get condition
        if config['condition'] == "identity":
            condition = batch['coef_cond'].to(config['device'])
            condition = condition.flatten(start_dim=1) # (batch,14)
        elif config['condition'] == "mlp":
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
    model_path = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/' \
    'results/0617_1913gostraight_h16_b64_normTrue/diffusion_ckpt_latest.pt'
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
    testing_num = 1000
    success = 0
    
    '''diffusion model inference'''
    # for id in tqdm(range(testing_num)):
    #     '''randomly generate a complex trajectory'''
    #     pos_list = []
    #     while len(pos_list) < 32:
    #         init_jnt = robot_s.rand_conf()
    #         pos_init, rotmat_init = robot_s.fk(jnt_values=init_jnt)
    #         scale = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
    #         # scale = np.random.choice([0.1, 0.2, 0.3])
    #         import mp_datagen_curveline as datagen
    #         workspace_points, coeffs = datagen.generate_random_cubic_curve(num_points=32, scale=scale, center=pos_init)
            
    #         pos_list, success_count = datagen.gen_jnt_list_from_pos_list(init_jnt=init_jnt,
    #             pos_list=workspace_points, robot=robot_s, obstacle_list=None, base=base,
    #             max_try_time=5.0, check_collision=True, visualize=False
    #         )

    #     '''inference the trajectory'''
    #     trajectory_window = torch.tensor(workspace_points[:config['horizon']]).to(config['device'])
    #     if config['condition'] == "mlp":
    #         condition = torch.tensor(trajectory_window, device=config['device']).unsqueeze(0).float()
    #         condition = condition.flatten(start_dim=1)
    #     prior = torch.zeros((n_samples, config['horizon'], config['action_dim']), device=config['device'])
    #     if n_samples != 1:
    #         condition = condition.repeat(n_samples, 1)
    #     with torch.no_grad():
    #         action, _ = agent.sample(prior=prior, n_samples=n_samples, sample_steps=config['sample_steps'], temperature=1.0,
    #                                 solver=solver, condition_cfg=condition, w_cfg = 1.0, use_ema=True)
    #     if config['normalize']:
    #         action = dataset.normalizer['obs']['jnt_pos'].unnormalize(np.array(action.to('cpu')))
    #     jnt_seed = action[0, 0, :]
    #     _, rot = robot_s.fk(jnt_values=jnt_seed)
    #     adjusted_jnt = robot_s.ik(tgt_pos=workspace_points[0], tgt_rotmat=rot, seed_jnt_values=jnt_seed)
    #     if adjusted_jnt is None:
    #         print("IK failed at initial position:", workspace_points[0])
        
    #     path = [adjusted_jnt] if adjusted_jnt is not None else []
    #     for _ in workspace_points:
    #         current_pos = np.array(_)
    #         adjusted_jnt = robot_s.ik(tgt_pos=np.array(_), tgt_rotmat=rot, 
    #                         seed_jnt_values=path[-1] if path else jnt_seed)
    #         if adjusted_jnt is not None:
    #             path.append(adjusted_jnt)
    #         else:
    #             print("IK failed at position:", current_pos)
    #             break
    #     if len(path) == 33:
    #         print(f"Trajectory {id} generated successfully.")
    #         success += 1
    
    # print(f"Total {testing_num} trajectories generated, {success} successful.")
    # print(f"Success rate: {success / testing_num * 100:.2f}%")

    '''randomly generate a complex trajectory'''
    pos_list = []
    traj_length = 16
    while len(pos_list) < 16:
        init_jnt = robot_s.rand_conf()
        pos_init, rotmat_init = robot_s.fk(jnt_values=init_jnt)
        # scale = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
        scale = np.random.choice([0.1, 0.2, 0.3])
        import mp_datagen_curveline as datagen
        workspace_points, coeffs = datagen.generate_random_cubic_curve(num_points=16, scale=scale, center=pos_init)
        
        pos_list, success_count = datagen.gen_jnt_list_from_pos_list(init_jnt=init_jnt,
            pos_list=workspace_points, robot=robot_s, obstacle_list=None, base=base,
            max_try_time=5.0, check_collision=True, visualize=False
        )
    # trajectory_generator = ComplexTrajectoryGenerator(num_segments=1, num_points_per_segment=32, 
    #                                                   scale_range=(0.1,0.3), center_init=init_pos)
    # test_trajectory = trajectory_generator.generate()
    AttachTraj2base(workspace_points, radius=0.0025)

    robot_s.goto_given_conf(pos_list[0])
    robot_s.gen_meshmodel(rgb=[0,1,0], alpha=0.2).attach_to(base)
    robot_s.goto_given_conf(pos_list[-1])
    robot_s.gen_meshmodel(rgb=[0,1,0], alpha=0.2).attach_to(base)

    '''inference the trajectory'''
    visualization = 'dynamic' # 'static' or 'dynamic'

    trajectory_window = torch.tensor(workspace_points[:config['horizon']]).to(config['device'])
    if config['condition'] == "mlp":
        condition = torch.tensor(trajectory_window, device=config['device']).unsqueeze(0).float()
        condition = condition.flatten(start_dim=1)
    prior = torch.zeros((n_samples, config['horizon'], config['action_dim']), device=config['device'])
    if n_samples != 1:
        condition = condition.repeat(n_samples, 1)
    with torch.no_grad():
        action, _ = agent.sample(prior=prior, n_samples=n_samples, sample_steps=config['sample_steps'], temperature=1.0,
                                solver=solver, condition_cfg=condition, w_cfg = 1.0, use_ema=True)
    if config['normalize']:
        action = dataset.normalizer['obs']['jnt_pos'].unnormalize(np.array(action.to('cpu')))
    jnt_seed = action[0, 0, :]
    if visualization == 'static':
        robot_s.goto_given_conf(jnt_seed)
        robot_s.gen_meshmodel(rgb=[1,0,0], alpha=0.2).attach_to(base)
    _, rot = robot_s.fk(jnt_values=jnt_seed)
    adjusted_jnt = robot_s.ik(tgt_pos=workspace_points[0], tgt_rotmat=rot, seed_jnt_values=jnt_seed)
    if adjusted_jnt is not None:
        if visualization == 'static':
            robot_s.goto_given_conf(adjusted_jnt)
            robot_s.gen_meshmodel(rgb=[0,0,1], alpha=0.2).attach_to(base)
    else:
        print("IK failed at initial position:", workspace_points[0])
    
    path = [adjusted_jnt] if adjusted_jnt is not None else []
    for _ in workspace_points[1:]:
        current_pos = np.array(_)
        adjusted_jnt = robot_s.ik(tgt_pos=np.array(_), tgt_rotmat=rot, 
                        seed_jnt_values=path[-1] if path else jnt_seed)
        if adjusted_jnt is not None:
            path.append(adjusted_jnt)
            if visualization == 'static':
                robot_s.goto_given_conf(adjusted_jnt)
                robot_s.gen_meshmodel(rgb=[0,0,1], alpha=0.2).attach_to(base)
        else:
            print("IK failed at position:", current_pos)
            break
    if len(path) == traj_length:
        print(f"Trajectory generated successfully.")
    else:
        print(f"Trajectory generation failed.")
        print(f"Generated {len(path)} waypoints, expected {traj_length} waypoints.")
    import helper_functions as hf
    if visualization == 'dynamic':
        hf.visualize_anime_path(base, robot_s, path)
    else:
        base.run()

else:
    raise ValueError("Illegal mode")