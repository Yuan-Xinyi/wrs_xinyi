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

from torch.optim.lr_scheduler import CosineAnnealingLR
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters
from ruckig_dataset import MotionPlanningDataset, ObstaclePlanningDataset
from torch.utils.data import random_split

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def validate(agent, val_dataloader, device):
    agent.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            x = batch['trajectory'].to(device)
            val_loss = agent.loss(x)
            total_val_loss += val_loss.item()
    return total_val_loss / len(val_dataloader)

# def gen_collision_free_start_goal(robot):
#     '''generate the start and goal conf'''
#     while True:
#         start_conf = robot.rand_conf()
#         goal_conf = robot.rand_conf()
#         robot.goto_given_conf(jnt_values=start_conf)
#         start_cc = robot.cc.is_collided()
#         robot.goto_given_conf(jnt_values=goal_conf)
#         goal_cc = robot.cc.is_collided()
#         if not start_cc and not goal_cc:
#             break
#     return start_conf, goal_conf

def plot_details(robot_s, jnt_pos_list, jnt_vel_list, jnt_acc_list):
    sampling_interval = 0.01
    
    plt.figure(figsize=(10, 12))
    plt.subplot(3, 1, 1)
    time_points = np.arange(0, len(jnt_pos_list) * sampling_interval, sampling_interval)[:len(jnt_pos_list)]
    for i in range(robot_s.n_dof):
        plt.plot(time_points, [p[i] for p in jnt_pos_list], label=f'DoF {i}')
    plt.ylabel('Position [m]')
    plt.legend()
    # plt.grid(True)

    plt.subplot(3, 1, 2)
    time_points = np.arange(0, len(jnt_vel_list) * sampling_interval, sampling_interval)[:len(jnt_vel_list)]
    for i in range(robot_s.n_dof):
        plt.plot(time_points, [v[i] for v in jnt_vel_list], label=f'DoF {i}')
    plt.ylabel('Velocity [m/s]')
    plt.legend()
    # plt.grid(True)

    plt.subplot(3, 1, 3)
    time_points = np.arange(0, len(jnt_acc_list) * sampling_interval, sampling_interval)[:len(jnt_acc_list)]
    for i in range(robot_s.n_dof):
        plt.plot(time_points, [a[i] for a in jnt_acc_list], label=f'DoF {i}')
    plt.ylabel('Acceleration [m/s²]')
    plt.xlabel('Time [s]')
    plt.legend()
    # plt.grid(True)

    plt.tight_layout()
    plt.show()    


'''load the config file'''
current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))
config_file = os.path.join(current_file_dir, 'ruckig_config.yaml')
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

'''dataset loading'''
if config['mode'] == "train":
    dataset_path = os.path.join(parent_dir, 'datasets', config['dataset_name'])

    dataset = MotionPlanningDataset(dataset_path, horizon=config['horizon'], obs_keys=config['obs_keys'], 
                                    pad_before=config['obs_steps']-1, pad_after=config['action_steps']-1, abs_action=config['abs_action'])

    train_dataset, val_dataset = random_split(
        dataset, lengths=[0.95, 0.05],
        generator=torch.Generator().manual_seed(config['seed'])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=2,
        shuffle=False,
        pin_memory=True
    )

# --------------- Create Diffusion Model -----------------
if config['mode'] == "train":
    set_seed(config['seed'])
assert config["nn"] == "chi_unet"
assert config['diffusion'] == "ddpm"

from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import ChiUNet1d, JannerUNet1d
from cleandiffuser.diffusion.ddpm import DDPM
from cleandiffuser.diffusion.diffusionsde import DiscreteDiffusionSDE

# --------------- Network Architecture -----------------
nn_diffusion = ChiUNet1d(
    config['action_dim'], config['obs_dim'], config['obs_steps'], model_dim=256, emb_dim=256, dim_mult=config['dim_mult'],
    obs_as_global_cond=True, timestep_emb_type="positional").to(config['device'])
nn_condition = IdentityCondition(dropout=0.0).to(config['device'])


print(f"======================= Parameter Report of Diffusion Model =======================")
report_parameters(nn_diffusion)


import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from wrs import wd, rm, mcm
robot_s = franka.FrankaResearch3(enable_cc=True)

'''define the robot joint limits'''
jnt_v_max = rm.np.asarray([rm.pi * 2 / 3] * robot_s.n_dof)
jnt_a_max = rm.np.asarray([rm.pi] * robot_s.n_dof)
jnt_config_range = robot_s.jnt_ranges
x_max = torch.zeros((1, config['horizon'], config['action_dim']), device=config['device'])
x_min = torch.zeros((1, config['horizon'], config['action_dim']), device=config['device'])

x_max[:, :, :7] = torch.tensor(jnt_v_max, device=config['device']) 
x_max[:, :, -7:] = torch.tensor(jnt_a_max, device=config['device']) 

x_min[:, :, :7] = torch.tensor(-jnt_v_max, device=config['device'])
x_min[:, :, -7:] = torch.tensor(-jnt_a_max, device=config['device'])

agent = DDPM(
    nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=config['device'],
    diffusion_steps=config['sample_steps'], x_max=x_max, x_min=x_min,
    optim_params={"lr": config['lr']})

lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=config['gradient_steps'])

if config['mode'] == "train":
    # --------------- Data Loading -----------------
    '''prepare the save path'''
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f"{TimeCode}_h{config['horizon']}_unnorm"
    current_file_dir = os.path.dirname(__file__)
    save_path = os.path.join(current_file_dir, 'results', rootpath)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project="rrt_ruckig", name=rootpath)

    # ----------------- Training ----------------------
    agent.train()
    n_gradient_step = 0
    diffusion_loss_list = []
    log = {'avg_loss_diffusion': 0.}
    start_time = time.time()
    
    for batch in loop_dataloader(train_loader):
        # get condition
        condition = batch['cond'].to(config['device'])
        action = batch['action'].to(config['device'])

        condition = condition.flatten(start_dim=1) # (64,12)
        
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
        
        if n_gradient_step != 0 and n_gradient_step % config['val_freq'] == 0:
            val_loss = validate(agent, val_dataloader, config['device'])
            print(f"Validation Loss: {val_loss}")
            wandb.log({'val_loss': val_loss}, commit=False)

        n_gradient_step += 1
        if n_gradient_step >= config['gradient_steps']:
            break
    wandb.finish()

elif config['mode'] == "inference":
    # ----------------- Inference ----------------------
    # model_path = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/results/0402_1328_h256_unnorm/diffusion_ckpt_latest.pt'    
    model_path = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/results/0403_1111_h128_unnorm/diffusion_ckpt_latest.pt'
    agent.load(model_path)
    agent.model.eval()
    agent.model_ema.eval()

    '''capture the image'''
    sys.path.append('/home/lqin/wrs_xinyi/wrs')
    import wrs.visualization.panda.world as wd
    from wrs import wd, rm, mcm
    import wrs.modeling.geometric_model as mgm
    
    # init
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    # inference
    solver = config['inference_solver']
    inference_steps = config['inference_samples']
    visualization = config['visualization']
    
    success_num = 0
    for _ in tqdm(range(config['episode_num'])):
        jnt_pos_list = []
        jnt_vel_list = []
        jnt_acc_list = []
        delta_t = 0.01

        import mp_datagen_obstacles_rrt_ruckig as mp_datagen
        import copy

        start_conf, goal_conf, obstacle_list, obstacle_info = mp_datagen.generate_obstacle_confs(robot_s, obstacle_num=6)
        START_CONF = copy.deepcopy(start_conf)
        GOAL_CONF = copy.deepcopy(goal_conf)
        tgt_pos, tgt_rotmat = robot_s.fk(jnt_values=goal_conf)
        print(f"Start Conf: {start_conf}, Goal Conf: {goal_conf}, Obstacle Info: {obstacle_info}")
        
        robot_s.goto_given_conf(jnt_values=goal_conf)
        robot_s.gen_meshmodel(alpha=0.2, rgb=[0,1,0]).attach_to(base)
        robot_s.goto_given_conf(jnt_values=start_conf)
        robot_s.gen_meshmodel(alpha=0.2, rgb=[0,0,1]).attach_to(base)

        assert config['normalize'] == False
        update_counter = 0
        condition = torch.zeros((1, config['obs_dim']*config['obs_steps']), device=config['device'])

        for _ in range(inference_steps):
            start_conf = robot_s.get_jnt_values()
            condition[:, :robot_s.n_dof] = torch.tensor(start_conf).to(config['device'])
            condition[:, robot_s.n_dof:2*robot_s.n_dof] = torch.tensor(goal_conf).to(config['device'])
            condition[:, 2*robot_s.n_dof:] = torch.tensor(obstacle_info).to(config['device'])
            
            with torch.no_grad():
                prior = torch.zeros((1, config['horizon'], config['action_dim']), device=config['device'])
                action, _ = agent.sample(prior=prior, n_samples=1, sample_steps=config['sample_steps'],
                                        solver=solver, condition_cfg=condition, w_cfg=1.0, use_ema=True)

            # sample actions and unnorm
            jnt_acc_pred = action[0, :,-7:].detach().to('cpu').numpy()
            jnt_pos_list.append(start_conf)
            jnt_vel_list.append(action[0, 0, :7].detach().to('cpu').numpy())
            
            for idx in range(jnt_acc_pred.shape[0]):
                # print(update_counter)
                robot_s.goto_given_conf(jnt_values=jnt_pos_list[-1])
                pred_pos, pred_rotmat = robot_s.fk(jnt_values=jnt_pos_list[-1])
                pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos, tgt_rotmat, pred_pos, pred_rotmat)
                print(f"Step: {update_counter}, pos_err: {pos_err} mms, rot_err: {rot_err} degrees")

                jnt_pos_list.append(jnt_pos_list[-1] + delta_t*jnt_vel_list[-1])  # p(t+1)
                jnt_acc_list.append(jnt_acc_pred[idx]) # a(t)
                jnt_vel_list.append(jnt_vel_list[-1] + delta_t*jnt_acc_pred[idx]) # v(t+1)

                if visualization:
                    robot_s.gen_meshmodel(alpha=0.2).attach_to(base)
                update_counter += 1
                # print(f"Step: {update_counter}, distance: {np.linalg.norm(robot_s.get_jnt_values() - goal_conf)}")

                if robot_s.cc.is_collided(obstacle_list=obstacle_list):
                    print("Collision detected!")
                    break

                if (pos_err < config['max_pos_err'] and rot_err < config['max_rot_err']):
                    print("Goal reached!")
                    break

                if update_counter > config['max_iter']:
                    print("Max iteration reached!")
                    break
            
            if pos_err < config['max_pos_err'] and rot_err < config['max_rot_err'] and not robot_s.cc.is_collided(obstacle_list=obstacle_list):
                success_num += 1
                print("Success!")
                break
        
        if visualization:
            # mp_datagen.visualize_anime_diffusion(robot=robot_s, path = jnt_pos_list, 
            #                                      start_conf=START_CONF, goal_conf=GOAL_CONF)
            jnt_pos_array, jnt_vel_arrat, jnt_acc_array = np.array(jnt_pos_list), np.array(jnt_vel_list), np.array(jnt_acc_list)
            np.savez('jnt_info.npz', jnt_pos=jnt_pos_array, jnt_vel=jnt_vel_arrat, jnt_acc=jnt_acc_array)
            plot_details(robot_s, jnt_pos_list, jnt_vel_list, jnt_acc_list)
            base.run()    

    print(f"Success rate: {success_num/config['episode_num']*100}%")

else:
    raise ValueError("Illegal mode")