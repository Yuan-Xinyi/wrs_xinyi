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
from ruckig_dataset import BSplineDataset
from torch.utils.data import random_split

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def update_bspline_with_new_time(spline, T_total_new):
    """
    根据新的 T_total 和 dt 更新 B-Spline 曲线及其导数。
    """
    s_fine = np.linspace(0, 1, 1000)
    q_s = spline(s_fine)
    dq_ds = spline.derivative(1)(s_fine)
    d2q_ds2 = spline.derivative(2)(s_fine)
    d3q_ds3 = spline.derivative(3)(s_fine)

    # 时间映射
    r_s = 1 / T_total_new
    q_t = q_s
    dq_dt = dq_ds * r_s
    d2q_dt2 = d2q_ds2 * (r_s ** 2)
    d3q_dt3 = d3q_ds3 * (r_s ** 3)
    t_fine = s_fine * T_total_new

    return t_fine, q_t, dq_dt, d2q_dt2, d3q_dt3

def update_bspline_with_new_time(spline, T_total_new):
    """
    根据新的 T_total 和 dt 更新 B-Spline 曲线及其导数。
    """
    s_fine = np.linspace(0, 1, 1000)
    q_s = spline(s_fine)
    dq_ds = spline.derivative(1)(s_fine)
    d2q_ds2 = spline.derivative(2)(s_fine)
    d3q_ds3 = spline.derivative(3)(s_fine)

    # 时间映射
    r_s = 1 / T_total_new
    q_t = q_s
    dq_dt = dq_ds * r_s
    d2q_dt2 = d2q_ds2 * (r_s ** 2)
    d3q_dt3 = d3q_ds3 * (r_s ** 3)
    t_fine = s_fine * T_total_new

    return t_fine, q_t, dq_dt, d2q_dt2, d3q_dt3

def plot_bspline(results, num_joints, overlay=True):
    """
    绘制 B-Spline 及其导数（无原始数据），比较不同 T_total。
    """
    if overlay:
        fig, axs = plt.subplots(num_joints, 4, figsize=(16, 4 * num_joints), sharex=True)
        colors = ['r', 'g', 'b', 'm', 'c']

        for idx, (T_total, t_fine, q_t, dq_dt, d2q_dt2, d3q_dt3) in enumerate(results):
            color = colors[idx % len(colors)]
            for j in range(num_joints):
                axs[j, 0].plot(t_fine, q_t[:, j], label=f'B-Spline (T={T_total}s)', color=color, linewidth=2)
                axs[j, 0].set_title(f"Position $q_{j}(t)$")
                axs[j, 0].set_ylabel("Position")
                axs[j, 0].legend()

                axs[j, 1].plot(t_fine, dq_dt[:, j], label=f'B-Spline Velocity (T={T_total}s)', color=color)
                axs[j, 1].set_title(f"Velocity $\\dot{{q}}_{j}(t)$")
                axs[j, 1].set_ylabel("Velocity")
                axs[j, 1].legend()

                axs[j, 2].plot(t_fine, d2q_dt2[:, j], label=f'B-Spline Acceleration (T={T_total}s)', color=color)
                axs[j, 2].set_title(f"Acceleration $\\ddot{{q}}_{j}(t)$")
                axs[j, 2].set_ylabel("Acceleration")
                axs[j, 2].legend()

                axs[j, 3].plot(t_fine, d3q_dt3[:, j], label=f'B-Spline Jerk (T={T_total}s)', color=color)
                axs[j, 3].set_title(f"Jerk $\\dddot{{q}}_{j}(t)$")
                axs[j, 3].set_ylabel("Jerk")
                axs[j, 3].legend()

    else:
        for idx, (T_total, t_fine, q_t, dq_dt, d2q_dt2, d3q_dt3) in enumerate(results):
            fig, axs = plt.subplots(num_joints, 4, figsize=(16, 4 * num_joints), sharex=True)
            for j in range(num_joints):
                axs[j, 0].plot(t_fine, q_t[:, j], label=f'B-Spline (T={T_total}s)', linewidth=2)
                axs[j, 0].set_title(f"Position $q_{j}(t)$")
                axs[j, 0].set_ylabel("Position")
                axs[j, 0].legend()

                axs[j, 1].plot(t_fine, dq_dt[:, j], label='B-Spline Velocity')
                axs[j, 1].set_title(f"Velocity $\\dot{{q}}_{j}(t)$")
                axs[j, 1].set_ylabel("Velocity")
                axs[j, 1].legend()

                axs[j, 2].plot(t_fine, d2q_dt2[:, j], label='B-Spline Acceleration')
                axs[j, 2].set_title(f"Acceleration $\\ddot{{q}}_{j}(t)$")
                axs[j, 2].set_ylabel("Acceleration")
                axs[j, 2].legend()

                axs[j, 3].plot(t_fine, d3q_dt3[:, j], label='B-Spline Jerk')
                axs[j, 3].set_title(f"Jerk $\\dddot{{q}}_{j}(t)$")
                axs[j, 3].set_ylabel("Jerk")
                axs[j, 3].legend()

    plt.tight_layout()
    plt.show()

def plot_control_points_comparison(control_points_original, control_points_generated, num_joints, title="Control Points Comparison"):
    """
    绘制原始控制点和生成控制点的比较。

    Parameters:
    - control_points_original: (numpy array) 原始控制点，形状为 (num_ctrl_pts, num_joints)。
    - control_points_generated: (numpy array) 生成的控制点，形状为 (num_ctrl_pts, num_joints)。
    - num_joints: (int) 关节数量。
    - title: (str) 图表标题。
    """
    fig, axs = plt.subplots(num_joints, 1, figsize=(10, 3 * num_joints), sharex=True)

    if num_joints == 1:
        axs = [axs]  # 确保 axs 是列表

    for j in range(num_joints):
        axs[j].plot(control_points_original[:, j], 'o-', label='Original Control Points', color='blue', markersize=5)
        axs[j].plot(control_points_generated[:, j], '*--', label='Generated Control Points', color='red', markersize=5)
        axs[j].set_title(f"Joint {j+1} Control Points")
        axs[j].set_ylabel("Control Point Value")
        axs[j].legend(loc="best")

    axs[-1].set_xlabel("Control Point Index")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def compare_bspline_two_methods(knots, control_points, action_points, degree, T_total_list, n_dof):
    """
    比较两种 B-Spline(基于 control_points 和 action_points)并绘制在一张图上
    """
    # 基于 control_points 构建的 B-Spline
    spline_control = BSpline(knots, control_points, degree)
    
    # 基于 action_points 构建的 B-Spline
    spline_action = BSpline(knots, action_points, degree)

    results_control = []
    results_action = []

    # 测试不同 T_total
    for T_total_new in T_total_list:
        print(f"\nTesting with T_total = {T_total_new}s (Control Points)")
        result_control = (T_total_new, *update_bspline_with_new_time(spline_control, T_total_new))
        results_control.append(result_control)

        print(f"\nTesting with T_total = {T_total_new}s (Action Points)")
        result_action = (T_total_new, *update_bspline_with_new_time(spline_action, T_total_new))
        results_action.append(result_action)

    # 绘制对比
    fig, axs = plt.subplots(n_dof, 4, figsize=(16, 4 * n_dof), sharex=True)
    colors = ['r', 'b']

    for idx, (T_total, t_fine, q_t, dq_dt, d2q_dt2, d3q_dt3) in enumerate(results_control):
        for j in range(n_dof):
            axs[j, 0].plot(t_fine, q_t[:, j], label=f'Control Points B-Spline (T={T_total}s)', color=colors[0], linewidth=2)
            axs[j, 1].plot(t_fine, dq_dt[:, j], color=colors[0])
            axs[j, 2].plot(t_fine, d2q_dt2[:, j], color=colors[0])
            axs[j, 3].plot(t_fine, d3q_dt3[:, j], color=colors[0])

    for idx, (T_total, t_fine, q_t, dq_dt, d2q_dt2, d3q_dt3) in enumerate(results_action):
        for j in range(n_dof):
            axs[j, 0].plot(t_fine, q_t[:, j], label=f'Action Points B-Spline (T={T_total}s)', color=colors[1], linewidth=2, linestyle='--')
            axs[j, 1].plot(t_fine, dq_dt[:, j], color=colors[1], linestyle='--')
            axs[j, 2].plot(t_fine, d2q_dt2[:, j], color=colors[1], linestyle='--')
            axs[j, 3].plot(t_fine, d3q_dt3[:, j], color=colors[1], linestyle='--')

    for ax in axs.flat:
        ax.legend()
        ax.set_xlabel("Time [s]")

    axs[0, 0].set_ylabel("Position")
    axs[0, 1].set_ylabel("Velocity")
    axs[0, 2].set_ylabel("Acceleration")
    axs[0, 3].set_ylabel("Jerk")

    plt.tight_layout()
    plt.show()

def third_vertex(A, B):
    A, B = np.array(A), np.array(B)
    AB = B - A
    mid = (A + B) / 2
    n = np.cross(AB, [0, 0, 1])
    if np.linalg.norm(n) < 1e-6:
        n = np.cross(AB, [0, 1, 0])
    n /= np.linalg.norm(n)
    return mid + (np.sqrt(3)/2 * np.linalg.norm(AB)) * n

'''load the config file'''
current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))
config_file = os.path.join(current_file_dir,'config', 'ruckig_straight_bspline_config.yaml')
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

'''dataset loading'''
dataset_path = os.path.join('/home/lqin', 'zarr_datasets', config['dataset_name'])

dataset = BSplineDataset(dataset_path, horizon=config['horizon'], obs_keys=config['obs_keys'], 
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
jnt_v_max = rm.np.asarray([rm.pi * 2 / 3] * robot_s.n_dof)
jnt_a_max = rm.np.asarray([rm.pi] * robot_s.n_dof)
jnt_config_range = torch.tensor(robot_s.jnt_ranges, device=config['device'])

if config['normalize']:
    x_max = torch.ones((1, config['horizon'], config['action_dim']), device=config['device']) * +1.0
    x_min = torch.ones((1, config['horizon'], config['action_dim']), device=config['device']) * -1.0
    print('*'*50)
    print("Using Normalized Action Space. the action space is normalized to [-1, 1]")
    print('*'*50)
else:
    x_max = (jnt_config_range[:,1]).repeat(1, config['horizon'], 1)
    x_min = (jnt_config_range[:,0]).repeat(1, config['horizon'], 1)
    print('*'*50)
    print("Using Absolute Action Space. the action space is absolute joint configuration")
    print('*'*50)

loss_weight = torch.ones((config['horizon'], config['action_dim']))
loss_weight[0, :] = config['action_loss_weight']

'''fix mask'''
# fix_mask = torch.zeros((config['horizon'], config['action_dim']), device=config['device'])
# fix_mask[0, :] = 1.
# fix_mask[-1:, :] = 1.

# agent = DDPM(
#     nn_diffusion=nn_diffusion, nn_condition=nn_condition, fix_mask = fix_mask,
#     device=config['device'], diffusion_steps=config['diffusion_steps'], x_max=x_max, x_min=x_min,
#     optim_params={"lr": config['lr']}, predict_noise=config['predict_noise'])

'''no fix mask'''
agent = DDPM(
    nn_diffusion=nn_diffusion, nn_condition=nn_condition, loss_weight=loss_weight,
    device=config['device'], diffusion_steps=config['diffusion_steps'], x_max=x_max, x_min=x_min,
    optim_params={"lr": config['lr']}, predict_noise=config['predict_noise'])

lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=config['gradient_steps'])

if config['mode'] == "train":
    # --------------- Data Loading -----------------
    '''prepare the save path'''
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f"{TimeCode}_{config['horizon']}h_{config['batch_size']}b_norm{config['normalize']}"
    save_path = os.path.join(current_file_dir, 'results', rootpath)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project="ruckig_bspline", name=rootpath)

    # ----------------- Training ----------------------
    agent.train()
    n_gradient_step = 0
    diffusion_loss_list = []
    log = {'avg_loss_diffusion': 0.}
    start_time = time.time()
    
    for batch in loop_dataloader(train_loader):
        # get condition
        if config['condition'] == "identity":
            condition = batch['cond'].to(config['device'])
            condition = condition.flatten(start_dim=1).to(torch.float32) # (batch,14)
        else:
            condition = None
        action = batch['control_points'].to(config['device']) # (batch,horizon,7)
        
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
    model_path = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/results/0608_1500_32h_64b_normTrue/diffusion_ckpt_latest.pt'
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
    n_samples = 200

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

    import numpy as np
    import torch
    import helper_functions as hf

    # 预设三角形三个点
    pos_1 = np.array([-0.56887997, -0.06892034, 0.321434])
    pos_2 = np.array([-0.39887989, -0.06892035, 0.32143399])
    pos_3 = third_vertex(pos_1, pos_2)  # 你已有这个函数

    triangle_edges = [[pos_1, pos_2], [pos_2, pos_3], [pos_3, pos_1]]

    # 可视化箭头
    for spos, epos in triangle_edges:
        mgm.gen_arrow(spos=np.array(spos), epos=np.array(epos), stick_radius=.005, rgb=[1, 0, 0]).attach_to(base)

    # 使用 diffusion 预测所有边的初始关节角
    conditions = np.array([np.concatenate([start, end]) for start, end in triangle_edges])
    conditions = torch.tensor(conditions, device=config['device']).float().unsqueeze(1).repeat(1, n_samples, 1)
    conditions = conditions.view(-1, 6)
    prior = torch.zeros((3 * n_samples, config['horizon'], config['action_dim']), device=config['device'])

    with torch.no_grad():
        actions, _ = agent.sample(
            prior=prior,
            n_samples=3 * n_samples,
            sample_steps=config['sample_steps'],
            temperature=1.0,
            solver=solver,
            condition_cfg=conditions,
            w_cfg=1.0,
            use_ema=True,
        )
        actions_np = dataset.normalizer['obs']['jnt_pos'].unnormalize(actions.cpu().numpy())  # (3 * n_samples, H, D)

    # 初始化
    jnt_list = []
    all_sub_jnt_lists = []
    sample_indices = [0] * len(triangle_edges)  # 每条边当前尝试的sample index
    edge_idx = 0

    while 0 <= edge_idx < len(triangle_edges):
        sample_idx = sample_indices[edge_idx]
        if sample_idx >= n_samples:
            if edge_idx == 0:
                print("Planning failed: all samples exhausted for first edge.")
                break
            # print(f"Edge {edge_idx} failed. Backtracking to edge {edge_idx - 1}")
            edge_idx -= 1
            jnt_list = jnt_list[:-len(all_sub_jnt_lists[-1])]
            all_sub_jnt_lists.pop()
            sample_indices[edge_idx] += 1  # 尝试下一个 sample
            continue

        # 当前边的信息
        pos_start, pos_goal = triangle_edges[edge_idx]
        disp_vec = pos_goal - pos_start
        distance = np.linalg.norm(disp_vec)
        direction = disp_vec / distance
        n_steps = int(distance / 0.01)
        # print(f"Edge {edge_idx} | Trying sample {sample_idx} | Distance: {distance:.4f}m | Steps: {n_steps}")

        # diffusion sample -> initial joint angle
        sample_id = edge_idx * n_samples + sample_idx
        pred_jnt_seed = actions_np[sample_id, 0, :]
        _, rot = robot_s.fk(jnt_values=pred_jnt_seed, toggle_jacobian=True, update=True)

        sub_jnt_list = []
        for step in range(1, n_steps + 1):
            pos = pos_start + direction * (0.01 * step)
            seed = all_sub_jnt_lists[-1][-1] if all_sub_jnt_lists else pred_jnt_seed
            res = robot_s.ik(tgt_pos=pos, tgt_rotmat=rot, seed_jnt_values=seed)
            if res is None:
                break
            sub_jnt_list.append(res)

        # 判断是否成功
        if len(sub_jnt_list) == n_steps:
            print(f"Edge {edge_idx} succeeded with sample {sample_idx}")
            all_sub_jnt_lists.append(sub_jnt_list)
            jnt_list.extend(sub_jnt_list)
            edge_idx += 1
            if edge_idx < len(triangle_edges):
                sample_indices[edge_idx] = 0  # ✅ 初始化下一个边的 sample index

        else:
            sample_indices[edge_idx] += 1  # 当前 edge 用下一个 sample 继续尝试

    # 成功后可视化整条轨迹
    if edge_idx == len(triangle_edges):
        print(repr(jnt_list))
        jnt_array = np.array(jnt_list)
        
        fig, axes = plt.subplots(7, 1, figsize=(10, 18), sharex=True)

        for i in range(7):
            axes[i].plot(jnt_array[:, i], label=f'Joint {i+1}')
            axes[i].set_ylabel(f'Joint {i+1}')
            axes[i].grid(True)
            axes[i].legend()

        axes[-1].set_xlabel('Step Index')
        plt.tight_layout()
        plt.show()
        
        hf.visualize_anime_path(base, robot_s, jnt_list)
    else:
        print("Trajectory planning failed.")



else:
    raise ValueError("Illegal mode")