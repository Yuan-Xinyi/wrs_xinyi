# -*- coding: utf-8 -*-
'''wrs reliance'''
from wrs import wd, rm, mcm
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
import wrs.modeling.geometric_model as mgm

'''self modules'''
# 如果你有自己的 cone_constraint 模块也可以用；此处我们实现一个稳定版本
# import rotation_cone_constraint as cone_constraint

'''global variables'''
import time
import pickle
import numpy as np
import torch
torch.autograd.set_detect_anomaly(False)  # 如需定位梯度问题可切 True

import matplotlib.pyplot as plt

# ======================== 初始化 ============================
xarm_gpu = xarm6_gpu.XArmLite6GPU()
xarm_sim = xarm6_sim.XArmLite6Miller(enable_cc=True)
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)

# 桌面与纸
table_size = np.array([1.5, 1.5, 0.05])
table_pos  = np.array([0.6, 0, -0.025])
table = mcm.gen_box(xyz_lengths=table_size, pos=table_pos, rgb=np.array([0.6, 0.4, 0.2]), alpha=1)
table.attach_to(base)

paper_size = np.array([1.0, 1.0, 0.002])
paper_pos = table_pos.copy()
paper_pos[2] = table_pos[2] + table_size[2]/2 + paper_size[2]/2
paper = mcm.gen_box(xyz_lengths=paper_size, pos=paper_pos, rgb=np.array([1, 1, 1]), alpha=1)
paper.attach_to(base)

device = xarm_gpu.device

# ======================== 实用函数 ============================

def circle_points(center, radius, num_points, z_fixed):
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    px = center[0] + radius * np.cos(theta)
    py = center[1] + radius * np.sin(theta)
    pz = np.full_like(px, z_fixed)
    return np.stack([px, py, pz], axis=1)  # (N,3)

def randomize_circles_batch(paper_pos, paper_size, batch_size=1, num_points=50, margin=0.12, device=None):
    """与之前版本一致：返回 (B,N,3) 与 {center:(B,3), radius:(B,)}，法向缺省为 -Z"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    px, py, pz = paper_pos
    lx, ly, lz = paper_size
    z = pz + lz / 2

    x_min, x_max = (px - lx/2) + margin, (px + lx/2) - margin
    y_min, y_max = (py - ly/2) + margin, (py + ly/2) - margin

    centers_x = torch.empty(batch_size, device=device).uniform_(x_min, x_max)
    centers_y = torch.empty(batch_size, device=device).uniform_(y_min, y_max)
    centers_z = torch.full((batch_size,), z, device=device)
    centers = torch.stack([centers_x, centers_y, centers_z], dim=1)

    left   = centers_x - x_min
    right  = x_max - centers_x
    bottom = centers_y - y_min
    top    = y_max - centers_y
    max_r  = torch.min(torch.min(left, right), torch.min(bottom, top))
    min_r  = torch.full_like(max_r, 0.1)
    max_r  = torch.clamp(max_r, min=min_r)
    radii  = torch.rand(batch_size, device=device) * (max_r - min_r) + min_r

    theta = torch.linspace(0, 2 * torch.pi, num_points + 1, device=device)[:-1]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    x_offsets = radii[:, None] * cos_t[None, :]
    y_offsets = radii[:, None] * sin_t[None, :]
    z_offsets = torch.zeros_like(x_offsets, device=device) + z

    pos_tensor = torch.stack([
        centers_x[:, None] + x_offsets,
        centers_y[:, None] + y_offsets,
        z_offsets
    ], dim=-1)  # (B,N,3)

    info = {'center': centers, 'radius': radii}
    return pos_tensor, info

def visualize_pos_batch(pos_batch_torch):
    pos_batch_np = pos_batch_torch.detach().cpu().numpy()
    for b in range(pos_batch_np.shape[0]):
        for t in range(pos_batch_np.shape[1]):
            mgm.gen_sphere(pos=pos_batch_np[b, t], radius=0.004, rgb=[0,1,0]).attach_to(base)

def visualize_jnt_traj(robot_sim, q_traj_torch, alpha=0.2, rgb=[0,0,1]):
    q_np = q_traj_torch.detach().cpu().numpy()
    for i in range(q_np.shape[0]):
        robot_sim.goto_given_conf(jnt_values=q_np[i])
        robot_sim.gen_meshmodel(alpha=alpha, rgb=rgb).attach_to(base)

def visualize_jnt_traj_video(robot_sim, q_traj_torch):
    q_np = q_traj_torch.detach().cpu().numpy()
    q_flat = q_np.reshape(-1, 6)
    import helper_functions as helper
    helper.visualize_anime_path(base, robot_sim, q_flat)

def circ_shift_tensor(t, shift):
    """循环移位（正值右移，负值左移），用于相位对齐"""
    shift = int(shift) % t.shape[0]
    if shift == 0:
        return t
    return torch.cat([t[-shift:], t[:-shift]], dim=0)

def phase_align(prev_pos, prev_q, new_pos):
    """在圆上做相位对齐：寻找一个循环移位，使 prev_pos 与 new_pos 尽量贴近"""
    N = prev_pos.shape[0]
    # 预计算
    errs = []
    for s in range(N):
        ppos_s = circ_shift_tensor(prev_pos, s)
        err = torch.mean(torch.norm(ppos_s - new_pos, dim=1))
        errs.append(err)
    s_best = int(torch.argmin(torch.stack(errs)).item())
    return circ_shift_tensor(prev_pos, s_best), circ_shift_tensor(prev_q, s_best), s_best

def cone_angle_loss_stable(z_axis, z_target, max_angle_deg=30.0):
    """
    z_axis: (N,3) 当前末端Z轴
    z_target: (3,) 期望法向
    return: 平滑可导的锥约束损失，避免 acos 数值不稳
    """
    phi = torch.deg2rad(torch.tensor(max_angle_deg, device=z_axis.device))
    z_axis = z_axis / (torch.norm(z_axis, dim=-1, keepdim=True).clamp(min=1e-8))
    z_target = z_target / (torch.norm(z_target).clamp(min=1e-8))
    cos_angle = torch.sum(z_axis * z_target, dim=-1).clamp(-0.999999, 0.999999)
    # 只有超过锥界时才惩罚：cos(theta) < cos(phi)
    viol = torch.relu(torch.cos(phi) - cos_angle)
    loss = torch.mean(viol ** 2)
    return loss, cos_angle

def optimize_trajectory_from_warm(
    xarm_gpu,
    pos_target,         # (N,3) torch
    q_warm,             # (N,6) torch
    steps=1500,
    lr=1e-2,
    lambda_pos=100.0,
    lambda_smooth=2.0,
    lambda_cone=1.0,
    cone_max_deg=30.0,
    log_every=50
):
    """用上一次的 q 作为初值；加入稳定 cone 约束；pos 为主，平滑、cone 次之。"""
    device = pos_target.device
    q_traj = q_warm.clone().detach().to(device).requires_grad_(True)
    opt = torch.optim.Adam([q_traj], lr=lr)
    z_target = torch.tensor([0, 0, -1], dtype=torch.float32, device=device)

    for step in range(steps):
        pos_fk, rot_fk = xarm_gpu.robot.fk_batch(q_traj)
        pos_err = pos_fk - pos_target

        # 位置项：均值 + 最大误差（避免均值掩盖单点大误差）
        per_point = torch.norm(pos_err, dim=1)
        pos_loss = per_point.mean() + 0.5 * per_point.max()

        # 平滑项：一阶差分
        smooth_loss = torch.mean((q_traj[1:] - q_traj[:-1]) ** 2)

        # 姿态锥约束（稳定版）
        z_axis = rot_fk[:, :, 2]
        cone_loss, cos_angle = cone_angle_loss_stable(z_axis, z_target, max_angle_deg=cone_max_deg)

        loss = lambda_pos * pos_loss + lambda_smooth * smooth_loss + lambda_cone * cone_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step % log_every == 0) or (step == steps - 1):
            mean_ang = torch.rad2deg(torch.acos(cos_angle.clamp(-0.9999, 0.9999))).mean().item()
            print(f"Step {step:03d}: loss={loss.item():.6f}, "
                  f"pos={pos_loss.item():.4f}, smooth={smooth_loss.item():.4f}, "
                  f"cone={cone_loss.item():.4f}, mean_angle={mean_ang:.2f}°")

    return q_traj.detach()

# ========= 批量相位对齐 =========
def batch_phase_align(pos_demo, q_demo, pos_batch):
    """
    pos_demo: (N,3)  示教位置
    q_demo:   (N,6)  示教关节
    pos_batch:(B,N,3) 新圆目标点
    return:
      q_warm_batch: (B,N,6)  每条圆对应的 warm-start 关节
      best_shift:   (B,)     每条圆选择的相位
    """
    device = pos_batch.device
    B, N, _ = pos_batch.shape

    # 预生成所有相位的“示教位置/关节”堆栈，形状：(N, S=N, dim)
    shifts = torch.arange(N, device=device)
    # 堆叠所有roll版本：(S, N, 3) / (S, N, 6)
    pos_rolls = torch.stack([torch.roll(pos_demo, shifts=int(s), dims=0) for s in range(N)], dim=0)  # (S,N,3)
    q_rolls   = torch.stack([torch.roll(q_demo,   shifts=int(s), dims=0) for s in range(N)], dim=0)  # (S,N,6)

    # 计算每个 batch 与每个相位的误差： (B,S)
    # 先扩展到同形状：(B,S,N,3)
    pos_batch_exp = pos_batch[:, None, :, :]                 # (B,1,N,3)
    pos_rolls_exp = pos_rolls[None, :, :, :]                 # (1,S,N,3)
    err = torch.norm(pos_batch_exp - pos_rolls_exp, dim=-1)  # (B,S,N)
    mean_err = err.mean(dim=-1)                              # (B,S)

    # 选每条圆的最佳相位
    best_shift = torch.argmin(mean_err, dim=1)               # (B,)

    # 按相位gather对应的 q
    # 先把 q_rolls 改成 (S,N,6) -> (1,S,N,6) -> (B,S,N,6)
    q_rolls_exp = q_rolls[None, :, :, :].expand(B, -1, -1, -1)  # (B,S,N,6)
    # 构造索引：(B,1,1,1) 广播到 (B,1,N,6) 在 dim=1 处选择
    idx = best_shift.view(B, 1, 1, 1).expand(-1, 1, N, 6)       # (B,1,N,6)
    q_warm_batch = torch.gather(q_rolls_exp, dim=1, index=idx).squeeze(1)  # (B,N,6)

    return q_warm_batch, best_shift


# ========= 批量并行优化主函数 =========
def optimize_trajectories_batch(
    xarm_gpu,
    pos_batch,              # (B,N,3) 圆上目标点（已固定）
    q_warm_batch,           # (B,N,6) 每条圆的初始关节（相位对齐后的示教）
    steps=1200,
    lr=1e-2,
    lambda_pos=120.0,
    lambda_smooth=2.0,
    lambda_cone=1.0,
    cone_max_deg=30.0,
    log_every=100,
):
    """
    并行优化 B 条圆的整条轨迹
    """
    device = pos_batch.device
    B, N, _ = pos_batch.shape

    q_traj = q_warm_batch.clone().detach().to(device).requires_grad_(True)  # (B,N,6)
    opt = torch.optim.Adam([q_traj], lr=lr)

    z_target = torch.tensor([0, 0, -1], dtype=torch.float32, device=device)

    for step in range(steps):
        # 展平到 (B*N,6) 走一次 FK，再 reshape 回来
        q_flat = q_traj.reshape(B * N, -1)
        pos_fk_flat, rot_fk_flat = xarm_gpu.robot.fk_batch(q_flat)  # (B*N,3), (B*N,3,3)
        pos_fk = pos_fk_flat.reshape(B, N, 3)
        rot_fk = rot_fk_flat.reshape(B, N, 3, 3)

        # --- 位置项（mean + 0.5*max，避免个别点炸偏）---
        pos_err = pos_fk - pos_batch                           # (B,N,3)
        per_point = torch.norm(pos_err, dim=-1)                # (B,N)
        pos_loss = per_point.mean() + 0.5 * per_point.max()

        # --- 平滑项 ---
        smooth_loss = torch.mean((q_traj[:, 1:, :] - q_traj[:, :-1, :]) ** 2)

        # --- 稳定的 cone 约束（超界才惩罚）---
        z_axis = rot_fk[:, :, :, 2]                            # (B,N,3)
        z_axis = z_axis / (torch.norm(z_axis, dim=-1, keepdim=True).clamp(min=1e-8))
        zt = z_target / torch.norm(z_target).clamp(min=1e-8)
        cos_angle = (z_axis @ zt)                              # (B,N)
        cos_angle = cos_angle.clamp(-0.999999, 0.999999)
        phi = torch.deg2rad(torch.tensor(cone_max_deg, device=device))
        viol = torch.relu(torch.cos(phi) - cos_angle)
        cone_loss = torch.mean(viol ** 2)

        loss = lambda_pos * pos_loss + lambda_smooth * smooth_loss + lambda_cone * cone_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step % log_every == 0) or (step == steps - 1):
            mean_ang = torch.rad2deg(torch.acos(cos_angle)).mean().item()
            print(f"[Batch] Step {step:03d}: loss={loss.item():.6f}, "
                  f"pos={pos_loss.item():.4f}, smooth={smooth_loss.item():.4f}, "
                  f"cone={cone_loss.item():.4f}, mean_angle={mean_ang:.2f}°")

    return q_traj.detach()  # (B,N,6)

def visualize_pos_batch(pos_batch):
    pos_batch_np = pos_batch.cpu().numpy()
    for b in range(pos_batch_np.shape[0]):
        for t in range(pos_batch_np.shape[1]):
            mgm.gen_sphere(pos_batch_np[b, t], 0.005, [0,1,0]).attach_to(base)

def gen_expand_shift_candidates(
    c0, r0,
    paper_pos, paper_size,
    num_points,
    d_center=0.03,      # 圆心漂移步长（m）
    dr=0.02,            # 半径增量（m）
    num_dirs=8,         # 方向数量（例如 8 个罗盘方向）
    r_min_abs=0.08,     # 最小允许半径（避免太小）
    margin=0.06,
    device=None
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    px, py, pz = paper_pos
    lx, ly, lz = paper_size
    z = pz + lz/2
    x_min, x_max = (px - lx/2) + margin, (px + lx/2) - margin
    y_min, y_max = (py - ly/2) + margin, (py + ly/2) - margin

    # 方向向量
    angles = torch.linspace(0, 2*torch.pi, steps=num_dirs+1, device=device)[:-1]  # (K,)
    dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)             # (K,2)

    # 候选圆心（仅 xy 变动）
    centers_xy = c0[:2][None, :] + d_center * dirs                                # (K,2)
    centers = torch.cat([centers_xy, torch.full((num_dirs,1), z, device=device)], dim=1)  # (K,3)
    
    # 候选半径
    r_cand = torch.full((num_dirs,), r0 + dr, device=device)                      # (K,)

    # 边界允许的最大发电半径（每个圆心不同）
    left   = centers[:, 0] - x_min
    right  = x_max - centers[:, 0]
    bottom = centers[:, 1] - y_min
    top    = y_max - centers[:, 1]
    max_r_boundary = torch.minimum(torch.minimum(left, right), torch.minimum(bottom, top))  # (K,)

    # 把半径裁剪到边界内，并做最小半径过滤
    radii = torch.minimum(r_cand, max_r_boundary)                                  # (K,)
    valid = radii >= r_min_abs                                                     # (K,)
    centers = centers[valid]
    radii   = radii[valid]

    if centers.shape[0] == 0:
        # 全部无效，返回空
        return (torch.empty(0, num_points, 3, device=device),
                {'center': centers, 'radius': radii})

    # 生成每条圆的采样点
    B = centers.shape[0]
    th = torch.linspace(0, 2*torch.pi, steps=num_points+1, device=device)[:-1]    # (N,)
    ct, st = torch.cos(th), torch.sin(th)
    pos_batch = torch.stack([
        centers[:, 0:1] + radii[:, None] * ct[None, :],
        centers[:, 1:2] + radii[:, None] * st[None, :],
        torch.full((B, num_points), z, device=device)
    ], dim=-1)  # (B,N,3)

    info = {'center': centers, 'radius': radii}
    return pos_batch, info



# ======================== 主流程（递进扩环 + 优化） ============================
if __name__ == "__main__":
    with open("0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro/circle_data.pkl", "rb") as f:
        data = pickle.load(f)
    q_demo = data["jnt"]
    pos_demo = data["circle"]
    circle_info = data["circle_info"]
    c0 = torch.as_tensor(circle_info["center"], dtype=torch.float32, device=device)
    r0 = float(circle_info["radius"])
    N = pos_demo.shape[0]

    num_expand_per_round = 64     # 每轮生成多少圈
    dr = 0.02                     # 每圈半径增量
    topk_keep = 32                 # 每轮保留多少条最大的圆
    max_rounds = 6               # 迭代轮数上限

    current_centers = [c0]
    current_radii = [r0]

    for round_idx in range(max_rounds):
        print(f"\n=== Round {round_idx+1}/{max_rounds} ===")

        # ------- 1. 生成扩展候选 -------
        all_pos = []
        all_info = {'center': [], 'radius': []}

        for ci, ri in zip(current_centers, current_radii):
            pos_batch, info_batch = gen_expand_shift_candidates(
                c0=ci, r0=ri,
                paper_pos=paper_pos, paper_size=paper_size,
                num_points=N,
                d_center=0.03,      # 圆心漂移步长（每轮偏移 3cm，可自行调）
                dr=dr,              # 半径增加 dr
                num_dirs=num_expand_per_round,  # 每轮生成 num_expand_per_round 个漂移方向
                r_min_abs=0.08,
                margin=0.06,
                device=device
            )
            if pos_batch.shape[0] > 0:
                all_pos.append(pos_batch)
                all_info['center'].append(info_batch['center'])
                all_info['radius'].append(info_batch['radius'])

        if len(all_pos) == 0:
            print("No valid new rings this round, stop iteration.")
            break

        pos_batch = torch.cat(all_pos, dim=0)
        all_info['center'] = torch.cat(all_info['center'], dim=0)
        all_info['radius'] = torch.cat(all_info['radius'], dim=0)
        print(f"Generated {pos_batch.shape[0]} total candidate rings.")

        # ------- 2. 相位对齐 + 优化 -------
        q_warm_batch, best_shift = batch_phase_align(pos_demo, q_demo, pos_batch)
        q_batch = optimize_trajectories_batch(
            xarm_gpu,
            pos_batch=pos_batch,
            q_warm_batch=q_warm_batch,
            steps=800,
            lr=1e-2,
            lambda_pos=100.0,
            lambda_smooth=2.0,
            lambda_cone=1.0,
            cone_max_deg=30.0,
            log_every=100
        )

        # ------- 3. 选出半径最大的 top-k -------
        radii = all_info['radius']
        topk = min(topk_keep, len(radii))
        top_idx = torch.topk(radii, k=topk).indices

        # 可选：如果想看半径选择结果
        print(f"Selected top-{topk} radii:", radii[top_idx].cpu().tolist())

        # ------- 4. 更新当前圆，用于下一轮扩展 -------
        current_centers = all_info['center'][top_idx]
        current_radii = radii[top_idx]

        # ------- 5. 可视化当前轮 -------
        # visualize_pos_batch(pos_batch[top_idx])
        # visualize_jnt_traj_video(xarm_sim, q_batch[top_idx])

        # 判断是否超出画布
        if (current_radii.max() + dr) > 0.5:  # 或者按边界条件终止
            print("Reached paper boundary, stopping.")
            break
    visualize_pos_batch(pos_batch[top_idx])
    visualize_jnt_traj_video(xarm_sim, q_batch[top_idx])
    base.run()

