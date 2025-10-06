import os
import math
import numpy as np
import torch
import torch.nn.functional as F

# ====== 依赖 wrs ======
from wrs import wd, rm, mcm  # 仅用于取 robot/参数（本脚本不启可视化）
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6

# -------------------- 基本设置 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
torch.manual_seed(0)
np.random.seed(0)

# -------------------- 构建机器人与场景几何 --------------------
# 不开启渲染窗口；只拿到机器人对象和场景尺寸（桌/纸）
robot = xarm6.XArmLite6Miller(enable_cc=False)
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)

# 桌 & 纸参数（仅用于设定纸面高度与圆心约束）
table_size = np.array([1.5, 1.5, 0.05])
table_pos = np.array([0.6, 0, -0.025])
paper_size = np.array([1.0, 1.0, 0.002])
paper_pos = table_pos.copy()
paper_pos[0] = paper_size[0] / 2.0
paper_pos[1] = 0.0
paper_pos[2] = table_pos[2] + table_size[2] / 2 + paper_size[2] / 2
paper_surface_z = float(paper_pos[2] + paper_size[2] / 2)

# -------------------- 超参数 --------------------
num_points = 96
dof = robot.manipulator.n_dof
safe_clearance = 0.06                   # 初始抬高以便 IK/收敛
alpha_r = 3.0                           # 半径奖励
lambda_smooth = 1e-2
lambda_jl = 1.0                         # 关节越界软惩罚
r_min, r_max = 0.05, 0.40               # 半径上下界
center_x_bounds = (0.30, 0.50)          # 圆心 x 软约束
center_y_abs_max = 0.20                 # 圆心 y 软约束
center_z_bounds = (paper_surface_z + 0.005, paper_surface_z + 0.020)

lr_q = 1e-2
lr_r = 5e-3
lr_c = 3e-3
steps_stage1 = 1000
steps_stage2 = 1500
steps_stage3 = 1500

# -------------------- 目标圆参数（变量） --------------------
# 初始圆心（抬高 safe_clearance）
initial_center = torch.tensor([0.35, 0.0, paper_surface_z + safe_clearance], dtype=DTYPE, device=device)
center = torch.nn.Parameter(initial_center.clone(), requires_grad=False)

radius = torch.nn.Parameter(torch.tensor(0.10, dtype=DTYPE, device=device), requires_grad=False)

# 均匀角度与单位圆方向（在 XY 平面）
thetas = torch.linspace(0, 2 * math.pi, num_points, device=device)
dirs = torch.stack([torch.cos(thetas), torch.sin(thetas), torch.zeros_like(thetas)], dim=1)  # (M,3)

# -------------------- 从 wrs 机器人构建 PoE（可微 FK） --------------------
def _torch_from_np(a):
    return torch.tensor(a, dtype=DTYPE, device=device)

def _skew3(v):  # (...,3) -> (...,3,3)
    O = torch.zeros((*v.shape[:-1], 3, 3), dtype=v.dtype, device=v.device)
    O[..., 0, 1] = -v[..., 2]; O[..., 0, 2] =  v[..., 1]
    O[..., 1, 0] =  v[..., 2]; O[..., 1, 2] = -v[..., 0]
    O[..., 2, 0] = -v[..., 1]; O[..., 2, 1] =  v[..., 0]
    return O

def _exp_so3_batch(omega, theta):  # omega:(3,), theta:(M,)
    M = theta.shape[0]
    wnorm = omega.norm()
    if wnorm < 1e-12:
        return torch.eye(3, dtype=DTYPE, device=device).expand(M, 3, 3).clone()
    w = omega / wnorm
    K = _skew3(w).expand(M, 3, 3)
    th = theta.view(-1, 1, 1)
    I = torch.eye(3, dtype=DTYPE, device=device).expand(M, 3, 3)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)

def _exp_se3_batch(Si, theta):  # Si:(6,), theta:(M,)
    w, v = Si[:3], Si[3:]
    M = theta.shape[0]
    if w.norm() < 1e-12:
        R = torch.eye(3, dtype=DTYPE, device=device).expand(M, 3, 3).clone()
        t = theta.view(-1, 1) * v.view(1, 3)
    else:
        R = _exp_so3_batch(w, theta)
        K = _skew3(w / (w.norm() + 1e-12)).expand(M, 3, 3)
        th = theta.view(-1, 1, 1)
        I = torch.eye(3, dtype=DTYPE, device=device).expand(M, 3, 3)
        V = I * th + (1 - torch.cos(th)) * K + (th - torch.sin(th)) * (K @ K)
        t = torch.einsum('mij,j->mi', V, v)
    T = torch.zeros((M, 4, 4), dtype=DTYPE, device=device)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0
    return T

def _build_poe_from_wrs(robot):
    """从 wrs 读取每节的 loc_rotmat/loc_pos/loc_motion_ax，构造 Slist 和 M（零位末端）。"""
    jnts = robot.manipulator.jlc.jnts
    DOF = robot.manipulator.n_dof
    # 基座外参（默认 I）
    T = torch.eye(4, dtype=DTYPE, device=device)
    omegas, qs = [], []
    for i in range(DOF):
        R_loc = _torch_from_np(jnts[i].loc_rotmat)
        p_loc = _torch_from_np(jnts[i].loc_pos)
        # 累乘父->当前零位
        T_next = T.clone()
        T_next[:3, :3] = T[:3, :3] @ R_loc
        T_next[:3, 3] = T[:3, 3] + T[:3, :3] @ p_loc
        T = T_next
        # 轴向（局部给的是单位 z，已被 loc_rotmat 旋到基座系）
        ax_local = _torch_from_np(jnts[i].loc_motion_ax)
        omega_i = T[:3, :3] @ ax_local
        q_i = T[:3, 3]  # 轴上一点（取该关节原点）
        omegas.append(omega_i)
        qs.append(q_i)
    # 零位末端（你的 TCP=I）
    M = T.clone()
    # 组装 Slist
    S_cols = []
    for w, qpt in zip(omegas, qs):
        v = -torch.cross(w, qpt)
        S_cols.append(torch.cat([w, v]))
    Slist = torch.stack(S_cols, dim=1)  # (6, DOF)
    return Slist, M

_Slist, _M = _build_poe_from_wrs(robot)  # 只构建一次
q_min_np = robot.manipulator.jlc.jnt_ranges[:, 0]
q_max_np = robot.manipulator.jlc.jnt_ranges[:, 1]
q_min = torch.tensor(q_min_np, dtype=DTYPE, device=device)
q_max = torch.tensor(q_max_np, dtype=DTYPE, device=device)

def fk_batch(q_batch: torch.Tensor):
    """可微正运动学：q_batch (M, DOF) -> 末端位置 (M,3)"""
    Mbs = q_batch.shape[0]
    Tq = torch.eye(4, dtype=DTYPE, device=device).unsqueeze(0).expand(Mbs, 4, 4).clone()
    DOF = _Slist.shape[1]
    for i in range(DOF):
        Tq = torch.einsum('mij,mjk->mik', Tq, _exp_se3_batch(_Slist[:, i], q_batch[:, i]))
    Tq = torch.einsum('mij,jk->mik', Tq, _M)
    p = Tq[:, :3, 3]
    return p

# -------------------- 初始化关节轨迹（尝试 IK，失败回 home） --------------------
# 初始位姿：笔尖朝 -Z（若失败自动回退）
R_DEFAULT = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, -1]])

try_pos = initial_center.detach().cpu().numpy().tolist()
init_jnt = robot.ik(tgt_pos=try_pos, tgt_rotmat=R_DEFAULT, option="single")
if init_jnt is None:
    print("[Warn] IK failed at the initial center; fallback to home_conf with small noise.")
    init_jnt = robot.manipulator.home_conf

q_init = np.tile(init_jnt, (num_points, 1)) + 0.01 * np.random.randn(num_points, dof)
q_vars = torch.nn.Parameter(torch.tensor(q_init, dtype=DTYPE, device=device), requires_grad=True)

# -------------------- 统一的优化步 --------------------
def run_optimization(optimizer, steps=1000, stage_name="Stage", center_trainable=False, radius_trainable=False):
    center.requires_grad_(center_trainable)
    radius.requires_grad_(radius_trainable)

    for step in range(steps):
        optimizer.zero_grad()

        # 当前目标/实际
        pos_actual = fk_batch(q_vars)                        # (M,3) 可微
        pos_target = center + radius * dirs                  # (M,3)

        # 位置贴合
        loss_pos = torch.mean(torch.sum((pos_actual - pos_target) ** 2, dim=1))
        # 平滑
        loss_smooth = torch.mean(torch.sum((q_vars[1:] - q_vars[:-1]) ** 2, dim=1))
        # 半径奖励（如果半径此阶段参与优化）
        loss_radius = -alpha_r * radius if radius.requires_grad else torch.tensor(0.0, dtype=DTYPE, device=device)

        # ---- 半径与圆心边界（软约束，全部用 relu）----
        pen = torch.tensor(0.0, dtype=DTYPE, device=device)
        # radius bounds
        pen = pen + (F.relu(r_min - radius) ** 2 + F.relu(radius - r_max) ** 2) * 1000.0
        # center bounds（仅当 center 可训练）
        if center.requires_grad:
            cx, cy, cz = center[0], center[1], center[2]
            pen = pen + (F.relu(center_x_bounds[0] - cx) ** 2 + F.relu(cx - center_x_bounds[1]) ** 2) * 1000.0
            pen = pen + (F.relu(torch.abs(cy) - center_y_abs_max) ** 2) * 1000.0
            pen = pen + (F.relu(center_z_bounds[0] - cz) ** 2 + F.relu(cz - center_z_bounds[1]) ** 2) * 1000.0

        # 关节软限位
        L_jl = (F.relu(q_min - q_vars) ** 2 + F.relu(q_vars - q_max) ** 2).mean()

        total_loss = loss_pos + lambda_smooth * loss_smooth + loss_radius + lambda_jl * L_jl + pen
        total_loss.backward()
        # 可选：裁剪避免数值突跳
        torch.nn.utils.clip_grad_norm_([q_vars], max_norm=1.0)
        optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            c_np = center.detach().cpu().numpy()
            print(f"[{stage_name:>12} {step:4d}] "
                  f"loss={total_loss.item():.6e} | "
                  f"pos={loss_pos.item():.3e} sm={loss_smooth.item():.3e} "
                  f"r={radius.item():.4f} c=({c_np[0]:.3f},{c_np[1]:.3f},{c_np[2]:.3f})")

# -------------------- 三阶段优化 --------------------
print("\n🚀 阶段 1：仅优化关节角 (r, c 固定)")
opt1 = torch.optim.Adam([q_vars], lr=lr_q)
run_optimization(opt1, steps=steps_stage1, stage_name="Pretrain Q", center_trainable=False, radius_trainable=False)

print("\n🚀 阶段 2：解锁半径（优化 q + r）")
opt2 = torch.optim.Adam([
    {"params": [q_vars], "lr": lr_q},
    {"params": [radius], "lr": lr_r},
])
run_optimization(opt2, steps=steps_stage2, stage_name="Optimize r+q", center_trainable=False, radius_trainable=True)

print("\n🚀 阶段 3：解锁圆心（优化 q + r + c）")
opt3 = torch.optim.Adam([
    {"params": [q_vars], "lr": lr_q},
    {"params": [radius, center], "lr": lr_c},
])
run_optimization(opt3, steps=steps_stage3, stage_name="Full optimize", center_trainable=True, radius_trainable=True)

# -------------------- 结果 --------------------
print("\n✅ 优化完成")
print("最大半径 r* =", float(radius.item()))
print("最优圆心 c* =", center.detach().cpu().numpy())
print("q 轨迹形状 =", tuple(q_vars.shape))

with torch.no_grad():
    cx, cy, cz = center.tolist()
    r = float(radius.item())
    th = torch.linspace(0, 2*math.pi, 720, device=device)
    circle_xyz = torch.stack([
        torch.full_like(th, cx) + r*torch.cos(th),
        torch.full_like(th, cy) + r*torch.sin(th),
        torch.full_like(th, cz)                      # 圆在纸面，z=cz
    ], dim=1)  # (720,3)

# 若要转 numpy 保存：
np_circle = circle_xyz.detach().cpu().numpy()  # shape = (720, 3)
for pos in np_circle:
    sphere = mgm.gen_sphere(radius=0.005, pos=pos, rgb=[1,0,0], alpha=1)
    sphere.attach_to(base)
for q in q_vars.detach().cpu().numpy():
    pos, _ = robot.fk(q)
    mgm.gen_sphere(radius=0.008, pos=pos, rgb=[0,0,1], alpha=1).attach_to(base)

import helper_functions as helper
helper.visualize_anime_path(base, robot, q_vars.detach().cpu().numpy())
