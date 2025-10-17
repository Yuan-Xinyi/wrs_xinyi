# -*- coding: utf-8 -*-
"""Grow-a-circle with Beam Search from a demonstration (seed) + FK-based optimization.

依赖: wrs 全家桶（你的工程已有）
示教数据: circle_data.pkl，包含 {"circle": (N,3), "jnt": (N,6), "circle_info": {"center": (3,), "radius": float}}
"""

# ===== wrs reliance =====
from wrs import wd, rm, mcm
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
import wrs.modeling.geometric_model as mgm

# ===== std & 3rd =====
import pickle
import numpy as np
import torch
torch.autograd.set_detect_anomaly(False)

from dataclasses import dataclass
import heapq

# ======================== Init ============================
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

# ======================== Helpers ============================

def paper_max_radius_for_center(center_xy, paper_pos, paper_size, margin=0.06):
    """给定圆心（xy），求在纸面内允许的最大半径（安全距 margin）"""
    cx = float(center_xy[0]); cy = float(center_xy[1])
    px = float(paper_pos[0]); py = float(paper_pos[1])
    lx = float(paper_size[0]); ly = float(paper_size[1])
    x_min, x_max = (px - lx/2) + margin, (px + lx/2) - margin
    y_min, y_max = (py - ly/2) + margin, (py + ly/2) - margin
    left   = cx - x_min
    right  = x_max - cx
    bottom = cy - y_min
    top    = y_max - cy
    return max(0.0, min(left, right, bottom, top))

def make_circle_points(center, radius, num_points, z_fixed, device):
    """基于 (center, radius) 在 z=z_fixed 的平面上采样 N 个圆点（torch 张量）"""
    center = torch.as_tensor(center, dtype=torch.float32, device=device)
    radius = torch.as_tensor(radius, dtype=torch.float32, device=device)
    z_fixed = torch.as_tensor(z_fixed, dtype=torch.float32, device=device)

    th = torch.linspace(0, 2*torch.pi, steps=num_points+1, device=device)[:-1]
    x = center[0] + radius * torch.cos(th)
    y = center[1] + radius * torch.sin(th)
    z = torch.full_like(x, z_fixed)
    return torch.stack([x, y, z], dim=-1)  # (N,3)

def circ_shift_tensor(t, shift):
    """循环移位（正右移）"""
    shift = int(shift) % t.shape[0]
    if shift == 0:
        return t
    return torch.cat([t[-shift:], t[:-shift]], dim=0)

def phase_align_xy(prev_pos, prev_q, new_pos):
    """仅用 XY 做相位对齐，返回相位对齐后的 (pos, q, shift)"""
    N = prev_pos.shape[0]
    errs = []
    for s in range(N):
        ppos_s = circ_shift_tensor(prev_pos, s)
        err = torch.norm((ppos_s[:, :2] - new_pos[:, :2]), dim=1).mean()
        errs.append(err)
    s_best = int(torch.argmin(torch.stack(errs)).item())
    return circ_shift_tensor(prev_pos, s_best), circ_shift_tensor(prev_q, s_best), s_best

def cone_angle_loss_stable(z_axis, z_target, max_angle_deg=30.0):
    """
    z_axis: (...,3) 当前末端Z轴；z_target: (3,) 目标法向（朝下 = [0,0,-1]）
    超过锥半角才惩罚（避免 acos 的数值不稳）
    """
    phi = torch.deg2rad(torch.tensor(max_angle_deg, device=z_axis.device))
    z_axis = z_axis / (torch.norm(z_axis, dim=-1, keepdim=True).clamp(min=1e-8))
    z_target = z_target / (torch.norm(z_target).clamp(min=1e-8))
    cos_angle = (z_axis * z_target).sum(dim=-1).clamp(-0.999999, 0.999999)
    viol = torch.relu(torch.cos(phi) - cos_angle)
    return (viol ** 2).mean(), cos_angle

def optimize_trajectory_from_warm(
    xarm_gpu,
    pos_target,         # (N,3) torch
    q_warm,             # (N,6) torch
    steps=1000,
    lr=1e-2,
    lambda_pos=120.0,
    lambda_smooth=2.0,
    lambda_cone=1.0,
    cone_max_deg=30.0,
    log_every=50
):
    """整条轨迹联合优化（以 q_warm 为初值）"""
    device = pos_target.device
    q_traj = q_warm.clone().detach().to(device).requires_grad_(True)
    opt = torch.optim.Adam([q_traj], lr=lr)
    z_target = torch.tensor([0, 0, -1], dtype=torch.float32, device=device)

    for step in range(steps):
        pos_fk, rot_fk = xarm_gpu.robot.fk_batch(q_traj)   # (N,3), (N,3,3)

        # 位置误差（mean + 0.5*max，避免少数点爆炸）
        per_point = torch.norm(pos_fk - pos_target, dim=-1)    # (N,)
        pos_loss = per_point.mean() + 0.5 * per_point.max()

        # 平滑项
        smooth_loss = torch.mean((q_traj[1:] - q_traj[:-1]) ** 2)

        # 姿态锥约束
        z_axis = rot_fk[:, :, 2]
        cone_loss, cos_angle = cone_angle_loss_stable(z_axis, z_target, max_angle_deg=cone_max_deg)

        loss = lambda_pos * pos_loss + lambda_smooth * smooth_loss + lambda_cone * cone_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step % log_every == 0) or (step == steps - 1):
            mean_ang = torch.rad2deg(torch.acos(cos_angle.clamp(-0.9999, 0.9999))).mean().item()
            print(f"Step {step:03d}: loss={loss.item():.6f}, pos={pos_loss.item():.4f}, "
                  f"smooth={smooth_loss.item():.4f}, cone={cone_loss.item():.4f}, "
                  f"mean_angle={mean_ang:.2f}°")

    return q_traj.detach()

@torch.no_grad()
def check_demo_feasible(xarm_gpu, pos_demo, q_demo, tol_mm=2.0):
    """不优化，直接用示教的 FK 对比示教的目标点，做 sanity check。"""
    pos_fk, _ = xarm_gpu.robot.fk_batch(q_demo)   # (N,3)
    err = torch.norm(pos_fk - pos_demo, dim=-1)   # (N,)
    pos_max = err.max().item()
    return pos_max < (tol_mm/1000.0), pos_max

def try_optimize_one(
    xarm_gpu,
    pos_demo, q_demo,                 # 用于 warm-start + 相位对齐
    center, radius,                   # 本次要检验的圆
    paper_pos, paper_size, margin=0.06,
    steps=800, lr=1e-2,
    pos_tol_mm=2.0, cone_max_deg=30.0
):
    """
    1) 基于 (center, radius) 采样圆；2) 与示教相位对齐得到 q_warm；
    3) 以 FK 误差为目标做轨迹优化；4) 若最大点误差 < pos_tol 则可行。
    """
    device = xarm_gpu.device
    # 边界上界：超界的直接判不可行
    r_max = paper_max_radius_for_center(center[:2], paper_pos, paper_size, margin)
    if radius > r_max + 1e-9:
        return False, None, None, None

    # 圆采样（z 平面用示教的均值，以免 z 不一致）
    z_fixed = float(pos_demo[:, 2].mean().item())
    pos_target = make_circle_points(center, radius, num_points=pos_demo.shape[0],
                                    z_fixed=z_fixed, device=device)  # (N,3)

    # 与示教相位对齐（仅 XY）
    _, q_warm, _ = phase_align_xy(pos_demo, q_demo, pos_target)

    # 优化
    q_opt = optimize_trajectory_from_warm(
        xarm_gpu, pos_target, q_warm,
        steps=steps, lr=lr,
        lambda_pos=120.0, lambda_smooth=2.0, lambda_cone=1.0,
        cone_max_deg=cone_max_deg, log_every=max(1, steps//6)
    )

    # 验证
    with torch.no_grad():
        pos_fk, _ = xarm_gpu.robot.fk_batch(q_opt)
        per_point = torch.norm(pos_fk - pos_target, dim=-1)
        pos_max = per_point.max().item()
        feasible = pos_max < (pos_tol_mm/1000.0)

    return feasible, pos_max, q_opt, pos_target

# ======================== Beam Search ============================

@dataclass
class Node:
    center: np.ndarray   # (3,)
    radius: float
    depth: int
    score: float         # 排序依据（半径越大越好）
    feasible: bool
    q_traj: torch.Tensor | None = None  # (N,6)
    pos_target: torch.Tensor | None = None

def expand_children(node: Node,
                    num_dirs=8,
                    d_center=0.02,    # 圆心每步漂移
                    growth=1.10,      # 半径乘法增量
                    additive_dr=None  # 或者用加法增量
                    ):
    """基于当前节点生成 proposal（不做可行性判断）"""
    angles = np.linspace(0, 2*np.pi, num_dirs, endpoint=False)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (K,2)
    if additive_dr is None:
        r_new = node.radius * growth
    else:
        r_new = node.radius + additive_dr

    centers = []
    for k in range(num_dirs):
        delta = d_center * dirs[k]
        c_new = np.array([node.center[0] + delta[0],
                          node.center[1] + delta[1],
                          node.center[2]], dtype=float)
        centers.append(c_new)
    return centers, r_new

def beam_search_grow_circle(
    xarm_gpu,
    pos_demo, q_demo,              # 示教 (N,3)/(N,6) (torch)
    circle_center0, radius0,       # 示教圆心/半径 (np / float)
    paper_pos, paper_size, margin=0.06,
    # beam & depth
    beam_width=6, max_depth=12,
    # 扩展参数
    num_dirs=8, d_center=0.02, growth=1.10, additive_dr=None,
    # 优化参数
    quick_steps=300, quick_tol_mm=3.0,      # 快速验算
    refine_steps=800, refine_tol_mm=2.0,    # 通过后再精炼
    lr=1e-2, cone_max_deg=30.0
):
    """从示教出发做树式扩展 + 剪枝，返回最佳节点与访问节点数"""
    device = xarm_gpu.device

    def push(pq, node): heapq.heappush(pq, (-node.score, node))
    def pop(pq): return heapq.heappop(pq)[1]

    # 1) 示教直接作为种子（**不重采样，不判定不可行**）
    best_node = Node(center=np.array(circle_center0, dtype=float),
                     radius=float(radius0), depth=0, score=float(radius0),
                     feasible=True, q_traj=q_demo.detach(), pos_target=pos_demo.detach())
    frontier = []
    push(frontier, best_node)
    visited = 1

    # 上界剪枝
    def upper_bound_r(center_xy):
        return paper_max_radius_for_center(center_xy, paper_pos, paper_size, margin)

    # 2) 层层扩展
    for depth in range(1, max_depth+1):
        if not frontier:
            break

        # 当前层 beam
        current_layer = []
        for _ in range(min(beam_width, len(frontier))):
            current_layer.append(pop(frontier))

        next_candidates = []
        for parent in current_layer:
            centers, r_prop = expand_children(parent, num_dirs=num_dirs,
                                              d_center=d_center, growth=growth,
                                              additive_dr=additive_dr)

            for c_new in centers:
                # 上界剪枝
                ub = upper_bound_r(c_new[:2])
                if ub <= best_node.radius + 1e-6:
                    continue
                if r_prop > ub + 1e-9:
                    continue

                # 快速检验
                ok, loss, q_est, pos_tgt = try_optimize_one(
                    xarm_gpu, pos_demo, q_demo,
                    center=c_new, radius=r_prop,
                    paper_pos=paper_pos, paper_size=paper_size, margin=margin,
                    steps=quick_steps, lr=lr, pos_tol_mm=quick_tol_mm, cone_max_deg=cone_max_deg
                )
                visited += 1
                if not ok:
                    continue

                # 精炼
                ok2, loss2, q_ref, pos_tgt2 = try_optimize_one(
                    xarm_gpu, pos_demo, q_demo,
                    center=c_new, radius=r_prop,
                    paper_pos=paper_pos, paper_size=paper_size, margin=margin,
                    steps=refine_steps, lr=lr, pos_tol_mm=refine_tol_mm, cone_max_deg=cone_max_deg
                )
                if ok2:
                    n = Node(center=c_new, radius=r_prop, depth=depth,
                             score=r_prop, feasible=True, q_traj=q_ref, pos_target=pos_tgt2)
                else:
                    n = Node(center=c_new, radius=r_prop, depth=depth,
                             score=r_prop, feasible=True, q_traj=q_est, pos_target=pos_tgt)

                next_candidates.append(n)
                if r_prop > best_node.radius:
                    best_node = n

        # 下一层：半径排序 + 截断
        next_candidates.sort(key=lambda n: n.radius, reverse=True)
        for n in next_candidates[:beam_width]:
            push(frontier, n)

        if not next_candidates:
            break

    return best_node, visited

# ======================== Visualization ============================

def visualize_circle_points(pos_t):
    pts = pos_t.detach().cpu().numpy()
    for i in range(pts.shape[0]):
        mgm.gen_sphere(pos=pts[i], radius=0.004, rgb=[1,0,0]).attach_to(base)

def visualize_traj_mesh(robot_sim, q_traj_t, alpha=0.28, rgb=[0,0,1]):
    q_np = q_traj_t.detach().cpu().numpy()
    for i in range(q_np.shape[0]):
        robot_sim.goto_given_conf(jnt_values=q_np[i])
        robot_sim.gen_meshmodel(alpha=alpha, rgb=rgb).attach_to(base)

# ======================== Main ============================

if __name__ == "__main__":
    # 载入示教（注意：这里统一为 TCP 坐标）
    with open("0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro/circle_data.pkl", "rb") as f:
        data = pickle.load(f)

    pos_demo = torch.as_tensor(data["circle"], dtype=torch.float32, device=device).clone()  # (N,3)
    q_demo   = torch.as_tensor(data["jnt"],    dtype=torch.float32, device=device).clone()  # (N,6)
    circle_info = data["circle_info"]
    c0 = np.array(circle_info["center"].to("cpu"), dtype=float)  # (3,)
    r0 = float(circle_info["radius"])

    # ——可视化示教（可注释）——
    # visualize_circle_points(pos_demo)
    # visualize_traj_mesh(xarm_sim, q_demo, alpha=0.15, rgb=[0.2,0.2,1.0])

    # Beam Search：从示教生长
    best, visited = beam_search_grow_circle(
        xarm_gpu,
        pos_demo=pos_demo, q_demo=q_demo,
        circle_center0=c0, radius0=r0,
        paper_pos=paper_pos, paper_size=paper_size, margin=0.06,
        beam_width=6, max_depth=12,
        num_dirs=8, d_center=0.02, growth=1.10, additive_dr=None,
        quick_steps=300, quick_tol_mm=3.0,
        refine_steps=800, refine_tol_mm=2.0,
        lr=1e-2, cone_max_deg=30.0
    )

    if best is None:
        print("种子不可行或无可行扩展。")
    else:
        print(f"[DONE] visited={visited}, best_radius={best.radius:.3f}, center={best.center}")

        # 画最佳结果
        mgm.gen_sphere(pos=best.center, radius=0.006, rgb=[0,1,0]).attach_to(base)
        visualize_circle_points(best.pos_target)
        visualize_traj_mesh(xarm_sim, best.q_traj, alpha=0.28, rgb=[0,0,1])

    base.run()
