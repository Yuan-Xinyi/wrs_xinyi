from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
LENGTH_PREDICTION_DIR = PARENT_DIR / 'length_prediction'
if str(LENGTH_PREDICTION_DIR) not in sys.path:
    sys.path.insert(0, str(LENGTH_PREDICTION_DIR))

from length_prediction.lnet_contrastive_fr3_rotation_cone_eval import build_tracker
from trajectory_generation.fr3_nullspace_straight import (
    directional_manipulability_batch,
    damped_pseudoinverse_batch,
    joint_limit_avoidance_velocity_batch,
    nullspace_projector_batch,
)


@dataclass
class RolloutObjectiveWeights:
    progress_reward: float = 1.0
    pos_error_weight: float = 40.0
    orientation_weight: float = 10.0
    joint_center_weight: float = 0.1
    collision_weight: float = 20.0
    low_mu_weight: float = 10.0
    control_weight: float = 0.01


def repeat_vec(vec_np: np.ndarray, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.repeat(vec_np[None, :].astype(np.float32), batch_size, axis=0)).to(device)


def q_from_raw(raw_q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    return center + half * torch.tanh(raw_q)


def inv_tanh_clamped(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def raw_from_q(q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    normalized = (q - center) / half.clamp_min(1e-6)
    return inv_tanh_clamped(normalized)


def sample_raw_q_batch(robot, batch_size: int, device: torch.device) -> torch.Tensor:
    q_np = robot.rand_conf_batch(int(batch_size)).detach().cpu().numpy().astype(np.float32)
    q = torch.from_numpy(q_np).to(device)
    lower = robot.jnt_ranges[:, 0].unsqueeze(0)
    upper = robot.jnt_ranges[:, 1].unsqueeze(0)
    return raw_from_q(q, lower, upper).detach()


def start_pose_score(
    tracker,
    q_batch: torch.Tensor,
    target_pos: torch.Tensor,
    target_normal: torch.Tensor,
    pos_weight: float,
    orientation_weight: float,
    joint_center_weight: float,
    collision_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    tcp_pos, tcp_rot = tracker.robot.fk_batch(q_batch)
    tcp_z = tcp_rot[:, :, 2]
    pos_err = torch.linalg.norm(tcp_pos - target_pos, dim=-1)
    cos_theta = torch.sum(tcp_z * target_normal, dim=-1)
    cos_theta_max = float(np.cos(tracker.config.theta_max))
    ori_violation = torch.relu(cos_theta_max - cos_theta)
    coll_cost = torch.relu(tracker.collision_fn(q_batch))
    lower = tracker.robot.jnt_ranges[:, 0].unsqueeze(0)
    upper = tracker.robot.jnt_ranges[:, 1].unsqueeze(0)
    center = 0.5 * (lower + upper)
    span = (upper - lower).clamp_min(1e-6)
    joint_center_penalty = torch.mean(((q_batch - center) / span) ** 2, dim=-1)
    score = (
        -float(pos_weight) * (pos_err ** 2)
        -float(orientation_weight) * (ori_violation ** 2)
        -float(joint_center_weight) * joint_center_penalty
        -float(collision_weight) * (coll_cost ** 2)
    )
    stats = {
        'pos_err': pos_err,
        'ori_violation': ori_violation,
        'collision_cost': coll_cost,
        'joint_center_penalty': joint_center_penalty,
    }
    return score, stats


def differentiable_control_rollout_score(
    tracker,
    q0_batch: torch.Tensor,
    direction_batch: torch.Tensor,
    target_normal_batch: torch.Tensor,
    control_seq: torch.Tensor,
    weights: RolloutObjectiveWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    q = q0_batch
    target_normal_batch = target_normal_batch / target_normal_batch.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    direction_batch = direction_batch / direction_batch.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    start_pos, _ = tracker.robot.fk_batch(q)
    lower = tracker.robot.jnt_ranges[:, 0].unsqueeze(0)
    upper = tracker.robot.jnt_ranges[:, 1].unsqueeze(0)
    cos_theta_max = float(np.cos(tracker.config.theta_max))
    total_penalty = torch.zeros(q.shape[0], device=q.device, dtype=q.dtype)
    total_control_penalty = torch.zeros_like(total_penalty)

    horizon_steps = int(control_seq.shape[1])
    for step_idx in range(horizon_steps):
        q = q + tracker.config.dt * control_seq[:, step_idx, :]
        tcp_pos, tcp_rot = tracker.robot.fk_batch(q)
        tcp_z = tcp_rot[:, :, 2]
        cos_theta = torch.sum(tcp_z * target_normal_batch, dim=-1)

        j_cols = []
        for dim in range(3):
            grad_dim = torch.autograd.grad(
                tcp_pos[:, dim].sum(),
                q,
                retain_graph=True,
                create_graph=True,
            )[0]
            j_cols.append(grad_dim)
        j_pos = torch.stack(j_cols, dim=1)
        mu_val = directional_manipulability_batch(j_pos, direction_batch, tracker.config.damping)

        expected_pos = start_pos + direction_batch * ((step_idx + 1) * tracker.config.dt * tracker.config.task_speed)
        pos_err = torch.linalg.norm(tcp_pos - expected_pos, dim=-1)
        ori_violation = torch.relu(cos_theta_max - cos_theta)
        lower_violation = torch.relu(lower - q)
        upper_violation = torch.relu(q - upper)
        boundary_penalty = torch.mean((lower_violation + upper_violation) ** 2, dim=-1)
        collision_penalty = torch.relu(tracker.collision_fn(q))
        low_mu_penalty = torch.relu(tracker.config.mu_threshold - mu_val)

        total_penalty = (
            total_penalty
            + float(weights.pos_error_weight) * (pos_err ** 2)
            + float(weights.orientation_weight) * (ori_violation ** 2)
            + float(weights.joint_center_weight) * boundary_penalty
            + float(weights.collision_weight) * (collision_penalty ** 2)
            + float(weights.low_mu_weight) * (low_mu_penalty ** 2)
        )
        total_control_penalty = total_control_penalty + float(weights.control_weight) * torch.mean(control_seq[:, step_idx, :] ** 2, dim=-1)

    end_pos, _ = tracker.robot.fk_batch(q)
    projected_length = torch.sum((end_pos - start_pos) * direction_batch, dim=-1)
    score = float(weights.progress_reward) * projected_length - total_penalty - total_control_penalty
    stats = {
        'projected_length': projected_length,
        'total_penalty': total_penalty,
        'total_control_penalty': total_control_penalty,
    }
    return score, stats


def optimize_start_pose_multistart(
    tracker,
    target_pos_np: np.ndarray,
    target_normal_np: np.ndarray,
    num_starts: int,
    iters: int,
    lr: float,
    pos_weight: float = 200.0,
    orientation_weight: float = 20.0,
    joint_center_weight: float = 0.1,
    collision_weight: float = 20.0,
) -> tuple[np.ndarray, dict]:
    device = tracker.robot.jnt_ranges.device
    lower = tracker.robot.jnt_ranges[:, 0].unsqueeze(0)
    upper = tracker.robot.jnt_ranges[:, 1].unsqueeze(0)
    raw_q = sample_raw_q_batch(tracker.robot, int(num_starts), device).requires_grad_(True)
    target_pos = repeat_vec(target_pos_np, int(num_starts), device)
    target_normal = repeat_vec(target_normal_np, int(num_starts), device)
    optimizer = torch.optim.Adam([raw_q], lr=float(lr))

    for _ in range(int(iters)):
        optimizer.zero_grad(set_to_none=True)
        q_batch = q_from_raw(raw_q, lower, upper)
        score, _ = start_pose_score(
            tracker=tracker,
            q_batch=q_batch,
            target_pos=target_pos,
            target_normal=target_normal,
            pos_weight=float(pos_weight),
            orientation_weight=float(orientation_weight),
            joint_center_weight=float(joint_center_weight),
            collision_weight=float(collision_weight),
        )
        loss = -score.mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        q_batch = q_from_raw(raw_q, lower, upper)
        score, stats = start_pose_score(
            tracker=tracker,
            q_batch=q_batch,
            target_pos=target_pos,
            target_normal=target_normal,
            pos_weight=float(pos_weight),
            orientation_weight=float(orientation_weight),
            joint_center_weight=float(joint_center_weight),
            collision_weight=float(collision_weight),
        )
    best_idx = int(torch.argmax(score).item())
    return q_batch.detach().cpu().numpy().astype(np.float32), {
        'best_idx': best_idx,
        'best_score': float(score[best_idx].item()),
        'best_pos_err_mm': float(stats['pos_err'][best_idx].item() * 1e3),
        'best_ori_violation': float(stats['ori_violation'][best_idx].item()),
        'best_collision_cost': float(stats['collision_cost'][best_idx].item()),
    }


def differentiable_closed_loop_rollout_score(
    tracker,
    q0_batch: torch.Tensor,
    direction_batch: torch.Tensor,
    target_normal_batch: torch.Tensor,
    horizon_steps: int,
    weights: RolloutObjectiveWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    q = q0_batch
    target_normal_batch = target_normal_batch / target_normal_batch.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    direction_batch = direction_batch / direction_batch.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    start_pos, _ = tracker.robot.fk_batch(q)
    lower = tracker.robot.jnt_ranges[:, 0].unsqueeze(0)
    upper = tracker.robot.jnt_ranges[:, 1].unsqueeze(0)
    cos_theta_max = float(np.cos(tracker.config.theta_max))
    total_penalty = torch.zeros(q.shape[0], device=q.device, dtype=q.dtype)
    total_control_penalty = torch.zeros_like(total_penalty)

    for step_idx in range(int(horizon_steps)):
        q_eval = q
        tcp_pos, tcp_rot = tracker.robot.fk_batch(q_eval)
        tcp_z = tcp_rot[:, :, 2]
        cos_theta = torch.sum(tcp_z * target_normal_batch, dim=-1)

        j_cols = []
        for dim in range(3):
            grad_dim = torch.autograd.grad(
                tcp_pos[:, dim].sum(),
                q_eval,
                retain_graph=True,
                create_graph=True,
            )[0]
            j_cols.append(grad_dim)
        j_pos = torch.stack(j_cols, dim=1)
        j_g = torch.autograd.grad(cos_theta.sum(), q_eval, retain_graph=True, create_graph=True)[0].unsqueeze(1)

        mu_val = directional_manipulability_batch(j_pos, direction_batch, tracker.config.damping)
        grad_mu = torch.autograd.grad(mu_val.sum(), q_eval, retain_graph=True, create_graph=True)[0]

        on_boundary = (cos_theta <= cos_theta_max).view(-1, 1, 1)
        j_task = torch.cat([j_pos, j_g * on_boundary], dim=1)
        j_pinv = damped_pseudoinverse_batch(j_task, tracker.config.damping)
        projector = nullspace_projector_batch(j_task, tracker.config.damping)

        v_pos = (tracker.config.task_speed * direction_batch).unsqueeze(-1)
        v_g = (tracker.config.boundary_gain * torch.clamp(cos_theta_max - cos_theta, min=0.0)).view(-1, 1, 1)
        v_g = v_g * on_boundary.squeeze(-1).float().unsqueeze(-1)
        v_task = torch.cat([v_pos, v_g], dim=1)

        q_dot_task = (j_pinv @ v_task).squeeze(-1)
        q_dot_joint_limit = tracker.config.joint_limit_gain * joint_limit_avoidance_velocity_batch(tracker.robot, q_eval)
        q_dot_null = tracker.config.null_gain * grad_mu + q_dot_joint_limit
        q_dot = q_dot_task + (projector @ q_dot_null.unsqueeze(-1)).squeeze(-1)
        q = q + tracker.config.dt * q_dot

        tcp_pos_next, tcp_rot_next = tracker.robot.fk_batch(q)
        tcp_z_next = tcp_rot_next[:, :, 2]
        cos_theta_next = torch.sum(tcp_z_next * target_normal_batch, dim=-1)
        expected_pos = start_pos + direction_batch * ((step_idx + 1) * tracker.config.dt * tracker.config.task_speed)
        pos_err = torch.linalg.norm(tcp_pos_next - expected_pos, dim=-1)
        ori_violation = torch.relu(cos_theta_max - cos_theta_next)
        lower_violation = torch.relu(lower - q)
        upper_violation = torch.relu(q - upper)
        boundary_penalty = torch.mean((lower_violation + upper_violation) ** 2, dim=-1)
        collision_penalty = torch.relu(tracker.collision_fn(q))
        low_mu_penalty = torch.relu(tracker.config.mu_threshold - mu_val)

        total_penalty = (
            total_penalty
            + float(weights.pos_error_weight) * (pos_err ** 2)
            + float(weights.orientation_weight) * (ori_violation ** 2)
            + float(weights.joint_center_weight) * boundary_penalty
            + float(weights.collision_weight) * (collision_penalty ** 2)
            + float(weights.low_mu_weight) * (low_mu_penalty ** 2)
        )
        total_control_penalty = total_control_penalty + float(weights.control_weight) * torch.mean(q_dot ** 2, dim=-1)

    end_pos, _ = tracker.robot.fk_batch(q)
    projected_length = torch.sum((end_pos - start_pos) * direction_batch, dim=-1)
    score = float(weights.progress_reward) * projected_length - total_penalty - total_control_penalty
    stats = {
        'projected_length': projected_length,
        'total_penalty': total_penalty,
        'total_control_penalty': total_control_penalty,
    }
    return score, stats
