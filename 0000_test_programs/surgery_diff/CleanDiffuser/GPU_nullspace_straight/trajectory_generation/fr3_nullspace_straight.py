import argparse
from dataclasses import dataclass

import jax
import jax2torch
import numpy as np
import torch

from wrs import wd, mcm
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.neuro._kinematics.jlchain as jlc
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as fr3_sim
from wrs.robot_sim.robots.franka_research_3.sphere_collision_checker import SphereCollisionChecker


class FrankaResearch3GPU:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.robot = jlc.JLChain(n_dof=7, pos=torch.zeros(3, device=self.device), rotmat=torch.eye(3, device=self.device))
        self._build_robot()
        self.robot.finalize()

    def _build_robot(self):
        r = self.robot
        d = self.device

        r.jnts[0].loc_pos = torch.tensor([0.0, 0.0, 0.333], dtype=torch.float32, device=d)
        r.jnts[0].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[0].motion_range = torch.tensor([-2.8973, 2.8973], dtype=torch.float32, device=d)

        r.jnts[1].loc_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=d)
        r.jnts[1].loc_rotmat = torch.tensor(rm.rotmat_from_euler(-1.57079632679, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[1].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[1].motion_range = torch.tensor([-1.8326, 1.8326], dtype=torch.float32, device=d)

        r.jnts[2].loc_pos = torch.tensor([0.0, -0.316, 0.0], dtype=torch.float32, device=d)
        r.jnts[2].loc_rotmat = torch.tensor(rm.rotmat_from_euler(1.57079632679, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[2].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[2].motion_range = torch.tensor([-2.8972, 2.8972], dtype=torch.float32, device=d)

        r.jnts[3].loc_pos = torch.tensor([0.0825, 0.0, 0.0], dtype=torch.float32, device=d)
        r.jnts[3].loc_rotmat = torch.tensor(rm.rotmat_from_euler(1.57079632679, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[3].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[3].motion_range = torch.tensor([-3.0718, -0.1222], dtype=torch.float32, device=d)

        r.jnts[4].loc_pos = torch.tensor([-0.0825, 0.384, 0.0], dtype=torch.float32, device=d)
        r.jnts[4].loc_rotmat = torch.tensor(rm.rotmat_from_euler(-1.57079632679, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[4].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[4].motion_range = torch.tensor([-2.8798, 2.8798], dtype=torch.float32, device=d)

        r.jnts[5].loc_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=d)
        r.jnts[5].loc_rotmat = torch.tensor(rm.rotmat_from_euler(1.57079632679, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[5].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[5].motion_range = torch.tensor([0.4364, 4.6251], dtype=torch.float32, device=d)

        r.jnts[6].loc_pos = torch.tensor([0.088, 0.0, 0.0], dtype=torch.float32, device=d)
        r.jnts[6].loc_rotmat = torch.tensor(rm.rotmat_from_euler(1.57079632679, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[6].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[6].motion_range = torch.tensor([-3.0543, 3.0543], dtype=torch.float32, device=d)

        # flange 0.107 + hand acting center 0.1034
        r._loc_flange_pos = torch.tensor([0.0, 0.0, 0.2104], dtype=torch.float32, device=d)



def normalize_batch(vec: torch.Tensor) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)



def random_unit_vectors_batch(batch_size: int, device: torch.device) -> torch.Tensor:
    return normalize_batch(torch.randn(batch_size, 3, device=device))



def project_direction_to_plane_batch(direction: torch.Tensor, target_normal: torch.Tensor) -> torch.Tensor:
    target_normal = normalize_batch(target_normal)
    proj_on_normal = torch.sum(direction * target_normal, dim=-1, keepdim=True)
    direction_in_plane = direction - proj_on_normal * target_normal
    tiny_mask = direction_in_plane.norm(dim=-1, keepdim=True) < 1e-8
    if tiny_mask.any():
        fallback = torch.zeros_like(direction_in_plane)
        fallback[..., 0] = 1.0
        fallback = fallback - torch.sum(fallback * target_normal, dim=-1, keepdim=True) * target_normal
        fallback = normalize_batch(fallback)
        direction_in_plane = torch.where(tiny_mask, fallback, direction_in_plane)
    return normalize_batch(direction_in_plane)



def position_jacobian_batch(robot, q_batch: torch.Tensor, create_graph: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    q_eval = q_batch.detach().clone().requires_grad_(True)
    tcp_pos, _ = robot.fk_batch(q_eval)
    grads = []
    for dim in range(3):
        grad_dim = torch.autograd.grad(
            tcp_pos[:, dim].sum(),
            q_eval,
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        grads.append(grad_dim)
    return torch.stack(grads, dim=1), q_eval



def damped_pseudoinverse_batch(j_mat: torch.Tensor, damping: float) -> torch.Tensor:
    batch, task_dim, _ = j_mat.shape
    eye = torch.eye(task_dim, device=j_mat.device, dtype=j_mat.dtype).unsqueeze(0).expand(batch, -1, -1)
    jj_t = j_mat @ j_mat.transpose(1, 2)
    return j_mat.transpose(1, 2) @ torch.linalg.inv(jj_t + (damping ** 2) * eye)



def nullspace_projector_batch(j_pos: torch.Tensor, damping: float) -> torch.Tensor:
    j_pinv = damped_pseudoinverse_batch(j_pos, damping)
    batch, _, dof = j_pos.shape
    eye = torch.eye(dof, device=j_pos.device, dtype=j_pos.dtype).unsqueeze(0).expand(batch, -1, -1)
    return eye - j_pinv @ j_pos



def directional_manipulability_batch(j_pos: torch.Tensor, direction: torch.Tensor, damping: float) -> torch.Tensor:
    batch = j_pos.shape[0]
    eye = torch.eye(3, device=j_pos.device, dtype=j_pos.dtype).unsqueeze(0).expand(batch, -1, -1)
    metric = j_pos @ j_pos.transpose(1, 2) + (damping ** 2) * eye
    dir_col = direction.unsqueeze(-1)
    values = dir_col.transpose(1, 2) @ torch.linalg.inv(metric) @ dir_col
    return values.squeeze(-1).squeeze(-1).clamp_min(1e-12).pow(-0.5)



def joints_in_range_mask(robot, q_batch: torch.Tensor) -> torch.Tensor:
    lower = robot.jnt_ranges[:, 0].unsqueeze(0)
    upper = robot.jnt_ranges[:, 1].unsqueeze(0)
    return ((q_batch >= lower) & (q_batch <= upper)).all(dim=1)



def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normal / max(np.linalg.norm(normal), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-12)
    return np.column_stack((x_axis, y_axis, z_axis))


@dataclass
class TrackerConfig:
    dt: float = 0.01
    task_speed: float = 0.1
    damping: float = 1e-3
    null_gain: float = 0.6
    theta_max: float = np.deg2rad(30.0)
    boundary_gain: float = 10.0
    max_steps: int = 2000
    mu_threshold: float = 0.01
    pos_error_threshold: float = 0.01


@dataclass
class BatchTrackerResult:
    q0_batch: torch.Tensor
    direction_batch: torch.Tensor
    target_normal_batch: torch.Tensor
    projected_length: torch.Tensor
    euclidean_length: torch.Tensor
    mean_mu: torch.Tensor
    steps: torch.Tensor
    termination_code: torch.Tensor
    best_idx: int
    q_path_best: list[np.ndarray]
    tcp_path_best: list[np.ndarray]


class GPUNullspaceStraightTracker:
    def __init__(self, robot, collision_fn, config: TrackerConfig, print_every: int = 50):
        self.robot = robot
        self.collision_fn = collision_fn
        self.config = config
        self.print_every = max(0, int(print_every))

    def sample_valid_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_list, d_list, n_list = [], [], []
        remaining = batch_size
        oversample = max(256, batch_size * 4)
        cos_theta_max = float(np.cos(self.config.theta_max))
        while remaining > 0:
            q_cand = self.robot.rand_conf_batch(oversample).to(device)
            n_cand = random_unit_vectors_batch(oversample, device=device)
            d_cand = project_direction_to_plane_batch(torch.randn(oversample, 3, device=device), n_cand)
            j_pos, _ = position_jacobian_batch(self.robot, q_cand, create_graph=False)
            mu = directional_manipulability_batch(j_pos, d_cand, self.config.damping)
            _, tcp_rot = self.robot.fk_batch(q_cand)
            tcp_z = tcp_rot[:, :, 2]
            cos_theta = torch.sum(tcp_z * n_cand, dim=-1)
            coll_cost = self.collision_fn(q_cand)
            valid_mask = (
                joints_in_range_mask(self.robot, q_cand)
                & (coll_cost <= 0.0)
                & (mu > self.config.mu_threshold)
                & (cos_theta > cos_theta_max)
            )
            if valid_mask.any():
                take_q = q_cand[valid_mask][:remaining]
                take_d = d_cand[valid_mask][:remaining]
                take_n = n_cand[valid_mask][:remaining]
                q_list.append(take_q)
                d_list.append(take_d)
                n_list.append(take_n)
                remaining -= take_q.shape[0]
                print(f'[sample] collected={batch_size - remaining}/{batch_size} valid starts')
        return torch.cat(q_list, dim=0), torch.cat(d_list, dim=0), torch.cat(n_list, dim=0)

    def run_batch(self, q0_batch: torch.Tensor, direction_batch: torch.Tensor, target_normal_batch: torch.Tensor) -> BatchTrackerResult:
        q = q0_batch.clone()
        target_normal_batch = normalize_batch(target_normal_batch)
        direction = project_direction_to_plane_batch(direction_batch, target_normal_batch)
        start_pos, _ = self.robot.fk_batch(q)
        batch_size = q.shape[0]
        cos_theta_max = float(np.cos(self.config.theta_max))
        active = torch.ones(batch_size, dtype=torch.bool, device=q.device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=q.device)
        termination_code = torch.full((batch_size,), 3, dtype=torch.long, device=q.device)
        step_counter = torch.zeros(batch_size, dtype=torch.long, device=q.device)
        mu_sum = torch.zeros(batch_size, dtype=torch.float32, device=q.device)
        mu_count = torch.zeros(batch_size, dtype=torch.float32, device=q.device)

        q_history = torch.empty((self.config.max_steps + 1, batch_size, q.shape[1]), device=q.device, dtype=q.dtype)
        tcp_history = torch.empty((self.config.max_steps + 1, batch_size, 3), device=q.device, dtype=start_pos.dtype)
        q_history[0] = q
        tcp_history[0] = start_pos

        for step_idx in range(self.config.max_steps):
            if not active.any():
                print(f'[step {step_idx:04d}] all samples terminated')
                break

            q_eval = q.detach().clone().requires_grad_(True)
            tcp_pos, tcp_rot = self.robot.fk_batch(q_eval)
            tcp_z = tcp_rot[:, :, 2]
            cos_theta = torch.sum(tcp_z * target_normal_batch, dim=-1)

            j_cols = []
            for dim in range(3):
                grad_dim = torch.autograd.grad(tcp_pos[:, dim].sum(), q_eval, retain_graph=True, create_graph=True)[0]
                j_cols.append(grad_dim)
            j_pos = torch.stack(j_cols, dim=1)
            j_g = torch.autograd.grad(cos_theta.sum(), q_eval, retain_graph=True, create_graph=False)[0].unsqueeze(1)

            mu_val = directional_manipulability_batch(j_pos, direction, self.config.damping)
            mu_sum[active] += mu_val[active]
            mu_count[active] += 1.0

            low_mu = active & (mu_val < self.config.mu_threshold)
            termination_code[low_mu] = 0
            done[low_mu] = True
            active = active & (~low_mu)
            if not active.any():
                break

            grad_mu = torch.autograd.grad(mu_val.sum(), q_eval, retain_graph=False, create_graph=False)[0]
            on_boundary = (cos_theta <= cos_theta_max).view(-1, 1, 1)
            j_task = torch.cat([j_pos, j_g * on_boundary], dim=1)
            j_pinv = damped_pseudoinverse_batch(j_task, self.config.damping)
            projector = nullspace_projector_batch(j_task, self.config.damping)

            v_pos = (self.config.task_speed * direction).unsqueeze(-1)
            v_g = (self.config.boundary_gain * torch.clamp(cos_theta_max - cos_theta, min=0.0)).view(-1, 1, 1)
            v_g = v_g * on_boundary.squeeze(-1).float().unsqueeze(-1)
            v_task = torch.cat([v_pos, v_g], dim=1)

            q_dot_task = (j_pinv @ v_task).squeeze(-1)
            q_dot_null = self.config.null_gain * grad_mu
            q_dot = q_dot_task + (projector @ q_dot_null.unsqueeze(-1)).squeeze(-1)
            q_next = q + self.config.dt * q_dot

            in_range = joints_in_range_mask(self.robot, q_next)
            joint_limit = active & (~in_range)
            termination_code[joint_limit] = 1
            done[joint_limit] = True

            coll_cost = self.collision_fn(q_next)
            collided = active & (~joint_limit) & (coll_cost > 0.0)
            termination_code[collided] = 2
            done[collided] = True

            tcp_pos_candidate, _ = self.robot.fk_batch(q_next)
            expected_pos = start_pos + direction * ((step_idx + 1) * self.config.dt * self.config.task_speed)
            pos_error = torch.linalg.norm(tcp_pos_candidate - expected_pos, dim=1)
            pos_failed = active & (~joint_limit) & (~collided) & (pos_error > self.config.pos_error_threshold)
            termination_code[pos_failed] = 4
            done[pos_failed] = True

            advance = active & in_range & (~collided) & (~pos_failed)
            q[advance] = q_next[advance]
            step_counter[advance] += 1
            active = advance

            tcp_pos_now, _ = self.robot.fk_batch(q)
            q_history[step_idx + 1] = q
            tcp_history[step_idx + 1] = tcp_pos_now

            if self.print_every > 0 and (step_idx + 1) % self.print_every == 0:
                delta_now = tcp_pos_now - start_pos
                proj_now = torch.sum(delta_now * direction, dim=1)
                print(f'[step {step_idx + 1:04d}] active={int(active.sum().item())}/{batch_size} mean_proj={proj_now.mean().item():.4f}m max_proj={proj_now.max().item():.4f}m mean_mu={mu_val.mean().item():.6f}')

        end_pos, _ = self.robot.fk_batch(q)
        delta = end_pos - start_pos
        projected_length = torch.sum(delta * direction, dim=1)
        euclidean_length = torch.linalg.norm(delta, dim=1)
        mean_mu = mu_sum / mu_count.clamp_min(1.0)
        maxed = (~done) & (step_counter >= self.config.max_steps)
        termination_code[maxed] = 3

        best_idx = int(torch.argmax(projected_length).item())
        best_num_points = int(step_counter[best_idx].item()) + 1
        q_path_best = [arr.detach().cpu().numpy().copy() for arr in q_history[:best_num_points, best_idx]]
        tcp_path_best = [arr.detach().cpu().numpy().copy() for arr in tcp_history[:best_num_points, best_idx]]

        return BatchTrackerResult(
            q0_batch=q0_batch,
            direction_batch=direction,
            target_normal_batch=target_normal_batch.detach().cpu().clone(),
            projected_length=projected_length,
            euclidean_length=euclidean_length,
            mean_mu=mean_mu,
            steps=step_counter,
            termination_code=termination_code,
            best_idx=best_idx,
            q_path_best=q_path_best,
            tcp_path_best=tcp_path_best,
        )



    def collect_batch_trajectories(
        self,
        q0_batch: torch.Tensor,
        direction_batch: torch.Tensor,
        target_normal_batch: torch.Tensor,
    ) -> list[dict]:
        q = q0_batch.clone()
        target_normal_batch = normalize_batch(target_normal_batch)
        direction = project_direction_to_plane_batch(direction_batch, target_normal_batch)
        start_pos, _ = self.robot.fk_batch(q)
        batch_size = q.shape[0]
        cos_theta_max = float(np.cos(self.config.theta_max))
        active = torch.ones(batch_size, dtype=torch.bool, device=q.device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=q.device)
        termination_code = torch.full((batch_size,), 3, dtype=torch.long, device=q.device)
        step_counter = torch.zeros(batch_size, dtype=torch.long, device=q.device)

        q_history = torch.empty((self.config.max_steps + 1, batch_size, q.shape[1]), device=q.device, dtype=q.dtype)
        tcp_history = torch.empty((self.config.max_steps + 1, batch_size, 3), device=q.device, dtype=start_pos.dtype)
        q_history[0] = q
        tcp_history[0] = start_pos

        for step_idx in range(self.config.max_steps):
            if not active.any():
                break

            q_eval = q.detach().clone().requires_grad_(True)
            tcp_pos, tcp_rot = self.robot.fk_batch(q_eval)
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
            j_g = torch.autograd.grad(
                cos_theta.sum(),
                q_eval,
                retain_graph=True,
                create_graph=False,
            )[0].unsqueeze(1)

            mu_val = directional_manipulability_batch(j_pos, direction, self.config.damping)
            low_mu = active & (mu_val < self.config.mu_threshold)
            termination_code[low_mu] = 0
            done[low_mu] = True
            active = active & (~low_mu)
            if not active.any():
                break

            grad_mu = torch.autograd.grad(mu_val.sum(), q_eval, retain_graph=False, create_graph=False)[0]
            on_boundary = (cos_theta <= cos_theta_max).view(-1, 1, 1)
            j_task = torch.cat([j_pos, j_g * on_boundary], dim=1)
            j_pinv = damped_pseudoinverse_batch(j_task, self.config.damping)
            projector = nullspace_projector_batch(j_task, self.config.damping)

            v_pos = (self.config.task_speed * direction).unsqueeze(-1)
            v_g = (self.config.boundary_gain * torch.clamp(cos_theta_max - cos_theta, min=0.0)).view(-1, 1, 1)
            v_g = v_g * on_boundary.squeeze(-1).float().unsqueeze(-1)
            v_task = torch.cat([v_pos, v_g], dim=1)

            q_dot_task = (j_pinv @ v_task).squeeze(-1)
            q_dot_null = self.config.null_gain * grad_mu
            q_dot = q_dot_task + (projector @ q_dot_null.unsqueeze(-1)).squeeze(-1)
            q_next = q + self.config.dt * q_dot

            in_range = joints_in_range_mask(self.robot, q_next)
            joint_limit = active & (~in_range)
            termination_code[joint_limit] = 1
            done[joint_limit] = True

            coll_cost = self.collision_fn(q_next)
            collided = active & (~joint_limit) & (coll_cost > 0.0)
            termination_code[collided] = 2
            done[collided] = True

            tcp_pos_candidate, _ = self.robot.fk_batch(q_next)
            expected_pos = start_pos + direction * ((step_idx + 1) * self.config.dt * self.config.task_speed)
            pos_error = torch.linalg.norm(tcp_pos_candidate - expected_pos, dim=1)
            pos_failed = active & (~joint_limit) & (~collided) & (pos_error > self.config.pos_error_threshold)
            termination_code[pos_failed] = 4
            done[pos_failed] = True

            advance = active & in_range & (~collided) & (~pos_failed)
            q[advance] = q_next[advance]
            step_counter[advance] += 1
            active = advance

            tcp_pos_now, _ = self.robot.fk_batch(q)
            q_history[step_idx + 1] = q
            tcp_history[step_idx + 1] = tcp_pos_now

        trajectories = []
        target_normal_cpu = target_normal_batch.detach().cpu().numpy().copy()
        termination_np = termination_code.detach().cpu().numpy()
        steps_np = step_counter.detach().cpu().numpy()

        for idx in range(batch_size):
            num_points = int(steps_np[idx]) + 1
            q_path_t = q_history[:num_points, idx].detach().clone()
            tcp_pos_t, tcp_rot_t = self.robot.fk_batch(q_path_t)
            j_pos_t, _ = position_jacobian_batch(self.robot, q_path_t, create_graph=False)
            dir_t = direction[idx:idx + 1].expand(num_points, -1)
            mu_t = directional_manipulability_batch(j_pos_t, dir_t, self.config.damping)
            tcp_z_t = tcp_rot_t[:, :, 2]
            cos_theta_t = torch.sum(tcp_z_t * target_normal_batch[idx:idx + 1].expand(num_points, -1), dim=-1)
            boundary_active_t = (cos_theta_t <= cos_theta_max)
            step_idx_t = torch.arange(num_points, device=q.device, dtype=torch.float32)
            expected_pos_t = start_pos[idx:idx + 1] + dir_t * (step_idx_t.unsqueeze(-1) * self.config.dt * self.config.task_speed)
            pos_error_t = torch.linalg.norm(tcp_pos_t - expected_pos_t, dim=1)

            start_pos_i = tcp_pos_t[0]
            end_pos_i = tcp_pos_t[-1]
            progress_length_t = torch.sum((tcp_pos_t - start_pos_i.unsqueeze(0)) * dir_t, dim=1)
            total_projected_length = float(progress_length_t[-1].item())
            remaining_length_t = torch.clamp(total_projected_length - progress_length_t, min=0.0)
            remaining_l2_to_end_t = torch.linalg.norm(end_pos_i.unsqueeze(0) - tcp_pos_t, dim=1)
            total_euclidean_length = float(torch.linalg.norm(end_pos_i - start_pos_i).item())
            is_terminal_t = torch.zeros(num_points, dtype=torch.bool, device=q.device)
            is_terminal_t[-1] = True

            trajectories.append({
                'trajectory_id': int(idx),
                'start_q': q_path_t[0].detach().cpu().numpy().copy(),
                'start_pos': start_pos_i.detach().cpu().numpy().copy(),
                'direction': direction[idx].detach().cpu().numpy().copy(),
                'target_normal': target_normal_cpu[idx].copy(),
                'termination_code': int(termination_np[idx]),
                'termination_reason': termination_label(int(termination_np[idx])),
                'num_points': num_points,
                'total_projected_length': total_projected_length,
                'total_euclidean_length': total_euclidean_length,
                'mean_mu': float(mu_t.mean().item()),
                'min_mu': float(mu_t.min().item()),
                'max_mu': float(mu_t.max().item()),
                'boundary_hit_count': int(boundary_active_t.sum().item()),
                'max_pos_error': float(pos_error_t.max().item()),
                'q': q_path_t.detach().cpu().numpy(),
                'tcp_pos': tcp_pos_t.detach().cpu().numpy(),
                'tcp_rotmat': tcp_rot_t.detach().cpu().numpy(),
                'mu': mu_t.detach().cpu().numpy(),
                'remaining_length': remaining_length_t.detach().cpu().numpy(),
                'remaining_euclidean_length': remaining_l2_to_end_t.detach().cpu().numpy(),
                'progress_length': progress_length_t.detach().cpu().numpy(),
                'pos_error': pos_error_t.detach().cpu().numpy(),
                'cos_theta': cos_theta_t.detach().cpu().numpy(),
                'boundary_active': boundary_active_t.detach().cpu().numpy().astype(np.uint8),
                'step_index': torch.arange(num_points, device=q.device, dtype=torch.int32).detach().cpu().numpy(),
                'is_terminal': is_terminal_t.detach().cpu().numpy().astype(np.uint8),
            })
        return trajectories


def termination_label(code: int) -> str:
    mapping = {
        0: 'low_mu',
        1: 'joint_limit',
        2: 'self_collision',
        3: 'max_steps',
        4: 'pos_tracking_error',
    }
    return mapping.get(int(code), 'unknown')



def visualize_best_sample(result: BatchTrackerResult, plane_size: float = 0.35, vis_mode: str = 'static') -> None:
    best_idx = result.best_idx
    best_direction = result.direction_batch[best_idx].detach().cpu().numpy()
    best_normal = result.target_normal_batch[best_idx].detach().cpu().numpy()
    best_q_path = result.q_path_best
    best_tcp_path = result.tcp_path_best

    world = wd.World(cam_pos=[2.6, 2.2, 1.4], lookat_pos=[0.0, 0.0, 0.5])
    mgm.gen_frame().attach_to(world)

    sim_robot = fr3_sim.FrankaResearch3(enable_cc=False)
    sim_robot.goto_given_conf(best_q_path[0])
    sim_robot.gen_meshmodel(alpha=0.18, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(world)
    sim_robot.goto_given_conf(best_q_path[-1])
    sim_robot.gen_meshmodel(alpha=0.85, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(world)

    line_segs = [(best_tcp_path[i], best_tcp_path[i + 1]) for i in range(len(best_tcp_path) - 1)]
    if line_segs:
        mgm.gen_linesegs(line_segs, thickness=4.0, rgb=np.array([0.1, 0.8, 0.2]), alpha=1.0).attach_to(world)

    start_pos = best_tcp_path[0]
    ideal_end = start_pos + result.projected_length[best_idx].item() * best_direction
    mgm.gen_arrow(spos=start_pos, epos=start_pos + 0.2 * best_direction, stick_radius=0.008, rgb=np.array([0.95, 0.15, 0.15]), alpha=1.0).attach_to(world)
    mgm.gen_linesegs([(start_pos, ideal_end)], thickness=2.0, rgb=np.array([0.15, 0.35, 0.95]), alpha=0.55).attach_to(world)

    plane_rotmat = rotation_matrix_from_normal(best_normal)
    plane_center = start_pos + 0.5 * plane_size * best_direction
    mcm.gen_box(xyz_lengths=np.array([plane_size, plane_size, 0.001]), pos=plane_center, rotmat=plane_rotmat, rgb=np.array([0.75, 0.85, 0.95]), alpha=0.25).attach_to(world)
    world.run()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='GPU-parallel nullspace straight-line tracker for Franka Research 3.')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--speed', type=float, default=0.1)
    parser.add_argument('--damping', type=float, default=1e-3)
    parser.add_argument('--null-gain', type=float, default=0.6)
    parser.add_argument('--theta-max-deg', type=float, default=30.0)
    parser.add_argument('--boundary-gain', type=float, default=10.0)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--mu-threshold', type=float, default=0.01)
    parser.add_argument('--pos-error-threshold', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--no-vis', action='store_true')
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    if args.device == 'cpu':
        raise RuntimeError('This script currently requires CUDA because wrs.neuro JLChain is CUDA-oriented.')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('CUDA is not available, but this Franka GPU script requires a CUDA device.')
    franka = FrankaResearch3GPU(device=device)
    robot = franka.robot

    cc_model = SphereCollisionChecker('wrs/robot_sim/robots/franka_research_3/franka_research_3_sphere_visuals.urdf')
    vmap_jax_cost = jax.jit(jax.vmap(cc_model.self_collision_cost, in_axes=(0, None, None)))
    collision_fn = jax2torch.jax2torch(lambda q_batch: vmap_jax_cost(q_batch, 1.0, -0.005))

    tracker = GPUNullspaceStraightTracker(
        robot=robot,
        collision_fn=collision_fn,
        print_every=args.print_every,
        config=TrackerConfig(
            dt=args.dt,
            task_speed=args.speed,
            damping=args.damping,
            null_gain=args.null_gain,
            theta_max=np.deg2rad(args.theta_max_deg),
            boundary_gain=args.boundary_gain,
            max_steps=args.max_steps,
            mu_threshold=args.mu_threshold,
            pos_error_threshold=args.pos_error_threshold,
        ),
    )

    q0_batch, direction_batch, target_normal_batch = tracker.sample_valid_batch(batch_size=args.batch_size, device=device)
    result = tracker.run_batch(q0_batch=q0_batch, direction_batch=direction_batch, target_normal_batch=target_normal_batch)

    proj = result.projected_length.detach().cpu().numpy()
    euclid = result.euclidean_length.detach().cpu().numpy()
    mean_mu = result.mean_mu.detach().cpu().numpy()
    steps = result.steps.detach().cpu().numpy()
    term = result.termination_code.detach().cpu().numpy()
    normals_np = result.target_normal_batch.detach().cpu().numpy()

    print(f'device = {device}')
    print(f'batch_size = {args.batch_size}')
    print(f'euclidean_length_max = {euclid.max():.4f} m')
    print(f'euclidean_length_mean = {euclid.mean():.4f} m')
    print(f'mean_mu_mean = {mean_mu.mean():.6f}')
    print('target_normal_mean =', np.array2string(normals_np.mean(axis=0), precision=4, separator=', '))
    print('best_sample_target_normal =', np.array2string(normals_np[result.best_idx], precision=4, separator=', '))
    print(f'theta_max_deg = {args.theta_max_deg:.2f}')
    print(f'boundary_gain = {args.boundary_gain:.4f}')
    print(f'pos_error_threshold = {args.pos_error_threshold:.4f} m')
    print(f'steps_mean = {steps.mean():.2f}')
    unique, counts = np.unique(term, return_counts=True)
    print('termination_hist =', {termination_label(int(k)): int(v) for k, v in zip(unique, counts)})
    print(f'best_sample_idx = {result.best_idx}')
    print(f'best_sample_projected_length = {proj[result.best_idx]:.4f} m')
    print(f'best_sample_euclidean_length = {euclid[result.best_idx]:.4f} m')
    print(f'best_sample_mean_mu = {mean_mu[result.best_idx]:.6f}')
    print(f'best_sample_steps = {int(steps[result.best_idx])}')
    print(f'best_sample_termination = {termination_label(int(term[result.best_idx]))}')

    if not args.no_vis:
        visualize_best_sample(result)


if __name__ == '__main__':
    main()
