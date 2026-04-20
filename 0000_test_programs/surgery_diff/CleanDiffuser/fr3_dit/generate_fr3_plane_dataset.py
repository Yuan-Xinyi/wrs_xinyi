#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import jax
import jax2torch
import numpy as np
import torch

import wrs.basis.robot_math as rm
import wrs.neuro._kinematics.jlchain as jlc
from wrs.robot_sim.robots.franka_research_3.sphere_collision_checker import SphereCollisionChecker


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_URDF = PROJECT_ROOT / "wrs" / "robot_sim" / "robots" / "franka_research_3" / "franka_research_3_ccsphere.urdf"
PEN_LENGTH = 0.15
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "pen_fr3_plane_trajectories.hdf5"


class PenFrankaResearch3GPU:
    def __init__(self, device: torch.device):
        self.device = device
        self.robot = jlc.JLChain(n_dof=7, pos=torch.zeros(3, device=device), rotmat=torch.eye(3, device=device))
        self._build_robot()
        self.robot.finalize()

    def _build_robot(self) -> None:
        r = self.robot
        d = self.device

        r.jnts[0].loc_pos = torch.tensor([0.0, 0.0, 0.333], dtype=torch.float32, device=d)
        r.jnts[0].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[0].motion_range = torch.tensor([-2.8973, 2.8973], dtype=torch.float32, device=d)

        r.jnts[1].loc_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=d)
        r.jnts[1].loc_rotmat = torch.tensor(rm.rotmat_from_euler(-np.pi / 2, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[1].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[1].motion_range = torch.tensor([-1.8326, 1.8326], dtype=torch.float32, device=d)

        r.jnts[2].loc_pos = torch.tensor([0.0, -0.316, 0.0], dtype=torch.float32, device=d)
        r.jnts[2].loc_rotmat = torch.tensor(rm.rotmat_from_euler(np.pi / 2, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[2].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[2].motion_range = torch.tensor([-2.8972, 2.8972], dtype=torch.float32, device=d)

        r.jnts[3].loc_pos = torch.tensor([0.0825, 0.0, 0.0], dtype=torch.float32, device=d)
        r.jnts[3].loc_rotmat = torch.tensor(rm.rotmat_from_euler(np.pi / 2, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[3].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[3].motion_range = torch.tensor([-3.0718, -0.1222], dtype=torch.float32, device=d)

        r.jnts[4].loc_pos = torch.tensor([-0.0825, 0.384, 0.0], dtype=torch.float32, device=d)
        r.jnts[4].loc_rotmat = torch.tensor(rm.rotmat_from_euler(-np.pi / 2, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[4].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[4].motion_range = torch.tensor([-2.8798, 2.8798], dtype=torch.float32, device=d)

        r.jnts[5].loc_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=d)
        r.jnts[5].loc_rotmat = torch.tensor(rm.rotmat_from_euler(np.pi / 2, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[5].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[5].motion_range = torch.tensor([0.4364, 4.6251], dtype=torch.float32, device=d)

        r.jnts[6].loc_pos = torch.tensor([0.088, 0.0, 0.0], dtype=torch.float32, device=d)
        r.jnts[6].loc_rotmat = torch.tensor(rm.rotmat_from_euler(np.pi / 2, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[6].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[6].motion_range = torch.tensor([-3.0543, 3.0543], dtype=torch.float32, device=d)

        r._loc_flange_pos = torch.tensor([0.0, 0.0, 0.2104 + PEN_LENGTH], dtype=torch.float32, device=d)


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def random_unit_vectors_batch(batch_size: int, device: torch.device) -> torch.Tensor:
    return normalize_batch(torch.randn(batch_size, 3, device=device))


def project_to_plane_batch(v: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    normal = normalize_batch(normal)
    v_plane = v - torch.sum(v * normal, dim=-1, keepdim=True) * normal
    tiny = v_plane.norm(dim=-1, keepdim=True) < 1e-8
    if tiny.any():
        fallback = torch.zeros_like(v_plane)
        fallback[..., 0] = 1.0
        fallback = fallback - torch.sum(fallback * normal, dim=-1, keepdim=True) * normal
        v_plane = torch.where(tiny, fallback, v_plane)
    return normalize_batch(v_plane)


def position_jacobian_batch(robot, q_batch: torch.Tensor, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
    q_eval = q_batch.detach().clone().requires_grad_(True)
    tcp_pos, _ = robot.fk_batch(q_eval)
    grads = []
    for dim in range(3):
        grads.append(torch.autograd.grad(tcp_pos[:, dim].sum(), q_eval, retain_graph=True, create_graph=create_graph)[0])
    return torch.stack(grads, dim=1), q_eval


def damped_pseudoinverse_batch(j: torch.Tensor, damping: float) -> torch.Tensor:
    batch, task_dim, _ = j.shape
    eye = torch.eye(task_dim, device=j.device, dtype=j.dtype).unsqueeze(0).expand(batch, -1, -1)
    return j.transpose(1, 2) @ torch.linalg.inv(j @ j.transpose(1, 2) + (damping ** 2) * eye)


def nullspace_projector_batch(j: torch.Tensor, damping: float) -> torch.Tensor:
    batch, _, dof = j.shape
    eye = torch.eye(dof, device=j.device, dtype=j.dtype).unsqueeze(0).expand(batch, -1, -1)
    return eye - damped_pseudoinverse_batch(j, damping) @ j


def directional_manipulability_batch(j_pos: torch.Tensor, direction: torch.Tensor, damping: float) -> torch.Tensor:
    batch = j_pos.shape[0]
    eye = torch.eye(3, device=j_pos.device, dtype=j_pos.dtype).unsqueeze(0).expand(batch, -1, -1)
    metric = j_pos @ j_pos.transpose(1, 2) + (damping ** 2) * eye
    d = direction.unsqueeze(-1)
    return (d.transpose(1, 2) @ torch.linalg.inv(metric) @ d).squeeze(-1).squeeze(-1).clamp_min(1e-12).pow(-0.5)


def joint_margin_mask(robot, q_batch: torch.Tensor, margin_ratio: float) -> torch.Tensor:
    lower = robot.jnt_ranges[:, 0].unsqueeze(0)
    upper = robot.jnt_ranges[:, 1].unsqueeze(0)
    span = upper - lower
    inner_lower = lower + margin_ratio * span
    inner_upper = upper - margin_ratio * span
    return ((q_batch >= inner_lower) & (q_batch <= inner_upper)).all(dim=1)


def sample_normals_near_tcp_z(tcp_z: torch.Tensor, max_angle_deg: float) -> torch.Tensor:
    batch = tcp_z.shape[0]
    noise = random_unit_vectors_batch(batch, tcp_z.device)
    tangent = project_to_plane_batch(noise, tcp_z)
    angles = torch.rand(batch, device=tcp_z.device) * np.deg2rad(max_angle_deg)
    return normalize_batch(torch.cos(angles).unsqueeze(1) * tcp_z + torch.sin(angles).unsqueeze(1) * tangent)


def termination_label(code: int) -> str:
    labels = {
        0: "low_mu",
        1: "joint_margin",
        2: "self_collision",
        3: "max_steps",
        4: "pos_tracking_error",
        5: "plane_clearance",
        6: "angle_violation",
    }
    return labels.get(int(code), "unknown")


@dataclass
class TrackerConfig:
    dt: float = 0.01
    task_speed: float = 0.1
    damping: float = 1e-3
    null_gain: float = 0.6
    joint_limit_gain: float = 0.2
    theta_max_deg: float = 30.0
    joint_margin_ratio: float = 0.05
    boundary_gain: float = 10.0
    max_steps: int = 2000
    mu_threshold: float = 0.01
    pos_error_threshold: float = 0.01


class PlaneConstrainedTracker:
    def __init__(self, robot, self_collision_fn, sphere_positions_fn, sphere_radii: np.ndarray, sphere_link_indices: np.ndarray, config: TrackerConfig):
        self.robot = robot
        self.self_collision_fn = self_collision_fn
        self.sphere_positions_fn = sphere_positions_fn
        tensor_device = robot.anchor.pos.device
        self.sphere_radii = torch.tensor(sphere_radii, dtype=torch.float32, device=tensor_device)
        self.sphere_link_indices = torch.tensor(sphere_link_indices, dtype=torch.long, device=tensor_device)
        self.config = config
        self.theta_cos = float(np.cos(np.deg2rad(config.theta_max_deg)))
        max_link_index = int(np.max(sphere_link_indices))
        protected_from_plane = max(2, max_link_index - 1)
        self.keep_mask = self.sphere_link_indices < protected_from_plane

    def plane_clearance_mask(self, q_batch: torch.Tensor, plane_point: torch.Tensor, plane_normal: torch.Tensor, plane_side: torch.Tensor) -> torch.Tensor:
        sphere_pos = self.sphere_positions_fn(q_batch)
        signed = torch.sum((sphere_pos - plane_point[:, None, :]) * plane_normal[:, None, :], dim=-1)
        signed = signed * plane_side[:, None]
        required = self.sphere_radii.unsqueeze(0)
        return (signed[:, self.keep_mask] >= required[:, self.keep_mask]).all(dim=1)

    def sample_valid_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_list, p_list, d_list, n_list, s_list = [], [], [], [], []
        remaining = batch_size
        oversample = max(256, batch_size * 8)
        trial = 0
        while remaining > 0:
            trial += 1
            q = self.robot.rand_conf_batch(oversample).to(device)
            q = q[joint_margin_mask(self.robot, q, self.config.joint_margin_ratio)]
            if q.shape[0] == 0:
                print(f"[sample] trial={trial} after-margin=0/{oversample}")
                continue
            tcp_pos, tcp_rot = self.robot.fk_batch(q)
            tcp_z = tcp_rot[:, :, 2]
            normals = sample_normals_near_tcp_z(tcp_z, self.config.theta_max_deg)
            directions = project_to_plane_batch(torch.randn_like(normals), normals)
            j_pos, _ = position_jacobian_batch(self.robot, q, create_graph=False)
            mu = directional_manipulability_batch(j_pos, directions, self.config.damping)
            coll = self.self_collision_fn(q)
            plane_point = tcp_pos
            plane_side = -torch.ones(q.shape[0], device=device, dtype=torch.float32)
            plane_ok = self.plane_clearance_mask(q, plane_point, normals, plane_side)
            valid = (mu > self.config.mu_threshold) & (coll <= 0.0) & plane_ok
            print(
                f"[sample] trial={trial} after-margin={q.shape[0]}/{oversample} "
                f"plane-ok={int(plane_ok.sum().item())} valid={int(valid.sum().item())} "
                f"collected={batch_size - remaining}/{batch_size}"
            )
            if valid.any():
                take = min(int(valid.sum().item()), remaining)
                idx = torch.where(valid)[0][:take]
                q_list.append(q[idx])
                p_list.append(plane_point[idx])
                d_list.append(directions[idx])
                n_list.append(normals[idx])
                s_list.append(plane_side[idx])
                remaining -= take
                print(f"[sample] collected={batch_size - remaining}/{batch_size} valid starts")
        return (
            torch.cat(q_list, dim=0),
            torch.cat(p_list, dim=0),
            torch.cat(d_list, dim=0),
            torch.cat(n_list, dim=0),
            torch.cat(s_list, dim=0),
        )

    def collect_batch_trajectories(
        self,
        q0_batch: torch.Tensor,
        plane_point_batch: torch.Tensor,
        direction_batch: torch.Tensor,
        plane_normal_batch: torch.Tensor,
        plane_side_batch: torch.Tensor,
    ) -> list[dict]:
        q = q0_batch.clone()
        direction = project_to_plane_batch(direction_batch, plane_normal_batch)
        start_pos, _ = self.robot.fk_batch(q)
        batch_size = q.shape[0]
        print(f"[rollout] batch_size={batch_size} max_steps={self.config.max_steps}")
        active = torch.ones(batch_size, dtype=torch.bool, device=q.device)
        termination = torch.full((batch_size,), 3, dtype=torch.long, device=q.device)
        steps = torch.zeros(batch_size, dtype=torch.long, device=q.device)

        q_hist = torch.empty((self.config.max_steps + 1, batch_size, q.shape[1]), device=q.device)
        tcp_hist = torch.empty((self.config.max_steps + 1, batch_size, 3), device=q.device)
        q_hist[0] = q
        tcp_hist[0] = start_pos

        for step_idx in range(self.config.max_steps):
            if not active.any():
                break

            q_eval = q.detach().clone().requires_grad_(True)
            tcp_pos, tcp_rot = self.robot.fk_batch(q_eval)
            tcp_z = tcp_rot[:, :, 2]
            cos_theta = torch.sum(tcp_z * plane_normal_batch, dim=-1)

            j_pos, _ = position_jacobian_batch(self.robot, q_eval, create_graph=True)
            j_g = torch.autograd.grad(cos_theta.sum(), q_eval, retain_graph=True, create_graph=False)[0].unsqueeze(1)
            mu = directional_manipulability_batch(j_pos, direction, self.config.damping)
            low_mu = active & (mu < self.config.mu_threshold)
            termination[low_mu] = 0
            active = active & (~low_mu)
            if not active.any():
                break

            grad_mu = torch.autograd.grad(
                mu.sum(),
                q_eval,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad_mu is None:
                grad_mu = torch.zeros_like(q_eval)
            on_boundary = (cos_theta <= self.theta_cos).view(-1, 1, 1)
            j_task = torch.cat([j_pos, j_g * on_boundary], dim=1)
            j_pinv = damped_pseudoinverse_batch(j_task, self.config.damping)
            projector = nullspace_projector_batch(j_task, self.config.damping)

            v_pos = (self.config.task_speed * direction).unsqueeze(-1)
            v_ang = (self.config.boundary_gain * torch.clamp(self.theta_cos - cos_theta, min=0.0)).view(-1, 1, 1)
            v_task = torch.cat([v_pos, v_ang * on_boundary.squeeze(-1).float().unsqueeze(-1)], dim=1)
            q_dot_task = (j_pinv @ v_task).squeeze(-1)
            lower = self.robot.jnt_ranges[:, 0].unsqueeze(0)
            upper = self.robot.jnt_ranges[:, 1].unsqueeze(0)
            center = 0.5 * (lower + upper)
            span = (upper - lower).clamp_min(1e-6)
            q_dot_joint = -self.config.joint_limit_gain * (q_eval.detach() - center) / span
            q_dot = q_dot_task + (projector @ (self.config.null_gain * grad_mu + q_dot_joint).unsqueeze(-1)).squeeze(-1)
            q_next = q + self.config.dt * q_dot

            margin_ok = joint_margin_mask(self.robot, q_next, self.config.joint_margin_ratio)
            fail_margin = active & (~margin_ok)
            termination[fail_margin] = 1

            coll = self.self_collision_fn(q_next)
            fail_coll = active & margin_ok & (coll > 0.0)
            termination[fail_coll] = 2

            plane_ok = self.plane_clearance_mask(q_next, plane_point_batch, plane_normal_batch, plane_side_batch)
            fail_plane = active & margin_ok & (~fail_coll) & (~plane_ok)
            termination[fail_plane] = 5

            tcp_pos_next, tcp_rot_next = self.robot.fk_batch(q_next)
            cos_theta_next = torch.sum(tcp_rot_next[:, :, 2] * plane_normal_batch, dim=-1)
            fail_angle = active & margin_ok & (~fail_coll) & (~fail_plane) & (cos_theta_next < self.theta_cos)
            termination[fail_angle] = 6

            expected_pos = start_pos + direction * ((step_idx + 1) * self.config.dt * self.config.task_speed)
            pos_err = torch.linalg.norm(tcp_pos_next - expected_pos, dim=1)
            fail_pos = active & margin_ok & (~fail_coll) & (~fail_plane) & (~fail_angle) & (pos_err > self.config.pos_error_threshold)
            termination[fail_pos] = 4

            advance = active & margin_ok & (~fail_coll) & (~fail_plane) & (~fail_angle) & (~fail_pos)
            q[advance] = q_next[advance]
            steps[advance] += 1
            active = advance

            if (step_idx + 1) % 100 == 0 or not active.any():
                print(
                    f"[rollout] step={step_idx + 1} active={int(active.sum().item())}/{batch_size} "
                    f"mean_step={float(steps.float().mean().item()):.1f}"
                )

            tcp_now, _ = self.robot.fk_batch(q)
            q_hist[step_idx + 1] = q
            tcp_hist[step_idx + 1] = tcp_now

        trajectories = []
        for i in range(batch_size):
            num_points = int(steps[i].item()) + 1
            q_path = q_hist[:num_points, i].detach().cpu().numpy().astype(np.float32)
            tcp_path = tcp_hist[:num_points, i].detach().cpu().numpy().astype(np.float32)
            progress = np.sum((tcp_path - tcp_path[0]) * direction[i].detach().cpu().numpy()[None, :], axis=1).astype(np.float32)
            trajectories.append(
                {
                    "start_q": q0_batch[i].detach().cpu().numpy().astype(np.float32),
                    "plane_point": plane_point_batch[i].detach().cpu().numpy().astype(np.float32),
                    "plane_normal": plane_normal_batch[i].detach().cpu().numpy().astype(np.float32),
                    "plane_side": float(plane_side_batch[i].item()),
                    "direction": direction[i].detach().cpu().numpy().astype(np.float32),
                    "termination_code": int(termination[i].item()),
                    "termination_reason": termination_label(int(termination[i].item())),
                    "num_points": num_points,
                    "total_projected_length": float(progress[-1]),
                    "q": q_path,
                    "tcp_pos": tcp_path,
                    "progress_length": progress,
                }
            )
        return trajectories


def init_hdf5(path: Path, config: TrackerConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["dt"] = config.dt
        f.attrs["task_speed"] = config.task_speed
        f.attrs["theta_max_deg"] = config.theta_max_deg
        f.attrs["joint_margin_ratio"] = config.joint_margin_ratio
        f.attrs["plane_clearance_m"] = 0.0
        f.attrs["robot_name"] = "pen_fr3"
        f.attrs["pen_length_m"] = PEN_LENGTH
        f.attrs["plane_collision_policy"] = "plane_through_pen_tcp_keep_arm_on_negative_normal_side_ignore_distal_two_links"
        f.attrs["num_trajectories"] = 0


def append_trajectories_hdf5(path: Path, trajectories: list[dict]) -> int:
    with h5py.File(path, "a") as f:
        start_idx = int(f.attrs.get("num_trajectories", 0))
        for offset, traj in enumerate(trajectories):
            g = f.create_group(f"traj_{start_idx + offset:06d}")
            for key, value in traj.items():
                if isinstance(value, str):
                    g.attrs[key] = value
                elif np.isscalar(value):
                    g.attrs[key] = value
                else:
                    g.create_dataset(key, data=value, compression="gzip")
        f.attrs["num_trajectories"] = start_idx + len(trajectories)
        return int(f.attrs["num_trajectories"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FR3 plane-constrained straight-line trajectories.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-trajectories", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--theta-max-deg", type=float, default=30.0)
    parser.add_argument("--joint-margin-ratio", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    fr3 = PenFrankaResearch3GPU(device)
    cc = SphereCollisionChecker(str(DEFAULT_URDF))
    self_collision_cost_fn = jax2torch.jax2torch(jax.jit(jax.vmap(cc.self_collision_cost, in_axes=(0, None, None))))
    self_collision_fn = lambda q: self_collision_cost_fn(q, 1.0, -0.005)
    sphere_positions_fn = jax2torch.jax2torch(jax.jit(jax.vmap(cc.compute_sphere_positions, in_axes=(0,))))

    tracker = PlaneConstrainedTracker(
        robot=fr3.robot,
        self_collision_fn=self_collision_fn,
        sphere_positions_fn=sphere_positions_fn,
        sphere_radii=np.asarray(cc.sphere_radii, dtype=np.float32),
        sphere_link_indices=np.asarray(cc.sphere_link_indices, dtype=np.int64),
        config=TrackerConfig(
            theta_max_deg=float(args.theta_max_deg),
            joint_margin_ratio=float(args.joint_margin_ratio),
            max_steps=int(args.max_steps),
        ),
    )

    init_hdf5(args.output, tracker.config)
    total_written = 0
    batch_idx = 0
    while total_written < int(args.num_trajectories):
        batch_idx += 1
        remaining = int(args.num_trajectories) - total_written
        take = min(int(args.batch_size), remaining)
        print(f"[batch] {batch_idx} target_batch={take} collected_total={total_written}/{int(args.num_trajectories)}")
        q0, plane_point, direction, plane_normal, plane_side = tracker.sample_valid_batch(take, device)
        batch_trajs = tracker.collect_batch_trajectories(q0, plane_point, direction, plane_normal, plane_side)
        batch_lengths = np.asarray([float(t["total_projected_length"]) for t in batch_trajs], dtype=np.float32)
        print(
            f"[length] batch={batch_idx} mean={float(batch_lengths.mean()):.4f}m "
            f"median={float(np.median(batch_lengths)):.4f}m min={float(batch_lengths.min()):.4f}m "
            f"max={float(batch_lengths.max()):.4f}m first3={batch_lengths[:3].tolist()}"
        )
        total_written = append_trajectories_hdf5(args.output, batch_trajs)
        print(f"[save] collected={total_written}/{int(args.num_trajectories)}")

    print(f"[done] wrote {total_written} trajectories to {args.output}")


if __name__ == "__main__":
    main()
