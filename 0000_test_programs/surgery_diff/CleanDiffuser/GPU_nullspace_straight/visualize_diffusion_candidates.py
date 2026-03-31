import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd

from kinematic_diffusion_common import DEFAULT_H5_PATH, DEFAULT_RUN_NAME, DEFAULT_WORKDIR, sample_q_length_from_condition
from sample_kinematic_dit_inpainting import JacobianCorrection, load_model, normalize_direction
import jax2torch
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller
from wrs.robot_sim.robots.xarmlite6_wg.sphere_collision_checker import SphereCollisionChecker
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
from xarmlite6_gpu_nullspave_straight_demo import GPUNullspaceStraightTracker, TrackerConfig, position_jacobian_batch, directional_manipulability_batch
import jax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize multiple diffusion candidate q solutions for the same (pos, direction) condition.')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--traj-id', type=str, default=None)
    parser.add_argument('--point-idx', type=int, default=0)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n-samples', type=int, default=128)
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.22)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--no-vis', action='store_true')
    return parser.parse_args()


def load_anchor(h5_path: Path, traj_id: str | None, point_idx: int | None, rng: np.random.Generator) -> dict:
    with h5py.File(h5_path, 'r') as f:
        traj_keys = sorted(f['trajectories'].keys())
        traj_key = traj_id if traj_id is not None else traj_keys[int(rng.integers(len(traj_keys)))]
        grp = f['trajectories'][traj_key]
        num_points = int(grp.attrs['num_points'])
        idx = int(point_idx) if point_idx is not None else int(rng.integers(num_points))
        return {
            'traj_key': traj_key,
            'point_idx': idx,
            'q': np.asarray(grp['q'][idx], dtype=np.float32),
            'pos': np.asarray(grp['tcp_pos'][idx], dtype=np.float32),
            'direction': normalize_direction(np.asarray(grp.attrs['direction'], dtype=np.float32)),
            'length': float(np.asarray(grp['remaining_length'][idx], dtype=np.float32)),
            'target_normal': normalize_direction(np.asarray(grp.attrs['target_normal'], dtype=np.float32)),
        }




def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normal / max(np.linalg.norm(normal), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-12)
    return np.column_stack((x_axis, y_axis, z_axis))


def build_tracker(device: torch.device) -> tuple[GPUNullspaceStraightTracker, torch.device]:
    xarm = xarm6_gpu.XArmLite6GPU(device=device)
    cc_model = SphereCollisionChecker('wrs/robot_sim/robots/xarmlite6_wg/xarm6_sphere_visuals.urdf')
    vmap_jax_cost = jax.jit(jax.vmap(cc_model.self_collision_cost, in_axes=(0, None, None)))
    collision_fn = jax2torch.jax2torch(lambda q_batch: vmap_jax_cost(q_batch, 1.0, -0.005))
    tracker = GPUNullspaceStraightTracker(
        robot=xarm.robot,
        collision_fn=collision_fn,
        config=TrackerConfig(),
        print_every=0,
    )
    return tracker, device


def directional_mu_batch(
    tracker: GPUNullspaceStraightTracker,
    tracker_device: torch.device,
    q_batch_np: np.ndarray,
    direction_np: np.ndarray,
) -> np.ndarray:
    q_batch = torch.from_numpy(q_batch_np.astype(np.float32)).to(tracker_device)
    direction_batch = torch.from_numpy(np.repeat(direction_np[None, :].astype(np.float32), q_batch_np.shape[0], axis=0)).to(tracker_device)
    j_pos, _ = position_jacobian_batch(tracker.robot, q_batch, create_graph=False)
    mu = directional_manipulability_batch(j_pos, direction_batch, tracker.config.damping)
    return mu.detach().cpu().numpy()


def rollout_lengths_batch(
    tracker: GPUNullspaceStraightTracker,
    tracker_device: torch.device,
    q_batch_np: np.ndarray,
    direction_np: np.ndarray,
    target_normal_np: np.ndarray,
) -> np.ndarray:
    q_batch = torch.from_numpy(q_batch_np.astype(np.float32)).to(tracker_device)
    direction_batch = torch.from_numpy(np.repeat(direction_np[None, :].astype(np.float32), q_batch_np.shape[0], axis=0)).to(tracker_device)
    target_normal_batch = torch.from_numpy(np.repeat(target_normal_np[None, :].astype(np.float32), q_batch_np.shape[0], axis=0)).to(tracker_device)
    result = tracker.run_batch(q0_batch=q_batch, direction_batch=direction_batch, target_normal_batch=target_normal_batch)
    return result.projected_length.detach().cpu().numpy()

def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed if args.seed is not None else int(np.random.SeedSequence().entropy))
    device = torch.device(args.device)
    _, stats, model, q_dim, diffusion_steps = load_model(args.bundle, device)
    anchor = load_anchor(args.h5_path, args.traj_id, args.point_idx, rng)

    condition = np.concatenate([anchor['pos'], anchor['direction']], axis=0).astype(np.float32)
    import time
    start_time = time.time()
    q_preds, pred_lengths, _ = sample_q_length_from_condition(
        model=model,
        stats=stats,
        condition=condition,
        device=device,
        q_dim=q_dim,
        n_samples=int(args.n_samples),
        sample_steps=int(args.sample_steps) if args.sample_steps is not None else int(diffusion_steps),
        temperature=float(args.temperature),
    )
    end_time = time.time()
    print(f"Sampling {len(q_preds)} candidates took {end_time - start_time:.2f} seconds")
    
    tracker, tracker_device = build_tracker(device)
    correction_robot = XArmLite6Miller(enable_cc=True)
    correction = JacobianCorrection(robot=correction_robot, damping=args.correction_damping)
    q_corrs = []
    pos_errs = []
    for q_pred in q_preds.astype(np.float32):
        q_corr, pos_err = correction.run(q_pred, anchor['pos'], max_iters=args.correction_iters, tol=args.correction_tol)
        q_corrs.append(q_corr)
        pos_errs.append(float(pos_err))
    q_corrs = np.asarray(q_corrs, dtype=np.float32)
    pos_errs = np.asarray(pos_errs, dtype=np.float32)

    all_q = np.concatenate([anchor['q'][None, :], q_corrs], axis=0)
    all_real_lengths = rollout_lengths_batch(tracker, tracker_device, all_q, anchor['direction'], anchor['target_normal'])
    all_mu = directional_mu_batch(tracker, tracker_device, all_q, anchor['direction'])
    gt_real_length = float(all_real_lengths[0])
    gt_mu = float(all_mu[0])
    candidate_real_lengths = all_real_lengths[1:]
    candidate_mu = all_mu[1:]
    print("label   idx  pred_len   real_len   mu")
    print(f"GT      --   {anchor['length']:.6f}   {gt_real_length:.6f}   {gt_mu:.6f}")
    print("------------------------------------------")

    best_idx = int(np.argmax(candidate_real_lengths)) if len(candidate_real_lengths) > 0 else -1
    if len(candidate_real_lengths) > 0:
        avg_pred_length = float(np.mean(pred_lengths))
        avg_real_length = float(np.mean(candidate_real_lengths))
        avg_mu = float(np.mean(candidate_mu))
        avg_pos_err = float(np.mean(pos_errs))
        min_idx = int(np.argmin(candidate_real_lengths))
        max_mu_idx = int(np.argmax(candidate_mu))
        max_pred_idx = int(np.argmax(pred_lengths))
        print(f"AVG     --   {avg_pred_length:.6f}   {avg_real_length:.6f}   {avg_mu:.6f}   pos_err={avg_pos_err:.6e}")
        print(f"MIN     {min_idx:02d}   {float(pred_lengths[min_idx]):.6f}   {float(candidate_real_lengths[min_idx]):.6f}   {float(candidate_mu[min_idx]):.6f}   pos_err={float(pos_errs[min_idx]):.6e}")
        print(f"MAXMU   {max_mu_idx:02d}   {float(pred_lengths[max_mu_idx]):.6f}   {float(candidate_real_lengths[max_mu_idx]):.6f}   {float(candidate_mu[max_mu_idx]):.6f}   pos_err={float(pos_errs[max_mu_idx]):.6e}")
        print(f"MAXPRED {max_pred_idx:02d}   {float(pred_lengths[max_pred_idx]):.6f}   {float(candidate_real_lengths[max_pred_idx]):.6f}   {float(candidate_mu[max_pred_idx]):.6f}   pos_err={float(pos_errs[max_pred_idx]):.6e}")
    if best_idx >= 0:
        print(f"BEST    {best_idx:02d}   {float(pred_lengths[best_idx]):.6f}   {float(candidate_real_lengths[best_idx]):.6f}   {float(candidate_mu[best_idx]):.6f}   pos_err={float(pos_errs[best_idx]):.6e}")
    print("------------------------------------------")
    if args.no_vis:
        return

    world = wd.World(cam_pos=[1.7, -1.5, 1.05], lookat_pos=[0.25, 0.0, 0.25])
    mgm.gen_frame().attach_to(world)
    start = anchor['pos']
    direction = anchor['direction']
    plane_size = 1.2
    plane_rotmat = rotation_matrix_from_normal(anchor['target_normal'])
    plane_center = start + 0.5 * plane_size * direction
    mcm.gen_box(
        xyz_lengths=[plane_size, plane_size, 0.001],
        pos=plane_center,
        rotmat=plane_rotmat,
        rgb=[180/255, 211/255, 217/255],
        alpha=0.5,
    ).attach_to(world)
    mgm.gen_sphere(start, radius=0.010, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)
    gt_end = start + direction * anchor['length']
    gt_color = np.array([0.1, 0.75, 0.2], dtype=np.float32)
    mgm.gen_stick(spos=start, epos=gt_end, radius=0.0045, rgb=gt_color, alpha=0.90).attach_to(world)
    mgm.gen_sphere(gt_end, radius=0.009, rgb=gt_color, alpha=0.95).attach_to(world)

    robot = XArmLite6Miller(enable_cc=True)
    robot.goto_given_conf(anchor['q'])
    robot.gen_meshmodel(rgb=np.array([0.1, 0.75, 0.2]), alpha=0.55, toggle_tcp_frame=True).attach_to(world)

    default_color = np.array([0.75, 0.75, 0.75], dtype=np.float32)
    best_color = np.array([0.15, 0.45, 0.95], dtype=np.float32)
    for idx, (q_corr, pred_length) in enumerate(zip(q_corrs, pred_lengths)):
        color = best_color if idx == best_idx else default_color
        alpha = 0.55 if idx == best_idx else 0.20
        robot.goto_given_conf(q_corr.astype(np.float32))
        robot.gen_meshmodel(rgb=color, alpha=alpha, toggle_tcp_frame=False).attach_to(world)
        pred_end = start + direction * float(pred_length)
        line_radius = 0.004 if idx == best_idx else 0.0025
        line_alpha = 1.0 if idx == best_idx else 0.22
        mgm.gen_stick(spos=start, epos=pred_end, radius=line_radius, rgb=color, alpha=line_alpha).attach_to(world)
        if idx == best_idx:
            mgm.gen_sphere(pred_end, radius=0.009, rgb=color, alpha=0.95).attach_to(world)

    world.run()


if __name__ == '__main__':
    main()
