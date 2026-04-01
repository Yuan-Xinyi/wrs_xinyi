import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import numpy as np
import torch

import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd

from diffusion import DEFAULT_RUN_NAME, DEFAULT_WORKDIR, sample_q_length_from_condition
from diffusion_sample import JacobianCorrection, load_model, normalize_direction
import jax2torch
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller
from wrs.robot_sim.robots.xarmlite6_wg.sphere_collision_checker import SphereCollisionChecker
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
from trajectory_generation.xarmlite6_gpu_nullspave_straight_demo import GPUNullspaceStraightTracker, TrackerConfig, position_jacobian_batch, directional_manipulability_batch
import jax


DIRECTION_AXIS = 0
NORMAL_AXIS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize diffusion candidates for a random GT joint configuration.')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n-samples', type=int, default=16)
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--no-vis', action='store_true')
    return parser.parse_args()


def sample_anchor(rng: np.random.Generator) -> dict:
    robot = XArmLite6Miller(enable_cc=True)
    for _ in range(2000):
        q = robot.rand_conf().astype(np.float32)
        robot.goto_given_conf(q)
        if robot.is_collided():
            continue
        pos, rotmat = robot.fk(q, update=False)
        rotmat = np.asarray(rotmat, dtype=np.float32)
        direction = normalize_direction(rotmat[:, DIRECTION_AXIS])
        target_normal = normalize_direction(rotmat[:, NORMAL_AXIS])
        return {
            'q': q,
            'pos': np.asarray(pos, dtype=np.float32),
            'rotmat': rotmat,
            'direction': direction,
            'target_normal': target_normal,
        }
    raise RuntimeError('Failed to sample a valid random GT configuration.')


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
    anchor = sample_anchor(rng)

    condition = np.concatenate([anchor['pos'], anchor['direction'], anchor['target_normal']], axis=0).astype(np.float32)
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
    print(f'Sampling {len(q_preds)} candidates took {end_time - start_time:.2f} seconds')

    tracker, tracker_device = build_tracker(device)
    correction_robot = XArmLite6Miller(enable_cc=True)
    correction = JacobianCorrection(robot=correction_robot, damping=args.correction_damping)
    q_corrs = []
    pos_errs = []
    for q_pred in q_preds.astype(np.float32):
        q_pred64 = q_pred.astype(np.float64)
        correction_robot.goto_given_conf(q_pred64)
        cur_pos, _ = correction_robot.fk(q_pred64, update=False)
        raw_pos_err = float(np.linalg.norm(anchor['pos'] - cur_pos))
        q_corr, _ = correction.run(q_pred, anchor['pos'], max_iters=args.correction_iters, tol=args.correction_tol)
        q_corrs.append(q_corr)
        pos_errs.append(raw_pos_err)
    q_corrs = np.asarray(q_corrs, dtype=np.float32)
    pos_errs = np.asarray(pos_errs, dtype=np.float32)

    all_q = np.concatenate([anchor['q'][None, :], q_corrs], axis=0)
    all_real_lengths = rollout_lengths_batch(tracker, tracker_device, all_q, anchor['direction'], anchor['target_normal'])
    all_mu = directional_mu_batch(tracker, tracker_device, all_q, anchor['direction'])
    gt_real_length = float(all_real_lengths[0])
    gt_mu = float(all_mu[0])
    candidate_real_lengths = all_real_lengths[1:]
    candidate_mu = all_mu[1:]
    min_pos_idx = int(np.argmin(pos_errs)) if len(pos_errs) > 0 else -1

    row_fmt = '{label:<8} {idx:>4} {pred:>10} {real:>10.6f} {mu:>10.6f} {pos:>12}'
    print('--------------------------------------------------------------------------')
    print(f"{'label':<8} {'idx':>4} {'pred_len':>10} {'real_len':>10} {'mu':>10} {'pos_err_mm':>12}")
    print('--------------------------------------------------------------------------')
    print(row_fmt.format(label='GT', idx='--', pred='--', real=gt_real_length, mu=gt_mu, pos='--'))
    if min_pos_idx >= 0:
        print(row_fmt.format(
            label='MINPOS',
            idx=f'{min_pos_idx:02d}',
            pred=f'{float(pred_lengths[min_pos_idx]):.6f}',
            real=float(candidate_real_lengths[min_pos_idx]),
            mu=float(candidate_mu[min_pos_idx]),
            pos=f'{float(pos_errs[min_pos_idx]) * 1e3:.3f}',
        ))
    print('--------------------------------------------------------------------------')
    for idx in range(len(q_preds)):
        print(row_fmt.format(
            label='CAND',
            idx=f'{idx:02d}',
            pred=f'{float(pred_lengths[idx]):.6f}',
            real=float(candidate_real_lengths[idx]),
            mu=float(candidate_mu[idx]),
            pos=f'{float(pos_errs[idx]) * 1e3:.3f}',
        ))
    print('--------------------------------------------------------------------------')
    if args.no_vis:
        return

    world = wd.World(cam_pos=[1.7, -1.5, 1.05], lookat_pos=[0.25, 0.0, 0.25])
    mgm.gen_frame().attach_to(world)
    mgm.gen_frame(pos=anchor['pos'], rotmat=anchor['rotmat'], ax_length=0.12).attach_to(world)
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
    gt_end = start + direction * gt_real_length
    gt_color = np.array([0.1, 0.75, 0.2], dtype=np.float32)
    mgm.gen_stick(spos=start, epos=gt_end, radius=0.0045, rgb=gt_color, alpha=0.90).attach_to(world)
    mgm.gen_sphere(gt_end, radius=0.009, rgb=gt_color, alpha=0.95).attach_to(world)

    robot = XArmLite6Miller(enable_cc=True)
    robot.goto_given_conf(anchor['q'])
    robot.gen_meshmodel(rgb=gt_color, alpha=0.55, toggle_tcp_frame=True).attach_to(world)

    if min_pos_idx >= 0:
        pred_color = np.array([0.15, 0.45, 0.95], dtype=np.float32)
        robot.goto_given_conf(q_preds[min_pos_idx].astype(np.float32))
        robot.gen_meshmodel(rgb=pred_color, alpha=0.55, toggle_tcp_frame=False).attach_to(world)
        pred_end = start + direction * float(pred_lengths[min_pos_idx])
        mgm.gen_stick(spos=start, epos=pred_end, radius=0.004, rgb=pred_color, alpha=0.95).attach_to(world)
        mgm.gen_sphere(pred_end, radius=0.009, rgb=pred_color, alpha=0.95).attach_to(world)

    world.run()


if __name__ == '__main__':
    main()
