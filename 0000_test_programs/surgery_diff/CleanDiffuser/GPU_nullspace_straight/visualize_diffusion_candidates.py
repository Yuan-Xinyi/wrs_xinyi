import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd

from kinematic_diffusion_common import DEFAULT_H5_PATH, DEFAULT_RUN_NAME, DEFAULT_WORKDIR, sample_q_length_from_condition
from sample_kinematic_dit_inpainting import load_model, normalize_direction
import jax2torch
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller
from wrs.robot_sim.robots.xarmlite6_wg.sphere_collision_checker import SphereCollisionChecker
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
from xarmlite6_gpu_nullspave_straight_demo import GPUNullspaceStraightTracker, TrackerConfig
import jax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize multiple diffusion candidate q solutions for the same (pos, direction) condition.')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--traj-id', type=str, default=None)
    parser.add_argument('--point-idx', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n-samples', type=int, default=32)
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.22)
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
        }




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


def rollout_length(tracker: GPUNullspaceStraightTracker, tracker_device: torch.device, q: np.ndarray, direction: np.ndarray) -> float:
    q_batch = torch.from_numpy(q.astype(np.float32)).unsqueeze(0).to(tracker_device)
    d_batch = torch.from_numpy(direction.astype(np.float32)).unsqueeze(0).to(tracker_device)
    n_batch = torch.zeros((1, 3), dtype=torch.float32, device=tracker_device)
    result = tracker.run_batch(q0_batch=q_batch, direction_batch=d_batch, target_normal_batch=n_batch)
    return float(result.projected_length[0].detach().cpu().item())

def color_cycle(n: int) -> list[np.ndarray]:
    hues = np.linspace(0.0, 1.0, max(n, 2), endpoint=False)
    colors = []
    for h in hues[:n]:
        rgb = np.array([np.sin(2*np.pi*(h + 0.0))*0.5 + 0.5, np.sin(2*np.pi*(h + 1/3))*0.5 + 0.5, np.sin(2*np.pi*(h + 2/3))*0.5 + 0.5], dtype=np.float32)
        colors.append(np.clip(rgb, 0.1, 0.95))
    return colors


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
    gt_real_length = rollout_length(tracker, tracker_device, anchor['q'], anchor['direction'])
    candidate_real_lengths = []
    print(f"GT {anchor['length']:.6f} {gt_real_length:.6f}")
    for idx, (q_pred, pred_length) in enumerate(zip(q_preds, pred_lengths)):
        real_length = rollout_length(tracker, tracker_device, q_pred.astype(np.float32), anchor['direction'])
        candidate_real_lengths.append(real_length)
        print(f"{idx:02d} {float(pred_length):.6f} {real_length:.6f}")

    best_idx = int(np.argmax(candidate_real_lengths)) if candidate_real_lengths else -1
    if best_idx >= 0:
        print(f"BEST {best_idx:02d} {float(pred_lengths[best_idx]):.6f} {float(candidate_real_lengths[best_idx]):.6f} | GT {anchor['length']:.6f} {gt_real_length:.6f}")

    if args.no_vis:
        return

    world = wd.World(cam_pos=[1.7, -1.5, 1.05], lookat_pos=[0.25, 0.0, 0.25])
    mgm.gen_frame().attach_to(world)
    start = anchor['pos']
    direction = anchor['direction']
    mgm.gen_arrow(spos=start, epos=start + 0.25 * direction, stick_radius=0.006, rgb=np.array([0.95, 0.15, 0.15])).attach_to(world)
    mgm.gen_sphere(start, radius=0.010, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)
    mgm.gen_stick(spos=start, epos=start + direction * anchor['length'], radius=0.0025, rgb=np.array([0.0, 0.4, 1.0]), alpha=0.4).attach_to(world)

    robot = XArmLite6Miller(enable_cc=True)
    robot.goto_given_conf(anchor['q'])
    robot.gen_meshmodel(rgb=np.array([0.1, 0.75, 0.2]), alpha=0.5, toggle_tcp_frame=True).attach_to(world)

    for color, q_pred, pred_length in zip(color_cycle(len(q_preds)), q_preds, pred_lengths):
        robot.goto_given_conf(q_pred.astype(np.float32))
        robot.gen_meshmodel(rgb=color, alpha=float(args.alpha), toggle_tcp_frame=False).attach_to(world)
        mgm.gen_stick(spos=start, epos=start + direction * float(pred_length), radius=0.002, rgb=color, alpha=0.25).attach_to(world)

    world.run()


if __name__ == '__main__':
    main()
