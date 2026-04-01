import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import h5py
import numpy as np
import torch

import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd

from diffusion import DEFAULT_H5_PATH, DEFAULT_RUN_NAME, DEFAULT_WORKDIR, sample_q_length_from_condition
from diffusion_sample import JacobianCorrection, load_model, normalize_direction
from trajectory_generation.xarmlite6_nullspace_straight_demo import NullspaceStraightTracker, TrackerConfig
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Minimal evaluation: compare diffusion-predicted q/L against ground truth and visualize both straight-line traces.')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--traj-id', type=str, default=None)
    parser.add_argument('--point-idx', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n-samples', type=int, default=8)
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--tracker-max-steps', type=int, default=400)
    parser.add_argument('--no-vis', action='store_true')
    return parser.parse_args()


def load_gt(h5_path: Path, traj_id: str | None, point_idx: int | None, rng: np.random.Generator) -> dict:
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


def choose_best(results: list[dict], gt_length: float) -> dict:
    return min(results, key=lambda r: (abs(r['track'].projected_length - gt_length), abs(r['pred_length'] - gt_length), r['pos_err']))


def visualize(robot: XArmLite6Miller, gt: dict, gt_track, pred: dict) -> None:
    world = wd.World(cam_pos=[1.7, -1.5, 1.05], lookat_pos=[0.25, 0.0, 0.25])
    mgm.gen_frame().attach_to(world)

    start_pos = gt['pos']
    direction = gt['direction']
    gt_ideal_end = start_pos + direction * gt['length']
    pred_ideal_end = start_pos + direction * pred['pred_length']

    gt_segs = [[gt_track.tcp_path[i], gt_track.tcp_path[i + 1]] for i in range(len(gt_track.tcp_path) - 1)]
    if gt_segs:
        mgm.gen_linesegs(gt_segs, thickness=0.005, rgb=np.array([0.1, 0.75, 0.2]), alpha=1.0).attach_to(world)
    raw_pred_segs = [[pred['raw_track'].tcp_path[i], pred['raw_track'].tcp_path[i + 1]] for i in range(len(pred['raw_track'].tcp_path) - 1)]
    if raw_pred_segs:
        mgm.gen_linesegs(raw_pred_segs, thickness=0.005, rgb=np.array([0.9, 0.1, 0.1]), alpha=1.0).attach_to(world)
    pred_segs = [[pred['track'].tcp_path[i], pred['track'].tcp_path[i + 1]] for i in range(len(pred['track'].tcp_path) - 1)]
    if pred_segs:
        mgm.gen_linesegs(pred_segs, thickness=0.005, rgb=np.array([1.0, 0.55, 0.0]), alpha=1.0).attach_to(world)

    mgm.gen_arrow(spos=start_pos, epos=start_pos + 0.25 * direction, stick_radius=0.006, rgb=np.array([0.9, 0.1, 0.1])).attach_to(world)
    mgm.gen_stick(spos=start_pos, epos=gt_ideal_end, radius=0.0025, rgb=np.array([0.0, 0.4, 1.0]), alpha=0.45).attach_to(world)
    mgm.gen_stick(spos=start_pos, epos=pred_ideal_end, radius=0.0025, rgb=np.array([1.0, 0.5, 0.0]), alpha=0.35).attach_to(world)
    mgm.gen_sphere(start_pos, radius=0.010, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)
    mgm.gen_sphere(gt_ideal_end, radius=0.010, rgb=np.array([0.0, 0.4, 1.0]), alpha=0.7).attach_to(world)
    mgm.gen_sphere(pred_ideal_end, radius=0.010, rgb=np.array([1.0, 0.5, 0.0]), alpha=0.7).attach_to(world)
    if len(gt_track.tcp_path) > 0:
        mgm.gen_sphere(gt_track.tcp_path[-1], radius=0.012, rgb=np.array([0.1, 0.75, 0.2]), alpha=1.0).attach_to(world)
    if len(pred['raw_track'].tcp_path) > 0:
        mgm.gen_sphere(pred['raw_track'].tcp_path[-1], radius=0.010, rgb=np.array([0.9, 0.1, 0.1]), alpha=1.0).attach_to(world)
    if len(pred['track'].tcp_path) > 0:
        mgm.gen_sphere(pred['track'].tcp_path[-1], radius=0.012, rgb=np.array([1.0, 0.55, 0.0]), alpha=1.0).attach_to(world)

    robot.goto_given_conf(gt['q'])
    robot.gen_meshmodel(rgb=np.array([0.1, 0.75, 0.2]), alpha=0.35, toggle_tcp_frame=True).attach_to(world)
    robot.goto_given_conf(pred['q_pred'])
    robot.gen_meshmodel(rgb=np.array([0.9, 0.1, 0.1]), alpha=0.35, toggle_tcp_frame=True).attach_to(world)
    robot.goto_given_conf(pred['q_corr'])
    robot.gen_meshmodel(rgb=np.array([1.0, 0.55, 0.0]), alpha=0.75, toggle_tcp_frame=True).attach_to(world)

    world.run()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed if args.seed is not None else int(np.random.SeedSequence().entropy))
    device = torch.device(args.device)
    _, stats, model, q_dim, diffusion_steps = load_model(args.bundle, device)
    gt = load_gt(args.h5_path, args.traj_id, args.point_idx, rng)

    robot = XArmLite6Miller(enable_cc=True)
    correction = JacobianCorrection(robot=robot, damping=float(args.correction_damping))
    tracker = NullspaceStraightTracker(robot=robot, config=TrackerConfig(max_steps=int(args.tracker_max_steps)))

    gt_track = tracker.run(gt['q'], gt['direction'])
    condition = np.concatenate([gt['pos'], gt['direction']], axis=0).astype(np.float32)
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

    results = []
    for q_pred, pred_length in zip(q_preds, pred_lengths):
        q_corr, pos_err = correction.run(q_pred, gt['pos'], max_iters=int(args.correction_iters), tol=float(args.correction_tol))
        raw_track = tracker.run(q_pred.astype(np.float32), gt['direction'])
        corr_track = tracker.run(q_corr, gt['direction'])
        results.append({
            'q_pred': q_pred.astype(np.float32),
            'q_corr': q_corr.astype(np.float32),
            'pred_length': float(pred_length),
            'pos_err': float(pos_err),
            'raw_track': raw_track,
            'track': corr_track,
        })

    best = choose_best(results, gt['length'])

    summary = {
        'gt': {
            'traj_key': gt['traj_key'],
            'point_idx': int(gt['point_idx']),
            'q': gt['q'].tolist(),
            'length': float(gt['length']),
            'real_track_length': float(gt_track.projected_length),
        },
        'pred': {
            'q_pred': best['q_pred'].tolist(),
            'q_corr': best['q_corr'].tolist(),
            'pred_length': float(best['pred_length']),
            'raw_track_length': float(best['raw_track'].projected_length),
            'real_track_length': float(best['track'].projected_length),
            'pos_err': float(best['pos_err']),
            'q_pred_to_gt_l2': float(np.linalg.norm(best['q_pred'] - gt['q'])),
            'q_corr_to_gt_l2': float(np.linalg.norm(best['q_corr'] - gt['q'])),
        },
    }
    print(json.dumps(summary, indent=2))

    if not args.no_vis:
        visualize(robot, gt, gt_track, best)


if __name__ == '__main__':
    main()
