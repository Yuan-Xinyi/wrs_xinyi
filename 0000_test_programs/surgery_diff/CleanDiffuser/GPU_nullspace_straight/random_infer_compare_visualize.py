import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd

from kinematic_diffusion_common import (
    DEFAULT_H5_PATH,
    DEFAULT_RUN_NAME,
    DEFAULT_WORKDIR,
    FeatureLayout,
    StandardScaler,
    canonicalize_quaternion_xyzw,
)
from sample_kinematic_dit_inpainting import (
    JacobianCorrection,
    build_model,
    normalize_direction,
    sample_tokens,
)
from xarmlite6_nullspace_straight_demo import TrackerConfig, NullspaceStraightTracker
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Randomly sample one GT token from HDF5, run diffusion inpainting, compare to GT, and visualize both trajectories.'
    )
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n-samples', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--tracker-max-steps', type=int, default=400)
    parser.add_argument('--show-all-pred-ends', action='store_true')
    parser.add_argument('--no-vis', action='store_true')
    return parser.parse_args()


def random_gt_sample(h5_path: Path, rng: np.random.Generator) -> dict:
    with h5py.File(h5_path, 'r') as f:
        traj_keys = sorted(f['trajectories'].keys())
        traj_key = traj_keys[int(rng.integers(len(traj_keys)))]
        grp = f['trajectories'][traj_key]
        num_points = int(grp.attrs['num_points'])
        point_idx = int(rng.integers(num_points))
        q = np.asarray(grp['q'][point_idx], dtype=np.float32)
        pos = np.asarray(grp['tcp_pos'][point_idx], dtype=np.float32)
        rotmat = np.asarray(grp['tcp_rotmat'][point_idx], dtype=np.float32)
        rot_q = canonicalize_quaternion_xyzw(rm.rotmat_to_quaternion(rotmat))
        direction = normalize_direction(np.asarray(grp.attrs['direction'], dtype=np.float32))
        length = float(np.asarray(grp['remaining_length'][point_idx], dtype=np.float32))
        mu = float(np.asarray(grp['mu'][point_idx], dtype=np.float32))
        return {
            'traj_key': traj_key,
            'point_idx': point_idx,
            'q': q,
            'pos': pos,
            'rot_q': rot_q,
            'direction': direction,
            'length': length,
            'mu': mu,
            'termination_reason': str(grp.attrs['termination_reason']),
            'traj_total_length': float(grp.attrs['total_projected_length']),
        }


def quat_angle_error_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = canonicalize_quaternion_xyzw(q1)
    q2 = canonicalize_quaternion_xyzw(q2)
    dot = float(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


@torch.no_grad()
def load_bundle(bundle_path: Path, device: torch.device):
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    layout = FeatureLayout(q_dim=int(bundle['layout_q_dim']))
    scaler = StandardScaler(
        mean=np.asarray(bundle['scaler_mean'], dtype=np.float32),
        std=np.asarray(bundle['scaler_std'], dtype=np.float32),
        data_min=np.asarray(bundle['scaler_min'], dtype=np.float32),
        data_max=np.asarray(bundle['scaler_max'], dtype=np.float32),
    )
    model = build_model(bundle, device)
    return bundle, layout, scaler, model


def evaluate_predictions(
    robot: XArmLite6Miller,
    gt: dict,
    samples: np.ndarray,
    layout: FeatureLayout,
    correction: JacobianCorrection,
    tracker: NullspaceStraightTracker,
) -> list[dict]:
    results = []
    for idx, token in enumerate(samples):
        q_pred = token[layout.q_slice].astype(np.float32)
        rot_pred = canonicalize_quaternion_xyzw(token[layout.rot_slice])
        q_corr, pos_err = correction.run(q_pred, gt['pos'], max_iters=correction_max_iters, tol=correction_tol)
        robot.goto_given_conf(q_corr)
        fk_pos, fk_rot = robot.fk(q_corr, update=False)
        fk_rot_q = canonicalize_quaternion_xyzw(rm.rotmat_to_quaternion(fk_rot))
        track = tracker.run(q_corr, gt['direction'])
        results.append(
            {
                'sample_idx': idx,
                'q_pred': q_pred,
                'q_corr': q_corr,
                'rot_pred_q': rot_pred,
                'fk_rot_q': fk_rot_q,
                'pos_err': float(pos_err),
                'q_l2_to_gt': float(np.linalg.norm(q_corr - gt['q'])),
                'rot_err_deg_pred_to_gt': quat_angle_error_deg(rot_pred, gt['rot_q']),
                'rot_err_deg_fk_to_gt': quat_angle_error_deg(fk_rot_q, gt['rot_q']),
                'track': track,
                'length_err_to_gt': float(abs(track.projected_length - gt['length'])),
            }
        )
    return results


def choose_best_prediction(predictions: list[dict], gt_length: float) -> dict:
    return min(
        predictions,
        key=lambda item: (
            abs(item['track'].projected_length - gt_length),
            item['pos_err'],
            item['q_l2_to_gt'],
        ),
    )


def visualize_comparison(robot: XArmLite6Miller, gt: dict, gt_track, best_pred: dict, all_preds: list[dict], show_all_pred_ends: bool) -> None:
    world = wd.World(cam_pos=[1.7, -1.5, 1.05], lookat_pos=[0.25, 0.0, 0.25])
    mgm.gen_frame().attach_to(world)

    start_pos = gt['pos']
    direction = gt['direction']
    gt_end_ideal = start_pos + direction * gt['length']
    pred_end_ideal = start_pos + direction * best_pred['track'].projected_length

    mgm.gen_arrow(
        spos=start_pos,
        epos=start_pos + 0.25 * direction,
        stick_radius=0.006,
        rgb=np.array([0.9, 0.1, 0.1]),
    ).attach_to(world)

    gt_line_segs = [[gt_track.tcp_path[i], gt_track.tcp_path[i + 1]] for i in range(len(gt_track.tcp_path) - 1)]
    if gt_line_segs:
        mgm.gen_linesegs(gt_line_segs, thickness=0.005, rgb=np.array([0.1, 0.75, 0.2]), alpha=1.0).attach_to(world)
    pred_line_segs = [[best_pred['track'].tcp_path[i], best_pred['track'].tcp_path[i + 1]] for i in range(len(best_pred['track'].tcp_path) - 1)]
    if pred_line_segs:
        mgm.gen_linesegs(pred_line_segs, thickness=0.005, rgb=np.array([1.0, 0.55, 0.0]), alpha=1.0).attach_to(world)

    mgm.gen_stick(spos=start_pos, epos=gt_end_ideal, radius=0.0025, rgb=np.array([0.0, 0.4, 1.0]), alpha=0.45).attach_to(world)
    mgm.gen_stick(spos=start_pos, epos=pred_end_ideal, radius=0.0025, rgb=np.array([1.0, 0.5, 0.0]), alpha=0.35).attach_to(world)

    mgm.gen_sphere(start_pos, radius=0.010, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)
    mgm.gen_sphere(gt_track.tcp_path[-1], radius=0.012, rgb=np.array([0.1, 0.75, 0.2]), alpha=1.0).attach_to(world)
    mgm.gen_sphere(best_pred['track'].tcp_path[-1], radius=0.012, rgb=np.array([1.0, 0.55, 0.0]), alpha=1.0).attach_to(world)

    if show_all_pred_ends:
        for pred in all_preds:
            mgm.gen_sphere(pred['track'].tcp_path[-1], radius=0.004, rgb=np.array([1.0, 0.7, 0.2]), alpha=0.4).attach_to(world)

    robot.goto_given_conf(gt['q'])
    robot.gen_meshmodel(rgb=np.array([0.1, 0.75, 0.2]), alpha=0.35, toggle_tcp_frame=True).attach_to(world)
    robot.goto_given_conf(best_pred['q_corr'])
    robot.gen_meshmodel(rgb=np.array([1.0, 0.55, 0.0]), alpha=0.75, toggle_tcp_frame=True).attach_to(world)

    world.run()


def main() -> None:
    global correction_max_iters, correction_tol
    args = parse_args()
    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)
    device = torch.device(args.device)

    bundle, layout, scaler, model = load_bundle(args.bundle, device)
    gt = random_gt_sample(args.h5_path, rng)

    robot = XArmLite6Miller(enable_cc=True)
    correction_max_iters = int(args.correction_iters)
    correction_tol = float(args.correction_tol)
    correction = JacobianCorrection(robot=robot, damping=float(args.correction_damping))
    tracker = NullspaceStraightTracker(robot=robot, config=TrackerConfig(max_steps=int(args.tracker_max_steps)))

    gt_track = tracker.run(gt['q'], gt['direction'])
    samples = sample_tokens(
        model=model,
        scaler=scaler,
        layout=layout,
        pos=gt['pos'],
        direction=gt['direction'],
        target_length=float(gt['length']),
        n_samples=int(args.n_samples),
        temperature=float(args.temperature),
        device=device,
    )
    predictions = evaluate_predictions(robot, gt, samples, layout, correction, tracker)
    best_pred = choose_best_prediction(predictions, gt['length'])

    print(f'[random] seed={seed}')
    print(f"[gt] traj={gt['traj_key']} point_idx={gt['point_idx']} length={gt['length']:.6f} mu={gt['mu']:.6f} traj_total={gt['traj_total_length']:.6f}")
    print(f"[gt] pos={np.array2string(gt['pos'], precision=4, separator=', ')}")
    print(f"[gt] direction={np.array2string(gt['direction'], precision=4, separator=', ')}")
    print(f"[gt] q={np.array2string(gt['q'], precision=4, separator=', ')}")
    print(f"[gt] rot_q={np.array2string(gt['rot_q'], precision=4, separator=', ')}")
    print(f"[gt] tracker_projected_length={gt_track.projected_length:.6f} termination={gt_track.termination_reason}")

    for pred in predictions:
        print(
            f"[pred {pred['sample_idx']}] pos_err={pred['pos_err']:.6f} q_l2={pred['q_l2_to_gt']:.6f} "
            f"rot_pred_err_deg={pred['rot_err_deg_pred_to_gt']:.3f} rot_fk_err_deg={pred['rot_err_deg_fk_to_gt']:.3f} "
            f"real_length={pred['track'].projected_length:.6f} length_err={pred['length_err_to_gt']:.6f} "
            f"termination={pred['track'].termination_reason}"
        )

    summary = {
        'seed': seed,
        'bundle': str(args.bundle),
        'gt': {
            'traj_key': gt['traj_key'],
            'point_idx': gt['point_idx'],
            'pos': gt['pos'].tolist(),
            'direction': gt['direction'].tolist(),
            'length': float(gt['length']),
            'mu': float(gt['mu']),
            'q': gt['q'].tolist(),
            'rot_q': gt['rot_q'].tolist(),
            'tracker_projected_length': float(gt_track.projected_length),
            'termination_reason': gt_track.termination_reason,
        },
        'best_pred': {
            'sample_idx': int(best_pred['sample_idx']),
            'q_corr': best_pred['q_corr'].tolist(),
            'rot_pred_q': best_pred['rot_pred_q'].tolist(),
            'fk_rot_q': best_pred['fk_rot_q'].tolist(),
            'pos_err': float(best_pred['pos_err']),
            'q_l2_to_gt': float(best_pred['q_l2_to_gt']),
            'rot_pred_err_deg': float(best_pred['rot_err_deg_pred_to_gt']),
            'rot_fk_err_deg': float(best_pred['rot_err_deg_fk_to_gt']),
            'real_projected_length': float(best_pred['track'].projected_length),
            'real_euclidean_length': float(best_pred['track'].euclidean_length),
            'mean_mu': float(best_pred['track'].mean_mu),
            'length_err_to_gt': float(best_pred['length_err_to_gt']),
            'termination_reason': best_pred['track'].termination_reason,
        },
    }
    print(json.dumps(summary, indent=2))

    if not args.no_vis:
        visualize_comparison(robot, gt, gt_track, best_pred, predictions, args.show_all_pred_ends)


if __name__ == '__main__':
    main()
