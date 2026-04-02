from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diffusion import DEFAULT_H5_PATH, DEFAULT_RUN_NAME, DEFAULT_WORKDIR, normalize_condition
from diffusion_sample import load_model
from diffusion_eval_batch_candidates_lnet import (
    batch_position_error_and_correction,
    build_tracker,
    load_lnet_contrastive_model,
    rollout_lengths_batch,
)
from diffusion_eval_contrastive_guidance import load_anchor, sample_with_guidance
from length_prediction.paths import LNET_CONTRASTIVE_RUNS_DIR
import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller


DEFAULT_LAMBDAS = [0.1, 1.0, 5.0, 10.0]
DEFAULT_LAMBDAS = [0.1]
LAMBDA_COLORS = {
    0.1: np.array([0.25, 0.45, 0.95], dtype=np.float32),
    1.0: np.array([0.10, 0.70, 0.95], dtype=np.float32),
    5.0: np.array([0.95, 0.55, 0.15], dtype=np.float32),
    10.0: np.array([0.95, 0.15, 0.15], dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Render final guided robot poses for one specified trajectory id in the WRS world.')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--lnet-contrastive-ckpt', type=Path, default=LNET_CONTRASTIVE_RUNS_DIR / 'lnet_contrastive_q_cond_to_length_sub10_pref' / 'lnet_contrastive_best.pt')
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument('--traj-id', type=str, default='traj_056550')
    parser.add_argument('--point-idx', type=int, default=16)
    parser.add_argument('--title', type=str, default='contrastive_guidance_case')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--lambdas', type=float, nargs='+', default=DEFAULT_LAMBDAS)
    parser.add_argument('--show-corrected', action='store_true', help='Render corrected q instead of raw generated q.')
    parser.add_argument('--jsonl-path', type=Path, default=None, help='Optional JSONL output path. Defaults to guidance_vis_cases/log.jsonl and appends records.')
    return parser.parse_args()


def fmt(x: float) -> str:
    return f'{float(x):.3f}'


def round_nested(obj):
    if isinstance(obj, float):
        return round(obj, 3)
    if isinstance(obj, dict):
        return {k: round_nested(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_nested(v) for v in obj]
    return obj


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normal / max(np.linalg.norm(normal), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-12)
    return np.column_stack((x_axis, y_axis, z_axis))


def rotation_matrix_from_direction_normal(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    x_axis = np.asarray(direction, dtype=np.float32)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    z_axis = np.asarray(normal, dtype=np.float32)
    z_axis = z_axis / max(np.linalg.norm(z_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        return rotation_matrix_from_normal(z_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-12)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / max(np.linalg.norm(z_axis), 1e-12)
    return np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)


def evaluate_case(anchor: dict, model, lnet_contrastive, tracker, tracker_device, stats: dict, q_dim: int, steps: int, temperature: float, correction_iters: int, correction_tol: float, correction_damping: float, lambda_values: list[float], device: torch.device):
    condition_raw = np.concatenate([anchor['pos'], anchor['direction'], anchor['target_normal']], axis=0).astype(np.float32)
    condition_norm = normalize_condition(condition_raw[None, :], stats)[0]
    x_dim = q_dim + 10
    prior_np = np.zeros((1, 1, x_dim), dtype=np.float32)
    prior_np[:, 0, q_dim:q_dim + 9] = condition_norm[None, :]
    prior = torch.from_numpy(prior_np).float().to(device)
    init_noise = torch.randn_like(prior)

    results = []
    for lam in lambda_values:
        results.append(
            sample_with_guidance(
                model=model,
                lnet_contrastive=lnet_contrastive,
                stats=stats,
                q_dim=q_dim,
                condition_raw_np=condition_raw,
                prior=prior,
                init_noise=init_noise,
                sample_steps=steps,
                temperature=float(temperature),
                lambda_guidance=float(lam),
                device=device,
            )
        )

    q_pred_batch = np.stack([r['final_q'] for r in results], axis=0).astype(np.float32)
    q_corr_batch, raw_pos_err = batch_position_error_and_correction(
        tracker.robot,
        q_pred_batch,
        anchor['pos'],
        correction_damping,
        correction_iters,
        correction_tol,
        tracker_device,
    )
    real_len_batch = rollout_lengths_batch(tracker, tracker_device, q_corr_batch, anchor['direction'], anchor['target_normal'])
    gt_real = float(rollout_lengths_batch(tracker, tracker_device, anchor['q'][None, :], anchor['direction'], anchor['target_normal'])[0])

    rows = []
    for idx, result in enumerate(results):
        rows.append({
            'lambda': float(result['lambda']),
            'final_score': float(result['final_score']),
            'diff_pred_length': float(result['final_pred_length']),
            'raw_pos_err_mm': float(raw_pos_err[idx] * 1e3),
            'guided_real_len': float(real_len_batch[idx]),
            'gain_vs_gt': float(real_len_batch[idx] - gt_real),
            'q_raw': q_pred_batch[idx],
            'q_corr': q_corr_batch[idx],
        })
    return gt_real, rows


def render_world(anchor: dict, gt_real: float, rows: list[dict], show_corrected: bool) -> None:
    world = wd.World(cam_pos=[1.7, -1.5, 1.05], lookat_pos=[0.25, 0.0, 0.25])
    mgm.gen_frame().attach_to(world)
    anchor_rotmat = rotation_matrix_from_direction_normal(anchor['direction'], anchor['target_normal'])
    mgm.gen_frame(pos=anchor['pos'], rotmat=anchor_rotmat, ax_length=0.12).attach_to(world)

    start = anchor['pos']
    direction = anchor['direction']
    plane_size = 1.2
    plane_rotmat = rotation_matrix_from_normal(anchor['target_normal'])
    plane_center = start + 0.5 * plane_size * direction
    mcm.gen_box(
        xyz_lengths=[plane_size, plane_size, 0.001],
        pos=plane_center,
        rotmat=plane_rotmat,
        rgb=[180 / 255, 211 / 255, 217 / 255],
        alpha=0.35,
    ).attach_to(world)
    mgm.gen_sphere(start, radius=0.010, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)

    gt_color = np.array([0.10, 0.75, 0.20], dtype=np.float32)
    gt_end = start + direction * gt_real
    mgm.gen_stick(spos=start, epos=gt_end, radius=0.0045, rgb=gt_color, alpha=0.9).attach_to(world)
    mgm.gen_sphere(gt_end, radius=0.009, rgb=gt_color, alpha=0.95).attach_to(world)

    robot = XArmLite6Miller(enable_cc=True)
    robot.goto_given_conf(anchor['q'])
    robot.gen_meshmodel(rgb=gt_color, alpha=0.45, toggle_tcp_frame=True).attach_to(world)

    for row in rows:
        lam = float(row['lambda'])
        color = LAMBDA_COLORS.get(lam, np.random.default_rng(int(lam * 1000)).uniform(0.1, 0.95, size=3).astype(np.float32))
        q_vis = row['q_corr'] if show_corrected else row['q_raw']
        robot.goto_given_conf(q_vis.astype(np.float32))
        robot.gen_meshmodel(rgb=color, alpha=0.35, toggle_tcp_frame=False).attach_to(world)
        end = start + direction * float(row['guided_real_len'])
        mgm.gen_stick(spos=start, epos=end, radius=0.0035, rgb=color, alpha=0.9).attach_to(world)
        mgm.gen_sphere(end, radius=0.0075, rgb=color, alpha=0.9).attach_to(world)

    world.run()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed if args.seed is not None else int(np.random.SeedSequence().entropy))
    device = torch.device(args.device)

    _, stats, model, q_dim, diffusion_steps = load_model(args.bundle, device)
    lnet_contrastive = load_lnet_contrastive_model(args.lnet_contrastive_ckpt, device)
    tracker, tracker_device = build_tracker(device)
    steps = int(args.sample_steps) if args.sample_steps is not None else int(diffusion_steps)

    lambda_values = [float(v) for v in args.lambdas]
    anchor = load_anchor(args.h5_path, args.traj_id, args.point_idx, rng)
    gt_real, rows = evaluate_case(
        anchor=anchor,
        model=model,
        lnet_contrastive=lnet_contrastive,
        tracker=tracker,
        tracker_device=tracker_device,
        stats=stats,
        q_dim=q_dim,
        steps=steps,
        temperature=float(args.temperature),
        correction_iters=int(args.correction_iters),
        correction_tol=float(args.correction_tol),
        correction_damping=float(args.correction_damping),
        lambda_values=lambda_values,
        device=device,
    )

    pieces = [f"{args.title} ({args.traj_id})", f"GT:{fmt(gt_real)}"]
    for row in rows:
        pieces.append(f"λ={float(row['lambda']):g}:{fmt(row['guided_real_len'])}")
    print(' | '.join(pieces))

    jsonl_path = args.jsonl_path
    if jsonl_path is None:
        jsonl_path = Path(__file__).resolve().parent / 'guidance_vis_cases' / 'log.jsonl'

    meta_record = {
        'record_type': 'meta',
        'title': args.title,
        'traj_id': args.traj_id,
        'point_idx': int(args.point_idx),
        'gt_dataset_length': round(float(anchor['gt_length']), 3),
        'gt_real_length': round(float(gt_real), 3),
        'render_q': 'corrected' if args.show_corrected else 'raw',
        'direction': round_nested(np.asarray(anchor['direction']).tolist()),
        'target_normal': round_nested(np.asarray(anchor['target_normal']).tolist()),
        'pos': round_nested(np.asarray(anchor['pos']).tolist()),
    }
    lambda_records = [
        {
            'record_type': 'lambda',
            'traj_id': args.traj_id,
            'point_idx': int(args.point_idx),
            'lambda': round(float(r['lambda']), 3),
            'guided_real_len': round(float(r['guided_real_len']), 3),
            'gain_vs_gt': round(float(r['gain_vs_gt']), 3),
            'final_score': round(float(r['final_score']), 3),
            'diff_pred_length': round(float(r['diff_pred_length']), 3),
            'raw_pos_err_mm': round(float(r['raw_pos_err_mm']), 3),
            'q_raw': round_nested(np.asarray(r['q_raw']).tolist()),
            'q_corr': round_nested(np.asarray(r['q_corr']).tolist()),
        }
        for r in rows
    ]

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(round_nested(meta_record), ensure_ascii=False, indent=2) + '\n')
        for record in lambda_records:
            f.write(json.dumps(round_nested(record), ensure_ascii=False, indent=2) + '\n')
        f.write('\n')
    print(f'[saved] {jsonl_path}')

    render_world(anchor, gt_real, rows, args.show_corrected)


if __name__ == '__main__':
    main()
