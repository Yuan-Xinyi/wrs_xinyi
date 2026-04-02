from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
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
from diffusion_eval_contrastive_guidance import sample_with_guidance
from length_prediction.paths import LNET_CONTRASTIVE_RUNS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch evaluate contrastive classifier guidance on random trajectory points.')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--lnet-contrastive-ckpt', type=Path, default=LNET_CONTRASTIVE_RUNS_DIR / 'lnet_contrastive_q_cond_to_length_sub10_pref' / 'lnet_contrastive_best.pt')
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num-cases', type=int, default=100)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.0, 0.1, 1.0, 5.0, 10.0])
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--print-every', type=int, default=10)
    return parser.parse_args()


def sample_random_anchor(h5_path: Path, rng: np.random.Generator) -> dict:
    with h5py.File(h5_path, 'r') as f:
        traj_root = f['trajectories']
        traj_keys = sorted(traj_root.keys())
        traj_id = str(rng.choice(traj_keys))
        grp = traj_root[traj_id]
        n = int(grp.attrs['num_points'])
        # point_idx = int(rng.integers(0, n)
        point_idx = 0
        q = np.asarray(grp['q'][point_idx], dtype=np.float32)
        pos = np.asarray(grp['tcp_pos'][point_idx], dtype=np.float32)
        direction = np.asarray(grp.attrs['direction'], dtype=np.float32)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        target_normal = np.asarray(grp.attrs['target_normal'], dtype=np.float32)
        target_normal = target_normal / max(float(np.linalg.norm(target_normal)), 1e-12)
        gt_length = float(np.asarray(grp['remaining_length'][point_idx], dtype=np.float32))
        return {
            'traj_id': traj_id,
            'point_idx': point_idx,
            'q': q,
            'pos': pos,
            'direction': direction,
            'target_normal': target_normal,
            'gt_length': gt_length,
        }


def fmt(x: float) -> str:
    return f'{float(x):.3f}'


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed if args.seed is not None else int(np.random.SeedSequence().entropy))
    device = torch.device(args.device)

    _, stats, model, q_dim, diffusion_steps = load_model(args.bundle, device)
    lnet_contrastive = load_lnet_contrastive_model(args.lnet_contrastive_ckpt, device)
    tracker, tracker_device = build_tracker(device)
    steps = int(args.sample_steps) if args.sample_steps is not None else int(diffusion_steps)

    lambda_values = [float(v) for v in args.lambdas]
    summary = {lam: [] for lam in lambda_values}
    gains = {lam: [] for lam in lambda_values}
    gt_reals = []
    rows = []

    for case_idx in range(args.num_cases):
        anchor = sample_random_anchor(args.h5_path, rng)
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
                    temperature=float(args.temperature),
                    lambda_guidance=float(lam),
                    device=device,
                )
            )

        q_pred_batch = np.stack([r['final_q'] for r in results], axis=0).astype(np.float32)
        q_corr_batch, _ = batch_position_error_and_correction(
            tracker.robot,
            q_pred_batch,
            anchor['pos'],
            args.correction_damping,
            args.correction_iters,
            args.correction_tol,
            tracker_device,
        )
        real_len_batch = rollout_lengths_batch(tracker, tracker_device, q_corr_batch, anchor['direction'], anchor['target_normal'])
        gt_real = float(rollout_lengths_batch(tracker, tracker_device, anchor['q'][None, :], anchor['direction'], anchor['target_normal'])[0])
        gt_reals.append(gt_real)

        row = {
            'traj_id': anchor['traj_id'],
            'point_idx': int(anchor['point_idx']),
            'gt_real': gt_real,
        }
        for idx, lam in enumerate(lambda_values):
            guided_real = float(real_len_batch[idx])
            gain = guided_real - gt_real
            row[f'lam_{lam:g}_real'] = guided_real
            row[f'lam_{lam:g}_gain'] = gain
            summary[lam].append(guided_real)
            gains[lam].append(gain)
        rows.append(row)

        pieces = [
            f"traj={anchor['traj_id']}",
            f"pt={anchor['point_idx']}",
            f"gt={fmt(gt_real)}",
        ]
        for lam in lambda_values:
            pieces.append(f"lam{lam:g}={fmt(row[f'lam_{lam:g}_real'])}")
            pieces.append(f"gain{lam:g}={fmt(row[f'lam_{lam:g}_gain'])}")
        print(' '.join(pieces))

        if args.print_every > 0 and ((case_idx + 1) % args.print_every == 0 or case_idx + 1 == args.num_cases):
            print(f'[progress] {case_idx + 1}/{args.num_cases}')

    agg = {
        'num_cases': int(args.num_cases),
        'gt_real_mean': float(np.mean(gt_reals)) if gt_reals else None,
        'lambdas': {},
    }
    for lam in lambda_values:
        agg['lambdas'][str(lam)] = {
            'guided_real_mean': float(np.mean(summary[lam])) if summary[lam] else None,
            'gain_vs_gt_mean': float(np.mean(gains[lam])) if gains[lam] else None,
            'gain_vs_gt_p50': float(np.percentile(gains[lam], 50)) if gains[lam] else None,
            'gain_vs_gt_p90': float(np.percentile(gains[lam], 90)) if gains[lam] else None,
        }
    print(json.dumps(agg, indent=2))


if __name__ == '__main__':
    main()
