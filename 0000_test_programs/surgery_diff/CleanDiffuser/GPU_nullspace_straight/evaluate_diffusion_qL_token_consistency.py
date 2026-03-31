import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from kinematic_diffusion_common import DEFAULT_H5_PATH, DEFAULT_RUN_NAME, DEFAULT_WORKDIR, sample_q_length_from_condition
from sample_kinematic_dit_inpainting import load_model, normalize_direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate diffusion predictions against the training target only: q and remaining_length.')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--traj-id', type=str, default=None)
    parser.add_argument('--point-idx', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n-samples', type=int, default=8)
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
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


def choose_best(predictions: list[dict], gt_q: np.ndarray, gt_length: float) -> dict:
    return min(
        predictions,
        key=lambda item: (
            abs(item['pred_length'] - gt_length),
            item['q_l2'],
        ),
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed if args.seed is not None else int(np.random.SeedSequence().entropy))
    device = torch.device(args.device)
    _, stats, model, q_dim, diffusion_steps = load_model(args.bundle, device)
    gt = load_gt(args.h5_path, args.traj_id, args.point_idx, rng)

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

    predictions = []
    for idx, (q_pred, pred_length) in enumerate(zip(q_preds, pred_lengths)):
        q_l2 = float(np.linalg.norm(q_pred - gt['q']))
        q_l1 = float(np.mean(np.abs(q_pred - gt['q'])))
        length_abs_err = float(abs(float(pred_length) - gt['length']))
        predictions.append({
            'sample_idx': idx,
            'q_pred': q_pred.astype(np.float32),
            'pred_length': float(pred_length),
            'q_l2': q_l2,
            'q_l1': q_l1,
            'length_abs_err': length_abs_err,
        })

    best = choose_best(predictions, gt['q'], gt['length'])

    print(f"[gt] traj={gt['traj_key']} point_idx={gt['point_idx']}")
    print(f"[gt] pos={np.array2string(gt['pos'], precision=4, separator=', ')}")
    print(f"[gt] direction={np.array2string(gt['direction'], precision=4, separator=', ')}")
    print(f"[gt] q={np.array2string(gt['q'], precision=4, separator=', ')}")
    print(f"[gt] remaining_length={gt['length']:.6f}")

    for pred in predictions:
        print(
            f"[pred {pred['sample_idx']}] q_l2={pred['q_l2']:.6f} q_l1={pred['q_l1']:.6f} "
            f"pred_length={pred['pred_length']:.6f} length_abs_err={pred['length_abs_err']:.6f}"
        )

    summary = {
        'gt': {
            'traj_key': gt['traj_key'],
            'point_idx': int(gt['point_idx']),
            'q': gt['q'].tolist(),
            'length': float(gt['length']),
        },
        'best_pred': {
            'sample_idx': int(best['sample_idx']),
            'q_pred': best['q_pred'].tolist(),
            'pred_length': float(best['pred_length']),
            'q_l2': float(best['q_l2']),
            'q_l1': float(best['q_l1']),
            'length_abs_err': float(best['length_abs_err']),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
