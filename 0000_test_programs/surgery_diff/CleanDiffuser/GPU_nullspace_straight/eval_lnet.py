from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from lnet import LNet

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_H5 = BASE_DIR / 'xarmlite6_gpu_trajectories_100000_sub10.hdf5'
DEFAULT_CKPT = BASE_DIR / 'lnet_runs' / 'lnet_q_cond_to_length_sub10' / 'lnet_best.pt'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a trained LNet on one dataset sample or a random one.')
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--traj-id', type=str, default=None)
    parser.add_argument('--point-idx', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


def load_model(ckpt_path: Path, device: torch.device) -> LNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = LNet(
        q_min=ckpt['q_min'],
        q_max=ckpt['q_max'],
        in_min=ckpt['in_min'],
        in_max=ckpt['in_max'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def load_sample(h5_path: Path, traj_id: str | None, point_idx: int | None, seed: int | None) -> dict:
    rng = np.random.default_rng(seed if seed is not None else int(np.random.SeedSequence().entropy))
    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f['trajectories'].keys())
        key = traj_id if traj_id is not None else keys[int(rng.integers(len(keys)))]
        grp = f['trajectories'][key]
        n = int(grp.attrs['num_points'])
        idx = int(point_idx) if point_idx is not None else int(rng.integers(n))
        return {
            'traj_key': key,
            'point_idx': idx,
            'q': np.asarray(grp['q'][idx], dtype=np.float32),
            'pos': np.asarray(grp['tcp_pos'][idx], dtype=np.float32),
            'direction': np.asarray(grp.attrs['direction'], dtype=np.float32),
            'normal': np.asarray(grp.attrs['target_normal'], dtype=np.float32),
            'target_length': float(np.asarray(grp['remaining_length'][idx], dtype=np.float32)),
        }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.ckpt, device)
    sample = load_sample(args.h5_path, args.traj_id, args.point_idx, args.seed)

    q = torch.from_numpy(sample['q']).float().unsqueeze(0).to(device)
    cond = torch.from_numpy(np.concatenate([sample['pos'], sample['direction'], sample['normal']], axis=0)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred = float(model(q, cond).item())
    grad = model.get_gradient(q, cond).detach().cpu().numpy()[0]

    result = {
        'traj_key': sample['traj_key'],
        'point_idx': sample['point_idx'],
        'q': sample['q'].tolist(),
        'cond': {
            'pos': sample['pos'].tolist(),
            'direction': sample['direction'].tolist(),
            'normal': sample['normal'].tolist(),
        },
        'target_length': sample['target_length'],
        'pred_length': pred,
        'abs_error': abs(pred - sample['target_length']),
        'dq_grad': grad.tolist(),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
