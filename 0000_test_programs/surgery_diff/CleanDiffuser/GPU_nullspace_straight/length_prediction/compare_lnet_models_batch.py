from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import jax
import jax2torch
import numpy as np
import torch
import sys

PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from lnet import LNet
from lnet_contrastive import LNetContrastive
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
from wrs.robot_sim.robots.xarmlite6_wg.sphere_collision_checker import SphereCollisionChecker
from trajectory_generation.xarm_nullspave_straight_gpu import GPUNullspaceStraightTracker, TrackerConfig

from paths import LNET_CONTRASTIVE_RUNS_DIR, LNET_RUNS_DIR

DEFAULT_LNET_CKPT = LNET_RUNS_DIR / 'lnet_q_cond_to_length_sub10' / 'lnet_best.pt'
DEFAULT_CONTRASTIVE_CKPT = LNET_CONTRASTIVE_RUNS_DIR / 'lnet_contrastive_q_cond_to_length_sub10' / 'lnet_contrastive_best.pt'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare LNet and contrastive LNet on fresh random rollout samples.')
    parser.add_argument('--lnet-ckpt', type=Path, default=DEFAULT_LNET_CKPT)
    parser.add_argument('--contrastive-ckpt', type=Path, default=DEFAULT_CONTRASTIVE_CKPT)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--rollout-batch-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_lnet(ckpt_path: Path, device: torch.device) -> LNet:
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


def load_contrastive(ckpt_path: Path, device: torch.device) -> LNetContrastive:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta_args = ckpt.get('args', {})
    model = LNetContrastive(
        q_min=ckpt['q_min'],
        q_max=ckpt['q_max'],
        in_min=ckpt['in_min'],
        in_max=ckpt['in_max'],
        pair_threshold=float(meta_args.get('pair_threshold', 0.05)),
        pair_margin=float(meta_args.get('pair_margin', 0.05)),
        mse_weight=float(meta_args.get('mse_weight', 0.2)),
        rank_weight=float(meta_args.get('rank_weight', 1.0)),
        max_pairs=int(meta_args.get('max_pairs', 4096)),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def build_tracker(device: torch.device) -> GPUNullspaceStraightTracker:
    xarm = xarm6_gpu.XArmLite6GPU(device=device)
    cc_model = SphereCollisionChecker('wrs/robot_sim/robots/xarmlite6_wg/xarm6_sphere_visuals.urdf')
    vmap_jax_cost = jax.jit(jax.vmap(cc_model.self_collision_cost, in_axes=(0, None, None)))
    collision_fn = jax2torch.jax2torch(lambda q_batch: vmap_jax_cost(q_batch, 1.0, -0.005))
    return GPUNullspaceStraightTracker(
        robot=xarm.robot,
        collision_fn=collision_fn,
        config=TrackerConfig(),
        print_every=0,
    )


def collect_random_rollout_points(num_samples: int, rollout_batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tracker = build_tracker(device)
    q_list = []
    cond_list = []
    length_list = []
    collected = 0

    while collected < num_samples:
        current_batch = max(1, min(int(rollout_batch_size), int(num_samples - collected)))
        q0_batch, direction_batch, target_normal_batch = tracker.sample_valid_batch(batch_size=current_batch, device=device)
        trajectories = tracker.collect_batch_trajectories(
            q0_batch=q0_batch,
            direction_batch=direction_batch,
            target_normal_batch=target_normal_batch,
        )
        for traj in trajectories:
            q_np = np.asarray(traj['q'], dtype=np.float32)
            pos_np = np.asarray(traj['tcp_pos'], dtype=np.float32)
            direction_np = np.asarray(traj['direction'], dtype=np.float32)
            normal_np = np.asarray(traj['target_normal'], dtype=np.float32)
            length_np = np.asarray(traj['remaining_length'], dtype=np.float32)
            cond_np = np.concatenate([
                pos_np,
                np.repeat(direction_np.reshape(1, 3), q_np.shape[0], axis=0),
                np.repeat(normal_np.reshape(1, 3), q_np.shape[0], axis=0),
            ], axis=1).astype(np.float32)
            remaining = num_samples - collected
            if remaining <= 0:
                break
            take = min(remaining, q_np.shape[0])
            q_list.append(torch.from_numpy(q_np[:take]))
            cond_list.append(torch.from_numpy(cond_np[:take]))
            length_list.append(torch.from_numpy(length_np[:take]))
            collected += take
            if collected >= num_samples:
                break

    return torch.cat(q_list, dim=0), torch.cat(cond_list, dim=0), torch.cat(length_list, dim=0)


def summarize(pred: np.ndarray, target: np.ndarray) -> dict:
    err = pred - target
    abs_err = np.abs(err)
    return {
        'mse': float(np.mean(err ** 2)),
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mae': float(np.mean(abs_err)),
        'max_abs_err': float(np.max(abs_err)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    q_all, cond_all, target_all = collect_random_rollout_points(args.num_samples, args.rollout_batch_size, device)
    lnet = load_lnet(args.lnet_ckpt, device)
    contrastive = load_contrastive(args.contrastive_ckpt, device)

    lnet_preds = []
    contrastive_preds = []
    target_np = target_all.numpy().astype(np.float32)

    with torch.no_grad():
        for start in range(0, q_all.shape[0], args.batch_size):
            end = min(start + args.batch_size, q_all.shape[0])
            q = q_all[start:end].to(device)
            cond = cond_all[start:end].to(device)
            lnet_pred = lnet(q, cond)
            _, contrastive_pred = contrastive(q, cond)
            lnet_preds.append(lnet_pred.detach().cpu())
            contrastive_preds.append(contrastive_pred.detach().cpu())

    lnet_np = torch.cat(lnet_preds, dim=0).numpy().astype(np.float32)
    contrastive_np = torch.cat(contrastive_preds, dim=0).numpy().astype(np.float32)

    lnet_stats = summarize(lnet_np, target_np)
    contrastive_stats = summarize(contrastive_np, target_np)
    contrastive_abs = np.abs(contrastive_np - target_np)
    lnet_abs = np.abs(lnet_np - target_np)
    better_mask = contrastive_abs < lnet_abs
    worse_mask = contrastive_abs > lnet_abs

    result = {
        'num_random_rollout_samples': int(target_np.shape[0]),
        'source': {
            'type': 'fresh_random_rollout',
            'seed': int(args.seed),
            'rollout_batch_size': int(args.rollout_batch_size),
        },
        'lnet': lnet_stats,
        'lnet_contrastive': contrastive_stats,
        'head_to_head': {
            'contrastive_better_count': int(np.sum(better_mask)),
            'lnet_better_count': int(np.sum(worse_mask)),
            'tie_count': int(target_np.shape[0] - np.sum(better_mask) - np.sum(worse_mask)),
            'mean_abs_err_gap': float(np.mean(contrastive_abs - lnet_abs)),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
