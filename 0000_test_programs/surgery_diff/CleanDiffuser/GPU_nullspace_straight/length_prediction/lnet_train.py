from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
try:
    import wandb
except ImportError:
    wandb = None

from lnet import LNet
from paths import DATASETS_DIR, LNET_RUNS_DIR
from wrs.robot_sim.robots.franka_research_3.franka_research_3 import FrankaResearch3
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller

BASE_DIR = Path(__file__).resolve().parent
GPU_NULLSPACE_DIR = BASE_DIR.parent
DIFFUSION_DIR = GPU_NULLSPACE_DIR / 'diffusion_inpainting'
TRAJECTORY_DIR = GPU_NULLSPACE_DIR / 'trajectory_generation'
if str(GPU_NULLSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(GPU_NULLSPACE_DIR))
if str(DIFFUSION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_DIR))
if str(TRAJECTORY_DIR) not in sys.path:
    sys.path.insert(0, str(TRAJECTORY_DIR))

DEFAULT_H5 = DATASETS_DIR / 'franka_research_3_gpu_trajectories_sub10_pref.hdf5'
DEFAULT_WORKDIR = LNET_RUNS_DIR
DEFAULT_RUN_NAME = 'lnet_eps_ranking_fr3_evolution'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train an FR3 LNet ranker with online hard-pair evolution.')
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--run-name', type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--steps-per-epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rank-margin', type=float, default=0.05)
    parser.add_argument('--length-threshold', type=float, default=0.02)
    parser.add_argument('--joint-noise-std', type=float, default=0.01)
    parser.add_argument('--clip-grad-norm', type=float, default=5.0)
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--boundary-weight-scale', type=float, default=2.0)
    parser.add_argument('--boundary-q4-margin-ratio', type=float, default=0.12)
    parser.add_argument('--max-val-pairs', type=int, default=16000)
    parser.add_argument('--buffer-capacity', type=int, default=50000)
    parser.add_argument('--buffer-mix-ratio', type=float, default=0.35)
    parser.add_argument('--evolve-every', type=int, default=1)
    parser.add_argument('--enable-self-evolve', action='store_true')
    parser.add_argument('--self-evolve-tasks', type=int, default=12)
    parser.add_argument('--self-evolve-candidates', type=int, default=32)
    parser.add_argument('--self-evolve-topk-hard', type=int, default=128)
    parser.add_argument('--self-evolve-oversample', type=int, default=256)
    parser.add_argument('--self-evolve-pos-tol-mm', type=float, default=2.0)
    parser.add_argument('--symmetry-augment-copies', type=int, default=1)
    parser.add_argument('--symmetry-yaw-deg', type=float, default=180.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--wandb-project', type=str, default='franka-lnet-ranking-evolution')
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
    return parser.parse_args()


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    denom = max(float(np.linalg.norm(vec)), 1e-12)
    return (vec / denom).astype(np.float32)


def rotate_about_z(vec: np.ndarray, yaw_rad: float) -> np.ndarray:
    c = float(math.cos(yaw_rad))
    s = float(math.sin(yaw_rad))
    rot = np.asarray([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return (rot @ np.asarray(vec, dtype=np.float32)).astype(np.float32)


def infer_robot_name(h5_path: Path) -> str:
    with h5py.File(h5_path, 'r') as f:
        return str(f.attrs.get('robot', 'franka_research_3')).lower()


def infer_q_limits(h5_path: Path) -> np.ndarray:
    robot_name = infer_robot_name(h5_path)
    if robot_name == 'franka_research_3':
        return FrankaResearch3(enable_cc=False).manipulator.jnt_ranges.astype(np.float32)
    if robot_name in {'xarm_lite6', 'xarmlite6', 'xarmlite6_miller'}:
        return XArmLite6Miller(enable_cc=False).jnt_ranges.astype(np.float32)
    raise ValueError(f'Unsupported robot type in HDF5: {robot_name}')


def compute_min_max(q: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([q, cond], dim=1)
    return x.min(dim=0).values, x.max(dim=0).values


def kendall_tau_from_pairs(score: np.ndarray, target: np.ndarray, max_pairs: int = 20000, rng: np.random.Generator | None = None) -> float:
    n = int(score.shape[0])
    if n < 2:
        return float('nan')
    i_idx, j_idx = np.triu_indices(n, k=1)
    if i_idx.size == 0:
        return float('nan')
    if i_idx.size > max_pairs:
        if rng is None:
            rng = np.random.default_rng(0)
        keep = rng.choice(i_idx.size, size=max_pairs, replace=False)
        i_idx = i_idx[keep]
        j_idx = j_idx[keep]
    d_score = score[i_idx] - score[j_idx]
    d_target = target[i_idx] - target[j_idx]
    valid = np.abs(d_target) > 1e-8
    if not np.any(valid):
        return float('nan')
    concordant = np.sign(d_score[valid] * d_target[valid])
    return float(np.mean(concordant))


@dataclass
class PairRecord:
    q_a: np.ndarray
    cond_a: np.ndarray
    len_a: float
    q_b: np.ndarray
    cond_b: np.ndarray
    len_b: float
    weight: float
    source: str
    inversion_gap: float = 0.0


class ContrastivePrefBaseDataset:
    def __init__(self, h5_path: Path):
        with h5py.File(h5_path, 'r') as f:
            if 'contrastive_pref' not in f:
                raise KeyError(f'Missing contrastive_pref group in {h5_path}.')
            pref = f['contrastive_pref']
            self.q = torch.from_numpy(np.asarray(pref['q'][:], dtype=np.float32))
            self.cond = torch.from_numpy(np.concatenate([
                np.asarray(pref['pos'][:], dtype=np.float32),
                np.asarray(pref['direction'][:], dtype=np.float32),
                np.asarray(pref['normal'][:], dtype=np.float32),
            ], axis=1))
            self.length = torch.from_numpy(np.asarray(pref['length'][:], dtype=np.float32))
            self.traj_idx = np.asarray(pref['traj_idx'][:], dtype=np.int32)
            self.point_idx = np.asarray(pref['point_idx'][:], dtype=np.int32)
            self.neighbor_offsets = np.asarray(pref['neighbor_offsets'][:], dtype=np.int64)
            self.neighbor_index = np.asarray(pref['neighbor_index'][:], dtype=np.int32)
            self.neighbor_score = np.asarray(pref['neighbor_score'][:], dtype=np.int32)
            self.valid_anchor_index = np.asarray(pref['valid_anchor_index'][:], dtype=np.int32)


def split_pref_dataset(base: ContrastivePrefBaseDataset, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_traj = np.unique(base.traj_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_traj)
    val_count = max(1, int(len(unique_traj) * val_ratio))
    val_traj = unique_traj[:val_count]
    train_traj = unique_traj[val_count:]
    if train_traj.size == 0:
        raise RuntimeError('No train trajectories left after split.')
    max_traj = int(unique_traj.max()) if unique_traj.size > 0 else -1
    train_traj_mask = np.zeros(max_traj + 1, dtype=bool)
    val_traj_mask = np.zeros(max_traj + 1, dtype=bool)
    train_traj_mask[train_traj] = True
    val_traj_mask[val_traj] = True
    train_sample_mask = train_traj_mask[base.traj_idx]
    val_sample_mask = val_traj_mask[base.traj_idx]
    if not np.any(train_sample_mask) or not np.any(val_sample_mask):
        raise RuntimeError('Train/val split is empty. Adjust val_ratio.')
    return train_sample_mask, val_sample_mask, train_traj, val_traj


class PrefPairSampler:
    def __init__(self, dataset: ContrastivePrefBaseDataset, allowed_sample_mask: np.ndarray, length_threshold: float, seed: int):
        self.dataset = dataset
        self.allowed_sample_mask = np.asarray(allowed_sample_mask, dtype=bool)
        self.length_threshold = float(length_threshold)
        self.rng = np.random.default_rng(seed)
        self.anchor_indices = self._select_anchor_indices()
        if self.anchor_indices.size == 0:
            raise RuntimeError('No valid preference anchors are available for this split.')

    def _select_anchor_indices(self) -> np.ndarray:
        kept = []
        for anchor in self.dataset.valid_anchor_index:
            anchor = int(anchor)
            if not self.allowed_sample_mask[anchor]:
                continue
            start = int(self.dataset.neighbor_offsets[anchor])
            end = int(self.dataset.neighbor_offsets[anchor + 1])
            neigh = self.dataset.neighbor_index[start:end]
            if neigh.size == 0:
                continue
            mask = self.allowed_sample_mask[neigh]
            if np.any(mask):
                kept.append(anchor)
        return np.asarray(kept, dtype=np.int32)

    def _sample_partner(self, anchor: int) -> int:
        start = int(self.dataset.neighbor_offsets[anchor])
        end = int(self.dataset.neighbor_offsets[anchor + 1])
        neigh = self.dataset.neighbor_index[start:end]
        score = self.dataset.neighbor_score[start:end]
        mask = self.allowed_sample_mask[neigh]
        neigh = neigh[mask]
        score = score[mask]
        if neigh.size == 0:
            return anchor
        weight = score.astype(np.float64)
        weight = weight - weight.min() + 1.0
        weight_sum = float(weight.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            return int(self.rng.choice(neigh))
        return int(self.rng.choice(neigh, p=weight / weight_sum))

    def _make_pair(self, a_idx: int, b_idx: int, q_limits: np.ndarray, boundary_scale: float, boundary_margin_ratio: float, source: str) -> PairRecord | None:
        len_a = float(self.dataset.length[a_idx].item())
        len_b = float(self.dataset.length[b_idx].item())
        if abs(len_a - len_b) <= self.length_threshold:
            return None
        q_a = self.dataset.q[a_idx].cpu().numpy().astype(np.float32)
        q_b = self.dataset.q[b_idx].cpu().numpy().astype(np.float32)
        cond_a = self.dataset.cond[a_idx].cpu().numpy().astype(np.float32)
        cond_b = self.dataset.cond[b_idx].cpu().numpy().astype(np.float32)
        weight = max(
            compute_boundary_weight(q_a, q_limits, boundary_scale, boundary_margin_ratio),
            compute_boundary_weight(q_b, q_limits, boundary_scale, boundary_margin_ratio),
        )
        if len_a >= len_b:
            return PairRecord(q_a, cond_a, len_a, q_b, cond_b, len_b, weight, source=source)
        return PairRecord(q_b, cond_b, len_b, q_a, cond_a, len_a, weight, source=source)

    def sample_pairs(self, batch_size: int, q_limits: np.ndarray, boundary_scale: float, boundary_margin_ratio: float) -> dict[str, torch.Tensor]:
        pairs: list[PairRecord] = []
        tries = 0
        max_tries = max(8 * batch_size, 128)
        while len(pairs) < batch_size and tries < max_tries:
            a_idx = int(self.rng.choice(self.anchor_indices))
            b_idx = self._sample_partner(a_idx)
            pair = self._make_pair(a_idx, b_idx, q_limits, boundary_scale, boundary_margin_ratio, source='dataset_pref')
            if pair is not None:
                pairs.append(pair)
            tries += 1
        if not pairs:
            raise RuntimeError('Failed to sample valid preference pairs from the base dataset.')
        return collate_pair_records(pairs)

    def build_eval_pairs(self, max_pairs: int, q_limits: np.ndarray, boundary_scale: float, boundary_margin_ratio: float) -> list[PairRecord]:
        pairs: list[PairRecord] = []
        for anchor in self.anchor_indices:
            start = int(self.dataset.neighbor_offsets[anchor])
            end = int(self.dataset.neighbor_offsets[anchor + 1])
            neigh = self.dataset.neighbor_index[start:end]
            score = self.dataset.neighbor_score[start:end]
            mask = self.allowed_sample_mask[neigh]
            neigh = neigh[mask]
            score = score[mask]
            if neigh.size == 0:
                continue
            order = np.argsort(-score, kind='stable')
            for partner in neigh[order]:
                pair = self._make_pair(int(anchor), int(partner), q_limits, boundary_scale, boundary_margin_ratio, source='val_pref')
                if pair is not None:
                    pairs.append(pair)
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break
        return pairs


class RankingExperienceDataset:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.records: list[PairRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def append(self, pairs: list[PairRecord]) -> int:
        if not pairs:
            return 0
        self.records.extend(pairs)
        if len(self.records) > self.capacity:
            keep = self.records[-self.capacity :]
            self.records = list(keep)
        return len(pairs)

    def sample(self, count: int) -> dict[str, torch.Tensor]:
        if len(self.records) == 0:
            raise RuntimeError('Experience buffer is empty.')
        idx = self.rng.integers(0, len(self.records), size=max(1, int(count)))
        pairs = [self.records[int(i)] for i in idx.tolist()]
        return collate_pair_records(pairs)


def compute_boundary_weight(q: np.ndarray, q_limits: np.ndarray, scale: float, margin_ratio: float) -> float:
    q = np.asarray(q, dtype=np.float32)
    q_limits = np.asarray(q_limits, dtype=np.float32)
    q4_idx = min(3, q.shape[0] - 1)
    span = max(float(q_limits[q4_idx, 1] - q_limits[q4_idx, 0]), 1e-6)
    dist_limit = min(float(q[q4_idx] - q_limits[q4_idx, 0]), float(q_limits[q4_idx, 1] - q[q4_idx]))
    cliff = max(0.0, 1.0 - dist_limit / max(span * float(margin_ratio), 1e-6))
    center = 0.5 * (q_limits[:, 0] + q_limits[:, 1])
    radius = 0.5 * np.maximum(q_limits[:, 1] - q_limits[:, 0], 1e-6)
    singular_proxy = float(np.exp(-np.linalg.norm((q - center) / radius)))
    return float(1.0 + scale * max(cliff, singular_proxy))


def collate_pair_records(pairs: list[PairRecord]) -> dict[str, torch.Tensor]:
    return {
        'q_a': torch.from_numpy(np.stack([p.q_a for p in pairs], axis=0).astype(np.float32)),
        'cond_a': torch.from_numpy(np.stack([p.cond_a for p in pairs], axis=0).astype(np.float32)),
        'len_a': torch.from_numpy(np.asarray([p.len_a for p in pairs], dtype=np.float32)),
        'q_b': torch.from_numpy(np.stack([p.q_b for p in pairs], axis=0).astype(np.float32)),
        'cond_b': torch.from_numpy(np.stack([p.cond_b for p in pairs], axis=0).astype(np.float32)),
        'len_b': torch.from_numpy(np.asarray([p.len_b for p in pairs], dtype=np.float32)),
        'weight': torch.from_numpy(np.asarray([p.weight for p in pairs], dtype=np.float32)),
    }


class FrankaRolloutExplorer:
    def __init__(self, device: torch.device, args: argparse.Namespace):
        self.device = device
        self.args = args
        self._loaded = False

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        from length_prediction.lnet_contrastive_fr3_rotation_cone_eval import (
            build_tracker,
            collect_candidate_qs,
            rollout_same_task,
        )

        self.collect_candidate_qs = collect_candidate_qs
        self.rollout_same_task = rollout_same_task
        self.tracker, self.tracker_device = build_tracker(self.device)
        self._loaded = True

    def sample_random_task(self) -> dict[str, np.ndarray]:
        self._lazy_load()
        q_batch, direction_batch, normal_batch = self.tracker.sample_valid_batch(batch_size=1, device=self.tracker_device)
        pos_batch, _ = self.tracker.robot.fk_batch(q_batch)
        return {
            'pos': pos_batch[0].detach().cpu().numpy().astype(np.float32),
            'direction': normalize(direction_batch[0].detach().cpu().numpy().astype(np.float32)),
            'normal': normalize(normal_batch[0].detach().cpu().numpy().astype(np.float32)),
        }

    def sample_candidates(self, task: dict[str, np.ndarray], num_candidates: int) -> np.ndarray:
        self._lazy_load()
        q_np, _ = self.collect_candidate_qs(
            self.tracker,
            self.tracker_device,
            task['pos'],
            task['direction'],
            task['normal'],
            int(num_candidates),
            int(self.args.self_evolve_oversample),
            float(self.args.self_evolve_pos_tol_mm),
            int(self.args.correction_iters),
            float(self.args.correction_tol),
            float(self.args.correction_damping),
        )
        return q_np.astype(np.float32)

    def rollout_lengths(self, q_batch_np: np.ndarray, task: dict[str, np.ndarray]) -> np.ndarray:
        self._lazy_load()
        return self.rollout_same_task(self.tracker, self.tracker_device, q_batch_np, task['direction'], task['normal']).astype(np.float32)


class LNetRankingTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        self.run_dir = DEFAULT_WORKDIR / args.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.q_limits_np = infer_q_limits(args.h5_path)
        self.q_limits = torch.from_numpy(self.q_limits_np)

        self.dataset = ContrastivePrefBaseDataset(args.h5_path)
        self.train_mask, self.val_mask, self.train_traj, self.val_traj = split_pref_dataset(self.dataset, args.val_ratio, args.seed)
        train_q = self.dataset.q[self.train_mask]
        train_cond = self.dataset.cond[self.train_mask]
        in_min, in_max = compute_min_max(train_q, train_cond)
        self.model = LNet(
            q_min=self.q_limits[:, 0],
            q_max=self.q_limits[:, 1],
            in_min=in_min,
            in_max=in_max,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.rank_loss_fn = torch.nn.MarginRankingLoss(margin=args.rank_margin, reduction='none')
        self.train_sampler = PrefPairSampler(self.dataset, self.train_mask, args.length_threshold, args.seed)
        self.val_sampler = PrefPairSampler(self.dataset, self.val_mask, args.length_threshold, args.seed + 1)
        self.val_pairs = self.val_sampler.build_eval_pairs(
            args.max_val_pairs,
            self.q_limits_np,
            args.boundary_weight_scale,
            args.boundary_q4_margin_ratio,
        )
        self.buffer = RankingExperienceDataset(args.buffer_capacity, args.seed + 99)
        self.rng = np.random.default_rng(args.seed)
        self.global_step = 0
        self.best_pair_acc = -float('inf')
        self.best_tau = -float('inf')
        self.use_wandb = args.wandb_mode != 'disabled' and wandb is not None
        self.explorer = None
        if args.enable_self_evolve:
            self.explorer = FrankaRolloutExplorer(self.device, args)
        if self.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or args.run_name,
                mode=args.wandb_mode,
                config=vars(args),
                dir=str(self.run_dir),
            )

    def mix_batch(self) -> dict[str, torch.Tensor]:
        use_buffer = len(self.buffer) > 0 and self.rng.random() < float(self.args.buffer_mix_ratio)
        if use_buffer:
            n_buffer = max(1, int(round(self.args.batch_size * self.args.buffer_mix_ratio)))
            n_base = max(1, self.args.batch_size - n_buffer)
            base = self.train_sampler.sample_pairs(
                n_base,
                self.q_limits_np,
                self.args.boundary_weight_scale,
                self.args.boundary_q4_margin_ratio,
            )
            replay = self.buffer.sample(n_buffer)
            return {
                k: torch.cat([base[k], replay[k]], dim=0)
                for k in base.keys()
            }
        return self.train_sampler.sample_pairs(
            self.args.batch_size,
            self.q_limits_np,
            self.args.boundary_weight_scale,
            self.args.boundary_q4_margin_ratio,
        )

    def prepare_batch(self, batch: dict[str, torch.Tensor], training: bool) -> dict[str, torch.Tensor]:
        out = {}
        for key, value in batch.items():
            out[key] = value.to(self.device)
        if training and self.args.joint_noise_std > 0.0:
            q_min = self.model.q_min
            q_max = self.model.q_max
            for key in ('q_a', 'q_b'):
                noise = torch.randn_like(out[key]) * float(self.args.joint_noise_std)
                out[key] = torch.max(torch.min(out[key] + noise, q_max), q_min)
        return out

    def compute_pair_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        score_a = self.model(batch['q_a'], batch['cond_a'])
        score_b = self.model(batch['q_b'], batch['cond_b'])
        target = torch.ones_like(score_a)
        raw_rank = self.rank_loss_fn(score_a, score_b, target)
        gap = batch['len_a'] - batch['len_b']
        valid = gap > float(self.args.length_threshold)
        weight = batch['weight']
        if valid.any():
            loss = (raw_rank[valid] * weight[valid]).sum() / weight[valid].sum().clamp_min(1e-6)
        else:
            loss = raw_rank.mean() * 0.0
        acc = (score_a > score_b).float()
        if valid.any():
            pair_acc = float(acc[valid].mean().detach())
            score_gap = float((score_a[valid] - score_b[valid]).mean().detach())
            mean_weight = float(weight[valid].mean().detach())
            pair_count = int(valid.sum().item())
        else:
            pair_acc = float('nan')
            score_gap = 0.0
            mean_weight = 0.0
            pair_count = 0
        return loss, {
            'rank_loss': float(loss.detach()),
            'pair_acc': pair_acc,
            'score_gap': score_gap,
            'pair_weight': mean_weight,
            'pair_count': pair_count,
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        if not self.val_pairs:
            return {
                'pair_acc': float('nan'),
                'rank_loss': float('nan'),
                'kendall_tau': float('nan'),
                'score_mean': float('nan'),
                'score_std': float('nan'),
            }
        losses = []
        pair_accs = []
        score_pool = []
        target_pool = []
        chunk = 512
        for start in range(0, len(self.val_pairs), chunk):
            part = collate_pair_records(self.val_pairs[start:start + chunk])
            batch = self.prepare_batch(part, training=False)
            score_a = self.model(batch['q_a'], batch['cond_a'])
            score_b = self.model(batch['q_b'], batch['cond_b'])
            target = torch.ones_like(score_a)
            rank = self.rank_loss_fn(score_a, score_b, target)
            valid = (batch['len_a'] - batch['len_b']) > float(self.args.length_threshold)
            if valid.any():
                weighted = (rank[valid] * batch['weight'][valid]).sum() / batch['weight'][valid].sum().clamp_min(1e-6)
                acc = float((score_a[valid] > score_b[valid]).float().mean().detach())
                losses.append(float(weighted.detach()))
                pair_accs.append(acc)
            score_pool.append(score_a.detach().cpu().numpy())
            score_pool.append(score_b.detach().cpu().numpy())
            target_pool.append(batch['len_a'].detach().cpu().numpy())
            target_pool.append(batch['len_b'].detach().cpu().numpy())
        score_all = np.concatenate(score_pool, axis=0).astype(np.float32)
        target_all = np.concatenate(target_pool, axis=0).astype(np.float32)
        tau = kendall_tau_from_pairs(score_all, target_all, max_pairs=self.args.max_val_pairs, rng=self.rng)
        return {
            'pair_acc': float(np.mean(pair_accs)) if pair_accs else float('nan'),
            'rank_loss': float(np.mean(losses)) if losses else float('nan'),
            'kendall_tau': float(tau),
            'score_mean': float(score_all.mean()),
            'score_std': float(score_all.std()),
        }

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        losses = []
        accs = []
        pair_counts = []
        for local_step in range(1, self.args.steps_per_epoch + 1):
            batch = self.prepare_batch(self.mix_batch(), training=True)
            loss, aux = self.compute_pair_loss(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.args.clip_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.args.clip_grad_norm))
            self.optimizer.step()
            self.global_step += 1
            losses.append(float(loss.detach()))
            if np.isfinite(aux['pair_acc']):
                accs.append(aux['pair_acc'])
            pair_counts.append(aux['pair_count'])
            if self.use_wandb:
                wandb.log({
                    'train/rank_loss': float(loss.detach()),
                    'train/ranking_accuracy': aux['pair_acc'],
                    'train/mean_pair_weight': aux['pair_weight'],
                    'train/score_gap': aux['score_gap'],
                    'train/pair_count': aux['pair_count'],
                    'train/buffer_size': len(self.buffer),
                    'epoch': epoch,
                    'step': self.global_step,
                }, step=self.global_step)
            if self.args.print_every > 0 and local_step % self.args.print_every == 0:
                mean_loss = float(np.mean(losses[-self.args.print_every :]))
                mean_acc = float(np.mean(accs[-self.args.print_every :])) if accs else float('nan')
                print(
                    f'[train] epoch={epoch:03d} step={self.global_step:06d} '
                    f'loss={mean_loss:.6f} rank_acc={mean_acc:.4f} '
                    f'buffer={len(self.buffer)} pairs={int(np.mean(pair_counts[-self.args.print_every:]))}'
                )
        return {
            'rank_loss': float(np.mean(losses)) if losses else float('nan'),
            'pair_acc': float(np.mean(accs)) if accs else float('nan'),
            'pair_count': float(np.mean(pair_counts)) if pair_counts else 0.0,
        }

    def make_symmetry_augmented_pairs(self, task: dict[str, np.ndarray], better_q: np.ndarray, better_len: float, worse_q: np.ndarray, worse_len: float, base_weight: float, inversion_gap: float) -> list[PairRecord]:
        cond = np.concatenate([task['pos'], task['direction'], task['normal']], axis=0).astype(np.float32)
        pairs = [
            PairRecord(
                q_a=better_q.astype(np.float32),
                cond_a=cond,
                len_a=float(better_len),
                q_b=worse_q.astype(np.float32),
                cond_b=cond,
                len_b=float(worse_len),
                weight=float(base_weight),
                source='rollout_hard_pair',
                inversion_gap=float(inversion_gap),
            )
        ]
        copies = max(0, int(self.args.symmetry_augment_copies))
        if copies <= 0:
            return pairs
        yaw = math.radians(float(self.args.symmetry_yaw_deg))
        q_min = self.q_limits_np[:, 0]
        q_max = self.q_limits_np[:, 1]
        for sign in (-1.0, 1.0):
            if len(pairs) - 1 >= copies:
                break
            yaw_step = sign * yaw
            aug_q_a = better_q.copy().astype(np.float32)
            aug_q_b = worse_q.copy().astype(np.float32)
            aug_q_a[0] = np.clip(aug_q_a[0] + yaw_step, q_min[0], q_max[0])
            aug_q_b[0] = np.clip(aug_q_b[0] + yaw_step, q_min[0], q_max[0])
            aug_task = {
                'pos': rotate_about_z(task['pos'], yaw_step),
                'direction': normalize(rotate_about_z(task['direction'], yaw_step)),
                'normal': normalize(rotate_about_z(task['normal'], yaw_step)),
            }
            aug_cond = np.concatenate([aug_task['pos'], aug_task['direction'], aug_task['normal']], axis=0).astype(np.float32)
            pairs.append(
                PairRecord(
                    q_a=aug_q_a,
                    cond_a=aug_cond,
                    len_a=float(better_len),
                    q_b=aug_q_b,
                    cond_b=aug_cond,
                    len_b=float(worse_len),
                    weight=float(base_weight),
                    source='symmetry_augmented_hard_pair',
                    inversion_gap=float(inversion_gap),
                )
            )
        return pairs

    def self_evolve(self, epoch: int) -> dict[str, float]:
        if self.explorer is None:
            return {
                'tasks': 0.0,
                'candidates': 0.0,
                'hard_pairs_added': 0.0,
                'mean_inversion_gap': 0.0,
                'mean_rollout_length': 0.0,
            }
        self.model.eval()
        hard_pairs: list[PairRecord] = []
        inversion_gaps = []
        mean_rollout_lengths = []
        total_candidates = 0
        for _ in range(int(self.args.self_evolve_tasks)):
            task = self.explorer.sample_random_task()
            q_candidates = self.explorer.sample_candidates(
                task,
                int(self.args.self_evolve_candidates),
            )
            cond_batch = np.repeat(
                np.concatenate([task['pos'], task['direction'], task['normal']], axis=0)[None, :].astype(np.float32),
                q_candidates.shape[0],
                axis=0,
            )
            q_t = torch.from_numpy(q_candidates).to(self.device)
            cond_t = torch.from_numpy(cond_batch).to(self.device)
            with torch.no_grad():
                pred_score = self.model(q_t, cond_t).detach().cpu().numpy().astype(np.float32)
            real_len = self.explorer.rollout_lengths(q_candidates, task)
            total_candidates += int(q_candidates.shape[0])
            mean_rollout_lengths.append(float(real_len.mean()))

            pred_order = np.argsort(-pred_score, kind='stable')
            real_order = np.argsort(-real_len, kind='stable')
            pred_rank = np.empty_like(pred_order)
            real_rank = np.empty_like(real_order)
            pred_rank[pred_order] = np.arange(pred_order.shape[0])
            real_rank[real_order] = np.arange(real_order.shape[0])

            local_pairs: list[tuple[float, int, int]] = []
            for i in range(q_candidates.shape[0]):
                for j in range(i + 1, q_candidates.shape[0]):
                    real_gap = float(real_len[i] - real_len[j])
                    if abs(real_gap) <= float(self.args.length_threshold):
                        continue
                    pred_gap = float(pred_score[i] - pred_score[j])
                    inversion = real_gap * pred_gap < 0.0
                    if not inversion:
                        continue
                    severity = abs(real_gap) + abs(pred_gap) + abs(float(pred_rank[i] - pred_rank[j]) - float(real_rank[i] - real_rank[j]))
                    local_pairs.append((severity, i, j))
            local_pairs.sort(key=lambda x: x[0], reverse=True)
            for severity, i, j in local_pairs[: int(self.args.self_evolve_topk_hard)]:
                if real_len[i] >= real_len[j]:
                    better_i, worse_i = i, j
                else:
                    better_i, worse_i = j, i
                weight = max(
                    compute_boundary_weight(q_candidates[better_i], self.q_limits_np, self.args.boundary_weight_scale, self.args.boundary_q4_margin_ratio),
                    compute_boundary_weight(q_candidates[worse_i], self.q_limits_np, self.args.boundary_weight_scale, self.args.boundary_q4_margin_ratio),
                ) * (1.0 + abs(float(real_len[better_i] - real_len[worse_i])))
                pair_records = self.make_symmetry_augmented_pairs(
                    task,
                    q_candidates[better_i],
                    float(real_len[better_i]),
                    q_candidates[worse_i],
                    float(real_len[worse_i]),
                    weight,
                    float(severity),
                )
                hard_pairs.extend(pair_records)
                inversion_gaps.append(float(severity))

        added = self.buffer.append(hard_pairs)
        summary = {
            'tasks': float(self.args.self_evolve_tasks),
            'candidates': float(total_candidates),
            'hard_pairs_added': float(added),
            'mean_inversion_gap': float(np.mean(inversion_gaps)) if inversion_gaps else 0.0,
            'mean_rollout_length': float(np.mean(mean_rollout_lengths)) if mean_rollout_lengths else 0.0,
        }
        print(
            f'[evolve] epoch={epoch:03d} tasks={int(summary["tasks"])} candidates={int(summary["candidates"])} '
            f'hard_pairs_added={int(summary["hard_pairs_added"])} '
            f'mean_inversion_gap={summary["mean_inversion_gap"]:.4f} buffer={len(self.buffer)}'
        )
        if self.use_wandb:
            wandb.log({
                'evolve/tasks': summary['tasks'],
                'evolve/candidates': summary['candidates'],
                'evolve/hard_pairs_added': summary['hard_pairs_added'],
                'evolve/mean_inversion_gap': summary['mean_inversion_gap'],
                'evolve/mean_rollout_length': summary['mean_rollout_length'],
                'evolve/buffer_size': len(self.buffer),
                'epoch': epoch,
                'step': self.global_step,
            }, step=self.global_step)
        return summary

    def save_checkpoint(self, epoch: int, eval_metrics: dict[str, float]) -> None:
        state = {
            'model': self.model.state_dict(),
            'q_min': self.q_limits[:, 0],
            'q_max': self.q_limits[:, 1],
            'in_min': self.model.in_min.detach().cpu(),
            'in_max': self.model.in_max.detach().cpu(),
            'args': vars(self.args),
            'epoch': epoch,
            'global_step': self.global_step,
            'best_pair_acc': self.best_pair_acc,
            'best_tau': self.best_tau,
            'eval_metrics': eval_metrics,
            'buffer_size': len(self.buffer),
        }
        torch.save(state, self.run_dir / 'lnet_latest.pt')
        improved = False
        if np.isfinite(eval_metrics['pair_acc']) and eval_metrics['pair_acc'] >= self.best_pair_acc:
            self.best_pair_acc = eval_metrics['pair_acc']
            improved = True
        if np.isfinite(eval_metrics['kendall_tau']) and eval_metrics['kendall_tau'] >= self.best_tau:
            self.best_tau = eval_metrics['kendall_tau']
            improved = True
        if improved:
            state['best_pair_acc'] = self.best_pair_acc
            state['best_tau'] = self.best_tau
            torch.save(state, self.run_dir / 'lnet_best.pt')
            print(f'[eval] new_best pair_acc={self.best_pair_acc:.4f} kendall_tau={self.best_tau:.4f}')

    def fit(self) -> None:
        history = []
        for epoch in range(1, self.args.epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            eval_metrics = self.evaluate()
            print(
                f'[eval] epoch={epoch:03d} val_rank_loss={eval_metrics["rank_loss"]:.6f} '
                f'val_ranking_accuracy={eval_metrics["pair_acc"]:.4f} '
                f'val_kendall_tau={eval_metrics["kendall_tau"]:.4f}'
            )
            if self.use_wandb:
                wandb.log({
                    'eval/rank_loss': eval_metrics['rank_loss'],
                    'eval/ranking_accuracy': eval_metrics['pair_acc'],
                    'eval/kendall_tau': eval_metrics['kendall_tau'],
                    'eval/score_mean': eval_metrics['score_mean'],
                    'eval/score_std': eval_metrics['score_std'],
                    'epoch': epoch,
                    'step': self.global_step,
                }, step=self.global_step)
            evolve_metrics = None
            if self.args.enable_self_evolve and epoch % max(1, int(self.args.evolve_every)) == 0:
                evolve_metrics = self.self_evolve(epoch)
            self.save_checkpoint(epoch, eval_metrics)
            history.append({
                'epoch': epoch,
                'train': train_metrics,
                'eval': eval_metrics,
                'evolve': evolve_metrics,
                'buffer_size': len(self.buffer),
            })

        metadata = {
            'args': to_jsonable(vars(self.args)),
            'num_train_points': int(np.sum(self.train_mask)),
            'num_val_points': int(np.sum(self.val_mask)),
            'num_train_trajectories': int(self.train_traj.shape[0]),
            'num_val_trajectories': int(self.val_traj.shape[0]),
            'num_train_anchors': int(self.train_sampler.anchor_indices.shape[0]),
            'num_val_anchors': int(self.val_sampler.anchor_indices.shape[0]),
            'num_val_pairs': int(len(self.val_pairs)),
            'final_buffer_size': int(len(self.buffer)),
            'best_pair_acc': self.best_pair_acc,
            'best_tau': self.best_tau,
            'history': to_jsonable(history),
        }
        (self.run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        if self.use_wandb:
            wandb.finish()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not args.h5_path.exists():
        raise FileNotFoundError(f'HDF5 dataset not found: {args.h5_path}')
    trainer = LNetRankingTrainer(args)
    trainer.fit()


if __name__ == '__main__':
    main()
