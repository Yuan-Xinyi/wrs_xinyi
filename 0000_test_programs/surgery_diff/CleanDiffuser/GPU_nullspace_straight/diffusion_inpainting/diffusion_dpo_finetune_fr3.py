from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset

try:
    import wandb
except ImportError:
    wandb = None

BASE_DIR = Path(__file__).resolve().parent
GPU_NULLSPACE_DIR = BASE_DIR.parent
if str(GPU_NULLSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(GPU_NULLSPACE_DIR))

from diffusion import (  # noqa: E402
    DEFAULT_WORKDIR,
    FeatureLayout,
    build_inpainting_x,
    save_bundle,
    set_seed,
)
from diffusion import create_model  # noqa: E402
from trajectory_generation.fr3_nullspace_straight import FrankaResearch3GPU  # noqa: E402

DATASETS_DIR = GPU_NULLSPACE_DIR / 'datasets'
RUNS_DIR = GPU_NULLSPACE_DIR / 'runs'
DEFAULT_PREF_H5 = DATASETS_DIR / 'franka_research_3_gpu_trajectories_sub10_pref.hdf5'
DEFAULT_REF_BUNDLE = RUNS_DIR / 'dit_kinematic_inpainting_runs' / 'ddpm32_dit_inpaint_qL_from_posdirnormal_fr3_sub10' / 'bundle_latest.pt'
DEFAULT_RUN_NAME = 'ddpm32_dit_inpaint_qL_from_posdirnormal_fr3_sub10_dpo_pref_cross_cond'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Sample-level cross-condition DPO fine-tuning for the FR3 inpainting diffusion model using the preference-indexed dataset.')
    parser.add_argument('--pref-h5', type=Path, default=DEFAULT_PREF_H5)
    parser.add_argument('--ref-bundle', type=Path, default=DEFAULT_REF_BUNDLE)
    parser.add_argument('--workdir', type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument('--run-name', type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=20260418)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--val-ratio', type=float, default=0.02)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dpo-beta', type=float, default=1.0)
    parser.add_argument('--recon-weight', type=float, default=0.1, help='Weight for the original diffusion reconstruction objective.')
    parser.add_argument('--consistency-weight', type=float, default=0.1, help='Global weight for FK-based condition consistency regularization.')
    parser.add_argument('--consistency-pos-weight', type=float, default=1.0, help='Relative weight of TCP position consistency.')
    parser.add_argument('--consistency-dir-weight', type=float, default=0.25, help='Relative weight of TCP x-axis alignment with the target direction.')
    parser.add_argument('--consistency-normal-weight', type=float, default=0.25, help='Relative weight of TCP z-axis alignment with the target normal.')
    parser.add_argument('--max-pairs-per-epoch', type=int, default=None)
    parser.add_argument('--print-every', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--wandb-project', type=str, default='franka-diffusion-dpo')
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--wandb-mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    return parser.parse_args()


def to_jsonable(obj):
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
    return obj


def load_bundle_model(bundle_path: Path, device: torch.device):
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    stats = {
        k: np.asarray(v, dtype=np.float32) if isinstance(v, (list, tuple, np.ndarray)) else v
        for k, v in bundle['stats'].items()
    }
    q_dim = int(stats['q_dim'])
    x_min = np.asarray(bundle['x_min'], dtype=np.float32)
    x_max = np.asarray(bundle['x_max'], dtype=np.float32)
    bundle_args = bundle.get('args', {})
    diffusion_steps = int(bundle_args.get('diffusion_steps', 32))
    model = create_model(device=device, x_min=x_min, x_max=x_max, diffusion_steps=diffusion_steps, q_dim=q_dim)
    model_path = bundle_path.with_name('model_best.pt')
    if not model_path.exists():
        model_path = bundle_path.with_name('model_latest.pt')
    model.load(str(model_path))
    return bundle, stats, model, x_min, x_max


class PreferenceTokenDataset:
    def __init__(self, pref_h5: Path, stats: dict):
        with h5py.File(pref_h5, 'r') as f:
            if 'contrastive_pref' not in f:
                raise KeyError(f'Missing contrastive_pref group in {pref_h5}.')
            pref = f['contrastive_pref']
            self.q = np.asarray(pref['q'][:], dtype=np.float32)
            self.pos = np.asarray(pref['pos'][:], dtype=np.float32)
            self.direction = np.asarray(pref['direction'][:], dtype=np.float32)
            self.normal = np.asarray(pref['normal'][:], dtype=np.float32)
            self.length = np.asarray(pref['length'][:], dtype=np.float32)
            self.traj_idx = np.asarray(pref['traj_idx'][:], dtype=np.int32)
            self.point_idx = np.asarray(pref['point_idx'][:], dtype=np.int32)
            self.neighbor_offsets = np.asarray(pref['neighbor_offsets'][:], dtype=np.int64)
            self.neighbor_index = np.asarray(pref['neighbor_index'][:], dtype=np.int32)
            self.neighbor_score = np.asarray(pref['neighbor_score'][:], dtype=np.int32)
            self.valid_anchor_index = np.asarray(pref['valid_anchor_index'][:], dtype=np.int32)

        # Each token is constructed under its own sample condition
        # (q, pos, direction, normal, length). This is the key ingredient
        # for cross-condition DPO: winner and loser are never forced to
        # share a single anchor condition.
        raw_tokens = np.concatenate(
            [
                self.q,
                self.pos,
                self.direction,
                self.normal,
                self.length.reshape(-1, 1),
            ],
            axis=1,
        ).astype(np.float32)
        layout = FeatureLayout(q_dim=int(stats['q_dim']))
        self.x = build_inpainting_x(raw_tokens, stats, layout).astype(np.float32)
        self.token_dim = int(self.x.shape[1])


class PreferencePairDataset(Dataset):
    def __init__(self, base: PreferenceTokenDataset, anchor_indices: np.ndarray, allowed_mask: np.ndarray, seed: int):
        self.base = base
        self.anchor_indices = np.asarray(anchor_indices, dtype=np.int32)
        self.allowed_mask = np.asarray(allowed_mask, dtype=bool)
        self.seed = int(seed)

    def __len__(self) -> int:
        return int(self.anchor_indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + int(idx))
        anchor = int(self.anchor_indices[idx])
        start = int(self.base.neighbor_offsets[anchor])
        end = int(self.base.neighbor_offsets[anchor + 1])
        neigh = self.base.neighbor_index[start:end]
        score = self.base.neighbor_score[start:end]
        mask = self.allowed_mask[neigh]
        neigh = neigh[mask]
        score = score[mask]
        if neigh.size == 0:
            partner = anchor
        else:
            weight = score.astype(np.float64)
            weight = weight - weight.min() + 1.0
            weight_sum = float(weight.sum())
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                partner = int(rng.choice(neigh))
            else:
                partner = int(rng.choice(neigh, p=weight / weight_sum))

        len_anchor = float(self.base.length[anchor])
        len_partner = float(self.base.length[partner])
        if len_anchor >= len_partner:
            winner, loser = anchor, partner
        else:
            winner, loser = partner, anchor
        # x_w and x_l each already encode their own task condition, so the
        # downstream DPO loss is evaluated in a cross-condition manner.
        x_w = torch.from_numpy(self.base.x[winner]).float().unsqueeze(0)
        x_l = torch.from_numpy(self.base.x[loser]).float().unsqueeze(0)
        return x_w, x_l


class PairBatchSampler(BatchSampler):
    def __init__(self, dataset_len: int, batch_size: int, seed: int, max_batches: int | None = None):
        self.dataset_len = int(dataset_len)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.max_batches = None if max_batches is None else int(max_batches)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.arange(self.dataset_len, dtype=np.int64)
        rng.shuffle(order)
        num_batches = 0
        for start in range(0, self.dataset_len, self.batch_size):
            if self.max_batches is not None and num_batches >= self.max_batches:
                break
            num_batches += 1
            yield order[start:start + self.batch_size].tolist()

    def __len__(self) -> int:
        raw = int(np.ceil(self.dataset_len / self.batch_size))
        if self.max_batches is None:
            return raw
        return min(raw, self.max_batches)


def build_split_datasets(base: PreferenceTokenDataset, val_ratio: float, seed: int) -> tuple[PreferencePairDataset, PreferencePairDataset, dict]:
    unique_traj = np.unique(base.traj_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_traj)
    val_traj_count = max(1, int(len(unique_traj) * val_ratio))
    val_traj = unique_traj[:val_traj_count]
    train_traj = unique_traj[val_traj_count:]
    if train_traj.size == 0:
        raise RuntimeError('No training trajectories left after split.')

    max_traj = int(unique_traj.max()) if unique_traj.size > 0 else -1
    train_traj_mask = np.zeros(max_traj + 1, dtype=bool)
    val_traj_mask = np.zeros(max_traj + 1, dtype=bool)
    train_traj_mask[train_traj] = True
    val_traj_mask[val_traj] = True
    train_sample_mask = train_traj_mask[base.traj_idx]
    val_sample_mask = val_traj_mask[base.traj_idx]

    def select_anchor_indices(sample_mask: np.ndarray) -> np.ndarray:
        kept = []
        for anchor in base.valid_anchor_index:
            anchor = int(anchor)
            if not sample_mask[anchor]:
                continue
            start = int(base.neighbor_offsets[anchor])
            end = int(base.neighbor_offsets[anchor + 1])
            neigh = base.neighbor_index[start:end]
            if neigh.size == 0:
                continue
            if np.any(sample_mask[neigh]):
                kept.append(anchor)
        return np.asarray(kept, dtype=np.int32)

    train_anchor = select_anchor_indices(train_sample_mask)
    val_anchor = select_anchor_indices(val_sample_mask)
    if train_anchor.size == 0 or val_anchor.size == 0:
        raise RuntimeError('Train/val split produced an empty anchor subset.')

    train_dataset = PreferencePairDataset(base, train_anchor, train_sample_mask, seed=seed)
    val_dataset = PreferencePairDataset(base, val_anchor, val_sample_mask, seed=seed + 100_000)
    stats = {
        'num_total_samples': int(base.length.shape[0]),
        'num_train_trajectories': int(train_traj.size),
        'num_val_trajectories': int(val_traj.size),
        'num_train_pairs': int(train_anchor.size),
        'num_val_pairs': int(val_anchor.size),
    }
    return train_dataset, val_dataset, stats


def compute_per_sample_denoising_loss(model, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, use_ema_backbone: bool) -> torch.Tensor:
    xt, t_used, eps_used = model.add_noise(x0, t=t, eps=eps)
    backbone = model.model_ema if use_ema_backbone else model.model
    pred = backbone['diffusion'](xt, t_used, None)
    if model.predict_noise:
        loss = (pred - eps_used) ** 2
    else:
        loss = (pred - x0) ** 2
    weighted = loss * model.loss_weight * (1.0 - model.fix_mask)
    return weighted.mean(dim=tuple(range(1, weighted.ndim)))


def _expand_schedule_like(value: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return value.view(-1, *([1] * (x.dim() - 1)))


def compute_policy_terms(model, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, use_ema_backbone: bool) -> tuple[torch.Tensor, torch.Tensor]:
    xt, t_used, eps_used = model.add_noise(x0, t=t, eps=eps)
    backbone = model.model_ema if use_ema_backbone else model.model
    pred = backbone['diffusion'](xt, t_used, None)
    if model.predict_noise:
        loss = (pred - eps_used) ** 2
        bar_alpha = _expand_schedule_like(model.bar_alpha[t_used], x0)
        x_theta = (xt - (1.0 - bar_alpha).sqrt() * pred) / bar_alpha.sqrt().clamp_min(1e-8)
    else:
        loss = (pred - x0) ** 2
        x_theta = pred
    x_theta = (1.0 - model.fix_mask) * x_theta + model.fix_mask * x0
    weighted = loss * model.loss_weight * (1.0 - model.fix_mask)
    return weighted.mean(dim=tuple(range(1, weighted.ndim))), x_theta


def denormalize_q_torch(q_norm: torch.Tensor, stats: dict, device: torch.device) -> torch.Tensor:
    q_mean = torch.as_tensor(stats['q_mean'], dtype=q_norm.dtype, device=device).view(1, 1, -1)
    q_std = torch.as_tensor(stats['q_std'], dtype=q_norm.dtype, device=device).view(1, 1, -1)
    return q_norm * q_std + q_mean


def decode_condition_torch(x0: torch.Tensor, q_dim: int, stats: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x0.device
    dtype = x0.dtype
    pos_mean = torch.as_tensor(stats['pos_mean'], dtype=dtype, device=device).view(1, 1, 3)
    pos_std = torch.as_tensor(stats['pos_std'], dtype=dtype, device=device).view(1, 1, 3)
    pos = x0[..., q_dim:q_dim + 3] * pos_std + pos_mean
    direction = x0[..., q_dim + 3:q_dim + 6]
    normal = x0[..., q_dim + 6:q_dim + 9]
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return pos, direction, normal


def compute_fk_consistency_loss(
    robot,
    x_theta: torch.Tensor,
    x_target: torch.Tensor,
    q_dim: int,
    stats: dict,
    pos_weight: float,
    dir_weight: float,
    normal_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    q_pred = denormalize_q_torch(x_theta[..., :q_dim], stats, x_theta.device).squeeze(1)
    pos_target, dir_target, normal_target = decode_condition_torch(x_target, q_dim, stats)
    pos_target = pos_target.squeeze(1)
    dir_target = dir_target.squeeze(1)
    normal_target = normal_target.squeeze(1)

    tcp_pos, tcp_rot = robot.fk_batch(q_pred)
    tcp_x = tcp_rot[:, :, 0]
    tcp_z = tcp_rot[:, :, 2]

    pos_loss = torch.linalg.norm(tcp_pos - pos_target, dim=1).mean()
    dir_loss = (1.0 - torch.sum(tcp_x * dir_target, dim=1).clamp(-1.0, 1.0)).mean()
    normal_loss = (1.0 - torch.sum(tcp_z * normal_target, dim=1).clamp(-1.0, 1.0)).mean()
    total = float(pos_weight) * pos_loss + float(dir_weight) * dir_loss + float(normal_weight) * normal_loss
    return total, {
        'pos': float(pos_loss.detach()),
        'dir': float(dir_loss.detach()),
        'normal': float(normal_loss.detach()),
    }


def dpo_loss_from_pairs(
    policy_model,
    ref_model,
    robot,
    stats: dict,
    q_dim: int,
    x_w: torch.Tensor,
    x_l: torch.Tensor,
    beta: float,
    recon_weight: float,
    consistency_weight: float,
    consistency_pos_weight: float,
    consistency_dir_weight: float,
    consistency_normal_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size = x_w.shape[0]
    device = x_w.device
    # Use the same diffusion timestep schedule for both winner and loser so
    # their relative comparison is not dominated by sampling noise.
    t = torch.randint(policy_model.diffusion_steps, (batch_size,), device=device)
    # Pairwise comparison is more stable when both samples share the same
    # diffusion noise realization.
    eps = torch.randn_like(x_w)

    policy_w, x_theta_w = compute_policy_terms(policy_model, x_w, t, eps, use_ema_backbone=False)
    policy_l, x_theta_l = compute_policy_terms(policy_model, x_l, t, eps, use_ema_backbone=False)
    with torch.no_grad():
        ref_w, _ = compute_policy_terms(ref_model, x_w, t, eps, use_ema_backbone=True)
        ref_l, _ = compute_policy_terms(ref_model, x_l, t, eps, use_ema_backbone=True)

    policy_log_ratio = policy_l - policy_w
    ref_log_ratio = ref_l - ref_w
    logits = float(beta) * (policy_log_ratio - ref_log_ratio)
    dpo_loss = -F.logsigmoid(logits).mean()

    recon_loss = 0.5 * (policy_w.mean() + policy_l.mean())
    fk_w, fk_terms_w = compute_fk_consistency_loss(
        robot=robot,
        x_theta=x_theta_w,
        x_target=x_w,
        q_dim=q_dim,
        stats=stats,
        pos_weight=consistency_pos_weight,
        dir_weight=consistency_dir_weight,
        normal_weight=consistency_normal_weight,
    )
    fk_l, fk_terms_l = compute_fk_consistency_loss(
        robot=robot,
        x_theta=x_theta_l,
        x_target=x_l,
        q_dim=q_dim,
        stats=stats,
        pos_weight=consistency_pos_weight,
        dir_weight=consistency_dir_weight,
        normal_weight=consistency_normal_weight,
    )
    fk_loss = 0.5 * (fk_w + fk_l)
    loss = dpo_loss + float(recon_weight) * recon_loss + float(consistency_weight) * fk_loss
    aux = {
        'dpo_loss': float(dpo_loss.detach()),
        'recon_loss': float(recon_loss.detach()),
        'fk_loss': float(fk_loss.detach()),
        'policy_w_loss': float(policy_w.mean().detach()),
        'policy_l_loss': float(policy_l.mean().detach()),
        'ref_w_loss': float(ref_w.mean().detach()),
        'ref_l_loss': float(ref_l.mean().detach()),
        'logit_mean': float(logits.mean().detach()),
        'reward_margin': float((policy_log_ratio - ref_log_ratio).mean().detach()),
        'pair_acc': float((policy_log_ratio > 0.0).float().mean().detach()),
        'fk_pos_loss': 0.5 * (fk_terms_w['pos'] + fk_terms_l['pos']),
        'fk_dir_loss': 0.5 * (fk_terms_w['dir'] + fk_terms_l['dir']),
        'fk_normal_loss': 0.5 * (fk_terms_w['normal'] + fk_terms_l['normal']),
    }
    return loss, aux


@torch.no_grad()
def evaluate(
    policy_model,
    ref_model,
    robot,
    stats: dict,
    q_dim: int,
    loader: DataLoader,
    beta: float,
    recon_weight: float,
    consistency_weight: float,
    consistency_pos_weight: float,
    consistency_dir_weight: float,
    consistency_normal_weight: float,
    max_batches: int | None = None,
) -> tuple[float, dict[str, float]]:
    policy_model.eval()
    ref_model.eval()
    losses = []
    pair_acc = []
    reward_margin = []
    recon_losses = []
    fk_losses = []
    for batch_idx, (x_w, x_l) in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        x_w = x_w.to(policy_model.device)
        x_l = x_l.to(policy_model.device)
        loss, aux = dpo_loss_from_pairs(
            policy_model=policy_model,
            ref_model=ref_model,
            robot=robot,
            stats=stats,
            q_dim=q_dim,
            x_w=x_w,
            x_l=x_l,
            beta=beta,
            recon_weight=recon_weight,
            consistency_weight=consistency_weight,
            consistency_pos_weight=consistency_pos_weight,
            consistency_dir_weight=consistency_dir_weight,
            consistency_normal_weight=consistency_normal_weight,
        )
        losses.append(float(loss.detach()))
        pair_acc.append(aux['pair_acc'])
        reward_margin.append(aux['reward_margin'])
        recon_losses.append(aux['recon_loss'])
        fk_losses.append(aux['fk_loss'])
    policy_model.train()
    return float(np.mean(losses)), {
        'pair_acc': float(np.mean(pair_acc)) if pair_acc else float('nan'),
        'reward_margin': float(np.mean(reward_margin)) if reward_margin else float('nan'),
        'recon_loss': float(np.mean(recon_losses)) if recon_losses else float('nan'),
        'fk_loss': float(np.mean(fk_losses)) if fk_losses else float('nan'),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    if not args.pref_h5.exists():
        raise FileNotFoundError(f'Preference dataset not found: {args.pref_h5}')
    if not args.ref_bundle.exists():
        raise FileNotFoundError(f'Reference diffusion bundle not found: {args.ref_bundle}')

    ref_bundle, stats, ref_model, x_min, x_max = load_bundle_model(args.ref_bundle, device)
    _, _, policy_model, _, _ = load_bundle_model(args.ref_bundle, device)
    robot_helper = FrankaResearch3GPU(device=device)
    ref_model.eval()
    for param in ref_model.model.parameters():
        param.requires_grad_(False)
    for param in ref_model.model_ema.parameters():
        param.requires_grad_(False)

    # Override optimizer for preference fine-tuning.
    policy_model.optimizer = torch.optim.AdamW(
        policy_model.model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    base = PreferenceTokenDataset(args.pref_h5, stats)
    train_dataset, val_dataset, split_stats = build_split_datasets(base, args.val_ratio, args.seed)

    max_batches = None
    if args.max_pairs_per_epoch is not None:
        max_batches = max(1, int(np.ceil(int(args.max_pairs_per_epoch) / max(int(args.batch_size), 1))))

    train_sampler = PairBatchSampler(len(train_dataset), args.batch_size, args.seed, max_batches=max_batches)
    val_sampler = PairBatchSampler(len(val_dataset), args.batch_size, args.seed + 10_000)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0)

    run_dir = args.workdir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        'pref_h5': str(args.pref_h5),
        'ref_bundle': str(args.ref_bundle),
        'split_stats': split_stats,
        'token_dim': int(base.token_dim),
        'q_dim': int(stats['q_dim']),
        'dpo_mode': 'cross_condition',
    }

    use_wandb = (args.wandb_mode != 'disabled') and (wandb is not None)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or args.run_name,
            dir=str(run_dir),
            mode=args.wandb_mode,
            config={
                'pref_h5': str(args.pref_h5),
                'ref_bundle': str(args.ref_bundle),
                'device': str(device),
                'epochs': int(args.epochs),
                'batch_size': int(args.batch_size),
                'dpo_beta': float(args.dpo_beta),
                'recon_weight': float(args.recon_weight),
                'consistency_weight': float(args.consistency_weight),
                'lr': float(args.lr),
                **split_stats,
            },
        )
    elif args.wandb_mode != 'disabled' and wandb is None:
        print('[config] wandb is not installed; metrics will only be printed locally.', flush=True)

    best_val = float('inf')
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        policy_model.train()
        epoch_losses = []
        epoch_acc = []
        epoch_margin = []
        for x_w, x_l in train_loader:
            global_step += 1
            x_w = x_w.to(device)
            x_l = x_l.to(device)
            loss, aux = dpo_loss_from_pairs(
                policy_model=policy_model,
                ref_model=ref_model,
                robot=robot_helper.robot,
                stats=stats,
                q_dim=int(stats['q_dim']),
                x_w=x_w,
                x_l=x_l,
                beta=float(args.dpo_beta),
                recon_weight=float(args.recon_weight),
                consistency_weight=float(args.consistency_weight),
                consistency_pos_weight=float(args.consistency_pos_weight),
                consistency_dir_weight=float(args.consistency_dir_weight),
                consistency_normal_weight=float(args.consistency_normal_weight),
            )
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy_model.model.parameters(), policy_model.grad_clip_norm) \
                if policy_model.grad_clip_norm else None
            policy_model.optimizer.step()
            policy_model.optimizer.zero_grad()
            policy_model.ema_update()

            epoch_losses.append(float(loss.detach()))
            epoch_acc.append(float(aux['pair_acc']))
            epoch_margin.append(float(aux['reward_margin']))

            if use_wandb:
                wandb.log(
                    {
                        'train/loss_step': float(loss.detach()),
                        'train/pair_acc_step': float(aux['pair_acc']),
                        'train/reward_margin_step': float(aux['reward_margin']),
                        'train/dpo_loss_step': float(aux['dpo_loss']),
                        'train/recon_loss_step': float(aux['recon_loss']),
                        'train/fk_loss_step': float(aux['fk_loss']),
                        'train/fk_pos_loss_step': float(aux['fk_pos_loss']),
                        'train/fk_dir_loss_step': float(aux['fk_dir_loss']),
                        'train/fk_normal_loss_step': float(aux['fk_normal_loss']),
                        'train/grad_norm_step': None if grad_norm is None else float(grad_norm),
                        'global_step': global_step,
                    },
                    step=global_step,
                )
            if args.print_every > 0 and global_step % int(args.print_every) == 0:
                print(
                    f'[train] epoch={epoch:03d} step={global_step:06d} '
                    f'loss={float(loss.detach()):.6f} pair_acc={aux["pair_acc"]:.4f} '
                    f'margin={aux["reward_margin"]:.6f} dpo={aux["dpo_loss"]:.6f} '
                    f'recon={aux["recon_loss"]:.6f} fk={aux["fk_loss"]:.6f} grad_norm={grad_norm}',
                    flush=True,
                )

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
        train_pair_acc = float(np.mean(epoch_acc)) if epoch_acc else float('nan')
        train_margin = float(np.mean(epoch_margin)) if epoch_margin else float('nan')
        print(
            f'[epoch] epoch={epoch:03d} train_loss={train_loss:.6f} '
            f'train_pair_acc={train_pair_acc:.4f} train_margin={train_margin:.6f}',
            flush=True,
        )
        if use_wandb:
            wandb.log(
                {
                    'train/loss_epoch': train_loss,
                    'train/pair_acc_epoch': train_pair_acc,
                    'train/reward_margin_epoch': train_margin,
                    'train/epoch': epoch,
                },
                step=global_step,
            )

        if epoch % int(args.eval_interval) == 0:
            val_loss, val_aux = evaluate(
                policy_model=policy_model,
                ref_model=ref_model,
                robot=robot_helper.robot,
                stats=stats,
                q_dim=int(stats['q_dim']),
                loader=val_loader,
                beta=float(args.dpo_beta),
                recon_weight=float(args.recon_weight),
                consistency_weight=float(args.consistency_weight),
                consistency_pos_weight=float(args.consistency_pos_weight),
                consistency_dir_weight=float(args.consistency_dir_weight),
                consistency_normal_weight=float(args.consistency_normal_weight),
            )
            print(
                f'[eval] epoch={epoch:03d} val_loss={val_loss:.6f} '
                f'val_pair_acc={val_aux["pair_acc"]:.4f} val_margin={val_aux["reward_margin"]:.6f} '
                f'val_recon={val_aux["recon_loss"]:.6f} val_fk={val_aux["fk_loss"]:.6f}',
                flush=True,
            )
            if use_wandb:
                wandb.log(
                    {
                        'eval/loss': val_loss,
                        'eval/pair_acc': val_aux['pair_acc'],
                        'eval/reward_margin': val_aux['reward_margin'],
                        'eval/recon_loss': val_aux['recon_loss'],
                        'eval/fk_loss': val_aux['fk_loss'],
                        'train/epoch': epoch,
                    },
                    step=global_step,
                )
            if val_loss < best_val:
                best_val = val_loss
                policy_model.save(str(run_dir / 'model_best.pt'))
                torch.save(
                    {
                        'stats': {k: np.asarray(v, dtype=np.float32) if isinstance(v, np.ndarray) else v for k, v in stats.items()},
                        'x_min': np.asarray(x_min, dtype=np.float32),
                        'x_max': np.asarray(x_max, dtype=np.float32),
                        'args': dict(vars(args)),
                        'metadata': metadata,
                        'best_val_loss': best_val,
                    },
                    run_dir / 'bundle_best.pt',
                )
                print(f'[eval] new_best_val={best_val:.6f}', flush=True)

        if epoch % int(args.save_interval) == 0:
            save_bundle(run_dir, policy_model, stats, x_min, x_max, args, metadata)

    save_bundle(run_dir, policy_model, stats, x_min, x_max, args, metadata)
    summary = {
        'args': to_jsonable(vars(args)),
        'metadata': to_jsonable(metadata),
        'best_val_loss': float(best_val),
    }
    (run_dir / 'dpo_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] best_val_loss={best_val:.6f} run_dir={run_dir}', flush=True)
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
