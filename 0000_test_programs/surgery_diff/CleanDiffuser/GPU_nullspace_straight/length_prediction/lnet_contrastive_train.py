from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None
from torch.utils.data import BatchSampler, DataLoader, Dataset

from lnet_contrastive import LNetContrastive
from paths import DEFAULT_H5_PREF, LNET_CONTRASTIVE_RUNS_DIR
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller

DEFAULT_WORKDIR = LNET_CONTRASTIVE_RUNS_DIR
DEFAULT_RUN_NAME = 'lnet_contrastive_q_cond_to_length_sub10_pref'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train contrastive LNet on the pref-indexed trajectory dataset.')
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5_PREF)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--run-name', type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-ratio', type=float, default=0.02)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pair-threshold', type=float, default=0.05)
    parser.add_argument('--pair-margin', type=float, default=0.05)
    parser.add_argument('--mse-weight', type=float, default=0.2)
    parser.add_argument('--rank-weight', type=float, default=1.0)
    parser.add_argument('--max-pairs', type=int, default=4096)
    parser.add_argument('--joint-noise-std', type=float, default=0.01)
    parser.add_argument('--print-every', type=int, default=200)
    parser.add_argument('--wandb-project', type=str, default='xarm-lnet-contrastive')
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='online')
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ContrastivePrefBaseDataset:
    def __init__(self, h5_path: Path):
        with h5py.File(h5_path, 'r') as f:
            if 'contrastive_pref' not in f:
                raise KeyError(f'Missing contrastive_pref group in {h5_path}. Run lnet_contrastive_pref.py first.')
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


class ContrastivePrefPairDataset(Dataset):
    def __init__(self, base: ContrastivePrefBaseDataset, anchor_indices: np.ndarray, allowed_sample_mask: np.ndarray):
        self.base = base
        self.anchor_indices = np.asarray(anchor_indices, dtype=np.int32)
        self.allowed_sample_mask = np.asarray(allowed_sample_mask, dtype=bool)

    def __len__(self) -> int:
        return int(self.anchor_indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        anchor = int(self.anchor_indices[idx])
        start = int(self.base.neighbor_offsets[anchor])
        end = int(self.base.neighbor_offsets[anchor + 1])
        neigh = self.base.neighbor_index[start:end]
        score = self.base.neighbor_score[start:end]
        mask = self.allowed_sample_mask[neigh]
        neigh = neigh[mask]
        score = score[mask]
        if neigh.size == 0:
            partner = anchor
        else:
            weight = score.astype(np.float64)
            weight = weight - weight.min() + 1.0
            weight_sum = float(weight.sum())
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                partner = int(np.random.choice(neigh))
            else:
                partner = int(np.random.choice(neigh, p=weight / weight_sum))
        return (
            self.base.q[anchor], self.base.cond[anchor], self.base.length[anchor],
            self.base.q[partner], self.base.cond[partner], self.base.length[partner],
        )


class PairBatchSampler(BatchSampler):
    def __init__(self, dataset_len: int, batch_size: int, seed: int):
        self.dataset_len = int(dataset_len)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = np.arange(self.dataset_len, dtype=np.int64)
        rng.shuffle(order)
        for start in range(0, self.dataset_len, self.batch_size):
            yield order[start:start + self.batch_size].tolist()

    def __len__(self) -> int:
        return int(np.ceil(self.dataset_len / self.batch_size))


def build_split_datasets(base: ContrastivePrefBaseDataset, val_ratio: float, seed: int) -> tuple[ContrastivePrefPairDataset, ContrastivePrefPairDataset, dict]:
    unique_traj = np.unique(base.traj_idx)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_traj)
    val_traj_count = max(1, int(len(unique_traj) * val_ratio))
    val_traj = unique_traj[:val_traj_count]
    train_traj = unique_traj[val_traj_count:]
    if train_traj.size == 0:
        raise RuntimeError('No train trajectories left after split.')

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
        raise RuntimeError('Train/val split produced an empty anchor subset. Adjust val_ratio or pref construction.')

    train_dataset = ContrastivePrefPairDataset(base, train_anchor, train_sample_mask)
    val_dataset = ContrastivePrefPairDataset(base, val_anchor, val_sample_mask)
    stats = {
        'num_total_samples': int(base.length.shape[0]),
        'num_total_valid_anchors': int(base.valid_anchor_index.shape[0]),
        'num_train_trajectories': int(train_traj.size),
        'num_val_trajectories': int(val_traj.size),
        'num_train_anchors': int(train_anchor.size),
        'num_val_anchors': int(val_anchor.size),
    }
    return train_dataset, val_dataset, stats


def compute_min_max(base: ContrastivePrefBaseDataset, anchor_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    q = base.q[torch.from_numpy(anchor_indices).long()]
    cond = base.cond[torch.from_numpy(anchor_indices).long()]
    x = torch.cat([q, cond], dim=1)
    return x.min(dim=0).values, x.max(dim=0).values


def augment_q(q: torch.Tensor, q_min: torch.Tensor, q_max: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0.0:
        return q
    noise = torch.randn_like(q) * noise_std
    return torch.max(torch.min(q + noise, q_max.view(1, -1)), q_min.view(1, -1))


def pair_accuracy(score_i: torch.Tensor, score_j: torch.Tensor, l_i: torch.Tensor, l_j: torch.Tensor, threshold: float) -> tuple[float, int]:
    diff = l_i - l_j
    valid = diff.abs() > threshold
    if not valid.any():
        return 0.0, 0
    sign = torch.sign(diff[valid])
    pred = torch.sign(score_i[valid] - score_j[valid])
    acc = (pred == sign).float().mean().item()
    return float(acc), int(valid.sum().item())


def evaluate(model: LNetContrastive, loader: DataLoader, device: torch.device) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    total_rank = 0.0
    total_mse = 0.0
    total_abs = 0.0
    total_pair_acc = 0.0
    total_count = 0
    total_pairs = 0
    with torch.no_grad():
        for q_i, cond_i, l_i, q_j, cond_j, l_j in loader:
            q_i = q_i.to(device)
            cond_i = cond_i.to(device)
            l_i = l_i.to(device)
            q_j = q_j.to(device)
            cond_j = cond_j.to(device)
            l_j = l_j.to(device)
            loss, aux = model.compute_pair_loss(q_i, cond_i, l_i, q_j, cond_j, l_j)
            score_i, pred_i = model(q_i, cond_i)
            score_j, pred_j = model(q_j, cond_j)
            acc, pair_cnt = pair_accuracy(score_i, score_j, l_i, l_j, model.pair_threshold)
            bs = q_i.shape[0]
            total_loss += float(loss) * bs
            total_rank += aux['rank_loss'] * bs
            total_mse += aux['mse_loss'] * bs
            total_abs += 0.5 * float((pred_i - l_i).abs().mean() + (pred_j - l_j).abs().mean()) * bs
            total_pair_acc += acc * bs
            total_count += bs
            total_pairs += pair_cnt
    denom = max(total_count, 1)
    return total_loss / denom, {
        'rank_loss': total_rank / denom,
        'mse_loss': total_mse / denom,
        'mae': total_abs / denom,
        'pair_acc': total_pair_acc / denom,
        'pair_count': total_pairs,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    run_dir = DEFAULT_WORKDIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    base = ContrastivePrefBaseDataset(args.h5_path)
    train_dataset, val_dataset, split_stats = build_split_datasets(base, args.val_ratio, args.seed)

    q_limits = torch.from_numpy(XArmLite6Miller(enable_cc=False).jnt_ranges.astype(np.float32))
    in_min, in_max = compute_min_max(base, train_dataset.anchor_indices)
    model = LNetContrastive(
        q_min=q_limits[:, 0],
        q_max=q_limits[:, 1],
        in_min=in_min,
        in_max=in_max,
        pair_threshold=args.pair_threshold,
        pair_margin=args.pair_margin,
        mse_weight=args.mse_weight,
        rank_weight=args.rank_weight,
        max_pairs=args.max_pairs,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_wandb = args.wandb_mode != 'disabled' and wandb is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name or args.run_name, mode=args.wandb_mode, config=vars(args), dir=str(run_dir))

    train_sampler = PairBatchSampler(len(train_dataset), args.batch_size, args.seed)
    val_sampler = PairBatchSampler(len(val_dataset), args.batch_size, args.seed + 10_000)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0)

    best_val = float('inf')
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        for q_i, cond_i, l_i, q_j, cond_j, l_j in train_loader:
            q_i = q_i.to(device)
            cond_i = cond_i.to(device)
            l_i = l_i.to(device)
            q_j = q_j.to(device)
            cond_j = cond_j.to(device)
            l_j = l_j.to(device)
            q_i = augment_q(q_i, model.q_min.squeeze(0), model.q_max.squeeze(0), args.joint_noise_std)
            q_j = augment_q(q_j, model.q_min.squeeze(0), model.q_max.squeeze(0), args.joint_noise_std)
            loss, aux = model.compute_pair_loss(q_i, cond_i, l_i, q_j, cond_j, l_j)
            score_i, _ = model(q_i, cond_i)
            score_j, _ = model(q_j, cond_j)
            acc, pair_cnt = pair_accuracy(score_i, score_j, l_i, l_j, model.pair_threshold)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            if use_wandb:
                wandb.log({
                    'train/loss': float(loss),
                    'train/rank_loss': aux['rank_loss'],
                    'train/mse_loss': aux['mse_loss'],
                    'train/pair_acc': acc,
                    'train/pair_count': pair_cnt,
                    'epoch': epoch,
                    'step': global_step,
                }, step=global_step)
            if args.print_every > 0 and global_step % args.print_every == 0:
                print(f'[train] epoch={epoch:03d} step={global_step:06d} loss={float(loss):.6f} rank={aux["rank_loss"]:.6f} mse={aux["mse_loss"]:.6f} acc={acc:.4f} pairs={pair_cnt}')

        val_sampler.set_epoch(epoch)
        val_loss, val_aux = evaluate(model, val_loader, device)
        print(f'[eval] epoch={epoch:03d} val_loss={val_loss:.6f} val_rank={val_aux["rank_loss"]:.6f} val_mse={val_aux["mse_loss"]:.6f} val_mae={val_aux["mae"]:.6f} val_pair_acc={val_aux["pair_acc"]:.4f} pairs={val_aux["pair_count"]}')
        if use_wandb:
            wandb.log({
                'eval/loss': val_loss,
                'eval/rank_loss': val_aux['rank_loss'],
                'eval/mse_loss': val_aux['mse_loss'],
                'eval/mae': val_aux['mae'],
                'eval/pair_acc': val_aux['pair_acc'],
                'eval/pair_count': val_aux['pair_count'],
                'epoch': epoch,
                'step': global_step,
            }, step=global_step)

        state = {
            'model': model.state_dict(),
            'q_min': q_limits[:, 0],
            'q_max': q_limits[:, 1],
            'in_min': in_min,
            'in_max': in_max,
            'args': vars(args),
            'best_val': best_val,
            'epoch': epoch,
        }
        torch.save(state, run_dir / 'lnet_contrastive_latest.pt')
        if val_loss < best_val:
            best_val = val_loss
            state['best_val'] = best_val
            torch.save(state, run_dir / 'lnet_contrastive_best.pt')
            print(f'[eval] new_best={best_val:.6f}')

    meta_payload = {
        'args': to_jsonable(vars(args)),
        'split_stats': to_jsonable(split_stats),
        'train_size': int(len(train_dataset)),
        'val_size': int(len(val_dataset)),
        'best_val': best_val,
    }
    (run_dir / 'metadata.json').write_text(json.dumps(meta_payload, indent=2))
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
