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
from torch.utils.data import DataLoader, Dataset, BatchSampler, random_split

from lnet_contrastive import LNetContrastive
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller

from paths import DEFAULT_H5_PREF, LNET_CONTRASTIVE_RUNS_DIR

BASE_DIR = Path(__file__).resolve().parent
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


class ContrastivePrefDataset(Dataset):
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
            self.neighbor_offsets = np.asarray(pref['neighbor_offsets'][:], dtype=np.int64)
            self.neighbor_index = np.asarray(pref['neighbor_index'][:], dtype=np.int32)
            self.valid_anchor_index = np.asarray(pref['valid_anchor_index'][:], dtype=np.int32)

    def __len__(self) -> int:
        return int(self.valid_anchor_index.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        anchor = int(self.valid_anchor_index[idx])
        start = int(self.neighbor_offsets[anchor])
        end = int(self.neighbor_offsets[anchor + 1])
        if end <= start:
            partner = anchor
        else:
            partner = int(np.random.choice(self.neighbor_index[start:end]))
        return (
            self.q[anchor], self.cond[anchor], self.length[anchor],
            self.q[partner], self.cond[partner], self.length[partner],
        )


class SameCondBatchSampler(BatchSampler):
    def __init__(self, dataset: ContrastivePrefDataset, batch_size: int, seed: int, subset_indices: list[int] | np.ndarray):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.subset_indices = np.asarray(subset_indices, dtype=np.int64)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = self.subset_indices.copy()
        rng.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            yield order[start:start + self.batch_size].tolist()

    def __len__(self) -> int:
        return int(np.ceil(len(self.subset_indices) / self.batch_size))


def compute_min_max(dataset: ContrastivePrefDataset, pair_subset_indices: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if pair_subset_indices is None:
        q = dataset.q
        cond = dataset.cond
    else:
        anchor = torch.from_numpy(dataset.valid_anchor_index[pair_subset_indices]).long()
        q = dataset.q[anchor]
        cond = dataset.cond[anchor]
    x = torch.cat([q, cond], dim=1)
    return x.min(dim=0).values, x.max(dim=0).values


def evaluate(model: LNetContrastive, loader: DataLoader, device: torch.device) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    total_rank = 0.0
    total_mse = 0.0
    total_abs = 0.0
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
            _, pred_i = model(q_i, cond_i)
            _, pred_j = model(q_j, cond_j)
            bs = q_i.shape[0]
            total_loss += float(loss) * bs
            total_rank += aux['rank_loss'] * bs
            total_mse += aux['mse_loss'] * bs
            total_abs += 0.5 * float((pred_i - l_i).abs().mean() + (pred_j - l_j).abs().mean()) * bs
            total_count += bs
            total_pairs += aux['pair_count']
    denom = max(total_count, 1)
    return total_loss / denom, {
        'rank_loss': total_rank / denom,
        'mse_loss': total_mse / denom,
        'mae': total_abs / denom,
        'pair_count': total_pairs,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    run_dir = DEFAULT_WORKDIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = ContrastivePrefDataset(args.h5_path)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).numpy()
    train_pair_idx = perm[:train_size]
    val_pair_idx = perm[train_size:]

    q_limits = torch.from_numpy(XArmLite6Miller(enable_cc=False).jnt_ranges.astype(np.float32))
    in_min, in_max = compute_min_max(dataset, train_pair_idx)
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

    train_sampler = SameCondBatchSampler(dataset, args.batch_size, args.seed, train_pair_idx)
    val_sampler = SameCondBatchSampler(dataset, args.batch_size, args.seed + 10_000, val_pair_idx)
    train_loader = DataLoader(dataset, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(dataset, batch_sampler=val_sampler, num_workers=0)

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
            loss, aux = model.compute_pair_loss(q_i, cond_i, l_i, q_j, cond_j, l_j)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            if use_wandb:
                wandb.log({
                    'train/loss': float(loss),
                    'train/rank_loss': aux['rank_loss'],
                    'train/mse_loss': aux['mse_loss'],
                    'train/pair_count': aux['pair_count'],
                    'epoch': epoch,
                    'step': global_step,
                }, step=global_step)
            if args.print_every > 0 and global_step % args.print_every == 0:
                print(f'[train] epoch={epoch:03d} step={global_step:06d} loss={float(loss):.6f} rank={aux["rank_loss"]:.6f} mse={aux["mse_loss"]:.6f} pairs={aux["pair_count"]}')

        val_sampler.set_epoch(epoch)
        val_loss, val_aux = evaluate(model, val_loader, device)
        print(f'[eval] epoch={epoch:03d} val_loss={val_loss:.6f} val_rank={val_aux["rank_loss"]:.6f} val_mse={val_aux["mse_loss"]:.6f} val_mae={val_aux["mae"]:.6f} pairs={val_aux["pair_count"]}')
        if use_wandb:
            wandb.log({
                'eval/loss': val_loss,
                'eval/rank_loss': val_aux['rank_loss'],
                'eval/mse_loss': val_aux['mse_loss'],
                'eval/mae': val_aux['mae'],
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
        'num_valid_pairs': int(len(dataset)),
        'train_size': int(train_size),
        'val_size': int(val_size),
        'best_val': best_val,
    }
    (run_dir / 'metadata.json').write_text(json.dumps(meta_payload, indent=2))
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
