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
from torch.utils.data import DataLoader, Dataset, random_split

from lnet_contrastive import LNetContrastive
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_H5 = BASE_DIR / 'xarmlite6_gpu_trajectories_100000_sub10.hdf5'
DEFAULT_WORKDIR = BASE_DIR / 'lnet_contrastive_runs'
DEFAULT_RUN_NAME = 'lnet_contrastive_q_cond_to_length_sub10'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train contrastive LNet on the sub10 trajectory dataset.')
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--run-name', type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument('--batch-size', type=int, default=1024)
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


class TrajectoryPointDataset(Dataset):
    def __init__(self, h5_path: Path):
        q_list = []
        cond_list = []
        length_list = []
        with h5py.File(h5_path, 'r') as f:
            for key in sorted(f['trajectories'].keys()):
                grp = f['trajectories'][key]
                q = np.asarray(grp['q'][:], dtype=np.float32)
                pos = np.asarray(grp['tcp_pos'][:], dtype=np.float32)
                direction = np.asarray(grp.attrs['direction'], dtype=np.float32)
                normal = np.asarray(grp.attrs['target_normal'], dtype=np.float32)
                cond = np.concatenate([
                    pos,
                    np.repeat(direction.reshape(1, 3), q.shape[0], axis=0),
                    np.repeat(normal.reshape(1, 3), q.shape[0], axis=0),
                ], axis=1).astype(np.float32)
                length = np.asarray(grp['remaining_length'][:], dtype=np.float32)
                q_list.append(q)
                cond_list.append(cond)
                length_list.append(length)
        self.q = torch.from_numpy(np.concatenate(q_list, axis=0))
        self.cond = torch.from_numpy(np.concatenate(cond_list, axis=0))
        self.length = torch.from_numpy(np.concatenate(length_list, axis=0))

    def __len__(self) -> int:
        return self.q.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q[idx], self.cond[idx], self.length[idx]


def compute_min_max(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(dataset, 'indices'):
        base = dataset.dataset
        idx = torch.as_tensor(dataset.indices, dtype=torch.long)
        q = base.q[idx]
        cond = base.cond[idx]
    else:
        q = dataset.q
        cond = dataset.cond
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
        for q, cond, target in loader:
            q = q.to(device)
            cond = cond.to(device)
            target = target.to(device)
            loss, aux = model.compute_loss(q, cond, target)
            _, pred_length = model(q, cond)
            bs = q.shape[0]
            total_loss += float(loss) * bs
            total_rank += aux['rank_loss'] * bs
            total_mse += aux['mse_loss'] * bs
            total_abs += float((pred_length - target).abs().mean()) * bs
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

    dataset = TrajectoryPointDataset(args.h5_path)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    q_limits = torch.from_numpy(XArmLite6Miller(enable_cc=False).jnt_ranges.astype(np.float32))
    in_min, in_max = compute_min_max(train_set)
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
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or args.run_name,
            mode=args.wandb_mode,
            config=vars(args),
            dir=str(run_dir),
        )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    best_val = float('inf')
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for q, cond, target in train_loader:
            q = q.to(device)
            cond = cond.to(device)
            target = target.to(device)
            loss, aux = model.compute_loss(q, cond, target)
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
                print(
                    f'[train] epoch={epoch:03d} step={global_step:06d} loss={float(loss):.6f} '
                    f'rank={aux["rank_loss"]:.6f} mse={aux["mse_loss"]:.6f} pairs={aux["pair_count"]}'
                )

        val_loss, val_aux = evaluate(model, val_loader, device)
        print(
            f'[eval] epoch={epoch:03d} val_loss={val_loss:.6f} val_rank={val_aux["rank_loss"]:.6f} '
            f'val_mse={val_aux["mse_loss"]:.6f} val_mae={val_aux["mae"]:.6f} pairs={val_aux["pair_count"]}'
        )
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
        'train_size': train_size,
        'val_size': val_size,
        'best_val': best_val,
    }
    (run_dir / 'metadata.json').write_text(json.dumps(meta_payload, indent=2))

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
