import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

from diffusion import (
    DEFAULT_CACHE_DIR,
    DEFAULT_H5_PATH,
    DEFAULT_RUN_NAME,
    DEFAULT_WORKDIR,
    InpaintingDataset,
    build_inpainting_x,
    compute_stats,
    create_loader,
    create_model,
    prepare_raw_token_cache,
    sample_q_length_from_condition,
    save_bundle,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train DDPM inpainting baseline for q and remaining-length generation from (pos, direction, normal).')
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument('--workdir', type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument('--run-name', type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument('--seed', type=int, default=20260330)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--diffusion-steps', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--eval-batch-size', type=int, default=1024)
    parser.add_argument('--val-size', type=int, default=2000)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--save-interval', type=int, default=5)
    parser.add_argument('--n-sample-demo', type=int, default=4)
    parser.add_argument('--sample-entry-idx', type=int, default=0)
    parser.add_argument('--max-trajectories', type=int, default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--wandb-project', type=str, default='xarm-diffusion-inpainting')
    parser.add_argument('--wandb-name', type=str, default=None)
    parser.add_argument('--wandb-mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, val_loader, device: torch.device):
    model.eval()
    losses = []
    for batch in val_loader:
        x0 = batch['x0'].to(device)
        loss = model.loss(x0)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.h5_path.exists():
        raise FileNotFoundError(f'HDF5 dataset not found: {args.h5_path}')

    tokens_path, layout, cache_meta = prepare_raw_token_cache(
        h5_path=args.h5_path,
        cache_dir=args.cache_dir,
        max_trajectories=args.max_trajectories,
    )
    raw_tokens = np.load(tokens_path, mmap_mode='r')
    total_samples = raw_tokens.shape[0]
    if args.max_samples is not None:
        total_samples = min(total_samples, int(args.max_samples))
    raw_tokens = np.asarray(raw_tokens[:total_samples], dtype=np.float32)

    if total_samples <= 1:
        raise RuntimeError('Need at least 2 entries to build train/val split.')
    val_size = min(max(1, int(args.val_size)), total_samples - 1)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(total_samples)
    val_indices = perm[:val_size]
    train_mask = np.ones(total_samples, dtype=bool)
    train_mask[val_indices] = False
    val_mask = ~train_mask

    train_raw = raw_tokens[train_mask]
    val_raw = raw_tokens[val_mask]
    stats = compute_stats(train_raw, layout)
    train_x = build_inpainting_x(train_raw, stats, layout)
    val_x = build_inpainting_x(val_raw, stats, layout)
    x_min = train_x.min(axis=0).astype(np.float32)
    x_max = train_x.max(axis=0).astype(np.float32)

    train_dataset = InpaintingDataset(train_x, train_raw[:, layout.length_slice].reshape(-1))
    val_dataset = InpaintingDataset(val_x, val_raw[:, layout.length_slice].reshape(-1))
    train_loader = create_loader(train_dataset, args.batch_size, weighted=True)
    val_loader = create_loader(val_dataset, args.eval_batch_size, weighted=False, shuffle=False)

    device = torch.device(args.device)
    model = create_model(device=device, x_min=x_min, x_max=x_max, diffusion_steps=args.diffusion_steps, q_dim=layout.q_dim)

    run_dir = args.workdir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = (args.wandb_mode != 'disabled') and (wandb is not None)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or args.run_name,
            dir=str(run_dir),
            mode=args.wandb_mode,
            config={
                'h5_path': str(args.h5_path),
                'device': str(device),
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'eval_batch_size': args.eval_batch_size,
                'val_size': args.val_size,
                'train_entries': int(len(train_dataset)),
                'val_entries': int(len(val_dataset)),
                'diffusion_steps': int(model.diffusion_steps),
                'q_dim': int(layout.q_dim),
                'token_dim': int(layout.token_dim),
            },
        )
    elif args.wandb_mode != 'disabled' and wandb is None:
        print('[Config] wandb is not installed; metrics will only be printed locally.', flush=True)

    metadata = {
        'h5_path': str(args.h5_path),
        'train_entries': int(len(train_dataset)),
        'val_entries': int(len(val_dataset)),
        'val_size': int(len(val_dataset)),
        'val_entry_indices': np.flatnonzero(val_mask).astype(np.int32).tolist(),
        'cache_meta': cache_meta,
    }

    print(f'[Config] device={device}', flush=True)
    print(
        f'[Config] train_entries={len(train_dataset)} | val_entries={len(val_dataset)} | '
        f'batch_size={args.batch_size} | diffusion_steps={model.diffusion_steps}',
        flush=True,
    )
    print(f'[Config] run_dir={run_dir}', flush=True)

    global_step = 0
    best_val = float('inf')
    started_at = time.time()

    demo_idx = int(np.clip(args.sample_entry_idx, 0, total_samples - 1))
    demo_condition = raw_tokens[demo_idx, layout.pos_slice.start:layout.normal_slice.stop]

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for batch in train_loader:
            global_step += 1
            x0 = batch['x0'].to(device)
            log = model.update(x0=x0)
            epoch_losses.append(float(log['loss']))
            grad_norm_value = None if log['grad_norm'] is None else float(log['grad_norm'])
            if use_wandb:
                wandb.log(
                    {
                        'train/loss_step': float(log['loss']),
                        'train/grad_norm_step': grad_norm_value,
                        'train/epoch': epoch,
                        'global_step': global_step,
                    },
                    step=global_step,
                )
            if global_step % args.log_interval == 0:
                print(
                    f'[Train] epoch={epoch:04d} step={global_step:07d} '
                    f'loss={log["loss"]:.6f} grad_norm={log["grad_norm"]}',
                    flush=True,
                )

        train_loss = float(np.mean(epoch_losses))
        print(f'[Epoch] epoch={epoch:04d} train_loss={train_loss:.6f}', flush=True)
        if use_wandb:
            wandb.log({'train/loss_epoch': train_loss, 'train/epoch': epoch, 'global_step': global_step}, step=global_step)

        if epoch % args.eval_interval == 0:
            val_loss = evaluate_model(model, val_loader, device)
            elapsed_min = (time.time() - started_at) / 60.0
            print(f'[Eval] epoch={epoch:04d} val_loss={val_loss:.6f} elapsed={elapsed_min:.1f} min', flush=True)
            if use_wandb:
                wandb.log(
                    {'eval/loss': val_loss, 'time/elapsed_min': elapsed_min, 'train/epoch': epoch, 'global_step': global_step},
                    step=global_step,
                )
            if val_loss < best_val:
                best_val = val_loss
                model.save(str(run_dir / 'model_best.pt'))
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
                print(f'[Eval] new_best_val={best_val:.6f}', flush=True)
                if use_wandb:
                    wandb.log({'eval/best_val_loss': best_val}, step=global_step)

        if epoch % args.save_interval == 0:
            demo_q, demo_len, _ = sample_q_length_from_condition(
                model=model,
                stats=stats,
                condition=demo_condition,
                device=device,
                q_dim=layout.q_dim,
                n_samples=args.n_sample_demo,
                sample_steps=args.diffusion_steps,
                temperature=1.0,
            )
            print(
                f'[SampleDemo] epoch={epoch:04d} entry_idx={demo_idx} '
                f'samples_q={np.array2string(demo_q, precision=4, separator=", ")} '
                f'samples_length={np.array2string(demo_len, precision=4, separator=", ")}',
                flush=True,
            )
            if use_wandb:
                wandb.log(
                    {
                        'sample/demo_mean_abs_q': float(np.mean(np.abs(demo_q))),
                        'sample/demo_std_q': float(np.std(demo_q)),
                        'sample/demo_mean_length': float(np.mean(demo_len)),
                        'sample/demo_std_length': float(np.std(demo_len)),
                    },
                    step=global_step,
                )
            save_bundle(run_dir, model, stats, x_min, x_max, args, metadata)

    save_bundle(run_dir, model, stats, x_min, x_max, args, metadata)
    print(f'[Done] best_val_loss={best_val:.6f} | run_dir={run_dir}', flush=True)
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
