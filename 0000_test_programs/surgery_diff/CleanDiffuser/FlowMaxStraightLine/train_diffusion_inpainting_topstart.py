import argparse
import json
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import wandb
except ImportError:
    wandb = None


BASE_DIR = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine")
CLEANDIFFUSER_ROOT = BASE_DIR.parent
if str(CLEANDIFFUSER_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEANDIFFUSER_ROOT))

from cleandiffuser.diffusion.ddpm import DDPM
from cleandiffuser.nn_diffusion.dit import DiT1d


DEFAULT_H5_PATH = BASE_DIR / "datasets" / "xarm_trail1_large_scale_top10.h5"
DEFAULT_WORKDIR = BASE_DIR / "diffusion_inpainting_runs"
DEFAULT_RUN_NAME = "ddpm_dit_inpaint_q_from_posdir"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_entries_from_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        done_mask = np.asarray(f["done_mask"][:], dtype=bool)
        valid_mask = np.asarray(f["top_valid_mask"][:], dtype=bool)
        start_q = np.asarray(f["top_start_q"][:], dtype=np.float32)
        start_pos = np.asarray(f["top_start_pos"][:], dtype=np.float32)
        direction_vec = np.asarray(f["top_direction_vec"][:], dtype=np.float32)
        line_length = np.asarray(f["top_line_length"][:], dtype=np.float32)

    entry_kernel_idx, entry_slot_idx = np.where(valid_mask & done_mask[:, None])
    if len(entry_kernel_idx) == 0:
        raise RuntimeError(f"No valid finished entries found in: {h5_path}")

    flat_start_q = start_q[entry_kernel_idx, entry_slot_idx]
    flat_start_pos = start_pos[entry_kernel_idx, entry_slot_idx]
    flat_direction_vec = direction_vec[entry_kernel_idx, entry_slot_idx]
    flat_line_length = line_length[entry_kernel_idx, entry_slot_idx]
    flat_condition = np.concatenate([flat_start_pos, flat_direction_vec], axis=1).astype(np.float32)

    return {
        "start_q": flat_start_q.astype(np.float32),
        "condition": flat_condition.astype(np.float32),
        "line_length": flat_line_length.astype(np.float32),
        "kernel_idx": entry_kernel_idx.astype(np.int32),
        "slot_idx": entry_slot_idx.astype(np.int32),
        "num_done_kernels": int(done_mask.sum()),
        "num_total_kernels": int(done_mask.shape[0]),
    }


def split_random_entries(num_entries: int, val_size: int, seed: int):
    if num_entries <= 1:
        raise RuntimeError("Need at least 2 entries to build train/val split.")
    val_size = min(max(1, val_size), num_entries - 1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_entries)
    val_indices = perm[:val_size]
    train_mask = np.ones(num_entries, dtype=bool)
    train_mask[val_indices] = False
    val_mask = ~train_mask
    return train_mask, val_mask


def compute_stats(train_q: np.ndarray, train_condition: np.ndarray):
    q_mean = train_q.mean(axis=0).astype(np.float32)
    q_std = np.maximum(train_q.std(axis=0).astype(np.float32), 1e-6)
    pos_mean = train_condition[:, :3].mean(axis=0).astype(np.float32)
    pos_std = np.maximum(train_condition[:, :3].std(axis=0).astype(np.float32), 1e-6)
    return {
        "q_mean": q_mean,
        "q_std": q_std,
        "pos_mean": pos_mean,
        "pos_std": pos_std,
    }


def normalize_q(q: np.ndarray, stats: dict):
    return ((q - stats["q_mean"]) / stats["q_std"]).astype(np.float32)


def denormalize_q(q: np.ndarray, stats: dict):
    return (q * stats["q_std"] + stats["q_mean"]).astype(np.float32)


def normalize_condition(condition: np.ndarray, stats: dict):
    pos = (condition[:, :3] - stats["pos_mean"]) / stats["pos_std"]
    direction = condition[:, 3:]
    return np.concatenate([pos.astype(np.float32), direction.astype(np.float32)], axis=1)


def build_inpainting_x(q_norm: np.ndarray, cond_norm: np.ndarray):
    return np.concatenate([q_norm, cond_norm], axis=1).astype(np.float32)


class InpaintingDataset(Dataset):
    def __init__(self, x0: np.ndarray, line_length: np.ndarray):
        self.x0 = torch.from_numpy(x0).float().unsqueeze(1)
        self.line_length = torch.from_numpy(line_length).float()

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, idx):
        return {
            "x0": self.x0[idx],
            "line_length": self.line_length[idx],
        }


def create_loader(dataset: InpaintingDataset, batch_size: int, weighted: bool):
    if weighted:
        weights = dataset.line_length.numpy().copy()
        weights = weights / np.maximum(weights.mean(), 1e-6)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)


def create_model(device: torch.device, x_min: np.ndarray, x_max: np.ndarray):
    x_dim = 12
    nn_diffusion = DiT1d(
        in_dim=x_dim,
        emb_dim=256,
        d_model=384,
        n_heads=6,
        depth=8,
        dropout=0.0,
    )
    fix_mask = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=np.float32)
    model = DDPM(
        nn_diffusion=nn_diffusion,
        nn_condition=None,
        fix_mask=fix_mask,
        loss_weight=np.ones((1, x_dim), dtype=np.float32),
        grad_clip_norm=1.0,
        diffusion_steps=1000,
        ema_rate=0.999,
        optim_params={"lr": 2e-4, "weight_decay": 1e-4},
        x_min=torch.tensor(x_min, dtype=torch.float32, device=device).view(1, 1, x_dim),
        x_max=torch.tensor(x_max, dtype=torch.float32, device=device).view(1, 1, x_dim),
        predict_noise=True,
        device=device,
    )
    return model


@torch.no_grad()
def evaluate_model(model, val_loader, device: torch.device):
    model.eval()
    losses = []
    for batch in val_loader:
        x0 = batch["x0"].to(device)
        loss = model.loss(x0)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def sample_q_from_condition(model, cond_norm: np.ndarray, stats: dict, device: torch.device, n_samples: int):
    prior = np.zeros((n_samples, 1, 12), dtype=np.float32)
    prior[:, 0, 6:] = cond_norm[None, :]
    prior_t = torch.from_numpy(prior).float().to(device)
    samples, _ = model.sample(
        prior=prior_t,
        n_samples=n_samples,
        sample_steps=model.diffusion_steps,
        use_ema=True,
        temperature=1.0,
    )
    q_norm = samples[:, 0, :6].detach().cpu().numpy()
    return denormalize_q(q_norm, stats)


def save_bundle(run_dir: Path, model, stats: dict, x_min: np.ndarray, x_max: np.ndarray, args, metadata: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / "model_latest.pt"))
    args_dict = to_jsonable(vars(args))
    metadata_dict = to_jsonable(metadata)
    payload = {
        "stats": {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()},
        "x_min": np.asarray(x_min, dtype=np.float32),
        "x_max": np.asarray(x_max, dtype=np.float32),
        "args": args_dict,
        "metadata": metadata,
    }
    torch.save(payload, run_dir / "bundle_latest.pt")
    with open(run_dir / "metadata_latest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": args_dict,
                "metadata": metadata_dict,
                "stats": {k: np.asarray(v).tolist() for k, v in stats.items()},
                "x_min": np.asarray(x_min).tolist(),
                "x_max": np.asarray(x_max).tolist(),
            },
            f,
            indent=2,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM inpainting baseline for top_start_q generation.")
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--n-sample-demo", type=int, default=4)
    parser.add_argument("--sample-entry-idx", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="xarm-diffusion-inpainting")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.h5_path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {args.h5_path}")

    entries = load_entries_from_h5(args.h5_path)
    train_mask, val_mask = split_random_entries(len(entries["start_q"]), args.val_size, args.seed)

    stats = compute_stats(entries["start_q"][train_mask], entries["condition"][train_mask])
    q_norm = normalize_q(entries["start_q"], stats)
    cond_norm = normalize_condition(entries["condition"], stats)
    x_all = build_inpainting_x(q_norm, cond_norm)
    x_min = x_all[train_mask].min(axis=0)
    x_max = x_all[train_mask].max(axis=0)

    train_dataset = InpaintingDataset(x_all[train_mask], entries["line_length"][train_mask])
    val_dataset = InpaintingDataset(x_all[val_mask], entries["line_length"][val_mask])
    train_loader = create_loader(train_dataset, args.batch_size, weighted=True)
    val_loader = create_loader(val_dataset, args.eval_batch_size, weighted=False)

    device = torch.device(args.device)
    model = create_model(device=device, x_min=x_min, x_max=x_max)

    run_dir = args.workdir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = (args.wandb_mode != "disabled") and (wandb is not None)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or args.run_name,
            dir=str(run_dir),
            mode=args.wandb_mode,
            config={
                "h5_path": str(args.h5_path),
                "device": str(device),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "val_size": args.val_size,
                "train_entries": int(len(train_dataset)),
                "val_entries": int(len(val_dataset)),
                "diffusion_steps": int(model.diffusion_steps),
            },
        )
    elif args.wandb_mode != "disabled" and wandb is None:
        print("[Config] wandb is not installed; metrics will only be printed locally.", flush=True)

    metadata = {
        "h5_path": str(args.h5_path),
        "train_entries": int(len(train_dataset)),
        "val_entries": int(len(val_dataset)),
        "done_kernels": int(entries["num_done_kernels"]),
        "total_kernels": int(entries["num_total_kernels"]),
        "val_size": int(len(val_dataset)),
        "val_entry_indices": np.flatnonzero(val_mask).astype(np.int32).tolist(),
    }

    print(f"[Config] device={device}", flush=True)
    print(
        f"[Config] done_kernels={entries['num_done_kernels']}/{entries['num_total_kernels']} | "
        f"entries={len(entries['start_q'])}",
        flush=True,
    )
    print(
        f"[Config] train_entries={len(train_dataset)} | val_entries={len(val_dataset)} | "
        f"batch_size={args.batch_size}",
        flush=True,
    )
    print(f"[Config] run_dir={run_dir}", flush=True)

    global_step = 0
    best_val = float("inf")
    started_at = time.time()

    demo_idx = int(np.clip(args.sample_entry_idx, 0, len(cond_norm) - 1))
    demo_cond_norm = cond_norm[demo_idx]

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for batch in train_loader:
            global_step += 1
            x0 = batch["x0"].to(device)
            log = model.update(x0=x0)
            epoch_losses.append(float(log["loss"]))
            grad_norm_value = None if log["grad_norm"] is None else float(log["grad_norm"])
            if use_wandb:
                wandb.log(
                    {
                        "train/loss_step": float(log["loss"]),
                        "train/grad_norm_step": grad_norm_value,
                        "train/epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )
            if global_step % args.log_interval == 0:
                print(
                    f"[Train] epoch={epoch:04d} step={global_step:07d} "
                    f"loss={log['loss']:.6f} grad_norm={log['grad_norm']}",
                    flush=True,
                )

        train_loss = float(np.mean(epoch_losses))
        print(f"[Epoch] epoch={epoch:04d} train_loss={train_loss:.6f}", flush=True)
        if use_wandb:
            wandb.log(
                {
                    "train/loss_epoch": train_loss,
                    "train/epoch": epoch,
                    "global_step": global_step,
                },
                step=global_step,
            )

        if epoch % args.eval_interval == 0:
            val_loss = evaluate_model(model, val_loader, device)
            elapsed_min = (time.time() - started_at) / 60.0
            print(
                f"[Eval] epoch={epoch:04d} val_loss={val_loss:.6f} elapsed={elapsed_min:.1f} min",
                flush=True,
            )
            if use_wandb:
                wandb.log(
                    {
                        "eval/loss": val_loss,
                        "time/elapsed_min": elapsed_min,
                        "train/epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )
            if val_loss < best_val:
                best_val = val_loss
                model.save(str(run_dir / "model_best.pt"))
                torch.save(
                    {
                        "stats": {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()},
                        "x_min": np.asarray(x_min, dtype=np.float32),
                        "x_max": np.asarray(x_max, dtype=np.float32),
                        "args": to_jsonable(vars(args)),
                        "metadata": metadata,
                        "best_val_loss": best_val,
                    },
                    run_dir / "bundle_best.pt",
                )
                print(f"[Eval] new_best_val={best_val:.6f}", flush=True)
                if use_wandb:
                    wandb.log({"eval/best_val_loss": best_val}, step=global_step)

        if epoch % args.save_interval == 0:
            demo_q = sample_q_from_condition(
                model=model,
                cond_norm=demo_cond_norm,
                stats=stats,
                device=device,
                n_samples=args.n_sample_demo,
            )
            print(
                f"[SampleDemo] epoch={epoch:04d} entry_idx={demo_idx} "
                f"samples={np.array2string(demo_q, precision=4, separator=', ')}",
                flush=True,
            )
            if use_wandb:
                wandb.log(
                    {
                        "sample/demo_mean_abs_q": float(np.mean(np.abs(demo_q))),
                        "sample/demo_std_q": float(np.std(demo_q)),
                    },
                    step=global_step,
                )
            save_bundle(run_dir, model, stats, x_min, x_max, args, metadata)

    save_bundle(run_dir, model, stats, x_min, x_max, args, metadata)
    print(f"[Done] best_val_loss={best_val:.6f} | run_dir={run_dir}", flush=True)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
