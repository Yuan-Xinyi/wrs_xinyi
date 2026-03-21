import argparse
import json
import math
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

from cleandiffuser.diffusion.rectifiedflow import DiscreteRectifiedFlow
from cleandiffuser.nn_condition.mlp import MLPCondition
from cleandiffuser.nn_diffusion.dit import DiT1d


DEFAULT_H5_PATH = BASE_DIR / "datasets" / "xarm_trail1_large_scale_top10.h5"
DEFAULT_WORKDIR = BASE_DIR / "flow_matching_topstart_runs"
DEFAULT_RUN_NAME = "dit_rectifiedflow_q_from_posdir"


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


def decode_direction_batch(direction_name_array: np.ndarray) -> np.ndarray:
    out = []
    mapping = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    for value in direction_name_array:
        if isinstance(value, bytes):
            key = value.decode("ascii")
        else:
            key = str(value)
        out.append(mapping[key])
    return np.stack(out, axis=0)


def load_entries_from_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        done_mask = np.asarray(f["done_mask"][:], dtype=bool)
        valid_mask = np.asarray(f["top_valid_mask"][:], dtype=bool)
        start_q = np.asarray(f["top_start_q"][:], dtype=np.float32)
        start_pos = np.asarray(f["top_start_pos"][:], dtype=np.float32)
        line_length = np.asarray(f["top_line_length"][:], dtype=np.float32)
        if "top_direction_vec" in f:
            direction_vec = np.asarray(f["top_direction_vec"][:], dtype=np.float32)
        else:
            direction_vec = decode_direction_batch(np.asarray(f["top_direction_name"][:]))
        kernel_q = np.asarray(f["kernel_qs"][:], dtype=np.float32)

    entry_kernel_idx, entry_slot_idx = np.where(valid_mask & done_mask[:, None])
    if len(entry_kernel_idx) == 0:
        raise RuntimeError(f"No valid finished entries found in: {h5_path}")

    flat_start_q = start_q[entry_kernel_idx, entry_slot_idx]
    flat_start_pos = start_pos[entry_kernel_idx, entry_slot_idx]
    flat_direction_vec = direction_vec[entry_kernel_idx, entry_slot_idx]
    flat_line_length = line_length[entry_kernel_idx, entry_slot_idx]
    flat_condition = np.concatenate([flat_start_pos, flat_direction_vec], axis=1).astype(np.float32)
    flat_kernel_q = kernel_q[entry_kernel_idx]

    return {
        "kernel_idx": entry_kernel_idx.astype(np.int32),
        "slot_idx": entry_slot_idx.astype(np.int32),
        "start_q": flat_start_q.astype(np.float32),
        "start_pos": flat_start_pos.astype(np.float32),
        "direction_vec": flat_direction_vec.astype(np.float32),
        "condition": flat_condition,
        "line_length": flat_line_length.astype(np.float32),
        "kernel_q": flat_kernel_q.astype(np.float32),
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


def compute_stats(train_start_q: np.ndarray, train_condition: np.ndarray):
    q_mean = train_start_q.mean(axis=0).astype(np.float32)
    q_std = train_start_q.std(axis=0).astype(np.float32)
    q_std = np.maximum(q_std, 1e-6)

    pos_mean = train_condition[:, :3].mean(axis=0).astype(np.float32)
    pos_std = train_condition[:, :3].std(axis=0).astype(np.float32)
    pos_std = np.maximum(pos_std, 1e-6)

    return {
        "q_mean": q_mean,
        "q_std": q_std,
        "pos_mean": pos_mean,
        "pos_std": pos_std,
    }


def normalize_start_q(q: np.ndarray, stats: dict):
    return ((q - stats["q_mean"]) / stats["q_std"]).astype(np.float32)


def denormalize_start_q(q: np.ndarray, stats: dict):
    return (q * stats["q_std"] + stats["q_mean"]).astype(np.float32)


def normalize_condition(cond: np.ndarray, stats: dict):
    pos = (cond[:, :3] - stats["pos_mean"]) / stats["pos_std"]
    direction = cond[:, 3:]
    return np.concatenate([pos.astype(np.float32), direction.astype(np.float32)], axis=1)


class TopStartFlowDataset(Dataset):
    def __init__(self, x0: np.ndarray, condition: np.ndarray, line_length: np.ndarray):
        self.x0 = torch.from_numpy(x0).float().unsqueeze(1)
        self.condition = torch.from_numpy(condition).float()
        self.line_length = torch.from_numpy(line_length).float()

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, idx):
        return {
            "x0": self.x0[idx],
            "condition": self.condition[idx],
            "line_length": self.line_length[idx],
        }


def create_model(device: torch.device, x_min: np.ndarray, x_max: np.ndarray):
    cond_dim = 6
    emb_dim = 256
    nn_condition = MLPCondition(
        in_dim=cond_dim,
        out_dim=emb_dim,
        hidden_dims=[256, 256],
        dropout=0.1,
    )
    nn_diffusion = DiT1d(
        in_dim=6,
        emb_dim=emb_dim,
        d_model=384,
        n_heads=6,
        depth=8,
        dropout=0.0,
    )
    model = DiscreteRectifiedFlow(
        nn_diffusion=nn_diffusion,
        nn_condition=nn_condition,
        grad_clip_norm=1.0,
        ema_rate=0.999,
        optim_params={"lr": 2e-4, "weight_decay": 1e-4},
        diffusion_steps=1000,
        x_min=torch.tensor(x_min, dtype=torch.float32, device=device).view(1, 1, 6),
        x_max=torch.tensor(x_max, dtype=torch.float32, device=device).view(1, 1, 6),
        device=device,
    )
    return model


def create_train_loader(dataset: TopStartFlowDataset, batch_size: int):
    weights = dataset.line_length.numpy().copy()
    weights = weights / np.maximum(weights.mean(), 1e-6)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(weights),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False)


def create_eval_loader(dataset: TopStartFlowDataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)


@torch.no_grad()
def evaluate_model(model, val_loader, device: torch.device):
    model.eval()
    losses = []
    for batch in val_loader:
        x0 = batch["x0"].to(device)
        condition = batch["condition"].to(device)
        loss = model.loss(x0, condition=condition)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def sample_joint_angles(model, condition_np: np.ndarray, stats: dict, device: torch.device, n_samples: int, sample_steps: int):
    condition_norm = normalize_condition(condition_np[None, :].astype(np.float32), stats)
    condition_tensor = torch.from_numpy(condition_norm).float().to(device)
    condition_tensor = condition_tensor.repeat(n_samples, 1)
    prior = torch.zeros((n_samples, 1, 6), dtype=torch.float32, device=device)
    samples, _ = model.sample(
        prior=prior,
        n_samples=n_samples,
        sample_steps=sample_steps,
        sample_step_schedule="uniform",
        use_ema=True,
        condition_cfg=condition_tensor,
        w_cfg=1.0,
    )
    samples_np = samples.squeeze(1).detach().cpu().numpy()
    return denormalize_start_q(samples_np, stats)


def save_training_bundle(run_dir: Path, model, stats: dict, x_min: np.ndarray, x_max: np.ndarray, args, metadata: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / "model_latest.pt"))
    args_dict = to_jsonable(vars(args))
    metadata_jsonable = to_jsonable(metadata)
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
                "metadata": metadata_jsonable,
                "stats": {k: np.asarray(v).tolist() for k, v in stats.items()},
                "x_min": np.asarray(x_min).tolist(),
                "x_max": np.asarray(x_max).tolist(),
            },
            f,
            indent=2,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional flow matching model for top_start_q generation.")
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
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--n-sample-demo", type=int, default=4)
    parser.add_argument("--sample-kernel-idx", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="xarm-flow-matching")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.h5_path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {args.h5_path}")

    device = torch.device(args.device)
    entries = load_entries_from_h5(args.h5_path)
    train_mask, val_mask = split_random_entries(len(entries["start_q"]), args.val_size, args.seed)

    train_stats = compute_stats(entries["start_q"][train_mask], entries["condition"][train_mask])
    x_all = normalize_start_q(entries["start_q"], train_stats)
    cond_all = normalize_condition(entries["condition"], train_stats)

    x_min = x_all[train_mask].min(axis=0)
    x_max = x_all[train_mask].max(axis=0)

    train_dataset = TopStartFlowDataset(x_all[train_mask], cond_all[train_mask], entries["line_length"][train_mask])
    val_dataset = TopStartFlowDataset(x_all[val_mask], cond_all[val_mask], entries["line_length"][val_mask])
    train_loader = create_train_loader(train_dataset, args.batch_size)
    val_loader = create_eval_loader(val_dataset, args.eval_batch_size)

    run_dir = args.workdir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(device=device, x_min=x_min, x_max=x_max)
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
                "sample_steps": args.sample_steps,
                "train_entries": int(len(train_dataset)),
                "val_entries": int(len(val_dataset)),
                "done_kernels": int(entries["num_done_kernels"]),
                "total_kernels": int(entries["num_total_kernels"]),
            },
        )
    elif args.wandb_mode != "disabled" and wandb is None:
        print("[Config] wandb is not installed; metrics will only be printed locally.", flush=True)

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
    metadata = {
        "h5_path": str(args.h5_path),
        "train_entries": int(len(train_dataset)),
        "val_entries": int(len(val_dataset)),
        "done_kernels": int(entries["num_done_kernels"]),
        "total_kernels": int(entries["num_total_kernels"]),
        "val_size": int(len(val_dataset)),
        "val_entry_indices": np.flatnonzero(val_mask).astype(np.int32).tolist(),
    }

    sample_kernel_matches = np.where(entries["kernel_idx"] == args.sample_kernel_idx)[0]
    demo_condition = None
    if len(sample_kernel_matches) > 0:
        demo_condition = entries["condition"][sample_kernel_matches[0]]

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for batch in train_loader:
            global_step += 1
            x0 = batch["x0"].to(device)
            condition = batch["condition"].to(device)
            log = model.update(x0=x0, condition=condition)
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

        if (epoch % args.eval_interval) == 0:
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
                        "stats": {k: np.asarray(v, dtype=np.float32) for k, v in train_stats.items()},
                        "x_min": np.asarray(x_min, dtype=np.float32),
                        "x_max": np.asarray(x_max, dtype=np.float32),
                        "args": vars(args),
                        "metadata": metadata,
                        "best_val_loss": best_val,
                    },
                    run_dir / "bundle_best.pt",
                )
                print(f"[Eval] new_best_val={best_val:.6f}", flush=True)
                if use_wandb:
                    wandb.log(
                        {
                            "eval/best_val_loss": best_val,
                            "train/epoch": epoch,
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

        if demo_condition is not None and (epoch % args.save_interval) == 0:
            demo_q = sample_joint_angles(
                model=model,
                condition_np=demo_condition,
                stats=train_stats,
                device=device,
                n_samples=args.n_sample_demo,
                sample_steps=args.sample_steps,
            )
            print(
                f"[SampleDemo] epoch={epoch:04d} kernel_idx={args.sample_kernel_idx} "
                f"samples={np.array2string(demo_q, precision=4, separator=', ')}",
                flush=True,
            )
            if use_wandb:
                wandb.log(
                    {
                        "sample/demo_mean_abs_q": float(np.mean(np.abs(demo_q))),
                        "sample/demo_std_q": float(np.std(demo_q)),
                        "train/epoch": epoch,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

        if (epoch % args.save_interval) == 0:
            save_training_bundle(run_dir, model, train_stats, x_min, x_max, args, metadata)

    save_training_bundle(run_dir, model, train_stats, x_min, x_max, args, metadata)
    print(f"[Done] best_val_loss={best_val:.6f} | run_dir={run_dir}", flush=True)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
