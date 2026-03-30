import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from kinematic_diffusion_common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_H5_PATH,
    DEFAULT_RUN_NAME,
    DEFAULT_WORKDIR,
    JointDistributionMasker,
    MaskConditionedDiT,
    MaskedDiffusionModel,
    ResidualMLPDenoiser,
    StandardScaler,
    TokenMemmapDataset,
    prepare_token_cache,
    set_seed,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a CleanDiffuser-style DiT inpainting model for XArm kinematic tokens.")
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--resume-bundle", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--epochs", type=int, default=100, help="Total target epoch count. Resume will continue until this epoch.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=64)
    parser.add_argument("--backbone", type=str, default="dit", choices=["dit", "mlp"])
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--wandb-project", type=str, default="xarm-kinematic-dit")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser


def build_model(args, token_dim: int, device: torch.device) -> MaskedDiffusionModel:
    if args.backbone == "dit":
        denoiser = MaskConditionedDiT(token_dim=token_dim)
    else:
        denoiser = ResidualMLPDenoiser(token_dim=token_dim)
    model = MaskedDiffusionModel(
        denoiser=denoiser,
        token_dim=token_dim,
        diffusion_steps=args.diffusion_steps,
        predict_x0=True,
        device=device,
    )
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, masker, unknown_weight, device):
    model.eval()
    loss_list = []
    masked_mse_list = []
    for batch in loader:
        x0 = batch["token"].to(device)
        known_mask = masker.sample(x0.shape[0], device)
        loss, aux = model.training_loss(x0, known_mask, unknown_weight=unknown_weight)
        loss_list.append(float(loss.item()))
        masked_mse_list.append(float(aux["masked_mse"]))
    model.train()
    return float(np.mean(loss_list)), float(np.mean(masked_mse_list))


def save_bundle(run_dir: Path, model, optimizer, scaler: StandardScaler, layout, args, metadata, filename: str = "bundle_latest.pt"):
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
        "scaler_min": scaler.data_min,
        "scaler_max": scaler.data_max,
        "layout_q_dim": layout.q_dim,
        "layout_token_dim": layout.token_dim,
        "args": vars(args),
        "metadata": metadata,
    }
    torch.save(payload, run_dir / filename)
    if filename == "bundle_latest.pt":
        with open(run_dir / "metadata_latest.json", "w", encoding="utf-8") as f:
            json.dump(payload["metadata"], f, indent=2)


def maybe_init_wandb(args, run_dir: Path, config: dict):
    if args.wandb_mode == "disabled":
        return None
    if wandb is None:
        print("[warn] wandb is not installed. Falling back to disabled logging.")
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or args.run_name,
        dir=str(run_dir),
        mode=args.wandb_mode,
        config=config,
    )
    return run


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    tokens_path, scaler, layout, cache_meta = prepare_token_cache(
        h5_path=args.h5_path,
        cache_dir=args.cache_dir,
        max_trajectories=args.max_trajectories,
    )
    total_samples = int(cache_meta["num_samples"])
    if args.max_samples is not None:
        total_samples = min(total_samples, int(args.max_samples))
    indices = np.arange(total_samples, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    val_count = min(max(1, int(total_samples * args.val_ratio)), max(total_samples - 1, 1))
    val_indices = np.sort(indices[:val_count])
    train_indices = np.sort(indices[val_count:])

    train_dataset = TokenMemmapDataset(tokens_path=tokens_path, indices=train_indices)
    val_dataset = TokenMemmapDataset(tokens_path=tokens_path, indices=val_indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = build_model(args, token_dim=layout.token_dim, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    masker = JointDistributionMasker(layout)
    unknown_weight = torch.ones(layout.token_dim, dtype=torch.float32, device=device)
    unknown_weight[layout.q_slice] = 2.0
    unknown_weight[layout.rot_slice] = 2.0

    run_dir = args.workdir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_bundle = args.resume_bundle
    if resume_bundle is None:
        candidate = run_dir / "bundle_latest.pt"
        if candidate.exists():
            resume_bundle = candidate

    best_val = float("inf")
    global_step = 0
    start_epoch = 1
    if resume_bundle is not None and Path(resume_bundle).exists():
        resume_payload = torch.load(resume_bundle, map_location=device, weights_only=False)
        model.load_state_dict(resume_payload["model_state"])
        if "optimizer_state" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        resume_meta = dict(resume_payload.get("metadata", {}))
        start_epoch = int(resume_meta.get("epoch", 0)) + 1
        best_val = float(resume_meta.get("best_val_loss", best_val))
        global_step = int(resume_meta.get("global_step", 0))
        print(f"[resume] bundle={resume_bundle} start_epoch={start_epoch} global_step={global_step} best_val={best_val:.6f}")

    print(f"[info] token_dim={layout.token_dim} q_dim={layout.q_dim}")
    if layout.token_dim != 18:
        print(f"[warn] current XArm dataset yields token_dim={layout.token_dim}, not 18. This is expected for q_dim={layout.q_dim}.")
    print(f"[info] train_samples={len(train_dataset)} val_samples={len(val_dataset)} cache={tokens_path}")

    wandb_run = maybe_init_wandb(
        args,
        run_dir,
        config={
            "token_dim": layout.token_dim,
            "q_dim": layout.q_dim,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "cache_meta": cache_meta,
            **vars(args),
        },
    )
    if wandb_run is not None and global_step > 0:
        wandb_run.log({"resume/global_step": global_step, "resume/start_epoch": start_epoch}, step=global_step)

    if start_epoch > args.epochs:
        print(f"[done] resume checkpoint already reached epoch {start_epoch - 1}, target epochs={args.epochs}")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_losses = []
        train_masked = []
        for batch in train_loader:
            x0 = batch["token"].to(device)
            known_mask = masker.sample(x0.shape[0], device)
            loss, aux = model.training_loss(x0, known_mask, unknown_weight=unknown_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))
            train_masked.append(float(aux["masked_mse"]))
            global_step += 1
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss_step": float(loss.item()),
                        "train/masked_mse_step": float(aux["masked_mse"]),
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )
            if args.log_interval > 0 and global_step % args.log_interval == 0:
                mean_loss = float(np.mean(train_losses[-args.log_interval:]))
                mean_masked = float(np.mean(train_masked[-args.log_interval:]))
                print(
                    f"[train] epoch={epoch} step={global_step} "
                    f"loss={mean_loss:.6f} masked_mse={mean_masked:.6f}"
                )

        val_loss, val_masked = evaluate(model, val_loader, masker, unknown_weight, device)
        epoch_time = time.perf_counter() - epoch_start
        train_loss_epoch = float(np.mean(train_losses))
        train_masked_epoch = float(np.mean(train_masked))
        print(
            f"[epoch] {epoch}/{args.epochs} "
            f"train_loss={train_loss_epoch:.6f} train_masked={train_masked_epoch:.6f} "
            f"val_loss={val_loss:.6f} val_masked={val_masked:.6f} time={epoch_time:.2f}s"
        )

        metadata = {
            "epoch": epoch,
            "best_val_loss": min(best_val, val_loss),
            "train_loss": train_loss_epoch,
            "train_masked_mse": train_masked_epoch,
            "val_loss": val_loss,
            "val_masked_mse": val_masked,
            "global_step": global_step,
            "cache_meta": cache_meta,
        }

        save_bundle(run_dir, model, optimizer, scaler, layout, args, metadata, filename="bundle_latest.pt")

        if val_loss < best_val:
            best_val = val_loss
            save_bundle(run_dir, model, optimizer, scaler, layout, args, metadata, filename="bundle_best.pt")
            torch.save(model.state_dict(), run_dir / "model_best.pt")
        if epoch % args.save_interval == 0:
            save_bundle(run_dir, model, optimizer, scaler, layout, args, metadata, filename=f"bundle_epoch_{epoch:04d}.pt")
            torch.save(model.state_dict(), run_dir / f"model_epoch_{epoch:04d}.pt")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch/train_loss": train_loss_epoch,
                    "epoch/train_masked_mse": train_masked_epoch,
                    "epoch/val_loss": val_loss,
                    "epoch/val_masked_mse": val_masked,
                    "epoch/time_sec": epoch_time,
                    "epoch/index": epoch,
                    "epoch/best_val_loss": best_val,
                },
                step=global_step,
            )

    final_meta = {
        "epoch": args.epochs,
        "best_val_loss": best_val,
        "global_step": global_step,
        "cache_meta": cache_meta,
    }
    save_bundle(run_dir, model, optimizer, scaler, layout, args, final_meta, filename="bundle_latest.pt")
    save_bundle(run_dir, model, optimizer, scaler, layout, args, final_meta, filename="bundle_final.pt")
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    if wandb_run is not None:
        wandb_run.log({"final/best_val_loss": best_val}, step=global_step)
        wandb_run.finish()
    print(f"[done] run_dir={run_dir} best_val_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
