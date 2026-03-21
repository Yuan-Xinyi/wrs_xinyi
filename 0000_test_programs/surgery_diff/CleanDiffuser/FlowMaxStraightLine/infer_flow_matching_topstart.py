import argparse
import sys
from pathlib import Path

import numpy as np
import torch


BASE_DIR = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine")
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from train_flow_matching_topstart import (
    DEFAULT_RUN_NAME,
    DEFAULT_WORKDIR,
    create_model,
    denormalize_start_q,
    normalize_condition,
)


DEFAULT_MODEL_DIR = DEFAULT_WORKDIR / DEFAULT_RUN_NAME


def parse_direction(direction: str):
    direction = direction.lower()
    mapping = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    if direction not in mapping:
        raise ValueError(f"Unsupported direction={direction!r}, expected one of x/y/z.")
    return mapping[direction]


def load_bundle(bundle_path: Path):
    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in payload["stats"].items()}
    x_min = np.asarray(payload["x_min"], dtype=np.float32)
    x_max = np.asarray(payload["x_max"], dtype=np.float32)
    args = payload.get("args", {})
    return stats, x_min, x_max, args, payload.get("metadata", {})


def load_model(model_dir: Path, device: torch.device):
    bundle_path = model_dir / "bundle_best.pt"
    model_path = model_dir / "model_best.pt"
    if not bundle_path.exists():
        bundle_path = model_dir / "bundle_latest.pt"
    if not model_path.exists():
        model_path = model_dir / "model_latest.pt"
    if not bundle_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Could not find trained bundle/model under: {model_dir}")

    stats, x_min, x_max, train_args, metadata = load_bundle(bundle_path)
    model = create_model(device=device, x_min=x_min, x_max=x_max)
    model.load(str(model_path))
    model.eval()
    return model, stats, train_args, metadata, model_path


@torch.no_grad()
def infer_samples(model, stats, condition, device, n_samples, sample_steps):
    condition_norm = normalize_condition(condition[None, :].astype(np.float32), stats)
    condition_tensor = torch.from_numpy(condition_norm).float().to(device).repeat(n_samples, 1)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Infer start joint angles from start_pos + direction using a trained flow matching model.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--direction", type=str, required=True, help="One of x/y/z.")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, stats, train_args, metadata, model_path = load_model(args.model_dir, device)

    start_pos = np.array([args.x, args.y, args.z], dtype=np.float32)
    direction_vec = parse_direction(args.direction)
    condition = np.concatenate([start_pos, direction_vec], axis=0).astype(np.float32)

    samples = infer_samples(
        model=model,
        stats=stats,
        condition=condition,
        device=device,
        n_samples=args.n_samples,
        sample_steps=args.sample_steps,
    )

    print(f"model_path: {model_path}")
    print(f"device: {device}")
    print(f"condition_start_pos: {np.array2string(start_pos, precision=4, separator=', ')}")
    print(f"condition_direction: {args.direction.lower()} -> {np.array2string(direction_vec, precision=4, separator=', ')}")
    if metadata:
        print(f"train_entries: {metadata.get('train_entries')}")
        print(f"val_entries: {metadata.get('val_entries')}")
    print("predicted_start_q:")
    for idx, q in enumerate(samples):
        print(f"  sample_{idx:02d}: {np.array2string(q, precision=5, separator=', ')}")


if __name__ == "__main__":
    main()
