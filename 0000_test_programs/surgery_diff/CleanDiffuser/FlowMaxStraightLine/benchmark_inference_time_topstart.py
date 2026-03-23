import argparse
import time
from pathlib import Path

import numpy as np
import torch

from infer_flow_matching_topstart import infer_samples as infer_flow_samples
from infer_flow_matching_topstart import load_model as load_flow_model
from validate_diffusion_inpainting_topstart import infer_samples as infer_diffusion_samples
from validate_diffusion_inpainting_topstart import load_model as load_diffusion_model
from validate_topstart_common import (
    BASE_DIR,
    DEFAULT_H5_PATH,
    choose_validation_subset,
    load_flat_entries_from_h5,
    load_validation_entry_indices,
)


DEFAULT_FLOW_MODEL_DIR = BASE_DIR / "flow_matching_topstart_runs" / "dit_rectifiedflow_q_from_posdir"
DEFAULT_DIFFUSION_MODEL_DIR = BASE_DIR / "diffusion_inpainting_runs" / "ddpm32_dit_inpaint_q_from_posdir"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark inference time of flow matching vs diffusion inpainting on the same validation conditions."
    )
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--flow-model-dir", type=Path, default=DEFAULT_FLOW_MODEL_DIR)
    parser.add_argument("--diffusion-model-dir", type=Path, default=DEFAULT_DIFFUSION_MODEL_DIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-val-samples", type=int, default=100)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--flow-sample-steps", type=int, default=32)
    parser.add_argument("--diffusion-sample-steps", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260321)
    return parser.parse_args()


def maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_conditions(h5_path: Path, bundle_path: Path, num_val_samples: int, seed: int):
    entries = load_flat_entries_from_h5(h5_path)
    val_entry_indices = load_validation_entry_indices(bundle_path)
    chosen_indices = choose_validation_subset(val_entry_indices, num_val_samples, seed)
    conditions = []
    for entry_idx in chosen_indices:
        start_pos = entries["start_pos"][entry_idx]
        direction_vec = entries["direction_vec"][entry_idx]
        condition = np.concatenate([start_pos, direction_vec], axis=0).astype(np.float32)
        conditions.append(condition)
    return conditions, chosen_indices


def benchmark_flow(model, stats, conditions, device: torch.device, n_candidates: int, sample_steps: int, warmup_runs: int):
    warmup_conditions = conditions[: min(len(conditions), warmup_runs)]
    for condition in warmup_conditions:
        _ = infer_flow_samples(
            model=model,
            stats=stats,
            condition=condition,
            device=device,
            n_samples=n_candidates,
            sample_steps=sample_steps,
        )
    maybe_sync(device)

    timings = []
    for condition in conditions:
        maybe_sync(device)
        t0 = time.perf_counter()
        _ = infer_flow_samples(
            model=model,
            stats=stats,
            condition=condition,
            device=device,
            n_samples=n_candidates,
            sample_steps=sample_steps,
        )
        maybe_sync(device)
        timings.append(time.perf_counter() - t0)
    return np.asarray(timings, dtype=np.float64)


def benchmark_diffusion(
    model,
    stats,
    conditions,
    device: torch.device,
    n_candidates: int,
    sample_steps: int,
    warmup_runs: int,
):
    warmup_conditions = conditions[: min(len(conditions), warmup_runs)]
    for condition in warmup_conditions:
        _ = infer_diffusion_samples(
            model=model,
            stats=stats,
            condition=condition,
            device=device,
            n_samples=n_candidates,
            sample_steps=sample_steps,
        )
    maybe_sync(device)

    timings = []
    for condition in conditions:
        maybe_sync(device)
        t0 = time.perf_counter()
        _ = infer_diffusion_samples(
            model=model,
            stats=stats,
            condition=condition,
            device=device,
            n_samples=n_candidates,
            sample_steps=sample_steps,
        )
        maybe_sync(device)
        timings.append(time.perf_counter() - t0)
    return np.asarray(timings, dtype=np.float64)


def print_summary(name: str, timings: np.ndarray, num_conditions: int, n_candidates: int):
    total = float(timings.sum())
    mean = float(timings.mean())
    std = float(timings.std())
    median = float(np.median(timings))
    per_candidate = mean / max(n_candidates, 1)
    print(f"{name}:")
    print(f"  total_time_s={total:.4f}")
    print(f"  mean_time_per_condition_s={mean:.6f}")
    print(f"  median_time_per_condition_s={median:.6f}")
    print(f"  std_time_per_condition_s={std:.6f}")
    print(f"  mean_time_per_candidate_s={per_candidate:.6f}")
    print(f"  throughput_conditions_per_s={num_conditions / max(total, 1e-12):.4f}")
    print(f"  throughput_candidates_per_s={(num_conditions * n_candidates) / max(total, 1e-12):.4f}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    flow_model, flow_stats, _, flow_metadata, flow_model_path = load_flow_model(args.flow_model_dir, device)
    diffusion_model, diffusion_stats, diffusion_metadata, diffusion_bundle_path, diffusion_model_path = load_diffusion_model(
        args.diffusion_model_dir, device
    )

    flow_bundle_path = args.flow_model_dir / "bundle_best.pt"
    if not flow_bundle_path.exists():
        flow_bundle_path = args.flow_model_dir / "bundle_latest.pt"

    conditions, chosen_indices = build_conditions(
        h5_path=args.h5_path,
        bundle_path=flow_bundle_path,
        num_val_samples=args.num_val_samples,
        seed=args.seed,
    )

    print(f"device: {device}")
    print(f"h5_path: {args.h5_path}")
    print(f"num_conditions: {len(conditions)}")
    print(f"n_candidates: {args.n_candidates}")
    print(f"flow_sample_steps: {args.flow_sample_steps}")
    print(f"diffusion_sample_steps: {args.diffusion_sample_steps}")
    print(f"warmup_runs: {args.warmup_runs}")
    print(f"flow_model_path: {flow_model_path}")
    print(f"diffusion_model_path: {diffusion_model_path}")
    print(f"flow_val_entries: {flow_metadata.get('val_entries') if flow_metadata else 'n/a'}")
    print(f"diffusion_val_entries: {diffusion_metadata.get('val_entries') if diffusion_metadata else 'n/a'}")
    print(f"first_chosen_entry_idx: {int(chosen_indices[0]) if len(chosen_indices) > 0 else 'n/a'}")

    flow_timings = benchmark_flow(
        model=flow_model,
        stats=flow_stats,
        conditions=conditions,
        device=device,
        n_candidates=args.n_candidates,
        sample_steps=args.flow_sample_steps,
        warmup_runs=args.warmup_runs,
    )
    diffusion_timings = benchmark_diffusion(
        model=diffusion_model,
        stats=diffusion_stats,
        conditions=conditions,
        device=device,
        n_candidates=args.n_candidates,
        sample_steps=args.diffusion_sample_steps,
        warmup_runs=args.warmup_runs,
    )

    print()
    print_summary("flow_matching", flow_timings, len(conditions), args.n_candidates)
    print()
    print_summary("diffusion_inpainting", diffusion_timings, len(conditions), args.n_candidates)
    print()
    print("ratio:")
    print(f"  diffusion_over_flow_mean={float(diffusion_timings.mean() / max(flow_timings.mean(), 1e-12)):.4f}")
    print(f"  diffusion_over_flow_total={float(diffusion_timings.sum() / max(flow_timings.sum(), 1e-12)):.4f}")


if __name__ == "__main__":
    main()
