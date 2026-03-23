import argparse
from pathlib import Path

import numpy as np
import torch

from train_diffusion_inpainting_topstart import create_model, denormalize_q, normalize_condition
from validate_topstart_common import (
    BASE_DIR,
    DEFAULT_H5_PATH,
    build_robot_and_contour,
    choose_validation_subset,
    create_timestamped_result_dir,
    direction_name_from_vec,
    evaluate_prediction_set,
    evaluate_q_with_ground_truth_method,
    load_flat_entries_from_h5,
    load_validation_entry_indices,
    save_metrics_json,
    save_plot,
    set_seed,
    summarize_results,
    torch_load,
    visualize_record_comparison,
)


DEFAULT_MODEL_DIR = BASE_DIR / "diffusion_inpainting_runs" / "ddpm32_dit_inpaint_q_from_posdir"
DEFAULT_RESULTS_DIR = BASE_DIR / "validation_diffusion_inpainting"


def load_model(model_dir: Path, device: torch.device):
    bundle_path = model_dir / "bundle_best.pt"
    model_path = model_dir / "model_best.pt"
    if not bundle_path.exists():
        bundle_path = model_dir / "bundle_latest.pt"
    if not model_path.exists():
        model_path = model_dir / "model_latest.pt"
    if not bundle_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Could not find trained bundle/model under: {model_dir}")

    payload = torch_load(bundle_path)
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in payload["stats"].items()}
    x_min = np.asarray(payload["x_min"], dtype=np.float32)
    x_max = np.asarray(payload["x_max"], dtype=np.float32)
    metadata = payload.get("metadata", {})
    train_args = payload.get("args", {})
    diffusion_steps = int(train_args.get("diffusion_steps", 32))
    model = create_model(device=device, x_min=x_min, x_max=x_max, diffusion_steps=diffusion_steps)
    model.load(str(model_path))
    model.eval()
    return model, stats, metadata, bundle_path, model_path


@torch.no_grad()
def infer_samples(model, stats, condition, device, n_samples, sample_steps=32):
    cond_norm = normalize_condition(condition[None, :].astype(np.float32), stats)[0]
    prior = np.zeros((n_samples, 1, 12), dtype=np.float32)
    prior[:, 0, 6:] = cond_norm[None, :]
    prior_t = torch.from_numpy(prior).float().to(device)
    samples, _ = model.sample(
        prior=prior_t,
        n_samples=n_samples,
        sample_steps=sample_steps,
        use_ema=True,
        temperature=1.0,
    )
    q_norm = samples[:, 0, :6].detach().cpu().numpy()
    return denormalize_q(q_norm, stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate diffusion inpainting top_start_q model against ground-truth trace_line_by_ik.")
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-val-samples", type=int, default=2000)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--visualize-rank", type=int, default=0)
    parser.add_argument("--visualize-only", action="store_true", help="Skip full evaluation and only visualize one random validation sample.")
    parser.add_argument("--random-visualize", action="store_true", help="Ignore seed in visualize-only mode and draw a different random validation sample each run.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    run_results_dir = create_timestamped_result_dir(args.results_dir, prefix="diffusion_inpainting_validation")

    device = torch.device(args.device)
    model, stats, _, bundle_path, model_path = load_model(args.model_dir, device)

    entries = load_flat_entries_from_h5(args.h5_path)
    val_entry_indices = load_validation_entry_indices(bundle_path)
    robot, contour = build_robot_and_contour()
    subset_seed = None if (args.visualize_only and args.random_visualize) else args.seed

    if args.visualize_only:
        chosen_indices = choose_validation_subset(val_entry_indices, 1, subset_seed)
    else:
        chosen_indices = choose_validation_subset(val_entry_indices, args.num_val_samples, subset_seed)

    records = []
    for rank, entry_idx in enumerate(chosen_indices):
        gt_q = entries["start_q"][entry_idx]
        start_pos = entries["start_pos"][entry_idx]
        direction_vec = entries["direction_vec"][entry_idx]
        condition = np.concatenate([start_pos, direction_vec], axis=0).astype(np.float32)
        predicted_qs = infer_samples(
            model=model,
            stats=stats,
            condition=condition,
            device=device,
            n_samples=args.n_candidates,
            sample_steps=args.sample_steps,
        )

        gt_result = evaluate_q_with_ground_truth_method(robot, contour, gt_q, direction_vec)
        pred_eval = evaluate_prediction_set(robot, contour, predicted_qs, direction_vec)
        records.append(
            {
                "rank": int(rank),
                "entry_idx": int(entry_idx),
                "kernel_idx": int(entries["kernel_idx"][entry_idx]),
                "slot_idx": int(entries["slot_idx"][entry_idx]),
                "direction_name": direction_name_from_vec(direction_vec),
                "start_pos": start_pos.tolist(),
                "gt_q": gt_q.tolist(),
                "top1_q": pred_eval["top1_q"].tolist(),
                "best8_q": pred_eval["best_q"].tolist(),
                "gt_length": float(gt_result["line_length"]),
                "top1_length": float(pred_eval["top1_result"]["line_length"]),
                "best8_length": float(pred_eval["best_result"]["line_length"]),
                "gt_reason": gt_result["termination_reason"],
                "gt_result": gt_result,
                "top1_result": pred_eval["top1_result"],
                "best8_result": pred_eval["best_result"],
            }
        )
        print(
            f"[ValidateDiffusion] {rank + 1}/{len(chosen_indices)} "
            f"kernel={entries['kernel_idx'][entry_idx]} slot={entries['slot_idx'][entry_idx]} "
            f"gt={gt_result['line_length']:.4f} "
            f"top1={pred_eval['top1_result']['line_length']:.4f} "
            f"best8={pred_eval['best_result']['line_length']:.4f}",
            flush=True,
        )

    if args.visualize_only:
        visualize_record_comparison(records[0], title="Diffusion Inpainting Validation Visualization")
        return

    metrics = summarize_results(records)
    fig_path = run_results_dir / "diffusion_inpainting_length_comparison.png"
    json_path = run_results_dir / "diffusion_inpainting_validation_metrics.json"
    save_plot(records, fig_path, title="Diffusion Inpainting vs Ground Truth")
    save_metrics_json(metrics, records, json_path)

    print(f"model_path: {model_path}")
    print(f"results_dir: {run_results_dir}")
    print(f"plot_path: {fig_path}")
    print(f"metrics_path: {json_path}")
    print(f"metrics: {metrics}")

    vis_rank = int(np.clip(args.visualize_rank, 0, len(records) - 1))
    visualize_record_comparison(records[vis_rank], title="Diffusion Inpainting Validation Visualization")


if __name__ == "__main__":
    main()
