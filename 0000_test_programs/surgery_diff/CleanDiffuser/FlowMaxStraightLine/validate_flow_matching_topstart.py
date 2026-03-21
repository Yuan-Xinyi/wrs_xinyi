import argparse
from pathlib import Path

import numpy as np
import torch

from infer_flow_matching_topstart import infer_samples, load_model
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
    visualize_record_comparison,
)


DEFAULT_MODEL_DIR = BASE_DIR / "flow_matching_topstart_runs" / "dit_rectifiedflow_q_from_posdir"
DEFAULT_RESULTS_DIR = BASE_DIR / "validation_flow_matching"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate flow matching top_start_q model against ground-truth trace_line_by_ik.")
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-val-samples", type=int, default=2000)
    parser.add_argument("--n-candidates", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260321)
    parser.add_argument("--visualize-rank", type=int, default=0, help="Which validated sample to open in robot visualization.")
    parser.add_argument("--visualize-only", action="store_true", help="Skip full evaluation and only visualize one random validation sample.")
    parser.add_argument("--random-visualize", action="store_true", help="Ignore seed in visualize-only mode and draw a different random validation sample each run.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    run_results_dir = create_timestamped_result_dir(args.results_dir, prefix="flow_matching_validation")

    device = torch.device(args.device)
    model, stats, _, _, model_path = load_model(args.model_dir, device)
    bundle_path = args.model_dir / "bundle_best.pt"
    if not bundle_path.exists():
        bundle_path = args.model_dir / "bundle_latest.pt"

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
            f"[ValidateFlow] {rank + 1}/{len(chosen_indices)} "
            f"kernel={entries['kernel_idx'][entry_idx]} slot={entries['slot_idx'][entry_idx]} "
            f"gt={gt_result['line_length']:.4f} "
            f"top1={pred_eval['top1_result']['line_length']:.4f} "
            f"best8={pred_eval['best_result']['line_length']:.4f}",
            flush=True,
        )

    if args.visualize_only:
        visualize_record_comparison(records[0], title="Flow Matching Validation Visualization")
        return

    metrics = summarize_results(records)
    fig_path = run_results_dir / "flow_matching_length_comparison.png"
    json_path = run_results_dir / "flow_matching_validation_metrics.json"
    save_plot(records, fig_path, title="Flow Matching vs Ground Truth")
    save_metrics_json(metrics, records, json_path)

    print(f"model_path: {model_path}")
    print(f"results_dir: {run_results_dir}")
    print(f"plot_path: {fig_path}")
    print(f"metrics_path: {json_path}")
    print(f"metrics: {metrics}")

    vis_rank = int(np.clip(args.visualize_rank, 0, len(records) - 1))
    visualize_record_comparison(records[vis_rank], title="Flow Matching Validation Visualization")


if __name__ == "__main__":
    main()
