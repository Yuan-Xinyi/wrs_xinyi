#!/usr/bin/env python3
"""
Compare joint-space rollout curves of multiple candidate solutions under the
same task condition.

Inputs:
- A task condition selected from the batch log by `--case-idx`.
- Candidate configurations are generated with the same functions used in
  `lnet_contrastive_fr3_rotation_cone_eval.py`.

What this script does:
- Samples many feasible candidate joint configurations under one fixed
  `(pos, direction, normal)`.
- Rolls out every candidate with the current tracker.
- Selects several representative candidates spanning short to long travel
  lengths.
- Plots their 7-joint rollout curves together so you can inspect whether the
  joint-space manifold looks continuous or split.

Outputs:
- `candidate_length_scatter_gain_*.png`
- `candidate_length_hist_gain_*.png`
- `candidate_length_hist_overlay.png`
- `candidate_q_pca_gain_*.png`
- `candidate_joint_curves_gain_*.png`
- `summary.json`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
LENGTH_PREDICTION_DIR = ROOT_DIR / "length_prediction"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(LENGTH_PREDICTION_DIR) not in sys.path:
    sys.path.insert(0, str(LENGTH_PREDICTION_DIR))

from length_prediction.lnet_contrastive_fr3_rotation_cone_eval import build_tracker, collect_candidate_qs
from length_prediction.lnet_contrastive_fr3_rotation_cone_eval_batch_analysis import DEFAULT_INPUT, parse_log


DEFAULT_OUTDIR = CURRENT_DIR / "outputs" / "candidate_joint_trajectories"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rollout joint curves of same-task candidate solutions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--case-idx", type=int, default=151)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--oversample", type=int, default=1024)
    parser.add_argument("--num-compare", type=int, default=6)
    parser.add_argument("--joint-limit-gain", type=float, default=None)
    parser.add_argument("--joint-limit-gains", type=str, default=None, help="Comma-separated gain list, e.g. 0.0,0.2,0.5")
    parser.add_argument("--pos-tol-mm", type=float, default=2.0)
    parser.add_argument("--correction-iters", type=int, default=50)
    parser.add_argument("--correction-tol", type=float, default=1e-4)
    parser.add_argument("--correction-damping", type=float, default=1e-3)
    return parser.parse_args()


def find_task_row(rows: list[dict], case_idx: int) -> dict:
    rows = [row for row in rows if "pos" in row and "direction" in row and "normal" in row]
    for row in rows:
        if int(row["case_idx"]) == int(case_idx):
            return row
    raise ValueError(f"case_idx={case_idx} not found in log.")


def pick_representative_indices(real_len_np: np.ndarray, num_compare: int) -> list[int]:
    order = np.argsort(real_len_np)
    if len(order) <= num_compare:
        return [int(i) for i in order]
    grid = np.linspace(0, len(order) - 1, num_compare)
    chosen = sorted({int(order[int(round(g))]) for g in grid})
    return chosen


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:2].T
    proj = xc @ basis
    return proj.astype(np.float32)


def gain_suffix(gain: float) -> str:
    return f"{gain:.3f}".replace("-", "m").replace(".", "p")


def density_curve(x: np.ndarray, num_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    if x.size == 1:
        grid = np.linspace(x[0] - 1e-3, x[0] + 1e-3, num_points, dtype=np.float64)
        density = np.zeros_like(grid)
        density[len(density) // 2] = 1.0
        return grid.astype(np.float32), density.astype(np.float32)

    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    if sigma <= 1e-8:
        sigma = max(std, 1e-3)
    bandwidth = 0.9 * sigma * (x.size ** (-1.0 / 5.0))
    bandwidth = max(bandwidth, 1e-3)

    xmin = float(x.min())
    xmax = float(x.max())
    pad = max(3.0 * bandwidth, 1e-3)
    grid = np.linspace(xmin - pad, xmax + pad, num_points, dtype=np.float64)
    diff = (grid[:, None] - x[None, :]) / bandwidth
    density = np.exp(-0.5 * diff ** 2).sum(axis=1) / (x.size * bandwidth * np.sqrt(2.0 * np.pi))
    return grid.astype(np.float32), density.astype(np.float32)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = parse_log(args.input)
    row = find_task_row(rows, int(args.case_idx))

    pos = np.asarray(row["pos"], dtype=np.float32)
    direction = np.asarray(row["direction"], dtype=np.float32)
    normal = np.asarray(row["normal"], dtype=np.float32)

    device = torch.device(args.device)
    tracker, tracker_device = build_tracker(device)

    q_batch_np, raw_err_np = collect_candidate_qs(
        tracker,
        tracker_device,
        pos,
        direction,
        normal,
        int(args.num_candidates),
        int(args.oversample),
        float(args.pos_tol_mm),
        int(args.correction_iters),
        float(args.correction_tol),
        float(args.correction_damping),
    )

    gains: list[float]
    if args.joint_limit_gains:
        gains = [float(x.strip()) for x in args.joint_limit_gains.split(",") if x.strip()]
    elif args.joint_limit_gain is not None:
        gains = [float(args.joint_limit_gain)]
    else:
        gains = [float(tracker.config.joint_limit_gain)]

    q_pca_np = pca_2d(q_batch_np)
    gain_summaries = []
    overlay_hist_data = []

    for gain in gains:
        tracker_gain, tracker_gain_device = build_tracker(device)
        tracker_gain.config.joint_limit_gain = float(gain)
        trajectories = tracker_gain.collect_batch_trajectories(
            q0_batch=torch.from_numpy(q_batch_np.astype(np.float32)).to(tracker_gain_device),
            direction_batch=torch.from_numpy(np.repeat(direction[None, :], q_batch_np.shape[0], axis=0).astype(np.float32)).to(tracker_gain_device),
            target_normal_batch=torch.from_numpy(np.repeat(normal[None, :], q_batch_np.shape[0], axis=0).astype(np.float32)).to(tracker_gain_device),
        )

        real_len_np = np.asarray([traj["total_projected_length"] for traj in trajectories], dtype=np.float32)
        num_points_np = np.asarray([traj["num_points"] for traj in trajectories], dtype=np.int32)
        termination_reason = [str(traj["termination_reason"]) for traj in trajectories]
        selected_indices = pick_representative_indices(real_len_np, int(args.num_compare))
        suffix = gain_suffix(gain)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
        ax.scatter(np.arange(len(real_len_np)), real_len_np, color="#4c78a8", s=8, alpha=0.75)
        ax.scatter(
            np.asarray(selected_indices, dtype=np.int32),
            real_len_np[selected_indices],
            color="#e45756",
            s=24,
            label="selected",
        )
        ax.set_xlabel("candidate index")
        ax.set_ylabel("real rollout length")
        ax.set_title(f"Candidate Travel Ability (gain={gain:.3f})")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.outdir / f"candidate_length_scatter_gain_{suffix}.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
        x_grid, y_density = density_curve(real_len_np)
        ax.plot(x_grid, y_density, color="#4c78a8", linewidth=2.5, label="density")
        ax.fill_between(x_grid, y_density, color="#4c78a8", alpha=0.20)
        ax.axvline(float(real_len_np.mean()), color="#e45756", linestyle="--", linewidth=1.5, label="mean")
        ax.set_xlabel("real rollout length")
        ax.set_ylabel("density")
        ax.set_title(f"Length Distribution (gain={gain:.3f})")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.outdir / f"candidate_length_hist_gain_{suffix}.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
        sc = ax.scatter(
            q_pca_np[:, 0],
            q_pca_np[:, 1],
            c=real_len_np,
            cmap="viridis",
            s=8,
            alpha=0.85,
        )
        ax.scatter(
            q_pca_np[selected_indices, 0],
            q_pca_np[selected_indices, 1],
            facecolors="none",
            edgecolors="red",
            s=28,
            linewidths=1.5,
            label="selected",
        )
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        ax.set_title(f"Candidate Joint Configurations in PCA Space (gain={gain:.3f})")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.colorbar(sc, ax=ax, label="real rollout length")
        fig.tight_layout()
        fig.savefig(args.outdir / f"candidate_q_pca_gain_{suffix}.png")
        plt.close(fig)

        fig, axes = plt.subplots(7, 1, figsize=(11, 15), dpi=220, sharex=False)
        distinct_colors = [
            "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
            "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
        ]
        colors = [distinct_colors[i % len(distinct_colors)] for i in range(len(selected_indices))]
        for color, idx in zip(colors, selected_indices):
            q_path = np.asarray(trajectories[idx]["q"], dtype=np.float32)
            x = np.arange(q_path.shape[0], dtype=np.int32)
            label = f"idx={idx} len={real_len_np[idx]:.3f} n={q_path.shape[0]}"
            for j in range(7):
                axes[j].plot(x, q_path[:, j], color=color, linewidth=2.0, label=label if j == 0 else None)
                axes[j].set_ylabel(f"q{j + 1}")
                axes[j].grid(alpha=0.2)
        axes[0].legend(fontsize=8, ncol=2)
        axes[-1].set_xlabel("rollout step")
        fig.suptitle(f"Representative Candidate Joint Trajectories (gain={gain:.3f})")
        fig.tight_layout()
        fig.savefig(args.outdir / f"candidate_joint_curves_gain_{suffix}.png")
        plt.close(fig)

        gain_summaries.append({
            "joint_limit_gain": float(gain),
            "selected_candidates": [
                {
                    "idx": int(idx),
                    "rollout_length": float(real_len_np[idx]),
                    "num_points": int(num_points_np[idx]),
                    "termination_reason": termination_reason[idx],
                }
                for idx in selected_indices
            ],
            "real_rollout_length": {
                "min": float(real_len_np.min()),
                "max": float(real_len_np.max()),
                "mean": float(real_len_np.mean()),
                "std": float(real_len_np.std()),
            },
            "saved_files": [
                f"candidate_length_scatter_gain_{suffix}.png",
                f"candidate_length_hist_gain_{suffix}.png",
                f"candidate_q_pca_gain_{suffix}.png",
                f"candidate_joint_curves_gain_{suffix}.png",
            ],
        })
        overlay_hist_data.append((float(gain), real_len_np.copy()))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(1, len(overlay_hist_data))))
    for color, (gain, real_len_np) in zip(colors, overlay_hist_data):
        x_grid, y_density = density_curve(real_len_np)
        ax.plot(x_grid, y_density, linewidth=2.2, color=color, label=f"gain={gain:.3f}")
    ax.set_xlabel("real rollout length")
    ax.set_ylabel("density")
    ax.set_title("Length Distribution Across Joint-Limit Gains")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "candidate_length_hist_overlay.png")
    plt.close(fig)

    summary = {
        "case_idx": int(args.case_idx),
        "task": {
            "pos": pos.tolist(),
            "direction": direction.tolist(),
            "normal": normal.tolist(),
        },
        "num_candidates": int(q_batch_np.shape[0]),
        "raw_err_mm": {
            "mean": float(raw_err_np.mean() * 1000.0),
            "max": float(raw_err_np.max() * 1000.0),
        },
        "joint_limit_gains": gains,
        "gain_results": gain_summaries,
        "shared_saved_files": [
            "candidate_length_hist_overlay.png",
        ],
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[saved] {args.outdir}")


if __name__ == "__main__":
    main()
