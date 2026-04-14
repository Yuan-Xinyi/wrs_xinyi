#!/usr/bin/env python3
"""
Re-rollout the pair stored in `selected_pair.json` and compare it against the
previous rollout stored in `selected_pair_with_paths.json`.

Inputs:
- `--pair-json`: contains the task condition and the two initial joint angles.
- `--old-rollout-json`: contains the previous rollout trajectories.

What this script does:
- Rebuilds the GPU tracker with the current code.
- Re-rolls out the short and long initial joint configurations from
  `selected_pair.json`.
- Compares the new trajectories against the previous trajectories stored in
  `selected_pair_with_paths.json`.

Outputs:
- `new_selected_pair_with_paths.json`
- `comparison_summary.json`
- `compare_short_joints.png`
- `compare_long_joints.png`
- `compare_long_short_overlay.png`
- `compare_short_joint_limit_gain_sweep.png`
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

from length_prediction.lnet_contrastive_fr3_rotation_cone_eval import build_tracker


DEFAULT_PAIR_JSON = CURRENT_DIR / "outputs" / "intro_pair" / "selected_pair.json"
DEFAULT_OLD_JSON = CURRENT_DIR / "outputs" / "intro_pair" / "selected_pair_with_paths.json"
DEFAULT_OUTDIR = CURRENT_DIR / "outputs" / "intro_pair" / "compare_rollout"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-rollout selected_pair.json and compare with the old saved rollout.")
    parser.add_argument("--pair-json", type=Path, default=DEFAULT_PAIR_JSON)
    parser.add_argument("--old-rollout-json", type=Path, default=DEFAULT_OLD_JSON)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--short-joint-limit-gains", type=str, default="0.0,0.1,0.2,0.4,0.8,1.0,1.2,1.5,2.0")
    return parser.parse_args()


def compare_one(name: str, old: dict, new: dict) -> dict:
    old_q = np.asarray(old["q"], dtype=np.float32)
    new_q = np.asarray(new["q"], dtype=np.float32)
    old_tcp = np.asarray(old["tcp_pos"], dtype=np.float32)
    new_tcp = np.asarray(new["tcp_pos"], dtype=np.float32)

    min_len = min(old_q.shape[0], new_q.shape[0])
    if min_len > 0:
        q_rmse = float(np.sqrt(np.mean((old_q[:min_len] - new_q[:min_len]) ** 2)))
        tcp_rmse = float(np.sqrt(np.mean((old_tcp[:min_len] - new_tcp[:min_len]) ** 2)))
    else:
        q_rmse = None
        tcp_rmse = None

    return {
        "name": name,
        "old_num_points": int(old_q.shape[0]),
        "new_num_points": int(new_q.shape[0]),
        "old_total_projected_length": float(old["total_projected_length"]),
        "new_total_projected_length": float(new["total_projected_length"]),
        "delta_projected_length": float(new["total_projected_length"] - old["total_projected_length"]),
        "old_termination_reason": str(old["termination_reason"]),
        "new_termination_reason": str(new["termination_reason"]),
        "q_rmse_on_overlap": q_rmse,
        "tcp_rmse_on_overlap": tcp_rmse,
    }


def save_joint_plot(old_q: np.ndarray, new_q: np.ndarray, path: Path, title: str) -> None:
    fig, axes = plt.subplots(7, 1, figsize=(10, 14), dpi=180, sharex=False)
    for j in range(7):
        ax = axes[j]
        ax.plot(np.arange(old_q.shape[0]), old_q[:, j], color="#d95f02", linewidth=2.0, label="old")
        ax.plot(np.arange(new_q.shape[0]), new_q[:, j], color="#1b9e77", linewidth=2.0, label="new")
        ax.set_ylabel(f"q{j + 1}")
        ax.grid(alpha=0.2)
        if j == 0:
            ax.legend()
    axes[-1].set_xlabel("step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_long_short_overlay_plot(
    old_short_q: np.ndarray,
    new_short_q: np.ndarray,
    old_long_q: np.ndarray,
    new_long_q: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(7, 1, figsize=(10, 14), dpi=180, sharex=False)
    for j in range(7):
        ax = axes[j]
        ax.plot(
            np.arange(old_short_q.shape[0]),
            old_short_q[:, j],
            color="#d95f02",
            linewidth=2.0,
            linestyle="-",
            label="short-old" if j == 0 else None,
        )
        ax.plot(
            np.arange(new_short_q.shape[0]),
            new_short_q[:, j],
            color="#1b9e77",
            linewidth=2.0,
            linestyle="-",
            label="short-new" if j == 0 else None,
        )
        ax.plot(
            np.arange(old_long_q.shape[0]),
            old_long_q[:, j],
            color="#d95f02",
            linewidth=2.0,
            linestyle="--",
            label="long-old" if j == 0 else None,
        )
        ax.plot(
            np.arange(new_long_q.shape[0]),
            new_long_q[:, j],
            color="#1b9e77",
            linewidth=2.0,
            linestyle="--",
            label="long-new" if j == 0 else None,
        )
        ax.set_ylabel(f"q{j + 1}")
        ax.grid(alpha=0.2)
        if j == 0:
            ax.legend(ncol=2)
    axes[-1].set_xlabel("step")
    fig.suptitle("Joint Comparison: Long (dashed) vs Short (solid)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_short_gain_sweep_plot(
    old_short_q: np.ndarray,
    old_long_q: np.ndarray,
    new_long_q: np.ndarray,
    sweep_results: list[dict],
    path: Path,
) -> None:
    fig, axes = plt.subplots(7, 1, figsize=(10, 14), dpi=180, sharex=False)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(x) for x in np.linspace(0.15, 0.9, max(1, len(sweep_results)))]
    for j in range(7):
        ax = axes[j]
        ax.plot(
            np.arange(old_short_q.shape[0]),
            old_short_q[:, j],
            color="#d95f02",
            linewidth=2.2,
            linestyle="-",
            label="old-short" if j == 0 else None,
        )
        ax.plot(
            np.arange(old_long_q.shape[0]),
            old_long_q[:, j],
            color="#d95f02",
            linewidth=2.0,
            linestyle="--",
            label="old-long" if j == 0 else None,
        )
        ax.plot(
            np.arange(new_long_q.shape[0]),
            new_long_q[:, j],
            color="#1b9e77",
            linewidth=2.0,
            linestyle="--",
            label="new-long" if j == 0 else None,
        )
        for color, item in zip(colors, sweep_results):
            q = np.asarray(item["q"], dtype=np.float32)
            ax.plot(
                np.arange(q.shape[0]),
                q[:, j],
                color=color,
                linewidth=1.8,
                linestyle="-",
                label=f"gain={item['joint_limit_gain']:.2f}" if j == 0 else None,
            )
        ax.set_ylabel(f"q{j + 1}")
        ax.grid(alpha=0.2)
        if j == 0:
            ax.legend(ncol=3, fontsize=9)
    axes[-1].set_xlabel("step")
    fig.suptitle("Short Trajectory Sweep over Joint-Limit Gain")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    pair_data = json.loads(args.pair_json.read_text(encoding="utf-8"))
    old_data = json.loads(args.old_rollout_json.read_text(encoding="utf-8"))

    pos = np.asarray(pair_data["task"]["pos"], dtype=np.float32)
    direction = np.asarray(pair_data["task"]["direction"], dtype=np.float32)
    normal = np.asarray(pair_data["task"]["normal"], dtype=np.float32)
    short_q0 = np.asarray(pair_data["short_solution"]["q"], dtype=np.float32)
    long_q0 = np.asarray(pair_data["long_solution"]["q"], dtype=np.float32)

    device = torch.device(args.device)
    tracker, tracker_device = build_tracker(device)

    q0_batch = np.stack([short_q0, long_q0], axis=0).astype(np.float32)
    direction_batch = np.repeat(direction[None, :], 2, axis=0).astype(np.float32)
    normal_batch = np.repeat(normal[None, :], 2, axis=0).astype(np.float32)

    trajectories = tracker.collect_batch_trajectories(
        q0_batch=torch.from_numpy(q0_batch).to(tracker_device),
        direction_batch=torch.from_numpy(direction_batch).to(tracker_device),
        target_normal_batch=torch.from_numpy(normal_batch).to(tracker_device),
    )
    new_short = trajectories[0]
    new_long = trajectories[1]

    new_data = {
        "case_idx": int(pair_data["case_idx"]),
        "task": pair_data["task"],
        "pair_metrics": pair_data["pair_metrics"],
        "short_solution": pair_data["short_solution"],
        "long_solution": pair_data["long_solution"],
        "short_trajectory": {
            "num_points": int(new_short["num_points"]),
            "termination_code": int(new_short["termination_code"]),
            "termination_reason": str(new_short["termination_reason"]),
            "total_projected_length": float(new_short["total_projected_length"]),
            "q": np.asarray(new_short["q"], dtype=np.float32).tolist(),
            "tcp_pos": np.asarray(new_short["tcp_pos"], dtype=np.float32).tolist(),
            "progress_length": np.asarray(new_short["progress_length"], dtype=np.float32).tolist(),
            "remaining_length": np.asarray(new_short["remaining_length"], dtype=np.float32).tolist(),
            "pos_error": np.asarray(new_short["pos_error"], dtype=np.float32).tolist(),
        },
        "long_trajectory": {
            "num_points": int(new_long["num_points"]),
            "termination_code": int(new_long["termination_code"]),
            "termination_reason": str(new_long["termination_reason"]),
            "total_projected_length": float(new_long["total_projected_length"]),
            "q": np.asarray(new_long["q"], dtype=np.float32).tolist(),
            "tcp_pos": np.asarray(new_long["tcp_pos"], dtype=np.float32).tolist(),
            "progress_length": np.asarray(new_long["progress_length"], dtype=np.float32).tolist(),
            "remaining_length": np.asarray(new_long["remaining_length"], dtype=np.float32).tolist(),
            "pos_error": np.asarray(new_long["pos_error"], dtype=np.float32).tolist(),
        },
    }
    (args.outdir / "new_selected_pair_with_paths.json").write_text(json.dumps(new_data, indent=2), encoding="utf-8")

    short_summary = compare_one("short", old_data["short_trajectory"], new_data["short_trajectory"])
    long_summary = compare_one("long", old_data["long_trajectory"], new_data["long_trajectory"])
    summary = {
        "case_idx": int(pair_data["case_idx"]),
        "short": short_summary,
        "long": long_summary,
    }
    (args.outdir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_joint_plot(
        np.asarray(old_data["short_trajectory"]["q"], dtype=np.float32),
        np.asarray(new_data["short_trajectory"]["q"], dtype=np.float32),
        args.outdir / "compare_short_joints.png",
        "Short Trajectory Joint Comparison",
    )
    save_joint_plot(
        np.asarray(old_data["long_trajectory"]["q"], dtype=np.float32),
        np.asarray(new_data["long_trajectory"]["q"], dtype=np.float32),
        args.outdir / "compare_long_joints.png",
        "Long Trajectory Joint Comparison",
    )
    save_long_short_overlay_plot(
        np.asarray(old_data["short_trajectory"]["q"], dtype=np.float32),
        np.asarray(new_data["short_trajectory"]["q"], dtype=np.float32),
        np.asarray(old_data["long_trajectory"]["q"], dtype=np.float32),
        np.asarray(new_data["long_trajectory"]["q"], dtype=np.float32),
        args.outdir / "compare_long_short_overlay.png",
    )

    sweep_gains = [float(x.strip()) for x in args.short_joint_limit_gains.split(",") if x.strip()]
    short_sweep_results = []
    for gain in sweep_gains:
        tracker_sweep, tracker_sweep_device = build_tracker(device)
        tracker_sweep.config.joint_limit_gain = float(gain)
        short_traj = tracker_sweep.collect_batch_trajectories(
            q0_batch=torch.from_numpy(short_q0[None, :].astype(np.float32)).to(tracker_sweep_device),
            direction_batch=torch.from_numpy(direction[None, :].astype(np.float32)).to(tracker_sweep_device),
            target_normal_batch=torch.from_numpy(normal[None, :].astype(np.float32)).to(tracker_sweep_device),
        )[0]
        short_sweep_results.append(
            {
                "joint_limit_gain": float(gain),
                "num_points": int(short_traj["num_points"]),
                "termination_reason": str(short_traj["termination_reason"]),
                "total_projected_length": float(short_traj["total_projected_length"]),
                "q": np.asarray(short_traj["q"], dtype=np.float32).tolist(),
            }
        )

    save_short_gain_sweep_plot(
        np.asarray(old_data["short_trajectory"]["q"], dtype=np.float32),
        np.asarray(old_data["long_trajectory"]["q"], dtype=np.float32),
        np.asarray(new_data["long_trajectory"]["q"], dtype=np.float32),
        short_sweep_results,
        args.outdir / "compare_short_joint_limit_gain_sweep.png",
    )
    summary["short_joint_limit_gain_sweep"] = short_sweep_results
    (args.outdir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[saved] {args.outdir}")


if __name__ == "__main__":
    main()
