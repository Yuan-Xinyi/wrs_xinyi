#!/usr/bin/env python3
"""
Dataset-only analysis script for the GPU_nullspace_straight project.

Inputs:
- `--dataset`: an HDF5 trajectory dataset under `datasets/`, for example an FR3 trajectory file.

Outputs:
- A dataset analysis directory containing summary statistics and dataset figures.

Functions:
- Summarize the dataset size, joint-space coverage, TCP coverage, and trajectory-length statistics.
- Save one histogram for trajectory projected length.
- Save three TCP voxelized density plots on the XY / XZ / YZ planes using the same color metric.

Typical usage:
- `python analyze_and_plot.py`
- `python analyze_and_plot.py --dataset path/to/your_dataset.hdf5`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
DEFAULT_DATASET = DATASETS_DIR / "franka_research_3_gpu_trajectories_sub10.hdf5"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset-only analysis script for GPU_nullspace_straight.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--voxel-bins", type=int, default=120)
    return parser.parse_args()


def stats_dict(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset
    outdir = args.outdir

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    outdir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as f:
        trajectory_root = f["trajectories"]
        trajectory_keys = sorted(trajectory_root.keys())
        if args.max_trajectories is not None:
            trajectory_keys = trajectory_keys[: args.max_trajectories]
        if not trajectory_keys:
            raise RuntimeError(f"No trajectories found in {dataset_path}")

        all_q = []
        all_tcp_pos = []
        all_remaining_length = []
        all_direction = []
        all_normal = []
        trajectory_num_points = []
        trajectory_projected_length = []

        for key in trajectory_keys:
            group = trajectory_root[key]
            q = np.asarray(group["q"][:], dtype=np.float32)
            tcp_pos = np.asarray(group["tcp_pos"][:], dtype=np.float32)
            remaining_length = np.asarray(group["remaining_length"][:], dtype=np.float32).reshape(-1)
            direction = np.asarray(group.attrs["direction"], dtype=np.float32).reshape(1, 3)
            normal = np.asarray(group.attrs["target_normal"], dtype=np.float32).reshape(1, 3)

            if "total_projected_length" in group.attrs:
                total_projected_length = float(group.attrs["total_projected_length"])
            else:
                total_projected_length = float(np.asarray(group["progress_length"][-1], dtype=np.float32))

            all_q.append(q)
            all_tcp_pos.append(tcp_pos)
            all_remaining_length.append(remaining_length)
            all_direction.append(np.repeat(direction, q.shape[0], axis=0))
            all_normal.append(np.repeat(normal, q.shape[0], axis=0))
            trajectory_num_points.append(int(q.shape[0]))
            trajectory_projected_length.append(total_projected_length)

    q_all = np.concatenate(all_q, axis=0)
    tcp_pos_all = np.concatenate(all_tcp_pos, axis=0)
    remaining_length_all = np.concatenate(all_remaining_length, axis=0)
    direction_all = np.concatenate(all_direction, axis=0)
    normal_all = np.concatenate(all_normal, axis=0)
    trajectory_num_points = np.asarray(trajectory_num_points, dtype=np.int32)
    trajectory_projected_length = np.asarray(trajectory_projected_length, dtype=np.float32)
    direction_normal_dot = np.sum(direction_all * normal_all, axis=1)

    summary = {
        "dataset_path": str(dataset_path),
        "num_trajectories": int(len(trajectory_keys)),
        "num_samples": int(q_all.shape[0]),
        "q_dim": int(q_all.shape[1]),
        "trajectory_num_points": stats_dict(trajectory_num_points),
        "trajectory_projected_length": stats_dict(trajectory_projected_length),
        "remaining_length": stats_dict(remaining_length_all),
        "tcp_pos_x": stats_dict(tcp_pos_all[:, 0]),
        "tcp_pos_y": stats_dict(tcp_pos_all[:, 1]),
        "tcp_pos_z": stats_dict(tcp_pos_all[:, 2]),
        "direction_normal_dot": stats_dict(direction_normal_dot),
        "q_min": [float(v) for v in q_all.min(axis=0)],
        "q_max": [float(v) for v in q_all.max(axis=0)],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.hist(trajectory_projected_length, bins=400, color="#82B0D2", alpha=1)
    fig.tight_layout()
    fig.savefig(outdir / "hist_trajectory_projected_length.png")
    plt.close(fig)

    xy_x_min = float(np.floor(tcp_pos_all[:, 0].min() / 0.5) * 0.5)
    xy_x_max = float(np.ceil(tcp_pos_all[:, 0].max() / 0.5) * 0.5)
    xy_y_min = float(np.floor(tcp_pos_all[:, 1].min() / 0.5) * 0.5)
    xy_y_max = float(np.ceil(tcp_pos_all[:, 1].max() / 0.5) * 0.5)
    xz_x_min = float(np.floor(tcp_pos_all[:, 0].min() / 0.5) * 0.5)
    xz_x_max = float(np.ceil(tcp_pos_all[:, 0].max() / 0.5) * 0.5)
    xz_y_min = float(np.floor(tcp_pos_all[:, 2].min() / 0.5) * 0.5)
    xz_y_max = float(np.ceil(tcp_pos_all[:, 2].max() / 0.5) * 0.5)
    yz_x_min = float(np.floor(tcp_pos_all[:, 1].min() / 0.5) * 0.5)
    yz_x_max = float(np.ceil(tcp_pos_all[:, 1].max() / 0.5) * 0.5)
    yz_y_min = float(np.floor(tcp_pos_all[:, 2].min() / 0.5) * 0.5)
    yz_y_max = float(np.ceil(tcp_pos_all[:, 2].max() / 0.5) * 0.5)

    xy_x_edges = np.linspace(xy_x_min, xy_x_max, int(args.voxel_bins) + 1, dtype=np.float64)
    xy_y_edges = np.linspace(xy_y_min, xy_y_max, int(args.voxel_bins) + 1, dtype=np.float64)
    xz_x_edges = np.linspace(xz_x_min, xz_x_max, int(args.voxel_bins) + 1, dtype=np.float64)
    xz_y_edges = np.linspace(xz_y_min, xz_y_max, int(args.voxel_bins) + 1, dtype=np.float64)
    yz_x_edges = np.linspace(yz_x_min, yz_x_max, int(args.voxel_bins) + 1, dtype=np.float64)
    yz_y_edges = np.linspace(yz_y_min, yz_y_max, int(args.voxel_bins) + 1, dtype=np.float64)

    tcp_xy_count, _, _ = np.histogram2d(tcp_pos_all[:, 0], tcp_pos_all[:, 1], bins=[xy_x_edges, xy_y_edges])
    tcp_xz_count, _, _ = np.histogram2d(tcp_pos_all[:, 0], tcp_pos_all[:, 2], bins=[xz_x_edges, xz_y_edges])
    tcp_yz_count, _, _ = np.histogram2d(tcp_pos_all[:, 1], tcp_pos_all[:, 2], bins=[yz_x_edges, yz_y_edges])
    tcp_count_max = float(max(tcp_xy_count.max(), tcp_xz_count.max(), tcp_yz_count.max(), 1.0))
    tcp_cmap = "viridis"
    tcp_norm = mcolors.Normalize(vmin=0.0, vmax=tcp_count_max)
    xy_x_ticks = np.arange(xy_x_min, xy_x_max + 0.001, 0.5)
    xy_y_ticks = np.arange(xy_y_min, xy_y_max + 0.001, 0.5)
    xz_x_ticks = np.arange(xz_x_min, xz_x_max + 0.001, 0.5)
    xz_y_ticks = np.arange(xz_y_min, xz_y_max + 0.001, 0.5)
    yz_x_ticks = np.arange(yz_x_min, yz_x_max + 0.001, 0.5)
    yz_y_ticks = np.arange(yz_y_min, yz_y_max + 0.001, 0.5)
    tcp_xy_masked = np.ma.masked_where(tcp_xy_count.T <= 0.0, tcp_xy_count.T)
    tcp_xz_masked = np.ma.masked_where(tcp_xz_count.T <= 0.0, tcp_xz_count.T)
    tcp_yz_masked = np.ma.masked_where(tcp_yz_count.T <= 0.0, tcp_yz_count.T)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.pcolormesh(
        xy_x_edges,
        xy_y_edges,
        tcp_xy_masked,
        cmap=tcp_cmap,
        norm=tcp_norm,
        shading="auto",
    )
    ax.set_xlim(xy_x_min, xy_x_max)
    ax.set_ylim(xy_y_min, xy_y_max)
    ax.set_xticks(xy_x_ticks)
    ax.set_yticks(xy_y_ticks)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(outdir / "scatter_tcp_xy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.pcolormesh(
        xz_x_edges,
        xz_y_edges,
        tcp_xz_masked,
        cmap=tcp_cmap,
        norm=tcp_norm,
        shading="auto",
    )
    ax.set_xlim(xz_x_min, xz_x_max)
    ax.set_ylim(xz_y_min, xz_y_max)
    ax.set_xticks(xz_x_ticks)
    ax.set_yticks(xz_y_ticks)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(outdir / "scatter_tcp_xz.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.pcolormesh(
        yz_x_edges,
        yz_y_edges,
        tcp_yz_masked,
        cmap=tcp_cmap,
        norm=tcp_norm,
        shading="auto",
    )
    ax.set_xlim(yz_x_min, yz_x_max)
    ax.set_ylim(yz_y_min, yz_y_max)
    ax.set_xticks(yz_x_ticks)
    ax.set_yticks(yz_y_ticks)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(outdir / "scatter_tcp_yz.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 2), dpi=300)
    colorbar = mcolorbar.ColorbarBase(ax, cmap=plt.get_cmap(tcp_cmap), norm=tcp_norm, orientation="horizontal")
    colorbar.set_ticks(np.linspace(0.0, tcp_count_max, num=5))
    fig.tight_layout()
    fig.savefig(outdir / "scatter_tcp_colorbar.png")
    plt.close(fig)

    print(f"[saved][dataset] {outdir}")


if __name__ == "__main__":
    main()
