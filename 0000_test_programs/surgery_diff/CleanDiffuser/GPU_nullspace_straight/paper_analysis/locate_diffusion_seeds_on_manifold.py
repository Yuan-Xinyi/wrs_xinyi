#!/usr/bin/env python3
"""
Locate diffusion-produced seeds on the same candidate manifold used in the
paper-analysis candidate trajectory experiment.

What this script does:
- Fix one task condition `(pos, direction, normal)` from the batch log.
- Sample many feasible candidates with the current tracker.
- Roll out those candidates to obtain their real travel lengths.
- Build a 2D PCA manifold from the candidate joint configurations.
- Sample both plain diffusion seeds and contrastive-guided diffusion seeds for
  the same task condition.
- Project those seeds into the same PCA space.
- Cluster the candidate manifold and assign each seed to its nearest manifold
  cluster, so you can see which branch / submanifold the seed falls in.

Outputs:
- `candidate_seed_pca_overlay.png`
- `candidate_seed_pca_arrows.png`
- `candidate_seed_cluster_boxplot.png`
- `candidate_seed_cluster_count.png`
- `candidate_seed_length_scatter.png`
- `candidate_seed_joint_curves.png`
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
DIFFUSION_DIR = ROOT_DIR / "diffusion_inpainting"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(LENGTH_PREDICTION_DIR) not in sys.path:
    sys.path.insert(0, str(LENGTH_PREDICTION_DIR))
if str(DIFFUSION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_DIR))

from length_prediction.lnet_contrastive_fr3_rotation_cone_eval import build_tracker, collect_candidate_qs
from length_prediction.lnet_contrastive_fr3_rotation_cone_eval_batch_analysis import DEFAULT_INPUT, parse_log
from diffusion_train import FRANKA_DEFAULT_RUN_NAME
from diffusion import DEFAULT_WORKDIR
from diffusion_sample import load_model
from diffusion_eval_batch_candidates_lnet import batch_position_error_and_correction, load_lnet_contrastive_model
from diffusion_eval_contrastive_guidance import sample_with_guidance


DEFAULT_OUTDIR = CURRENT_DIR / "outputs" / "diffusion_seed_manifold"
DEFAULT_BUNDLE = DEFAULT_WORKDIR / FRANKA_DEFAULT_RUN_NAME / "bundle_latest.pt"
DEFAULT_LNET_CONTRASTIVE_CKPT = ROOT_DIR / "runs" / "lnet_contrastive_runs" / "lnet_contrastive_q_cond_to_length_fr3_sub10_pref" / "lnet_contrastive_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate diffusion seeds on the candidate manifold.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--lnet-contrastive-ckpt", type=Path, default=DEFAULT_LNET_CONTRASTIVE_CKPT)
    parser.add_argument("--case-idx", type=int, default=151)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-candidates", type=int, default=512)
    parser.add_argument("--oversample", type=int, default=4096)
    parser.add_argument("--n-seeds", type=int, default=32)
    parser.add_argument("--num-clusters", type=int, default=6)
    parser.add_argument("--guided-lambda", type=float, default=5.0)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--joint-limit-gain", type=float, default=0.0)
    parser.add_argument("--fixed-guidance-step", type=float, default=1.0)
    parser.add_argument("--guidance-grad-eps", type=float, default=1e-6)
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


def pca_fit_transform(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:2].T
    proj = xc @ basis
    return proj.astype(np.float32), mean.reshape(-1).astype(np.float32), basis.astype(np.float32)


def pca_transform(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x - mean[None, :]) @ basis).astype(np.float32)


def kmeans_np(x: np.ndarray, k: int, max_iters: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(x.shape[0], size=k, replace=False)
    centers = x[init_idx].copy()
    labels = np.zeros(x.shape[0], dtype=np.int32)

    for _ in range(max_iters):
        dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist2, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = x[mask].mean(axis=0)
    return labels, centers.astype(np.float32)


def assign_clusters(seed_pca: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dist2 = ((seed_pca[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(dist2, axis=1).astype(np.int32)


def pick_representative_indices(lengths: np.ndarray, max_items: int = 4) -> list[int]:
    order = np.argsort(lengths)
    if len(order) <= max_items:
        return [int(i) for i in order]
    grid = np.linspace(0, len(order) - 1, max_items)
    chosen = sorted({int(order[int(round(g))]) for g in grid})
    return chosen


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
    tracker.config.joint_limit_gain = float(args.joint_limit_gain)

    q_candidates, raw_err_np = collect_candidate_qs(
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

    candidate_trajectories = tracker.collect_batch_trajectories(
        q0_batch=torch.from_numpy(q_candidates.astype(np.float32)).to(tracker_device),
        direction_batch=torch.from_numpy(np.repeat(direction[None, :], q_candidates.shape[0], axis=0).astype(np.float32)).to(tracker_device),
        target_normal_batch=torch.from_numpy(np.repeat(normal[None, :], q_candidates.shape[0], axis=0).astype(np.float32)).to(tracker_device),
    )
    candidate_real_len = np.asarray([traj["total_projected_length"] for traj in candidate_trajectories], dtype=np.float32)

    q_pca, q_mean, q_basis = pca_fit_transform(q_candidates)
    labels, centers = kmeans_np(q_pca, int(args.num_clusters), seed=int(args.case_idx))

    _, stats, model, q_dim, diffusion_steps = load_model(args.bundle, device)
    lnet_contrastive = load_lnet_contrastive_model(args.lnet_contrastive_ckpt, device)
    condition = np.concatenate([pos, direction, normal], axis=0).astype(np.float32)
    sample_steps = int(args.sample_steps) if args.sample_steps is not None else int(diffusion_steps)

    x_dim = q_dim + 10
    cond_norm = np.concatenate([
        ((pos - np.asarray(stats["pos_mean"], dtype=np.float32)) / np.asarray(stats["pos_std"], dtype=np.float32)).astype(np.float32),
        direction.astype(np.float32),
        normal.astype(np.float32),
    ], axis=0)
    plain_q_list = []
    plain_pred_len_list = []
    guided_q_list = []
    guided_pred_len_list = []
    for _ in range(int(args.n_seeds)):
        prior_np = np.zeros((1, 1, x_dim), dtype=np.float32)
        prior_np[:, 0, q_dim:q_dim + 9] = cond_norm[None, :]
        prior = torch.from_numpy(prior_np).float().to(device)
        init_noise = torch.randn_like(prior)
        plain_result = sample_with_guidance(
            model=model,
            lnet_contrastive=lnet_contrastive,
            stats=stats,
            q_dim=q_dim,
            condition_raw_np=condition,
            prior=prior,
            init_noise=init_noise.clone(),
            sample_steps=sample_steps,
            temperature=float(args.temperature),
            lambda_guidance=0.0,
            device=device,
            fixed_guidance_step=float(args.fixed_guidance_step),
            guidance_grad_eps=float(args.guidance_grad_eps),
        )
        guided_result = sample_with_guidance(
            model=model,
            lnet_contrastive=lnet_contrastive,
            stats=stats,
            q_dim=q_dim,
            condition_raw_np=condition,
            prior=prior,
            init_noise=init_noise,
            sample_steps=sample_steps,
            temperature=float(args.temperature),
            lambda_guidance=float(args.guided_lambda),
            device=device,
            fixed_guidance_step=float(args.fixed_guidance_step),
            guidance_grad_eps=float(args.guidance_grad_eps),
        )
        plain_q_list.append(np.asarray(plain_result["final_q"], dtype=np.float32))
        plain_pred_len_list.append(float(plain_result["final_pred_length"]))
        guided_q_list.append(np.asarray(guided_result["final_q"], dtype=np.float32))
        guided_pred_len_list.append(float(guided_result["final_pred_length"]))

    q_seed = np.stack(plain_q_list, axis=0).astype(np.float32)
    pred_len_seed = np.asarray(plain_pred_len_list, dtype=np.float32)
    guided_q_seed = np.stack(guided_q_list, axis=0).astype(np.float32)
    guided_pred_len_seed = np.asarray(guided_pred_len_list, dtype=np.float32)

    plain_q_corr, plain_raw_pos_err = batch_position_error_and_correction(
        tracker.robot,
        q_seed.astype(np.float32),
        pos,
        float(args.correction_damping),
        int(args.correction_iters),
        float(args.correction_tol),
        tracker_device,
    )
    guided_q_corr, guided_raw_pos_err = batch_position_error_and_correction(
        tracker.robot,
        guided_q_seed.astype(np.float32),
        pos,
        float(args.correction_damping),
        int(args.correction_iters),
        float(args.correction_tol),
        tracker_device,
    )

    plain_seed_pca = pca_transform(plain_q_corr, q_mean, q_basis)
    guided_seed_pca = pca_transform(guided_q_corr, q_mean, q_basis)
    plain_seed_trajectories = tracker.collect_batch_trajectories(
        q0_batch=torch.from_numpy(plain_q_corr.astype(np.float32)).to(tracker_device),
        direction_batch=torch.from_numpy(np.repeat(direction[None, :], q_seed.shape[0], axis=0).astype(np.float32)).to(tracker_device),
        target_normal_batch=torch.from_numpy(np.repeat(normal[None, :], q_seed.shape[0], axis=0).astype(np.float32)).to(tracker_device),
    )
    plain_seed_real_len = np.asarray([traj["total_projected_length"] for traj in plain_seed_trajectories], dtype=np.float32)
    guided_seed_trajectories = tracker.collect_batch_trajectories(
        q0_batch=torch.from_numpy(guided_q_corr.astype(np.float32)).to(tracker_device),
        direction_batch=torch.from_numpy(np.repeat(direction[None, :], guided_q_seed.shape[0], axis=0).astype(np.float32)).to(tracker_device),
        target_normal_batch=torch.from_numpy(np.repeat(normal[None, :], guided_q_seed.shape[0], axis=0).astype(np.float32)).to(tracker_device),
    )
    guided_seed_real_len = np.asarray([traj["total_projected_length"] for traj in guided_seed_trajectories], dtype=np.float32)
    q_delta_norm = np.linalg.norm(guided_q_corr - plain_q_corr, axis=1).astype(np.float32)
    pca_delta_norm = np.linalg.norm(guided_seed_pca - plain_seed_pca, axis=1).astype(np.float32)
    pred_len_delta = (guided_pred_len_seed - pred_len_seed).astype(np.float32)
    real_len_delta = (guided_seed_real_len - plain_seed_real_len).astype(np.float32)

    plain_seed_cluster = assign_clusters(plain_seed_pca, centers)
    guided_seed_cluster = assign_clusters(guided_seed_pca, centers)

    cluster_stats = []
    for cid in range(int(args.num_clusters)):
        mask = labels == cid
        if not np.any(mask):
            continue
        cluster_stats.append(
            {
                "cluster_id": cid,
                "num_candidates": int(mask.sum()),
                "mean_rollout_length": float(candidate_real_len[mask].mean()),
                "std_rollout_length": float(candidate_real_len[mask].std()),
                "min_rollout_length": float(candidate_real_len[mask].min()),
                "max_rollout_length": float(candidate_real_len[mask].max()),
            }
        )

    seed_cluster_histogram = []
    for cid in range(int(args.num_clusters)):
        plain_seed_count = int((plain_seed_cluster == cid).sum())
        guided_seed_count = int((guided_seed_cluster == cid).sum())
        if plain_seed_count == 0 and guided_seed_count == 0 and not np.any(labels == cid):
            continue
        seed_cluster_histogram.append(
            {
                "cluster_id": cid,
                "num_plain_seeds": plain_seed_count,
                "num_guided_seeds": guided_seed_count,
            }
        )

    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    sc = ax.scatter(q_pca[:, 0], q_pca[:, 1], c=candidate_real_len, cmap="viridis", s=8, alpha=0.8)
    ax.scatter(plain_seed_pca[:, 0], plain_seed_pca[:, 1], c="#e45756", s=55, marker="*", edgecolors="black", linewidths=0.7, label="plain seeds")
    ax.scatter(guided_seed_pca[:, 0], guided_seed_pca[:, 1], c="#4c78a8", s=40, marker="X", edgecolors="white", linewidths=0.5, label="guided seeds")
    ax.scatter(centers[:, 0], centers[:, 1], c="white", s=70, marker="x", linewidths=2.0, label="cluster centers")
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.set_title("Plain vs Guided Diffusion Seeds on Candidate Manifold")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.colorbar(sc, ax=ax, label="candidate real rollout length")
    fig.tight_layout()
    fig.savefig(args.outdir / "candidate_seed_pca_overlay.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=220)
    sc = ax.scatter(q_pca[:, 0], q_pca[:, 1], c=candidate_real_len, cmap="viridis", s=8, alpha=0.35)
    for idx in range(len(q_seed)):
        ax.plot(
            [plain_seed_pca[idx, 0], guided_seed_pca[idx, 0]],
            [plain_seed_pca[idx, 1], guided_seed_pca[idx, 1]],
            color="#8c8c8c",
            linewidth=0.9,
            alpha=0.7,
        )
    ax.scatter(plain_seed_pca[:, 0], plain_seed_pca[:, 1], c="#e45756", s=42, marker="*", edgecolors="black", linewidths=0.5, label="plain")
    ax.scatter(guided_seed_pca[:, 0], guided_seed_pca[:, 1], c="#4c78a8", s=28, marker="X", edgecolors="white", linewidths=0.4, label="guided")
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.set_title("Paired Plain-to-Guided Seed Shift in PCA Space")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.colorbar(sc, ax=ax, label="candidate real rollout length")
    fig.tight_layout()
    fig.savefig(args.outdir / "candidate_seed_pca_arrows.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=220)
    valid_cluster_ids = [item["cluster_id"] for item in cluster_stats]
    box_data = [candidate_real_len[labels == cid] for cid in valid_cluster_ids]
    ax.boxplot(box_data, positions=np.arange(len(valid_cluster_ids)), widths=0.6)
    for i, cid in enumerate(valid_cluster_ids):
        plain_mask = plain_seed_cluster == cid
        guided_mask = guided_seed_cluster == cid
        if np.any(plain_mask):
            x = np.full(int(plain_mask.sum()), i, dtype=np.float32)
            jitter = np.linspace(-0.14, -0.02, int(plain_mask.sum()), dtype=np.float32)
            ax.scatter(x + jitter, plain_seed_real_len[plain_mask], color="#e45756", s=28, zorder=3, label="plain" if i == 0 else None)
        if np.any(guided_mask):
            x = np.full(int(guided_mask.sum()), i, dtype=np.float32)
            jitter = np.linspace(0.02, 0.14, int(guided_mask.sum()), dtype=np.float32)
            ax.scatter(x + jitter, guided_seed_real_len[guided_mask], color="#4c78a8", s=28, zorder=3, label="guided" if i == 0 else None)
    ax.set_xticks(np.arange(len(valid_cluster_ids)))
    ax.set_xticklabels([f"cluster {cid}" for cid in valid_cluster_ids], rotation=15)
    ax.set_ylabel("real rollout length")
    ax.set_title("Which Manifold Cluster Plain and Guided Seeds Fall Into")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "candidate_seed_cluster_boxplot.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=220)
    hist_cluster_ids = [item["cluster_id"] for item in seed_cluster_histogram]
    hist_plain_counts = [item["num_plain_seeds"] for item in seed_cluster_histogram]
    hist_guided_counts = [item["num_guided_seeds"] for item in seed_cluster_histogram]
    x = np.arange(len(hist_cluster_ids), dtype=np.float32)
    ax.bar(x - 0.18, hist_plain_counts, width=0.36, color="#e45756", alpha=0.9, label="plain")
    ax.bar(x + 0.18, hist_guided_counts, width=0.36, color="#4c78a8", alpha=0.9, label="guided")
    ax.set_xticks(np.arange(len(hist_cluster_ids)))
    ax.set_xticklabels([f"cluster {cid}" for cid in hist_cluster_ids], rotation=15)
    ax.set_ylabel("number of seeds")
    ax.set_title("How Plain and Guided Seeds Distribute Across Manifold Clusters")
    ax.grid(alpha=0.2, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "candidate_seed_cluster_count.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=220)
    plain_x = np.arange(len(plain_seed_real_len), dtype=np.int32)
    guided_x = np.arange(len(guided_seed_real_len), dtype=np.int32)
    ax.scatter(plain_x, plain_seed_real_len, color="#e45756", s=18, alpha=0.9, label="plain")
    ax.scatter(guided_x, guided_seed_real_len, color="#4c78a8", s=18, alpha=0.9, label="guided")
    ax.axhline(float(plain_seed_real_len.mean()), color="#e45756", linestyle="--", linewidth=1.8, alpha=0.9)
    ax.axhline(float(guided_seed_real_len.mean()), color="#4c78a8", linestyle="--", linewidth=1.8, alpha=0.9)
    ax.set_xlabel("seed index")
    ax.set_ylabel("real rollout length")
    ax.set_title("Plain vs Guided Seed Rollout Lengths")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "candidate_seed_length_scatter.png")
    plt.close(fig)

    plain_selected = pick_representative_indices(plain_seed_real_len, max_items=4)
    guided_selected = pick_representative_indices(guided_seed_real_len, max_items=4)
    fig, axes = plt.subplots(7, 1, figsize=(11, 15), dpi=220, sharex=False)
    plain_colors = ["#f28e8b", "#e15759", "#c73e45", "#8f2328"]
    guided_colors = ["#9ecae9", "#4c78a8", "#2f5597", "#1d3557"]
    for color, idx in zip(plain_colors, plain_selected):
        q_path = np.asarray(plain_seed_trajectories[idx]["q"], dtype=np.float32)
        x = np.arange(q_path.shape[0], dtype=np.int32)
        label = f"plain idx={idx} len={plain_seed_real_len[idx]:.3f}"
        for j in range(7):
            axes[j].plot(x, q_path[:, j], color=color, linewidth=2.0, linestyle="-", label=label if j == 0 else None)
            axes[j].set_ylabel(f"q{j + 1}")
    for color, idx in zip(guided_colors, guided_selected):
        q_path = np.asarray(guided_seed_trajectories[idx]["q"], dtype=np.float32)
        x = np.arange(q_path.shape[0], dtype=np.int32)
        label = f"guided idx={idx} len={guided_seed_real_len[idx]:.3f}"
        for j in range(7):
            axes[j].plot(x, q_path[:, j], color=color, linewidth=2.0, linestyle="--", label=label if j == 0 else None)
    axes[0].legend(loc="best", ncol=2, fontsize=9)
    axes[-1].set_xlabel("rollout step")
    fig.suptitle("Representative Plain vs Guided Seed Joint Trajectories")
    fig.tight_layout()
    fig.savefig(args.outdir / "candidate_seed_joint_curves.png")
    plt.close(fig)

    summary = {
        "case_idx": int(args.case_idx),
        "bundle": str(args.bundle),
        "lnet_contrastive_ckpt": str(args.lnet_contrastive_ckpt),
        "joint_limit_gain": float(tracker.config.joint_limit_gain),
        "guided_lambda": float(args.guided_lambda),
        "num_candidates": int(q_candidates.shape[0]),
        "num_plain_seeds": int(q_seed.shape[0]),
        "num_guided_seeds": int(guided_q_seed.shape[0]),
        "num_clusters": int(args.num_clusters),
        "candidate_mean_rollout_length": float(candidate_real_len.mean()),
        "candidate_median_rollout_length": float(np.median(candidate_real_len)),
        "candidate_max_rollout_length": float(candidate_real_len.max()),
        "plain_seed_mean_rollout_length": float(plain_seed_real_len.mean()),
        "guided_seed_mean_rollout_length": float(guided_seed_real_len.mean()),
        "plain_raw_pos_err_mm_mean": float(plain_raw_pos_err.mean() * 1000.0),
        "plain_raw_pos_err_mm_max": float(plain_raw_pos_err.max() * 1000.0),
        "guided_raw_pos_err_mm_mean": float(guided_raw_pos_err.mean() * 1000.0),
        "guided_raw_pos_err_mm_max": float(guided_raw_pos_err.max() * 1000.0),
        "mean_q_delta_norm": float(q_delta_norm.mean()),
        "max_q_delta_norm": float(q_delta_norm.max()),
        "mean_pca_delta_norm": float(pca_delta_norm.mean()),
        "max_pca_delta_norm": float(pca_delta_norm.max()),
        "mean_predicted_length_delta": float(pred_len_delta.mean()),
        "mean_real_rollout_length_delta": float(real_len_delta.mean()),
        "raw_err_mm": {
            "mean": float(raw_err_np.mean() * 1000.0),
            "max": float(raw_err_np.max() * 1000.0),
        },
        "cluster_stats": cluster_stats,
        "seed_cluster_histogram": seed_cluster_histogram,
        "plain_seed_stats": [
            {
                "seed_idx": int(i),
                "predicted_length": float(pred_len_seed[i]),
                "raw_pos_err_mm": float(plain_raw_pos_err[i] * 1000.0),
                "real_rollout_length": float(plain_seed_real_len[i]),
                "assigned_cluster": int(plain_seed_cluster[i]),
            }
            for i in range(len(q_seed))
        ],
        "guided_seed_stats": [
            {
                "seed_idx": int(i),
                "predicted_length": float(guided_pred_len_seed[i]),
                "raw_pos_err_mm": float(guided_raw_pos_err[i] * 1000.0),
                "real_rollout_length": float(guided_seed_real_len[i]),
                "assigned_cluster": int(guided_seed_cluster[i]),
            }
            for i in range(len(guided_q_seed))
        ],
        "saved_files": [
            "candidate_seed_pca_overlay.png",
            "candidate_seed_pca_arrows.png",
            "candidate_seed_cluster_boxplot.png",
            "candidate_seed_cluster_count.png",
            "candidate_seed_length_scatter.png",
            "candidate_seed_joint_curves.png",
        ],
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[saved] {args.outdir}")


if __name__ == "__main__":
    main()
