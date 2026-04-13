#!/usr/bin/env python3
"""
Unified analysis and plotting entry point for the GPU_nullspace_straight project.

Inputs:
- `--dataset`: an HDF5 trajectory dataset under `datasets/`, for example an FR3 or XArm trajectory file.
- `--lnet-log`: a ranking/evaluation log such as
  `length_prediction/lnet_contrastive_fr3_rotation_cone_eval_batch.jsonl`.
- `--guidance-log`: a guidance visualization log such as
  `diffusion_inpainting/guidance_vis_cases/log.jsonl`.

Outputs:
- A selected output directory containing figures (`.png`), summaries (`.json`), and small helper text files.
- Each mode writes its own subdirectory, so the script can be run repeatedly without affecting other analyses.

Functions:
- `dataset`: inspect dataset scale and coverage, including remaining-length distribution, TCP coverage,
  joint-space coverage, and direction/normal geometry.
- `lnet-log`: analyze pairwise ranking logs and visualize how often the score-selected candidate matches
  the true best candidate.
- `intro-case`: rank cases that are suitable for the paper's introduction figure, namely cases where a
  locally attractive candidate has clearly worse future rollout length than the globally better candidate.
- `guidance-log`: analyze contrastive-guidance logs and summarize lambda sensitivity, gains, and failure trends.
- `all`: run every analysis whose input file exists.

Typical usage:
- `python analyze_and_plot.py --mode dataset`
- `python analyze_and_plot.py --mode lnet-log`
- `python analyze_and_plot.py --mode intro-case`
- `python analyze_and_plot.py --mode guidance-log`
- `python analyze_and_plot.py --mode all`
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
LENGTH_PREDICTION_DIR = ROOT_DIR / "length_prediction"
DIFFUSION_INPAINTING_DIR = ROOT_DIR / "diffusion_inpainting"

DEFAULT_DATASET = DATASETS_DIR / "franka_research_3_gpu_trajectories_sub10.hdf5"
DEFAULT_LNET_LOG = LENGTH_PREDICTION_DIR / "lnet_contrastive_fr3_rotation_cone_eval_batch.jsonl"
DEFAULT_GUIDANCE_LOG = DIFFUSION_INPAINTING_DIR / "guidance_vis_cases" / "log.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_INTRO_REPLAY_SCRIPT = LENGTH_PREDICTION_DIR / "vis_lnet_contrastive_fr3_extreme_failures.py"

LNET_TASK_LINE = re.compile(
    r"^\[(?P<case_idx>\d+)\]\s+task:\s+"
    r"pos=(?P<pos>\[[^\]]+\])\s+"
    r"direction=(?P<direction>\[[^\]]+\])\s+"
    r"normal=(?P<normal>\[[^\]]+\])$"
)
LNET_RESULT_LINE = re.compile(
    r"^\[(?P<case_idx>\d+)\]\s+"
    r"real_best:\s+idx=(?P<real_idx>\d+)\s+score=(?P<real_score>[-+]?\d*\.?\d+)\s+real_len=(?P<real_len>[-+]?\d*\.?\d+)\s+"
    r"score_best:\s+idx=(?P<score_idx>\d+)\s+score=(?P<score_score>[-+]?\d*\.?\d+)\s+real_len=(?P<score_len>[-+]?\d*\.?\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified paper-analysis script for GPU_nullspace_straight.")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["dataset", "lnet-log", "intro-case", "guidance-log", "all"],
        help="Which analysis to run.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--lnet-log", type=Path, default=DEFAULT_LNET_LOG)
    parser.add_argument("--guidance-log", type=Path, default=DEFAULT_GUIDANCE_LOG)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--scatter-limit", type=int, default=30000)
    parser.add_argument("--intro-top-k", type=int, default=20)
    parser.add_argument("--intro-min-real-len", type=float, default=0.30)
    parser.add_argument("--intro-max-ratio", type=float, default=0.90)
    parser.add_argument("--intro-min-score-adv", type=float, default=0.0)
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


def read_lnet_rows(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    pending_task: dict[int, dict] = {}

    for raw_line in log_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        task_match = LNET_TASK_LINE.match(line)
        if task_match:
            case_idx = int(task_match.group("case_idx"))
            pending_task[case_idx] = {
                "pos": json.loads(task_match.group("pos")),
                "direction": json.loads(task_match.group("direction")),
                "normal": json.loads(task_match.group("normal")),
            }
            continue

        result_match = LNET_RESULT_LINE.match(line)
        if not result_match:
            continue

        case_idx = int(result_match.group("case_idx"))
        row = {
            "case_idx": case_idx,
            "real_best_idx": int(result_match.group("real_idx")),
            "real_best_score": float(result_match.group("real_score")),
            "real_best_len": float(result_match.group("real_len")),
            "score_best_idx": int(result_match.group("score_idx")),
            "score_best_score": float(result_match.group("score_score")),
            "score_best_len": float(result_match.group("score_len")),
        }
        if case_idx in pending_task:
            row.update(pending_task[case_idx])
        rows.append(row)

    return rows


def read_guidance_rows(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    current_block: list[str] = []
    brace_depth = 0

    for raw_line in log_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if brace_depth == 0 and line.startswith("{") and line.endswith("}"):
            try:
                rows.append(json.loads(line))
                continue
            except json.JSONDecodeError:
                pass

        if "{" in line or brace_depth > 0:
            brace_depth += line.count("{")
            brace_depth -= line.count("}")
            current_block.append(raw_line)
            if brace_depth == 0 and current_block:
                try:
                    rows.append(json.loads("\n".join(current_block)))
                except json.JSONDecodeError:
                    pass
                current_block = []

    return rows


def analyze_dataset(args: argparse.Namespace) -> None:
    dataset_path = args.dataset
    outdir = args.outdir / "dataset"
    if not dataset_path.exists():
        print(f"[skip][dataset] missing: {dataset_path}")
        return

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

        for key in trajectory_keys:
            group = trajectory_root[key]
            q = np.asarray(group["q"][:], dtype=np.float32)
            tcp_pos = np.asarray(group["tcp_pos"][:], dtype=np.float32)
            remaining_length = np.asarray(group["remaining_length"][:], dtype=np.float32).reshape(-1)
            direction = np.asarray(group.attrs["direction"], dtype=np.float32).reshape(1, 3)
            normal = np.asarray(group.attrs["target_normal"], dtype=np.float32).reshape(1, 3)

            all_q.append(q)
            all_tcp_pos.append(tcp_pos)
            all_remaining_length.append(remaining_length)
            all_direction.append(np.repeat(direction, q.shape[0], axis=0))
            all_normal.append(np.repeat(normal, q.shape[0], axis=0))
            trajectory_num_points.append(int(q.shape[0]))

    q_all = np.concatenate(all_q, axis=0)
    tcp_pos_all = np.concatenate(all_tcp_pos, axis=0)
    remaining_length_all = np.concatenate(all_remaining_length, axis=0)
    direction_all = np.concatenate(all_direction, axis=0)
    normal_all = np.concatenate(all_normal, axis=0)
    trajectory_num_points = np.asarray(trajectory_num_points, dtype=np.int32)
    direction_normal_dot = np.sum(direction_all * normal_all, axis=1)

    summary = {
        "dataset_path": str(dataset_path),
        "num_trajectories": int(len(trajectory_keys)),
        "num_samples": int(q_all.shape[0]),
        "q_dim": int(q_all.shape[1]),
        "trajectory_num_points": stats_dict(trajectory_num_points),
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
    ax.hist(remaining_length_all, bins=400, color="#82B0D2", alpha=0.88)
    ax.set_xlabel("remaining length")
    ax.set_ylabel("count")
    ax.set_title("Remaining Length Distribution")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "hist_remaining_length.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.hist(direction_normal_dot, bins=80, color="#bc4749", alpha=0.88)
    ax.set_xlabel("direction · target_normal")
    ax.set_ylabel("count")
    ax.set_title("Direction/Normal Alignment")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "hist_direction_normal_dot.png")
    plt.close(fig)

    if tcp_pos_all.shape[0] > args.scatter_limit:
        subset_idx = np.linspace(0, tcp_pos_all.shape[0] - 1, num=args.scatter_limit, dtype=np.int64)
    else:
        subset_idx = np.arange(tcp_pos_all.shape[0], dtype=np.int64)

    tcp_color_value = remaining_length_all[subset_idx]
    tcp_color_min = float(tcp_color_value.min()) if tcp_color_value.size > 0 else 0.0
    tcp_color_max = float(tcp_color_value.max()) if tcp_color_value.size > 0 else 1.0

    fig, ax = plt.subplots(figsize=(6.2, 5.8), dpi=300)
    scatter = ax.scatter(
        tcp_pos_all[subset_idx, 0],
        tcp_pos_all[subset_idx, 1],
        s=8,
        alpha=0.55,
        c=tcp_color_value,
        cmap="viridis",
        vmin=tcp_color_min,
        vmax=tcp_color_max,
        edgecolors="none",
    )
    ax.set_xlabel("tcp_x")
    ax.set_ylabel("tcp_y")
    ax.set_title("TCP Coverage on XY Plane")
    ax.grid(alpha=0.20)
    fig.colorbar(scatter, ax=ax, label="remaining_length")
    fig.tight_layout()
    fig.savefig(outdir / "scatter_tcp_xy.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.8), dpi=300)
    scatter = ax.scatter(
        tcp_pos_all[subset_idx, 0],
        tcp_pos_all[subset_idx, 2],
        s=8,
        alpha=0.55,
        c=tcp_color_value,
        cmap="viridis",
        vmin=tcp_color_min,
        vmax=tcp_color_max,
        edgecolors="none",
    )
    ax.set_xlabel("tcp_x")
    ax.set_ylabel("tcp_z")
    ax.set_title("TCP Coverage on XZ Plane")
    ax.grid(alpha=0.20)
    fig.colorbar(scatter, ax=ax, label="remaining_length")
    fig.tight_layout()
    fig.savefig(outdir / "scatter_tcp_xz.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.8), dpi=300)
    scatter = ax.scatter(
        tcp_pos_all[subset_idx, 1],
        tcp_pos_all[subset_idx, 2],
        s=8,
        alpha=0.55,
        c=tcp_color_value,
        cmap="viridis",
        vmin=tcp_color_min,
        vmax=tcp_color_max,
        edgecolors="none",
    )
    ax.set_xlabel("tcp_y")
    ax.set_ylabel("tcp_z")
    ax.set_title("TCP Coverage on YZ Plane")
    ax.grid(alpha=0.20)
    fig.colorbar(scatter, ax=ax, label="remaining_length")
    fig.tight_layout()
    fig.savefig(outdir / "scatter_tcp_yz.png")
    plt.close(fig)

    if q_all.shape[1] >= 2:
        q_subset_idx = subset_idx if q_all.shape[0] > args.scatter_limit else np.arange(q_all.shape[0], dtype=np.int64)
        fig, ax = plt.subplots(figsize=(6.2, 5.8), dpi=300)
        ax.scatter(q_all[q_subset_idx, 0], q_all[q_subset_idx, 1], s=8, alpha=0.35, color="#1d3557", edgecolors="none")
        ax.set_xlabel("q1")
        ax.set_ylabel("q2")
        ax.set_title("Joint Coverage: q1 vs q2")
        ax.grid(alpha=0.20)
        fig.tight_layout()
        fig.savefig(outdir / "scatter_q1_q2.png")
        plt.close(fig)

    if q_all.shape[1] >= 4:
        q_subset_idx = subset_idx if q_all.shape[0] > args.scatter_limit else np.arange(q_all.shape[0], dtype=np.int64)
        fig, ax = plt.subplots(figsize=(6.2, 5.8), dpi=300)
        ax.scatter(q_all[q_subset_idx, 2], q_all[q_subset_idx, 3], s=8, alpha=0.35, color="#6a994e", edgecolors="none")
        ax.set_xlabel("q3")
        ax.set_ylabel("q4")
        ax.set_title("Joint Coverage: q3 vs q4")
        ax.grid(alpha=0.20)
        fig.tight_layout()
        fig.savefig(outdir / "scatter_q3_q4.png")
        plt.close(fig)

    print(f"[saved][dataset] {outdir}")


def analyze_lnet_log(args: argparse.Namespace) -> None:
    log_path = args.lnet_log
    outdir = args.outdir / "lnet_log"
    if not log_path.exists():
        print(f"[skip][lnet-log] missing: {log_path}")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    rows = read_lnet_rows(log_path)
    if not rows:
        raise RuntimeError(f"No parsable rows found in {log_path}")

    real_best_len = np.asarray([row["real_best_len"] for row in rows], dtype=np.float64)
    score_best_len = np.asarray([row["score_best_len"] for row in rows], dtype=np.float64)
    gap_score_minus_real = score_best_len - real_best_len
    ratio_score_over_real = score_best_len / np.clip(real_best_len, 1e-8, None)
    exact_hit = np.asarray([row["real_best_idx"] == row["score_best_idx"] for row in rows], dtype=np.float64)
    severe_fail = np.asarray([score_best_len[i] <= 0.5 * real_best_len[i] for i in range(len(rows))], dtype=np.float64)
    zero_like_fail = np.asarray([score_best_len[i] <= 1e-6 for i in range(len(rows))], dtype=np.float64)

    summary = {
        "log_path": str(log_path),
        "num_cases": int(len(rows)),
        "real_best_len": stats_dict(real_best_len),
        "score_best_len": stats_dict(score_best_len),
        "gap_score_best_minus_real_best": stats_dict(gap_score_minus_real),
        "ratio_score_best_over_real_best": stats_dict(ratio_score_over_real),
        "exact_hit_rate": float(exact_hit.mean()),
        "severe_fail_rate_le_50pct": float(severe_fail.mean()),
        "zero_like_fail_rate": float(zero_like_fail.mean()),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    axis_limit = max(float(real_best_len.max(initial=0.0)), float(score_best_len.max(initial=0.0))) * 1.03 if len(rows) else 1.0
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=300)
    ax.scatter(real_best_len, score_best_len, s=18, alpha=0.72, color="#2364aa", edgecolors="none")
    ax.plot([0.0, axis_limit], [0.0, axis_limit], "--", color="#aa3a3a", linewidth=1.2)
    ax.set_xlabel("real_best_len")
    ax.set_ylabel("score_best_len")
    ax.set_title("Score-Best vs Real-Best Length")
    ax.set_xlim(0.0, axis_limit)
    ax.set_ylim(0.0, axis_limit)
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "scatter_score_best_vs_real_best.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.hist(gap_score_minus_real, bins=80, color="#3fa34d", alpha=0.88)
    ax.set_xlabel("score_best_len - real_best_len")
    ax.set_ylabel("count")
    ax.set_title("Gap Distribution")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "hist_gap_score_best_minus_real_best.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.hist(ratio_score_over_real, bins=80, color="#f0a202", alpha=0.88)
    ax.set_xlabel("score_best_len / real_best_len")
    ax.set_ylabel("count")
    ax.set_title("Retention Ratio Distribution")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "hist_ratio_score_best_over_real_best.png")
    plt.close(fig)

    case_idx = np.asarray([row["case_idx"] for row in rows], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=300)
    ax.plot(case_idx, real_best_len, label="real_best_len", color="#1b9e77", linewidth=1.2)
    ax.plot(case_idx, score_best_len, label="score_best_len", color="#d95f02", linewidth=1.0, alpha=0.85)
    ax.set_xlabel("case_idx")
    ax.set_ylabel("length")
    ax.set_title("Best Length Across Cases")
    ax.grid(alpha=0.20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "curve_best_lengths_by_case.png")
    plt.close(fig)

    print(f"[saved][lnet-log] {outdir}")


def analyze_intro_case(args: argparse.Namespace) -> None:
    log_path = args.lnet_log
    outdir = args.outdir / "intro_case"
    if not log_path.exists():
        print(f"[skip][intro-case] missing: {log_path}")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    rows = read_lnet_rows(log_path)
    rows = [row for row in rows if "pos" in row and "direction" in row and "normal" in row]
    if not rows:
        raise RuntimeError(f"No task-annotated rows found in {log_path}")

    ranked_rows = []
    for row in rows:
        real_best_len = float(row["real_best_len"])
        score_best_len = float(row["score_best_len"])
        real_best_score = float(row["real_best_score"])
        score_best_score = float(row["score_best_score"])
        gap_real_minus_score = real_best_len - score_best_len
        ratio_score_over_real = score_best_len / max(real_best_len, 1e-8)
        score_advantage = score_best_score - real_best_score
        different_selection = int(row["score_best_idx"]) != int(row["real_best_idx"])

        qualifies = (
            different_selection
            and real_best_len >= args.intro_min_real_len
            and ratio_score_over_real <= args.intro_max_ratio
            and score_advantage >= args.intro_min_score_adv
            and gap_real_minus_score > 0.0
        )

        intro_rank_score = (
            3.0 * gap_real_minus_score
            + 1.2 * max(0.0, 1.0 - ratio_score_over_real)
            + 0.8 * max(0.0, score_advantage)
            + 0.25 * real_best_len
        )

        ranked_rows.append(
            {
                **row,
                "gap_real_minus_score_best": float(gap_real_minus_score),
                "ratio_score_best_over_real_best": float(ratio_score_over_real),
                "score_advantage_of_score_best": float(score_advantage),
                "different_selection": bool(different_selection),
                "qualifies": bool(qualifies),
                "intro_rank_score": float(intro_rank_score),
            }
        )

    qualified_rows = [row for row in ranked_rows if row["qualifies"]]
    if qualified_rows:
        ranked_rows = sorted(qualified_rows, key=lambda row: row["intro_rank_score"], reverse=True)
    else:
        ranked_rows = sorted(ranked_rows, key=lambda row: row["intro_rank_score"], reverse=True)

    top_rows = ranked_rows[: max(1, int(args.intro_top_k))]
    best_row = top_rows[0]

    summary = {
        "log_path": str(log_path),
        "selection_rule": {
            "min_real_len": float(args.intro_min_real_len),
            "max_ratio": float(args.intro_max_ratio),
            "min_score_adv": float(args.intro_min_score_adv),
            "requires_different_selection": True,
        },
        "num_rows_with_task": int(len(rows)),
        "num_qualified_rows": int(sum(1 for row in ranked_rows if row["qualifies"])),
        "best_case_idx": int(best_row["case_idx"]),
        "best_case": best_row,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / "top_intro_candidates.json").write_text(json.dumps({"rows": top_rows}, indent=2))

    readme_text = "\n".join(
        [
            "# Introduction Figure Candidate",
            "",
            "This file lists the strongest candidate for a 'local greedy vs global better' introduction figure.",
            "",
            f"- Selected case: `{best_row['case_idx']}`",
            f"- Real-best length: `{best_row['real_best_len']:.4f}`",
            f"- Score-best length: `{best_row['score_best_len']:.4f}`",
            f"- Absolute gap: `{best_row['gap_real_minus_score_best']:.4f}`",
            f"- Retention ratio: `{best_row['ratio_score_best_over_real_best']:.4f}`",
            f"- Score advantage of score-best: `{best_row['score_advantage_of_score_best']:.4f}`",
            "",
            "Recommended replay command:",
            "",
            "```bash",
            f"/home/lqin/miniconda3/envs/wrs/bin/python {DEFAULT_INTRO_REPLAY_SCRIPT} --input {log_path} --case-idx {best_row['case_idx']}",
            "```",
            "",
            "Suggested narrative:",
            "",
            "Under the same TCP task condition, the locally preferred solution receives a higher learned score,",
            "but the globally better solution preserves substantially longer feasible rollout length.",
            "This illustrates why one-step attractiveness is insufficient for future task potential.",
            "",
        ]
    )
    (outdir / "README.md").write_text(readme_text)

    gap_real_minus_score = np.asarray([row["gap_real_minus_score_best"] for row in ranked_rows], dtype=np.float64)
    ratio_score_over_real = np.asarray([row["ratio_score_best_over_real_best"] for row in ranked_rows], dtype=np.float64)
    score_advantage = np.asarray([row["score_advantage_of_score_best"] for row in ranked_rows], dtype=np.float64)
    real_best_len = np.asarray([row["real_best_len"] for row in ranked_rows], dtype=np.float64)
    qualifies = np.asarray([row["qualifies"] for row in ranked_rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.hist(gap_real_minus_score, bins=80, color="#5a189a", alpha=0.88)
    ax.set_xlabel("real_best_len - score_best_len")
    ax.set_ylabel("count")
    ax.set_title("Intro-Case Gap Distribution")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "hist_intro_gap.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.hist(ratio_score_over_real, bins=80, color="#118ab2", alpha=0.88)
    ax.set_xlabel("score_best_len / real_best_len")
    ax.set_ylabel("count")
    ax.set_title("Intro-Case Retention Ratio Distribution")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "hist_intro_ratio.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.8), dpi=300)
    scatter = ax.scatter(real_best_len, gap_real_minus_score, s=18, alpha=0.72, c=score_advantage, cmap="viridis", edgecolors="none")
    ax.set_xlabel("real_best_len")
    ax.set_ylabel("real_best_len - score_best_len")
    ax.set_title("Candidate Strength for Introduction Figure")
    ax.grid(alpha=0.20)
    fig.colorbar(scatter, ax=ax, label="score_best_score - real_best_score")
    fig.tight_layout()
    fig.savefig(outdir / "scatter_intro_strength.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.8), dpi=300)
    ax.scatter(ratio_score_over_real, gap_real_minus_score, s=18, alpha=0.55, color="#adb5bd", edgecolors="none")
    if np.any(qualifies):
        ax.scatter(
            ratio_score_over_real[qualifies],
            gap_real_minus_score[qualifies],
            s=22,
            alpha=0.80,
            color="#d00000",
            edgecolors="none",
        )
    ax.axvline(float(args.intro_max_ratio), linestyle="--", linewidth=1.1, color="#1d3557")
    ax.axhline(0.0, linestyle="--", linewidth=1.1, color="#1d3557")
    ax.set_xlabel("score_best_len / real_best_len")
    ax.set_ylabel("real_best_len - score_best_len")
    ax.set_title("Qualified vs Unqualified Intro Cases")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "scatter_intro_ratio_vs_gap.png")
    plt.close(fig)

    print(f"[saved][intro-case] {outdir}")


def analyze_guidance_log(args: argparse.Namespace) -> None:
    log_path = args.guidance_log
    outdir = args.outdir / "guidance_log"
    if not log_path.exists():
        print(f"[skip][guidance-log] missing: {log_path}")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    rows = read_guidance_rows(log_path)
    meta_rows = [row for row in rows if row.get("record_type") == "meta"]
    lambda_rows = [row for row in rows if row.get("record_type") == "lambda"]
    if not lambda_rows:
        raise RuntimeError(f"No guidance lambda records found in {log_path}")

    gain_vs_gt = np.asarray([float(row["gain_vs_gt"]) for row in lambda_rows], dtype=np.float64)
    guided_real_len = np.asarray([float(row["guided_real_len"]) for row in lambda_rows], dtype=np.float64)
    final_score = np.asarray([float(row["final_score"]) for row in lambda_rows], dtype=np.float64)
    raw_pos_err_mm = np.asarray([float(row["raw_pos_err_mm"]) for row in lambda_rows], dtype=np.float64)
    lambda_value = np.asarray([float(row["lambda"]) for row in lambda_rows], dtype=np.float64)
    unique_lambda_values = sorted({float(v) for v in lambda_value.tolist()})

    per_lambda = {}
    for lam in unique_lambda_values:
        mask = np.isclose(lambda_value, lam)
        per_lambda[str(lam)] = {
            "count": int(mask.sum()),
            "guided_real_len": stats_dict(guided_real_len[mask]),
            "gain_vs_gt": stats_dict(gain_vs_gt[mask]),
            "final_score": stats_dict(final_score[mask]),
            "raw_pos_err_mm": stats_dict(raw_pos_err_mm[mask]),
            "positive_gain_rate": float(np.mean(gain_vs_gt[mask] > 0.0)),
            "high_error_rate_gt_20mm": float(np.mean(raw_pos_err_mm[mask] > 20.0)),
        }

    grouped_cases: dict[tuple[str, int], list[dict]] = {}
    for row in lambda_rows:
        key = (str(row.get("traj_id", "unknown")), int(row.get("point_idx", -1)))
        grouped_cases.setdefault(key, []).append(row)

    best_lambda_counts = {str(lam): 0 for lam in unique_lambda_values}
    for items in grouped_cases.values():
        best_item = max(items, key=lambda item: float(item["guided_real_len"]))
        best_lambda_counts[str(float(best_item["lambda"]))] += 1

    summary = {
        "log_path": str(log_path),
        "num_meta_records": int(len(meta_rows)),
        "num_lambda_records": int(len(lambda_rows)),
        "num_unique_cases": int(len(grouped_cases)),
        "lambda_values": unique_lambda_values,
        "guided_real_len_all": stats_dict(guided_real_len),
        "gain_vs_gt_all": stats_dict(gain_vs_gt),
        "final_score_all": stats_dict(final_score),
        "raw_pos_err_mm_all": stats_dict(raw_pos_err_mm),
        "per_lambda": per_lambda,
        "best_lambda_counts": best_lambda_counts,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.hist(guided_real_len, bins=80, color="#3a86ff", alpha=0.88)
    ax.set_xlabel("guided_real_len")
    ax.set_ylabel("count")
    ax.set_title("Guided Real Length Distribution")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "hist_guided_real_len.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.hist(gain_vs_gt, bins=80, color="#2a9d8f", alpha=0.88)
    ax.set_xlabel("guided_real_len - gt_real_len")
    ax.set_ylabel("count")
    ax.set_title("Guidance Gain Distribution")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "hist_gain_vs_gt.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.hist(raw_pos_err_mm, bins=80, color="#e76f51", alpha=0.88)
    ax.set_xlabel("raw_pos_err_mm")
    ax.set_ylabel("count")
    ax.set_title("Raw Position Error Before Correction")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "hist_raw_pos_err_mm.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.8), dpi=300)
    ax.scatter(final_score, guided_real_len, s=16, alpha=0.65, c=lambda_value, cmap="plasma", edgecolors="none")
    ax.set_xlabel("final_score")
    ax.set_ylabel("guided_real_len")
    ax.set_title("Guided Length vs Contrastive Score")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "scatter_score_vs_guided_real_len.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.8), dpi=300)
    ax.scatter(raw_pos_err_mm, gain_vs_gt, s=16, alpha=0.65, c=lambda_value, cmap="viridis", edgecolors="none")
    ax.set_xlabel("raw_pos_err_mm")
    ax.set_ylabel("gain_vs_gt")
    ax.set_title("Gain vs Pre-Correction Position Error")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "scatter_pos_err_vs_gain.png")
    plt.close(fig)

    lambda_x = np.asarray(unique_lambda_values, dtype=np.float64)
    lambda_gain_mean = np.asarray([per_lambda[str(lam)]["gain_vs_gt"]["mean"] for lam in unique_lambda_values], dtype=np.float64)
    lambda_gain_std = np.asarray([per_lambda[str(lam)]["gain_vs_gt"]["std"] or 0.0 for lam in unique_lambda_values], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=300)
    ax.errorbar(lambda_x, lambda_gain_mean, yerr=lambda_gain_std, marker="o", linewidth=1.5, color="#6a4c93")
    ax.set_xlabel("lambda")
    ax.set_ylabel("gain_vs_gt")
    ax.set_title("Lambda Sensitivity of Guidance Gain")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "curve_lambda_vs_gain.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=300)
    bar_labels = [str(lam) for lam in unique_lambda_values]
    bar_values = np.asarray([best_lambda_counts[str(lam)] for lam in unique_lambda_values], dtype=np.int32)
    ax.bar(bar_labels, bar_values, color="#577590", alpha=0.9)
    ax.set_xlabel("lambda")
    ax.set_ylabel("num cases where lambda is best")
    ax.set_title("Best Lambda Counts Across Cases")
    ax.grid(axis="y", alpha=0.20)
    fig.tight_layout()
    fig.savefig(outdir / "bar_best_lambda_counts.png")
    plt.close(fig)

    print(f"[saved][guidance-log] {outdir}")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "all":
        modes = ["dataset", "lnet-log", "intro-case", "guidance-log"]
    else:
        modes = [args.mode]

    for mode in modes:
        if mode == "dataset":
            analyze_dataset(args)
        elif mode == "lnet-log":
            analyze_lnet_log(args)
        elif mode == "intro-case":
            analyze_intro_case(args)
        elif mode == "guidance-log":
            analyze_guidance_log(args)


if __name__ == "__main__":
    main()
