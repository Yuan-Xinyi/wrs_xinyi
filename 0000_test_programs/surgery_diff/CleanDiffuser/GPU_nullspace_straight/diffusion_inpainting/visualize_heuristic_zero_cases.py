#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
LENGTH_PREDICTION_DIR = ROOT_DIR / "length_prediction"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(LENGTH_PREDICTION_DIR) not in sys.path:
    sys.path.insert(0, str(LENGTH_PREDICTION_DIR))

import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd
from wrs.robot_sim.robots.franka_research_3.franka_research_3 import FrankaResearch3

from baseline_eval_fr3_common import DEFAULT_TASKS_JSONL, load_tasks_from_jsonl, rollout_large_batch
from baseline_eval_heuristic_fr3_long_tasks import METHOD_LABELS, METHOD_ORDER, compute_candidate_features, select_best_index
from length_prediction.lnet_contrastive_fr3_rotation_cone_eval import (
    batch_position_error_and_correction,
    build_tracker,
    collect_candidate_qs,
)
from trajectory_generation.fr3_nullspace_straight import GPUNullspaceStraightTracker, TrackerConfig


DEFAULT_CASES_JSONL = CURRENT_DIR / "baseline_eval_heuristic_fr3_long_tasks_cases.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect zero-rollout heuristic cases in WRS.")
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--tasks-jsonl", type=Path, default=DEFAULT_TASKS_JSONL)
    parser.add_argument("--method", type=str, choices=METHOD_ORDER, default="joint_limit_margin")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--zero-tol", type=float, default=1e-9)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--task-index", type=int, default=None, help="Exact task_index to inspect.")
    parser.add_argument("--zero-rank", type=int, default=0, help="Pick the k-th zero case under the selected method.")
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--oversample", type=int, default=512)
    parser.add_argument("--pos-tol-mm", type=float, default=2.0)
    parser.add_argument("--correction-iters", type=int, default=50)
    parser.add_argument("--correction-tol", type=float, default=1e-4)
    parser.add_argument("--correction-damping", type=float, default=1e-3)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--short-horizon-steps", type=int, default=200)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_idx, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Failed to parse JSON on line {line_idx} of {path}: {exc}") from exc
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normal / max(np.linalg.norm(normal), 1e-12)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-12)
    return np.column_stack((x_axis, y_axis, z_axis))


def list_zero_cases(rows: list[dict], method: str, zero_tol: float, limit: int) -> list[dict]:
    zeros = [row for row in rows if abs(float(row["results"][method]["selected_real"])) <= zero_tol]
    print(f"[filter] method={method} zero_tol={zero_tol:.1e} matches={len(zeros)}/{len(rows)}")
    for row in zeros[: max(0, int(limit))]:
        res = row["results"][method]
        print(
            f"  task_index={int(row['task_index'])} oracle_rank={int(row['oracle_rank_among_6000'])} "
            f"selected_idx={int(res['selected_sample_idx'])} short_len={float(res['short_horizon_length']):.3f} "
            f"raw_pos_err_mm={float(res['raw_pos_err_mm']):.3f}"
        )
    return zeros


def select_target_row(zero_rows: list[dict], task_index: int | None, zero_rank: int) -> dict:
    if task_index is not None:
        for row in zero_rows:
            if int(row["task_index"]) == int(task_index):
                return row
        raise ValueError(f"task_index={task_index} not found among zero cases.")
    if not zero_rows:
        raise ValueError("No zero cases to select from.")
    idx = int(zero_rank)
    if idx < 0 or idx >= len(zero_rows):
        raise ValueError(f"zero-rank={idx} out of range for {len(zero_rows)} matches.")
    return zero_rows[idx]


def compute_method_scores(method: str, features: dict[str, np.ndarray], short_lengths: np.ndarray) -> np.ndarray:
    if method == "short_rollout":
        return short_lengths.astype(np.float32)
    return features[method].astype(np.float32)


def visualize_task(task: dict, q_selected: np.ndarray, selected_real: float, q_best: np.ndarray, best_real: float, zero_indices: np.ndarray, q_corrected: np.ndarray, real_lengths: np.ndarray, method: str) -> None:
    start = task["pos"]
    direction = task["direction"]
    normal = task["target_normal"]

    world = wd.World(cam_pos=[1.8, -1.6, 1.1], lookat_pos=[0.2, 0.0, 0.35])
    mgm.gen_frame().attach_to(world)
    mgm.gen_sphere(start, radius=0.01, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)
    mgm.gen_arrow(spos=start, epos=start + direction * 0.20, rgb=np.array([1.0, 0.15, 0.15]), alpha=0.9).attach_to(world)
    mgm.gen_arrow(spos=start, epos=start + normal * 0.20, rgb=np.array([0.15, 0.45, 1.0]), alpha=0.9).attach_to(world)
    plane_rotmat = rotation_matrix_from_normal(normal)
    plane_center = start + 0.30 * direction
    mcm.gen_box(
        xyz_lengths=[0.70, 0.70, 0.001],
        pos=plane_center,
        rotmat=plane_rotmat,
        rgb=[0.8, 0.85, 0.9],
        alpha=0.22,
    ).attach_to(world)
    mgm.gen_frame(pos=start, rotmat=plane_rotmat, ax_length=0.09).attach_to(world)

    robot = FrankaResearch3(enable_cc=True)

    gt_end = start + direction * float(task["gt_real"])
    mgm.gen_stick(spos=start, epos=gt_end, radius=0.0045, rgb=np.array([0.10, 0.82, 0.20]), alpha=0.95).attach_to(world)
    mgm.gen_sphere(gt_end, radius=0.009, rgb=np.array([0.10, 0.82, 0.20]), alpha=0.95).attach_to(world)

    faint_zero = zero_indices[: min(12, len(zero_indices))]
    for idx in faint_zero:
        q = q_corrected[int(idx)]
        robot.goto_given_conf(q.astype(np.float32))
        robot.gen_meshmodel(rgb=np.array([0.65, 0.65, 0.65]), alpha=0.06, toggle_tcp_frame=False).attach_to(world)
        end = start + direction * float(real_lengths[int(idx)])
        mgm.gen_stick(spos=start, epos=end, radius=0.0025, rgb=np.array([0.70, 0.70, 0.70]), alpha=0.35).attach_to(world)

    robot.goto_given_conf(q_best.astype(np.float32))
    robot.gen_meshmodel(rgb=np.array([0.10, 0.82, 0.20]), alpha=0.42, toggle_tcp_frame=True).attach_to(world)
    best_end = start + direction * float(best_real)
    mgm.gen_stick(spos=start, epos=best_end, radius=0.005, rgb=np.array([0.10, 0.82, 0.20]), alpha=0.95).attach_to(world)
    mgm.gen_sphere(best_end, radius=0.010, rgb=np.array([0.10, 0.82, 0.20]), alpha=0.95).attach_to(world)

    robot.goto_given_conf(q_selected.astype(np.float32))
    robot.gen_meshmodel(rgb=np.array([0.95, 0.18, 0.18]), alpha=0.58, toggle_tcp_frame=True).attach_to(world)
    sel_end = start + direction * float(selected_real)
    mgm.gen_stick(spos=start, epos=sel_end, radius=0.005, rgb=np.array([0.95, 0.18, 0.18]), alpha=0.95).attach_to(world)
    mgm.gen_sphere(sel_end, radius=0.010, rgb=np.array([0.95, 0.18, 0.18]), alpha=0.95).attach_to(world)

    print(f"[vis] method={METHOD_LABELS[method]} selected_real={selected_real:.4f} best_real={best_real:.4f} zero_candidates={len(zero_indices)}")
    world.run()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.cases_jsonl)
    zero_rows = list_zero_cases(rows, args.method, float(args.zero_tol), int(args.limit))
    if args.list_only:
        return

    target_row = select_target_row(zero_rows, args.task_index, args.zero_rank)
    tasks = load_tasks_from_jsonl(args.tasks_jsonl)
    task_by_index = {int(t["task_index"]): t for t in tasks}
    task_index = int(target_row["task_index"])
    if task_index not in task_by_index:
        raise KeyError(f"task_index={task_index} not found in {args.tasks_jsonl}")
    task = task_by_index[task_index]

    device = torch.device(args.device)
    tracker, tracker_device = build_tracker(device)
    short_config = TrackerConfig(**vars(tracker.config))
    short_config.max_steps = int(args.short_horizon_steps)
    short_tracker = GPUNullspaceStraightTracker(
        robot=tracker.robot,
        collision_fn=tracker.collision_fn,
        config=short_config,
        print_every=0,
    )

    q_candidates, _ = collect_candidate_qs(
        tracker=tracker,
        tracker_device=tracker_device,
        target_pos=task["pos"],
        direction=task["direction"],
        normal=task["target_normal"],
        num_candidates=int(args.num_candidates),
        oversample=int(args.oversample),
        pos_tol_mm=float(args.pos_tol_mm),
        correction_iters=int(args.correction_iters),
        correction_tol=float(args.correction_tol),
        correction_damping=float(args.correction_damping),
    )
    q_corrected, raw_pos_err = batch_position_error_and_correction(
        tracker.robot,
        q_candidates.astype(np.float32),
        task["pos"],
        float(args.correction_damping),
        int(args.correction_iters),
        float(args.correction_tol),
        tracker_device,
    )
    features = compute_candidate_features(
        tracker=tracker,
        q_batch_np=q_corrected,
        target_pos_np=task["pos"],
        direction_np=task["direction"],
        normal_np=task["target_normal"],
    )
    short_lengths = rollout_large_batch(
        tracker=short_tracker,
        tracker_device=tracker_device,
        q_batch=q_corrected.astype(np.float32),
        direction_batch=np.repeat(task["direction"][None, :], q_corrected.shape[0], axis=0).astype(np.float32),
        target_normal_batch=np.repeat(task["target_normal"][None, :], q_corrected.shape[0], axis=0).astype(np.float32),
        chunk_size=min(int(args.rollout_batch_size), max(1, int(q_corrected.shape[0]))),
    )
    real_lengths = rollout_large_batch(
        tracker=tracker,
        tracker_device=tracker_device,
        q_batch=q_corrected.astype(np.float32),
        direction_batch=np.repeat(task["direction"][None, :], q_corrected.shape[0], axis=0).astype(np.float32),
        target_normal_batch=np.repeat(task["target_normal"][None, :], q_corrected.shape[0], axis=0).astype(np.float32),
        chunk_size=min(int(args.rollout_batch_size), max(1, int(q_corrected.shape[0]))),
    )
    scores = compute_method_scores(args.method, features, short_lengths)
    selected_idx = select_best_index(scores)
    best_idx = int(np.argmax(real_lengths))
    zero_indices = np.where(np.abs(real_lengths) <= float(args.zero_tol))[0].astype(np.int64)

    print(
        f"[case] task_index={task_index} oracle_rank={int(task['oracle_rank_among_6000'])} gt_real={float(task['gt_real']):.4f} "
        f"stored_selected_real={float(target_row['results'][args.method]['selected_real']):.4f}"
    )
    print(
        f"[replay] selected_idx={selected_idx} score={float(scores[selected_idx]):.6f} "
        f"short_len={float(short_lengths[selected_idx]):.6f} real_len={float(real_lengths[selected_idx]):.6f} "
        f"raw_pos_err_mm={float(raw_pos_err[selected_idx]) * 1e3:.3f}"
    )
    print(
        f"[replay] best_idx={best_idx} best_real={float(real_lengths[best_idx]):.6f} "
        f"zero_candidates={len(zero_indices)}/{len(real_lengths)}"
    )
    if len(zero_indices) > 0:
        print("[replay] first_zero_indices=" + ", ".join(str(int(v)) for v in zero_indices[:20]))

    visualize_task(
        task=task,
        q_selected=q_corrected[selected_idx],
        selected_real=float(real_lengths[selected_idx]),
        q_best=q_corrected[best_idx],
        best_real=float(real_lengths[best_idx]),
        zero_indices=zero_indices,
        q_corrected=q_corrected,
        real_lengths=real_lengths,
        method=args.method,
    )


if __name__ == "__main__":
    main()
