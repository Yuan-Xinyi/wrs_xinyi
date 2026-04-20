#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd
from wrs.robot_sim.robots.franka_research_3.franka_research_3 import FrankaResearch3

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from lnet_contrastive_fr3_rotation_cone_eval import (
    build_tracker,
    collect_candidate_qs,
    rollout_same_task,
    sample_task_anchor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample one random FR3 task, collect candidates, and visualize near-zero rollout failures."
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--oversample", type=int, default=512)
    parser.add_argument("--zero-threshold", type=float, default=0.02, help="Treat rollout lengths <= this value (m) as failures.")
    parser.add_argument("--max-zero-vis", type=int, default=12, help="How many near-zero candidates to render.")
    parser.add_argument("--rollout-batch-size", type=int, default=16, help="Chunk size for full rollout evaluation.")
    parser.add_argument("--pos-tol-mm", type=float, default=2.0)
    parser.add_argument("--correction-iters", type=int, default=50)
    parser.add_argument("--correction-tol", type=float, default=1e-4)
    parser.add_argument("--correction-damping", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--list-only", action="store_true", help="Print the failing candidates without opening WRS.")
    return parser.parse_args()


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normal / max(float(np.linalg.norm(normal)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(z_axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-12)
    return np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)


def print_summary(anchor: dict, real_lengths: np.ndarray, zero_indices: np.ndarray, best_idx: int) -> None:
    print("=== Random Task ===")
    print(f"anchor_q = {np.array2string(anchor['q'], precision=4, suppress_small=True)}")
    print(f"pos      = {np.array2string(anchor['pos'], precision=4, suppress_small=True)}")
    print(f"dir      = {np.array2string(anchor['direction'], precision=4, suppress_small=True)}")
    print(f"normal   = {np.array2string(anchor['normal'], precision=4, suppress_small=True)}")
    print("")
    print("=== Rollout Summary ===")
    print(
        f"num_candidates={real_lengths.shape[0]} | "
        f"best_idx={best_idx} best_len={float(real_lengths[best_idx]):.4f} m | "
        f"near_zero_count={zero_indices.shape[0]}"
    )
    if zero_indices.size == 0:
        print("No near-zero candidates found under the current threshold.")
        return
    print("")
    print("=== Near-Zero Candidates ===")
    for idx in zero_indices.tolist():
        print(f"idx={idx:03d} real_len={float(real_lengths[idx]):.4f} m")


def print_candidate_joints(q_candidates: np.ndarray, real_lengths: np.ndarray, zero_indices: np.ndarray, best_idx: int) -> None:
    if zero_indices.size > 0:
        print("")
        print("=== Near-Zero Candidate Joint Angles ===")
        for idx in zero_indices.tolist():
            print(
                f"idx={idx:03d} real_len={float(real_lengths[idx]):.4f} m | "
                f"q={np.array2string(q_candidates[idx], precision=4, suppress_small=True)}"
            )
    print("")
    print("=== Best Candidate Joint Angles ===")
    print(
        f"idx={best_idx:03d} real_len={float(real_lengths[best_idx]):.4f} m | "
        f"q={np.array2string(q_candidates[best_idx], precision=4, suppress_small=True)}"
    )


def visualize(anchor: dict, q_candidates: np.ndarray, real_lengths: np.ndarray, zero_indices: np.ndarray, best_idx: int, max_zero_vis: int) -> None:
    start = anchor["pos"]
    direction = anchor["direction"]
    normal = anchor["normal"]

    world = wd.World(cam_pos=[1.8, -1.6, 1.1], lookat_pos=[0.2, 0.0, 0.35])
    mgm.gen_frame().attach_to(world)
    mgm.gen_sphere(start, radius=0.01, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)
    mgm.gen_arrow(spos=start, epos=start + direction * 0.20, rgb=np.array([1.0, 0.12, 0.12]), alpha=0.95).attach_to(world)
    mgm.gen_arrow(spos=start, epos=start + normal * 0.20, rgb=np.array([0.10, 0.45, 1.0]), alpha=0.95).attach_to(world)
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

    robot.goto_given_conf(anchor["q"].astype(np.float32))
    robot.gen_meshmodel(rgb=np.array([0.05, 0.75, 0.20]), alpha=0.28, toggle_tcp_frame=True).attach_to(world)

    best_q = q_candidates[best_idx]
    best_len = float(real_lengths[best_idx])
    robot.goto_given_conf(best_q.astype(np.float32))
    robot.gen_meshmodel(rgb=np.array([0.10, 0.85, 0.20]), alpha=0.55, toggle_tcp_frame=True).attach_to(world)
    mgm.gen_stick(spos=start, epos=start + direction * best_len, radius=0.005, rgb=np.array([0.10, 0.85, 0.20]), alpha=0.95).attach_to(world)

    for idx in zero_indices[: max_zero_vis].tolist():
        q = q_candidates[idx]
        real_len = float(real_lengths[idx])
        robot.goto_given_conf(q.astype(np.float32))
        robot.gen_meshmodel(rgb=np.array([0.92, 0.18, 0.18]), alpha=0.12, toggle_tcp_frame=False).attach_to(world)
        mgm.gen_stick(spos=start, epos=start + direction * real_len, radius=0.003, rgb=np.array([0.92, 0.18, 0.18]), alpha=0.55).attach_to(world)

    world.run()


def rollout_in_chunks(
    tracker,
    tracker_device: torch.device,
    q_candidates: np.ndarray,
    direction: np.ndarray,
    normal: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    batch_size = max(1, int(batch_size))
    outputs: list[np.ndarray] = []
    for start in range(0, q_candidates.shape[0], batch_size):
        stop = min(start + batch_size, q_candidates.shape[0])
        outputs.append(
            rollout_same_task(
                tracker=tracker,
                tracker_device=tracker_device,
                q_batch_np=q_candidates[start:stop],
                direction=direction,
                normal=normal,
            )
        )
        if tracker_device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(outputs, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))

    device = torch.device(args.device)
    tracker, tracker_device = build_tracker(device)
    anchor = sample_task_anchor(tracker, tracker_device)

    q_candidates, _ = collect_candidate_qs(
        tracker=tracker,
        tracker_device=tracker_device,
        target_pos=anchor["pos"],
        direction=anchor["direction"],
        normal=anchor["normal"],
        num_candidates=int(args.num_candidates),
        oversample=int(args.oversample),
        pos_tol_mm=float(args.pos_tol_mm),
        correction_iters=int(args.correction_iters),
        correction_tol=float(args.correction_tol),
        correction_damping=float(args.correction_damping),
    )
    real_lengths = rollout_in_chunks(
        tracker=tracker,
        tracker_device=tracker_device,
        q_candidates=q_candidates,
        direction=anchor["direction"],
        normal=anchor["normal"],
        batch_size=int(args.rollout_batch_size),
    )

    best_idx = int(np.argmax(real_lengths))
    zero_indices = np.where(real_lengths <= float(args.zero_threshold))[0]

    print_summary(anchor, real_lengths, zero_indices, best_idx)
    print_candidate_joints(q_candidates, real_lengths, zero_indices, best_idx)

    if not args.list_only:
        visualize(
            anchor=anchor,
            q_candidates=q_candidates,
            real_lengths=real_lengths,
            zero_indices=zero_indices,
            best_idx=best_idx,
            max_zero_vis=int(args.max_zero_vis),
        )


if __name__ == "__main__":
    main()
