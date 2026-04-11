from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
from wrs.robot_sim.robots.franka_research_3.franka_research_3 import FrankaResearch3

from lnet_contrastive_fr3_rotation_cone_eval_batch_analysis import parse_log, DEFAULT_INPUT
from lnet_contrastive_fr3_rotation_cone_eval import (
    DEFAULT_CKPT,
    build_tracker,
    collect_candidate_qs,
    load_model,
    rollout_same_task,
    rotation_matrix_from_normal,
)


def termination_label(code: int) -> str:
    mapping = {
        0: 'low_mu',
        1: 'joint_limit',
        2: 'self_collision',
        3: 'max_steps',
        4: 'pos_tracking_error',
    }
    return mapping.get(int(code), 'unknown')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize extreme failure cases from Franka contrastive q-ranking batch log in WRS world.')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--case-idx', type=int, default=None, help='Specific case index from the log to visualize.')
    parser.add_argument('--mode', type=str, default='worst_gap', choices=['worst_gap', 'worst_ratio'])
    parser.add_argument('--num-candidates', type=int, default=64)
    parser.add_argument('--oversample', type=int, default=512)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--pos-tol-mm', type=float, default=2.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    return parser.parse_args()


def select_row(rows: list[dict], case_idx: int | None, mode: str) -> dict:
    rows_with_task = [r for r in rows if 'pos' in r and 'direction' in r and 'normal' in r]
    if not rows_with_task:
        raise RuntimeError('No task information found in the log. Re-run batch evaluation with task lines enabled.')
    if case_idx is not None:
        for row in rows_with_task:
            if int(row['case_idx']) == int(case_idx):
                return row
        raise ValueError(f'case_idx={case_idx} not found in {len(rows_with_task)} logged rows')
    if mode == 'worst_ratio':
        return min(rows_with_task, key=lambda r: r['score_best_len'] / max(r['real_best_len'], 1e-8))
    return min(rows_with_task, key=lambda r: r['score_best_len'] - r['real_best_len'])


def joint_limit_margin(robot, q_np: np.ndarray) -> float:
    lower = robot.jnt_ranges[:, 0].detach().cpu().numpy()
    upper = robot.jnt_ranges[:, 1].detach().cpu().numpy()
    return float(np.minimum(q_np - lower, upper - q_np).min())


def evaluate_task(row: dict, ckpt: Path, device: torch.device, args: argparse.Namespace):
    model = load_model(ckpt, device)
    tracker, tracker_device = build_tracker(device)
    q_batch_np, _ = collect_candidate_qs(
        tracker,
        tracker_device,
        np.asarray(row['pos'], dtype=np.float32),
        np.asarray(row['direction'], dtype=np.float32),
        np.asarray(row['normal'], dtype=np.float32),
        int(args.num_candidates),
        int(args.oversample),
        float(args.pos_tol_mm),
        int(args.correction_iters),
        float(args.correction_tol),
        float(args.correction_damping),
    )
    pos_batch_np = np.repeat(np.asarray(row['pos'], dtype=np.float32)[None, :], q_batch_np.shape[0], axis=0)
    direction = np.asarray(row['direction'], dtype=np.float32)
    normal = np.asarray(row['normal'], dtype=np.float32)
    q_batch = torch.from_numpy(q_batch_np).to(device)
    cond_batch = torch.from_numpy(np.concatenate([
        pos_batch_np,
        np.repeat(direction[None, :], q_batch_np.shape[0], axis=0).astype(np.float32),
        np.repeat(normal[None, :], q_batch_np.shape[0], axis=0).astype(np.float32),
    ], axis=1)).to(device)
    with torch.no_grad():
        score_batch, _ = model(q_batch, cond_batch)
    score_np = score_batch.detach().cpu().numpy().astype(np.float32).reshape(-1)
    real_len_np = rollout_same_task(tracker, tracker_device, q_batch_np, direction, normal)
    q_sel = q_batch_np.astype(np.float32)
    sel_idx = np.array([int(np.argmax(real_len_np)), int(np.argmax(score_np))], dtype=np.int64)
    q_two = torch.from_numpy(q_sel[sel_idx]).to(tracker_device)
    d_two = torch.from_numpy(np.repeat(direction[None, :], 2, axis=0).astype(np.float32)).to(tracker_device)
    n_two = torch.from_numpy(np.repeat(normal[None, :], 2, axis=0).astype(np.float32)).to(tracker_device)
    traj_two = tracker.collect_batch_trajectories(q_two, d_two, n_two)
    return q_batch_np, score_np, real_len_np, traj_two, tracker


def print_failure_diagnostics(row: dict, score_np: np.ndarray, real_len_np: np.ndarray, traj_two: list[dict], tracker) -> tuple[int, int]:
    best_real_idx = int(np.argmax(real_len_np))
    best_score_idx = int(np.argmax(score_np))
    logged_gap = float(row['score_best_len']) - float(row['real_best_len'])
    logged_ratio = float(row['score_best_len']) / max(float(row['real_best_len']), 1e-8)
    replay_gap = float(real_len_np[best_score_idx]) - float(real_len_np[best_real_idx])
    replay_ratio = float(real_len_np[best_score_idx]) / max(float(real_len_np[best_real_idx]), 1e-8)
    score_order = np.argsort(-score_np)
    topk = score_order[: min(5, score_order.shape[0])]
    print(f"[reason][logged] gap={logged_gap:.4f} ratio={logged_ratio:.4f} real_best={float(row['real_best_len']):.4f} score_best={float(row['score_best_len']):.4f}")
    print(f"[reason][replay] gap={replay_gap:.4f} ratio={replay_ratio:.4f} real_best_idx={best_real_idx} score_best_idx={best_score_idx}")
    print(f"[reason][replay] real_best: score={float(score_np[best_real_idx]):.4f} real_len={float(real_len_np[best_real_idx]):.4f}")
    print(f"[reason][replay] score_best: score={float(score_np[best_score_idx]):.4f} real_len={float(real_len_np[best_score_idx]):.4f}")
    print('[reason][top_score_candidates]')
    for rank, idx in enumerate(topk, start=1):
        print(f"  rank={rank} idx={int(idx)} score={float(score_np[idx]):.4f} real_len={float(real_len_np[idx]):.4f}")
    labels = ['real_best', 'score_best']
    for label, traj in zip(labels, traj_two):
        q0 = np.asarray(traj['start_q'], dtype=np.float32)
        jl_margin = joint_limit_margin(tracker.robot, q0)
        print(
            f"[reason][kinematics] {label}: termination={traj['termination_reason']} "
            f"steps={int(traj['num_points'])-1} min_mu={float(traj['min_mu']):.6f} "
            f"mean_mu={float(traj['mean_mu']):.6f} max_pos_error={float(traj['max_pos_error']):.4f} "
            f"boundary_hits={int(traj['boundary_hit_count'])} joint_margin={jl_margin:.6f}"
        )
    return best_real_idx, best_score_idx


def render_case(row: dict, q_batch_np: np.ndarray, score_np: np.ndarray, real_len_np: np.ndarray, traj_two: list[dict], tracker) -> None:
    best_real_idx, best_score_idx = print_failure_diagnostics(row, score_np, real_len_np, traj_two, tracker)
    pos = np.asarray(row['pos'], dtype=np.float32)
    direction = np.asarray(row['direction'], dtype=np.float32)
    normal = np.asarray(row['normal'], dtype=np.float32)

    world = wd.World(cam_pos=[1.8, -1.6, 1.1], lookat_pos=[0.2, 0.0, 0.35])
    mgm.gen_frame().attach_to(world)
    mgm.gen_sphere(pos, radius=0.01, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)
    plane_rotmat = rotation_matrix_from_normal(normal)
    plane_center = pos + 0.25 * direction
    mcm.gen_box(xyz_lengths=[0.6, 0.6, 0.001], pos=plane_center, rotmat=plane_rotmat, rgb=[0.8, 0.85, 0.9], alpha=0.2).attach_to(world)
    mgm.gen_frame(pos=pos, rotmat=plane_rotmat, ax_length=0.09).attach_to(world)

    robot = FrankaResearch3(enable_cc=True)
    faint = min(24, q_batch_np.shape[0])
    for q in q_batch_np[:faint]:
        robot.goto_given_conf(q.astype(np.float32))
        robot.gen_meshmodel(rgb=np.array([0.6, 0.6, 0.6]), alpha=0.05, toggle_tcp_frame=False).attach_to(world)

    cases = [
        ('real_best', best_real_idx, np.array([0.10, 0.85, 0.20], dtype=np.float32)),
        ('score_best', best_score_idx, np.array([0.95, 0.20, 0.20], dtype=np.float32)),
    ]
    for label, idx, color in cases:
        robot.goto_given_conf(q_batch_np[idx].astype(np.float32))
        robot.gen_meshmodel(rgb=color, alpha=0.40, toggle_tcp_frame=True).attach_to(world)
        end = pos + direction * float(real_len_np[idx])
        mgm.gen_stick(spos=pos, epos=end, radius=0.005, rgb=color, alpha=0.95).attach_to(world)
        mgm.gen_sphere(end, radius=0.01, rgb=color, alpha=0.95).attach_to(world)
        print(f'[vis] {label}: idx={idx} score={float(score_np[idx]):.4f} real_len={float(real_len_np[idx]):.4f}')

    world.run()


def main() -> None:
    args = parse_args()
    rows = parse_log(args.input)
    row = select_row(rows, args.case_idx, args.mode)
    print(f'[selected] case={row["case_idx"]} logged_real_best_len={row["real_best_len"]:.4f} logged_score_best_len={row["score_best_len"]:.4f}')
    print(f'[task] pos={row["pos"]} direction={row["direction"]} normal={row["normal"]}')
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(int(args.seed))
    q_batch_np, score_np, real_len_np, traj_two, tracker = evaluate_task(row, args.ckpt, torch.device(args.device), args)
    render_case(row, q_batch_np, score_np, real_len_np, traj_two, tracker)


if __name__ == '__main__':
    main()
