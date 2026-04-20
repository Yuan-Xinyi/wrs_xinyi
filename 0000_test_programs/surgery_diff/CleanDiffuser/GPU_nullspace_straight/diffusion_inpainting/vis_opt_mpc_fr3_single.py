from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd
from wrs.robot_sim.robots.franka_research_3.franka_research_3 import FrankaResearch3

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
LENGTH_PREDICTION_DIR = PARENT_DIR / 'length_prediction'
if str(LENGTH_PREDICTION_DIR) not in sys.path:
    sys.path.insert(0, str(LENGTH_PREDICTION_DIR))

from baseline_eval_fr3_common import DEFAULT_TASKS_JSONL, load_tasks_from_jsonl, rollout_large_batch
from baseline_eval_opt_fr3_long_tasks import (
    compute_surrogate_scores,
    optimize_candidates,
    topk_indices_desc,
    valid_mask_after_correction,
)
from length_prediction.lnet_contrastive_fr3_rotation_cone_eval import (
    batch_position_error_and_correction,
    build_tracker,
    collect_candidate_qs,
)
from trajectory_generation.fr3_nullspace_straight import GPUNullspaceStraightTracker, TrackerConfig


METHOD_COLORS = {
    'gt': np.array([0.12, 0.75, 0.22], dtype=np.float32),
    'opt': np.array([0.12, 0.42, 0.92], dtype=np.float32),
    'mpc': np.array([0.95, 0.35, 0.15], dtype=np.float32),
    'raw_opt': np.array([0.55, 0.72, 0.98], dtype=np.float32),
    'raw_mpc': np.array([0.98, 0.72, 0.48], dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize one FR3 long-task case for Opt-based and MPC-based baselines.')
    parser.add_argument('--tasks-jsonl', type=Path, default=DEFAULT_TASKS_JSONL)
    parser.add_argument('--task-index', type=int, default=0, help='Zero-based row index in tasks-jsonl.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-candidates', type=int, default=32)
    parser.add_argument('--oversample', type=int, default=256)
    parser.add_argument('--pos-tol-mm', type=float, default=2.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--opt-steps', type=int, default=20)
    parser.add_argument('--opt-lr', type=float, default=0.03)
    parser.add_argument('--shortlist-size', type=int, default=8)
    parser.add_argument('--short-horizon-steps', type=int, default=400)
    parser.add_argument('--rerank-weight', type=float, default=0.10)
    parser.add_argument('--joint-center-weight', type=float, default=0.05)
    parser.add_argument('--normal-weight', type=float, default=0.20)
    parser.add_argument('--position-weight', type=float, default=10.0)
    parser.add_argument('--mpc-coarse-steps', type=int, default=40)
    parser.add_argument('--mpc-fine-steps', type=int, default=160)
    parser.add_argument('--mpc-topk', type=int, default=8)
    parser.add_argument('--rollout-batch-size', type=int, default=128)
    parser.add_argument('--show-rollout-meshes', action='store_true')
    parser.add_argument('--rollout-mesh-count', type=int, default=8)
    parser.add_argument('--show-tcp-frame', action='store_true')
    return parser.parse_args()


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-12)


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normalize(normal)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = normalize(np.cross(helper, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)


def rotation_matrix_from_direction_normal(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    x_axis = normalize(direction)
    z_axis = normalize(normal)
    y_axis = np.cross(z_axis, x_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        return rotation_matrix_from_normal(z_axis)
    y_axis = normalize(y_axis)
    z_axis = normalize(np.cross(x_axis, y_axis))
    return np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)


def rollout_single(tracker, tracker_device: torch.device, q_np: np.ndarray, direction: np.ndarray, target_normal: np.ndarray) -> dict:
    q_batch = torch.from_numpy(q_np[None, :].astype(np.float32)).to(tracker_device)
    d_batch = torch.from_numpy(direction[None, :].astype(np.float32)).to(tracker_device)
    n_batch = torch.from_numpy(target_normal[None, :].astype(np.float32)).to(tracker_device)
    rollout = tracker.run_batch(q0_batch=q_batch, direction_batch=d_batch, target_normal_batch=n_batch)
    return {
        'real_length': float(rollout.projected_length[0].detach().cpu().item()),
        'steps': int(rollout.steps[0].detach().cpu().item()),
        'termination_code': int(rollout.termination_code[0].detach().cpu().item()),
        'q_path': np.asarray(rollout.q_path_best, dtype=np.float32),
        'tcp_path': np.asarray(rollout.tcp_path_best, dtype=np.float32),
    }


def solve_opt_case(
    tracker,
    tracker_device: torch.device,
    short_tracker,
    task: dict,
    args: argparse.Namespace,
) -> dict:
    t0 = time.perf_counter()
    q_candidates, _ = collect_candidate_qs(
        tracker=tracker,
        tracker_device=tracker_device,
        target_pos=task['pos'],
        direction=task['direction'],
        normal=task['target_normal'],
        num_candidates=int(args.num_candidates),
        oversample=int(args.oversample),
        pos_tol_mm=float(args.pos_tol_mm),
        correction_iters=int(args.correction_iters),
        correction_tol=float(args.correction_tol),
        correction_damping=float(args.correction_damping),
    )
    t1 = time.perf_counter()

    initial_scores = compute_surrogate_scores(
        tracker=tracker,
        q_batch_np=q_candidates,
        target_pos_np=task['pos'],
        direction_np=task['direction'],
        normal_np=task['target_normal'],
        joint_center_weight=float(args.joint_center_weight),
        normal_weight=float(args.normal_weight),
        position_weight=float(args.position_weight),
        create_graph=False,
    ).detach().cpu().numpy().astype(np.float32)
    shortlist_idx = topk_indices_desc(initial_scores, int(args.shortlist_size))
    q_short = q_candidates[shortlist_idx].astype(np.float32)

    t2 = time.perf_counter()
    q_optimized, _ = optimize_candidates(
        tracker=tracker,
        q_init_np=q_short,
        target_pos_np=task['pos'],
        direction_np=task['direction'],
        normal_np=task['target_normal'],
        opt_steps=int(args.opt_steps),
        opt_lr=float(args.opt_lr),
        joint_center_weight=float(args.joint_center_weight),
        normal_weight=float(args.normal_weight),
        position_weight=float(args.position_weight),
    )
    t3 = time.perf_counter()

    q_corrected, raw_pos_err = batch_position_error_and_correction(
        tracker.robot,
        q_optimized,
        task['pos'],
        float(args.correction_damping),
        int(args.correction_iters),
        float(args.correction_tol),
        tracker_device,
    )
    final_scores = compute_surrogate_scores(
        tracker=tracker,
        q_batch_np=q_corrected,
        target_pos_np=task['pos'],
        direction_np=task['direction'],
        normal_np=task['target_normal'],
        joint_center_weight=float(args.joint_center_weight),
        normal_weight=float(args.normal_weight),
        position_weight=float(args.position_weight),
        create_graph=False,
    ).detach().cpu().numpy().astype(np.float32)

    short_lengths = rollout_large_batch(
        tracker=short_tracker,
        tracker_device=tracker_device,
        q_batch=q_corrected.astype(np.float32),
        direction_batch=np.repeat(task['direction'][None, :], q_corrected.shape[0], axis=0).astype(np.float32),
        target_normal_batch=np.repeat(task['target_normal'][None, :], q_corrected.shape[0], axis=0).astype(np.float32),
        chunk_size=min(int(args.rollout_batch_size), max(1, int(q_corrected.shape[0]))),
    )
    valid_mask = valid_mask_after_correction(tracker, q_corrected, task['target_normal'])
    if bool(np.any(valid_mask)):
        rank_score = short_lengths + float(args.rerank_weight) * final_scores
        rank_score[~valid_mask] = -np.inf
        best_local = int(np.argmax(rank_score))
    else:
        best_local = int(np.argmax(short_lengths + float(args.rerank_weight) * final_scores))

    best_q = q_corrected[best_local].astype(np.float32)
    rollout = rollout_single(tracker, tracker_device, best_q, task['direction'], task['target_normal'])
    t4 = time.perf_counter()
    return {
        'method': 'opt',
        'selected_sample_idx': int(shortlist_idx[best_local]),
        'shortlist_size': int(q_corrected.shape[0]),
        'short_horizon_length': float(short_lengths[best_local]),
        'final_score': float(final_scores[best_local]),
        'raw_pos_err_mm': float(raw_pos_err[best_local] * 1e3),
        'q_corr': best_q,
        'timing': {
            'candidate_collection_s': float(t1 - t0),
            'optimization_s': float(t3 - t2),
            'short_rollout_s': float(t4 - t3),
            'total_pre_eval_s': float(t4 - t0),
        },
        **rollout,
    }


def solve_mpc_case(
    tracker,
    tracker_device: torch.device,
    coarse_tracker,
    fine_tracker,
    task: dict,
    args: argparse.Namespace,
) -> dict:
    t0 = time.perf_counter()
    q_candidates, raw_err = collect_candidate_qs(
        tracker=tracker,
        tracker_device=tracker_device,
        target_pos=task['pos'],
        direction=task['direction'],
        normal=task['target_normal'],
        num_candidates=int(args.num_candidates),
        oversample=int(args.oversample),
        pos_tol_mm=float(args.pos_tol_mm),
        correction_iters=int(args.correction_iters),
        correction_tol=float(args.correction_tol),
        correction_damping=float(args.correction_damping),
    )
    t1 = time.perf_counter()
    q_batch = q_candidates.astype(np.float32)
    d_batch = np.repeat(task['direction'][None, :], q_batch.shape[0], axis=0).astype(np.float32)
    n_batch = np.repeat(task['target_normal'][None, :], q_batch.shape[0], axis=0).astype(np.float32)
    coarse_lengths = rollout_large_batch(
        tracker=coarse_tracker,
        tracker_device=tracker_device,
        q_batch=q_batch,
        direction_batch=d_batch,
        target_normal_batch=n_batch,
        chunk_size=int(args.rollout_batch_size),
    )
    t2 = time.perf_counter()
    topk = max(1, min(int(args.mpc_topk), int(q_batch.shape[0])))
    top_idx = np.argsort(-coarse_lengths, kind='stable')[:topk]
    fine_lengths = rollout_large_batch(
        tracker=fine_tracker,
        tracker_device=tracker_device,
        q_batch=q_batch[top_idx].astype(np.float32),
        direction_batch=d_batch[top_idx].astype(np.float32),
        target_normal_batch=n_batch[top_idx].astype(np.float32),
        chunk_size=min(int(args.rollout_batch_size), topk),
    )
    best_local = int(np.argmax(fine_lengths))
    best_idx = int(top_idx[best_local])
    best_q = q_candidates[best_idx].astype(np.float32)
    rollout = rollout_single(tracker, tracker_device, best_q, task['direction'], task['target_normal'])
    t3 = time.perf_counter()
    return {
        'method': 'mpc',
        'selected_sample_idx': int(best_idx),
        'mpc_topk': int(topk),
        'coarse_horizon_length': float(coarse_lengths[best_idx]),
        'fine_horizon_length': float(fine_lengths[best_local]),
        'raw_pos_err_mm': float(raw_err[best_idx] * 1e3),
        'q_corr': best_q,
        'timing': {
            'candidate_collection_s': float(t1 - t0),
            'mpc_coarse_s': float(t2 - t1),
            'mpc_fine_s': float(t3 - t2),
            'total_pre_eval_s': float(t3 - t0),
        },
        **rollout,
    }


def add_path(world, tcp_path: np.ndarray, color: np.ndarray, radius: float) -> None:
    if tcp_path.shape[0] < 2:
        return
    for idx in range(tcp_path.shape[0] - 1):
        mgm.gen_stick(spos=tcp_path[idx], epos=tcp_path[idx + 1], radius=radius, rgb=color, alpha=0.92).attach_to(world)
    mgm.gen_sphere(tcp_path[0], radius=radius * 1.8, rgb=color, alpha=0.95).attach_to(world)
    mgm.gen_sphere(tcp_path[-1], radius=radius * 2.0, rgb=color, alpha=0.95).attach_to(world)


def add_rollout_meshes(
    world,
    robot: FrankaResearch3,
    q_path: np.ndarray,
    color: np.ndarray,
    count: int,
    alpha: float,
    show_tcp_frame: bool,
) -> None:
    if q_path.ndim != 2 or q_path.shape[0] == 0:
        return
    count = max(1, min(int(count), int(q_path.shape[0])))
    indices = np.linspace(0, q_path.shape[0] - 1, count, dtype=int)
    used: list[int] = []
    for idx in indices.tolist():
        if used and idx == used[-1]:
            continue
        used.append(idx)
    for idx in used:
        robot.goto_given_conf(np.asarray(q_path[idx], dtype=np.float32))
        robot.gen_meshmodel(rgb=color, alpha=alpha, toggle_tcp_frame=show_tcp_frame).attach_to(world)


def render_world(task: dict, gt_row: dict, rows: list[dict], args: argparse.Namespace) -> None:
    world = wd.World(cam_pos=[1.7, -1.5, 1.05], lookat_pos=[0.25, 0.0, 0.25])
    mgm.gen_frame().attach_to(world)
    anchor_rotmat = rotation_matrix_from_direction_normal(task['direction'], task['target_normal'])
    start_pos = task['pos']
    mgm.gen_frame(pos=start_pos, rotmat=anchor_rotmat, ax_length=0.12).attach_to(world)

    plane_size = 1.2
    plane_rotmat = rotation_matrix_from_normal(task['target_normal'])
    plane_center = start_pos + 0.5 * plane_size * task['direction']
    mcm.gen_box(
        xyz_lengths=[plane_size, plane_size, 0.001],
        pos=plane_center,
        rotmat=plane_rotmat,
        rgb=[180 / 255, 211 / 255, 217 / 255],
        alpha=0.35,
    ).attach_to(world)

    gt_color = METHOD_COLORS['gt']
    gt_end = start_pos + task['direction'] * float(gt_row['real_length'])
    mgm.gen_stick(spos=start_pos, epos=gt_end, radius=0.0048, rgb=gt_color, alpha=0.90).attach_to(world)
    mgm.gen_sphere(gt_end, radius=0.009, rgb=gt_color, alpha=0.95).attach_to(world)

    robot = FrankaResearch3(enable_cc=True)
    robot.goto_given_conf(np.asarray(gt_row['q_corr'], dtype=np.float32))
    robot.gen_meshmodel(rgb=gt_color, alpha=0.38, toggle_tcp_frame=bool(args.show_tcp_frame)).attach_to(world)

    for row in rows:
        color = METHOD_COLORS[row['method']]
        raw_color = METHOD_COLORS[f'raw_{row["method"]}']
        add_path(world, np.asarray(row['tcp_path'], dtype=np.float32), color=color, radius=0.0038)
        mgm.gen_stick(
            spos=start_pos,
            epos=np.asarray(row['tcp_path'][0], dtype=np.float32),
            radius=0.0024,
            rgb=raw_color,
            alpha=0.70,
        ).attach_to(world)
        robot.goto_given_conf(np.asarray(row['q_corr'], dtype=np.float32))
        robot.gen_meshmodel(rgb=color, alpha=0.35, toggle_tcp_frame=bool(args.show_tcp_frame)).attach_to(world)
        if args.show_rollout_meshes:
            add_rollout_meshes(
                world=world,
                robot=robot,
                q_path=np.asarray(row['q_path'], dtype=np.float32),
                color=color,
                count=int(args.rollout_mesh_count),
                alpha=0.10,
                show_tcp_frame=False,
            )

    world.run()


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)

    tasks = load_tasks_from_jsonl(args.tasks_jsonl)
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise RuntimeError(f'task_index={args.task_index} is out of range for {len(tasks)} tasks in {args.tasks_jsonl}')
    task = tasks[int(args.task_index)]

    tracker, tracker_device = build_tracker(device)
    short_config = TrackerConfig(**vars(tracker.config))
    short_config.max_steps = int(args.short_horizon_steps)
    short_tracker = GPUNullspaceStraightTracker(
        robot=tracker.robot,
        collision_fn=tracker.collision_fn,
        config=short_config,
        print_every=0,
    )
    coarse_config = TrackerConfig(**vars(tracker.config))
    coarse_config.max_steps = int(args.mpc_coarse_steps)
    coarse_tracker = GPUNullspaceStraightTracker(
        robot=tracker.robot,
        collision_fn=tracker.collision_fn,
        config=coarse_config,
        print_every=0,
    )
    fine_config = TrackerConfig(**vars(tracker.config))
    fine_config.max_steps = int(args.mpc_fine_steps)
    fine_tracker = GPUNullspaceStraightTracker(
        robot=tracker.robot,
        collision_fn=tracker.collision_fn,
        config=fine_config,
        print_every=0,
    )

    gt_q = task.get('gt_best_q')
    if gt_q is None:
        gt_q = None
    gt_row = {
        'method': 'gt',
        'q_corr': np.zeros(7, dtype=np.float32),
        'real_length': float(task['gt_real']),
    }

    opt_row = solve_opt_case(tracker, tracker_device, short_tracker, task, args)
    mpc_row = solve_mpc_case(tracker, tracker_device, coarse_tracker, fine_tracker, task, args)

    # Use the better of the two candidate solutions to mark a concrete robot pose when the oracle q is unavailable.
    gt_row['q_corr'] = opt_row['q_corr'] if opt_row['real_length'] >= mpc_row['real_length'] else mpc_row['q_corr']

    print(
        json.dumps(
            {
                'task_index': int(task['task_index']),
                'oracle_rank_among_6000': int(task['oracle_rank_among_6000']),
                'gt_real': float(task['gt_real']),
                'opt': {
                    'real_length': float(opt_row['real_length']),
                    'raw_pos_err_mm': float(opt_row['raw_pos_err_mm']),
                    'timing': opt_row['timing'],
                    'short_horizon_length': float(opt_row['short_horizon_length']),
                },
                'mpc': {
                    'real_length': float(mpc_row['real_length']),
                    'raw_pos_err_mm': float(mpc_row['raw_pos_err_mm']),
                    'timing': mpc_row['timing'],
                    'coarse_horizon_length': float(mpc_row['coarse_horizon_length']),
                    'fine_horizon_length': float(mpc_row['fine_horizon_length']),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    render_world(task=task, gt_row=gt_row, rows=[opt_row, mpc_row], args=args)


if __name__ == '__main__':
    main()
