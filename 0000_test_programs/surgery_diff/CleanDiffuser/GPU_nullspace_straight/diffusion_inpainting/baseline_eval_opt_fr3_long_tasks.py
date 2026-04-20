from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
LENGTH_PREDICTION_DIR = PARENT_DIR / 'length_prediction'
if str(LENGTH_PREDICTION_DIR) not in sys.path:
    sys.path.insert(0, str(LENGTH_PREDICTION_DIR))

from baseline_eval_fr3_common import DEFAULT_TASKS_JSONL, load_tasks_from_jsonl, mean_std, rollout_large_batch, to_jsonable
from length_prediction.lnet_contrastive_fr3_rotation_cone_eval import (
    batch_position_error_and_correction,
    build_tracker,
    collect_candidate_qs,
)
from trajectory_generation.fr3_nullspace_straight import GPUNullspaceStraightTracker, TrackerConfig, directional_manipulability_batch, joints_in_range_mask


DEFAULT_OUTPUT_JSON = BASE_DIR / 'baseline_eval_opt_fr3_long_tasks_summary.json'
DEFAULT_CASES_JSONL = BASE_DIR / 'baseline_eval_opt_fr3_long_tasks_cases.jsonl'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a simple optimization-based FR3 baseline on the long-task benchmark.')
    parser.add_argument('--tasks-jsonl', type=Path, default=DEFAULT_TASKS_JSONL)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument('--cases-jsonl', type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-cases', type=int, default=2000)
    parser.add_argument('--num-candidates', type=int, default=64)
    parser.add_argument('--oversample', type=int, default=512)
    parser.add_argument('--pos-tol-mm', type=float, default=2.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--opt-steps', type=int, default=40)
    parser.add_argument('--opt-lr', type=float, default=0.03)
    parser.add_argument('--joint-center-weight', type=float, default=0.05)
    parser.add_argument('--normal-weight', type=float, default=0.20)
    parser.add_argument('--position-weight', type=float, default=10.0)
    parser.add_argument('--short-horizon-steps', type=int, default=120)
    parser.add_argument('--shortlist-size', type=int, default=16)
    parser.add_argument('--rerank-weight', type=float, default=0.10)
    parser.add_argument('--print-every', type=int, default=20)
    parser.add_argument('--rollout-batch-size', type=int, default=256)
    return parser.parse_args()


def compute_surrogate_scores_torch(
    tracker,
    q_batch: torch.Tensor,
    target_pos: torch.Tensor,
    direction: torch.Tensor,
    normal: torch.Tensor,
    joint_center_weight: float,
    normal_weight: float,
    position_weight: float,
    create_graph: bool,
) -> torch.Tensor:
    if q_batch.requires_grad:
        q_eval = q_batch
    else:
        q_eval = q_batch.detach().clone().requires_grad_(True)
    tcp_pos, tcp_rot = tracker.robot.fk_batch(q_eval)
    grads = []
    for dim in range(3):
        grad_dim = torch.autograd.grad(
            tcp_pos[:, dim].sum(),
            q_eval,
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        grads.append(grad_dim)
    j_pos = torch.stack(grads, dim=1)
    mu = directional_manipulability_batch(j_pos, direction, tracker.config.damping)
    tcp_z = tcp_rot[:, :, 2]
    normal_align = torch.sum(tcp_z * normal, dim=-1)
    pos_err = torch.linalg.norm(tcp_pos - target_pos, dim=-1)

    lower = tracker.robot.jnt_ranges[:, 0].unsqueeze(0)
    upper = tracker.robot.jnt_ranges[:, 1].unsqueeze(0)
    center = 0.5 * (lower + upper)
    span = (upper - lower).clamp_min(1e-6)
    centered = (q_eval - center) / span
    joint_center_penalty = torch.mean(centered * centered, dim=-1)

    scores = (
        torch.log(mu.clamp_min(1e-6))
        + float(normal_weight) * normal_align
        - float(position_weight) * pos_err
        - float(joint_center_weight) * joint_center_penalty
    )
    return scores


def compute_surrogate_scores(
    tracker,
    q_batch_np: np.ndarray,
    target_pos_np: np.ndarray,
    direction_np: np.ndarray,
    normal_np: np.ndarray,
    joint_center_weight: float,
    normal_weight: float,
    position_weight: float,
    create_graph: bool,
) -> torch.Tensor:
    device = tracker.robot.jnt_ranges.device
    q_batch = torch.from_numpy(q_batch_np.astype(np.float32)).to(device)
    target_pos = torch.from_numpy(np.repeat(target_pos_np[None, :], q_batch_np.shape[0], axis=0).astype(np.float32)).to(device)
    direction = torch.from_numpy(np.repeat(direction_np[None, :], q_batch_np.shape[0], axis=0).astype(np.float32)).to(device)
    normal = torch.from_numpy(np.repeat(normal_np[None, :], q_batch_np.shape[0], axis=0).astype(np.float32)).to(device)
    return compute_surrogate_scores_torch(
        tracker=tracker,
        q_batch=q_batch,
        target_pos=target_pos,
        direction=direction,
        normal=normal,
        joint_center_weight=joint_center_weight,
        normal_weight=normal_weight,
        position_weight=position_weight,
        create_graph=create_graph,
    )


def optimize_candidates(
    tracker,
    q_init_np: np.ndarray,
    target_pos_np: np.ndarray,
    direction_np: np.ndarray,
    normal_np: np.ndarray,
    opt_steps: int,
    opt_lr: float,
    joint_center_weight: float,
    normal_weight: float,
    position_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    device = tracker.robot.jnt_ranges.device
    q_param = torch.tensor(q_init_np.astype(np.float32), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([q_param], lr=float(opt_lr))
    lower = tracker.robot.jnt_ranges[:, 0].unsqueeze(0)
    upper = tracker.robot.jnt_ranges[:, 1].unsqueeze(0)
    target_pos = torch.from_numpy(np.repeat(target_pos_np[None, :], q_init_np.shape[0], axis=0).astype(np.float32)).to(device)
    direction = torch.from_numpy(np.repeat(direction_np[None, :], q_init_np.shape[0], axis=0).astype(np.float32)).to(device)
    normal = torch.from_numpy(np.repeat(normal_np[None, :], q_init_np.shape[0], axis=0).astype(np.float32)).to(device)

    for _ in range(int(opt_steps)):
        optimizer.zero_grad(set_to_none=True)
        scores = compute_surrogate_scores_torch(
            tracker=tracker,
            q_batch=q_param,
            target_pos=target_pos,
            direction=direction,
            normal=normal,
            joint_center_weight=joint_center_weight,
            normal_weight=normal_weight,
            position_weight=position_weight,
            create_graph=True,
        )
        loss = -scores.mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            q_param.clamp_(lower, upper)

    q_opt_np = q_param.detach().cpu().numpy().astype(np.float32)
    final_scores = compute_surrogate_scores(
        tracker=tracker,
        q_batch_np=q_opt_np,
        target_pos_np=target_pos_np,
        direction_np=direction_np,
        normal_np=normal_np,
        joint_center_weight=joint_center_weight,
        normal_weight=normal_weight,
        position_weight=position_weight,
        create_graph=False,
    ).detach().cpu().numpy().astype(np.float32)
    return q_opt_np, final_scores


def valid_mask_after_correction(tracker, q_batch_np: np.ndarray, normal_np: np.ndarray) -> np.ndarray:
    device = tracker.robot.jnt_ranges.device
    q_batch = torch.from_numpy(q_batch_np.astype(np.float32)).to(device)
    _, tcp_rot = tracker.robot.fk_batch(q_batch)
    tcp_z = tcp_rot[:, :, 2]
    target_normal = torch.from_numpy(np.repeat(normal_np[None, :], q_batch_np.shape[0], axis=0).astype(np.float32)).to(device)
    cos_theta = torch.sum(tcp_z * target_normal, dim=-1)
    valid = (
        joints_in_range_mask(tracker.robot, q_batch)
        & (tracker.collision_fn(q_batch) <= 0.0)
        & (cos_theta > float(np.cos(tracker.config.theta_max)))
    )
    return valid.detach().cpu().numpy().astype(bool)


def topk_indices_desc(values: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), int(values.shape[0])))
    order = np.argsort(-values, kind='stable')
    return order[:k].astype(np.int64)


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
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
    tasks = load_tasks_from_jsonl(args.tasks_jsonl, args.num_cases)

    args.cases_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.cases_jsonl.write_text('', encoding='utf-8')

    selected_qs: list[np.ndarray] = []
    selected_directions: list[np.ndarray] = []
    selected_normals: list[np.ndarray] = []
    buffered_case_rows: list[dict] = []

    aggregate = {
        'gt_real': [],
        'selected_real': [],
        'gain_vs_gt': [],
        'oracle_ratio': [],
        'raw_pos_err_mm': [],
        'candidate_collection_s': [],
        'optimization_s': [],
        'repair_s': [],
        'short_rollout_s': [],
        'rollout_time_s': [],
        'total_case_time_s': [],
        'surrogate_score': [],
        'short_horizon_length': [],
    }

    t_global0 = time.perf_counter()
    for task_idx, task in enumerate(tasks, start=1):
        case_t0 = time.perf_counter()

        cand_t0 = time.perf_counter()
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
        cand_t1 = time.perf_counter()

        pre_score_t0 = time.perf_counter()
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
        pre_score_t1 = time.perf_counter()

        opt_t0 = time.perf_counter()
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
        opt_t1 = time.perf_counter()

        repair_t0 = time.perf_counter()
        q_corrected, raw_pos_err = batch_position_error_and_correction(
            tracker.robot,
            q_optimized,
            task['pos'],
            float(args.correction_damping),
            int(args.correction_iters),
            float(args.correction_tol),
            tracker_device,
        )
        repair_t1 = time.perf_counter()
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
        short_rollout_t0 = time.perf_counter()
        short_lengths = rollout_large_batch(
            tracker=short_tracker,
            tracker_device=tracker_device,
            q_batch=q_corrected.astype(np.float32),
            direction_batch=np.repeat(task['direction'][None, :], q_corrected.shape[0], axis=0).astype(np.float32),
            target_normal_batch=np.repeat(task['target_normal'][None, :], q_corrected.shape[0], axis=0).astype(np.float32),
            chunk_size=min(int(args.rollout_batch_size), max(1, int(q_corrected.shape[0]))),
        )
        short_rollout_t1 = time.perf_counter()

        valid_mask = valid_mask_after_correction(tracker, q_corrected, task['target_normal'])
        if bool(np.any(valid_mask)):
            local_scores = (short_lengths + float(args.rerank_weight) * final_scores).astype(np.float32)
            local_scores[~valid_mask] = -np.inf
            best_idx = int(np.argmax(local_scores))
        else:
            best_idx = int(np.argmax(short_lengths + float(args.rerank_weight) * final_scores))

        selected_qs.append(q_corrected[best_idx].astype(np.float32))
        selected_directions.append(task['direction'].astype(np.float32))
        selected_normals.append(task['target_normal'].astype(np.float32))

        candidate_collection_s = float(cand_t1 - cand_t0)
        prescore_s = float(pre_score_t1 - pre_score_t0)
        optimization_s = float(opt_t1 - opt_t0)
        repair_s = float(repair_t1 - repair_t0)
        short_rollout_s = float(short_rollout_t1 - short_rollout_t0)
        aggregate['gt_real'].append(float(task['gt_real']))
        aggregate['raw_pos_err_mm'].append(float(raw_pos_err[best_idx] * 1e3))
        aggregate['candidate_collection_s'].append(candidate_collection_s)
        aggregate['optimization_s'].append(optimization_s + prescore_s)
        aggregate['repair_s'].append(repair_s)
        aggregate['short_rollout_s'].append(short_rollout_s)
        aggregate['surrogate_score'].append(float(final_scores[best_idx]))
        aggregate['short_horizon_length'].append(float(short_lengths[best_idx]))

        buffered_case_rows.append({
            'task_index': int(task['task_index']),
            'oracle_rank_among_6000': int(task['oracle_rank_among_6000']),
            'task_category': task['task_category'],
            'gt_real': float(task['gt_real']),
            'direction': task['direction'].astype(np.float32),
            'target_normal': task['target_normal'].astype(np.float32),
            'selected_sample_idx': int(shortlist_idx[best_idx]),
            'selected_from_num_samples': int(q_candidates.shape[0]),
            'shortlist_size': int(q_corrected.shape[0]),
            'final_score': float(final_scores[best_idx]),
            'short_horizon_length': float(short_lengths[best_idx]),
            'raw_pos_err_mm': float(raw_pos_err[best_idx] * 1e3),
            'q_corrected': q_corrected[best_idx].astype(np.float32),
            'candidate_collection_s': candidate_collection_s,
            'prescore_s': prescore_s,
            'optimization_s': optimization_s,
            'repair_s': repair_s,
            'short_rollout_s': short_rollout_s,
            'case_time_prefix_s': float(time.perf_counter() - case_t0),
        })

        if task_idx % max(1, int(args.print_every)) == 0 or task_idx == len(tasks):
            print(
                f'[opt-baseline] task {task_idx}/{len(tasks)} '
                f'gt={task["gt_real"]:.3f} cand_t={candidate_collection_s:.3f}s '
                f'opt_t={optimization_s:.3f}s short_t={short_rollout_s:.3f}s '
                f'short_len={float(short_lengths[best_idx]):.3f}',
                flush=True,
            )

    rollout_t0 = time.perf_counter()
    selected_real = rollout_large_batch(
        tracker=tracker,
        tracker_device=tracker_device,
        q_batch=np.stack(selected_qs, axis=0).astype(np.float32),
        direction_batch=np.stack(selected_directions, axis=0).astype(np.float32),
        target_normal_batch=np.stack(selected_normals, axis=0).astype(np.float32),
        chunk_size=int(args.rollout_batch_size),
    )
    rollout_t1 = time.perf_counter()
    rollout_total_s = float(rollout_t1 - rollout_t0)
    rollout_share_s = rollout_total_s / max(len(buffered_case_rows), 1)

    for row, selected_real_i in zip(buffered_case_rows, selected_real.tolist()):
        gain_vs_gt = float(selected_real_i - row['gt_real'])
        oracle_ratio = 100.0 * float(selected_real_i) / max(float(row['gt_real']), 1e-6)
        total_case_time_s = float(row['case_time_prefix_s']) + rollout_share_s

        aggregate['selected_real'].append(float(selected_real_i))
        aggregate['gain_vs_gt'].append(gain_vs_gt)
        aggregate['oracle_ratio'].append(oracle_ratio)
        aggregate['rollout_time_s'].append(rollout_share_s)
        aggregate['total_case_time_s'].append(total_case_time_s)

        payload = {
            'method': 'opt_based',
            'task_index': int(row['task_index']),
            'oracle_rank_among_6000': int(row['oracle_rank_among_6000']),
            'task_category': row['task_category'],
            'gt_real': float(row['gt_real']),
            'direction': row['direction'],
            'target_normal': row['target_normal'],
            'selected_sample_idx': int(row['selected_sample_idx']),
            'selected_from_num_samples': int(row['selected_from_num_samples']),
            'shortlist_size': int(row['shortlist_size']),
            'final_score': float(row['final_score']),
            'short_horizon_length': float(row['short_horizon_length']),
            'selected_real': float(selected_real_i),
            'gain_vs_gt': gain_vs_gt,
            'percent_of_oracle': oracle_ratio,
            'raw_pos_err_mm': float(row['raw_pos_err_mm']),
            'q_corrected': row['q_corrected'],
            'candidate_collection_s': float(row['candidate_collection_s']),
            'prescore_s': float(row['prescore_s']),
            'optimization_s': float(row['optimization_s']),
            'repair_s': float(row['repair_s']),
            'short_rollout_s': float(row['short_rollout_s']),
            'rollout_time_share_s': rollout_share_s,
            'total_case_time_s': total_case_time_s,
        }
        with args.cases_jsonl.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + '\n')

    summary = {
        'method': 'opt_based',
        'num_cases': int(len(tasks)),
        'tasks_jsonl': str(args.tasks_jsonl),
        'cases_jsonl': str(args.cases_jsonl),
        'num_candidates': int(args.num_candidates),
        'oversample': int(args.oversample),
        'opt_steps': int(args.opt_steps),
        'opt_lr': float(args.opt_lr),
        'short_horizon_steps': int(args.short_horizon_steps),
        'shortlist_size': int(args.shortlist_size),
        'rerank_weight': float(args.rerank_weight),
        'joint_center_weight': float(args.joint_center_weight),
        'normal_weight': float(args.normal_weight),
        'position_weight': float(args.position_weight),
        'selected_real': mean_std(aggregate['selected_real']),
        'gain_vs_gt': mean_std(aggregate['gain_vs_gt']),
        'percent_of_oracle': mean_std(aggregate['oracle_ratio']),
        'raw_pos_err_mm': mean_std(aggregate['raw_pos_err_mm']),
        'candidate_collection_s': mean_std(aggregate['candidate_collection_s']),
        'optimization_s': mean_std(aggregate['optimization_s']),
        'repair_s': mean_std(aggregate['repair_s']),
        'short_rollout_s': mean_std(aggregate['short_rollout_s']),
        'rollout_time_s': mean_std(aggregate['rollout_time_s']),
        'total_case_time_s': mean_std(aggregate['total_case_time_s']),
        'surrogate_score': mean_std(aggregate['surrogate_score']),
        'short_horizon_length': mean_std(aggregate['short_horizon_length']),
        'wall_clock_s': float(time.perf_counter() - t_global0),
    }
    args.output_json.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
