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
from length_prediction.lnet_contrastive_fr3_rotation_cone_eval import build_tracker, collect_candidate_qs
from trajectory_generation.fr3_nullspace_straight import GPUNullspaceStraightTracker, TrackerConfig


DEFAULT_OUTPUT_JSON = BASE_DIR / 'baseline_eval_mpc_fr3_long_tasks_summary.json'
DEFAULT_CASES_JSONL = BASE_DIR / 'baseline_eval_mpc_fr3_long_tasks_cases.jsonl'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a simple MPC-style FR3 baseline on the long-task benchmark.')
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
    parser.add_argument('--mpc-coarse-steps', type=int, default=40)
    parser.add_argument('--mpc-fine-steps', type=int, default=160)
    parser.add_argument('--mpc-topk', type=int, default=8)
    parser.add_argument('--print-every', type=int, default=20)
    parser.add_argument('--rollout-batch-size', type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)

    full_tracker, tracker_device = build_tracker(device)
    coarse_config = TrackerConfig(**vars(full_tracker.config))
    coarse_config.max_steps = int(args.mpc_coarse_steps)
    coarse_tracker = GPUNullspaceStraightTracker(
        robot=full_tracker.robot,
        collision_fn=full_tracker.collision_fn,
        config=coarse_config,
        print_every=0,
    )
    fine_config = TrackerConfig(**vars(full_tracker.config))
    fine_config.max_steps = int(args.mpc_fine_steps)
    fine_tracker = GPUNullspaceStraightTracker(
        robot=full_tracker.robot,
        collision_fn=full_tracker.collision_fn,
        config=fine_config,
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
        'candidate_collection_s': [],
        'mpc_coarse_s': [],
        'mpc_fine_s': [],
        'rollout_time_s': [],
        'total_case_time_s': [],
        'mpc_coarse_length': [],
        'mpc_fine_length': [],
    }

    t_global0 = time.perf_counter()
    for task_idx, task in enumerate(tasks, start=1):
        case_t0 = time.perf_counter()

        cand_t0 = time.perf_counter()
        q_candidates, raw_err = collect_candidate_qs(
            tracker=full_tracker,
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

        coarse_t0 = time.perf_counter()
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
        coarse_t1 = time.perf_counter()
        topk = max(1, min(int(args.mpc_topk), int(q_batch.shape[0])))
        top_idx = np.argsort(-coarse_lengths, kind='stable')[:topk]
        fine_t0 = time.perf_counter()
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
        fine_t1 = time.perf_counter()

        selected_qs.append(q_candidates[best_idx].astype(np.float32))
        selected_directions.append(task['direction'].astype(np.float32))
        selected_normals.append(task['target_normal'].astype(np.float32))

        candidate_collection_s = float(cand_t1 - cand_t0)
        coarse_s = float(coarse_t1 - coarse_t0)
        fine_s = float(fine_t1 - fine_t0)
        aggregate['gt_real'].append(float(task['gt_real']))
        aggregate['candidate_collection_s'].append(candidate_collection_s)
        aggregate['mpc_coarse_s'].append(coarse_s)
        aggregate['mpc_fine_s'].append(fine_s)
        aggregate['mpc_coarse_length'].append(float(coarse_lengths[best_idx]))
        aggregate['mpc_fine_length'].append(float(fine_lengths[best_local]))

        buffered_case_rows.append({
            'task_index': int(task['task_index']),
            'oracle_rank_among_6000': int(task['oracle_rank_among_6000']),
            'task_category': task['task_category'],
            'gt_real': float(task['gt_real']),
            'direction': task['direction'].astype(np.float32),
            'target_normal': task['target_normal'].astype(np.float32),
            'selected_sample_idx': int(best_idx),
            'selected_from_num_samples': int(q_candidates.shape[0]),
            'mpc_topk': int(topk),
            'final_score': float(fine_lengths[best_local]),
            'coarse_horizon_length': float(coarse_lengths[best_idx]),
            'fine_horizon_length': float(fine_lengths[best_local]),
            'raw_pos_err_mm': float(raw_err[best_idx] * 1e3),
            'q_corrected': q_candidates[best_idx].astype(np.float32),
            'candidate_collection_s': candidate_collection_s,
            'mpc_coarse_s': coarse_s,
            'mpc_fine_s': fine_s,
            'case_time_prefix_s': float(time.perf_counter() - case_t0),
        })

        if task_idx % max(1, int(args.print_every)) == 0 or task_idx == len(tasks):
            print(
                f'[mpc-baseline] task {task_idx}/{len(tasks)} '
                f'gt={task["gt_real"]:.3f} cand_t={candidate_collection_s:.3f}s '
                f'coarse_t={coarse_s:.3f}s fine_t={fine_s:.3f}s '
                f'coarse={float(coarse_lengths[best_idx]):.3f} fine={float(fine_lengths[best_local]):.3f}',
                flush=True,
            )

    rollout_t0 = time.perf_counter()
    selected_real = rollout_large_batch(
        tracker=full_tracker,
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
            'method': 'mpc_based',
            'task_index': int(row['task_index']),
            'oracle_rank_among_6000': int(row['oracle_rank_among_6000']),
            'task_category': row['task_category'],
            'gt_real': float(row['gt_real']),
            'direction': row['direction'],
            'target_normal': row['target_normal'],
            'selected_sample_idx': int(row['selected_sample_idx']),
            'selected_from_num_samples': int(row['selected_from_num_samples']),
            'mpc_topk': int(row['mpc_topk']),
            'final_score': float(row['final_score']),
            'coarse_horizon_length': float(row['coarse_horizon_length']),
            'fine_horizon_length': float(row['fine_horizon_length']),
            'selected_real': float(selected_real_i),
            'gain_vs_gt': gain_vs_gt,
            'percent_of_oracle': oracle_ratio,
            'raw_pos_err_mm': float(row['raw_pos_err_mm']),
            'q_corrected': row['q_corrected'],
            'candidate_collection_s': float(row['candidate_collection_s']),
            'mpc_coarse_s': float(row['mpc_coarse_s']),
            'mpc_fine_s': float(row['mpc_fine_s']),
            'rollout_time_share_s': rollout_share_s,
            'total_case_time_s': total_case_time_s,
            'mpc_coarse_steps': int(args.mpc_coarse_steps),
            'mpc_fine_steps': int(args.mpc_fine_steps),
        }
        with args.cases_jsonl.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + '\n')

    summary = {
        'method': 'mpc_based',
        'num_cases': int(len(tasks)),
        'tasks_jsonl': str(args.tasks_jsonl),
        'cases_jsonl': str(args.cases_jsonl),
        'num_candidates': int(args.num_candidates),
        'oversample': int(args.oversample),
        'mpc_coarse_steps': int(args.mpc_coarse_steps),
        'mpc_fine_steps': int(args.mpc_fine_steps),
        'mpc_topk': int(args.mpc_topk),
        'selected_real': mean_std(aggregate['selected_real']),
        'gain_vs_gt': mean_std(aggregate['gain_vs_gt']),
        'percent_of_oracle': mean_std(aggregate['oracle_ratio']),
        'raw_pos_err_mm': mean_std([float(row['raw_pos_err_mm']) for row in buffered_case_rows]),
        'candidate_collection_s': mean_std(aggregate['candidate_collection_s']),
        'mpc_coarse_s': mean_std(aggregate['mpc_coarse_s']),
        'mpc_fine_s': mean_std(aggregate['mpc_fine_s']),
        'rollout_time_s': mean_std(aggregate['rollout_time_s']),
        'total_case_time_s': mean_std(aggregate['total_case_time_s']),
        'mpc_coarse_length': mean_std(aggregate['mpc_coarse_length']),
        'mpc_fine_length': mean_std(aggregate['mpc_fine_length']),
        'wall_clock_s': float(time.perf_counter() - t_global0),
    }
    args.output_json.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
