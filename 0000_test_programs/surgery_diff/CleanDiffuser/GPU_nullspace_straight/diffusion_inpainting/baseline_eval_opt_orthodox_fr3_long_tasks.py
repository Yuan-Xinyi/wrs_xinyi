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

from baseline_eval_fr3_common import DEFAULT_TASKS_JSONL, load_tasks_from_jsonl, mean_std, rollout_large_batch, to_jsonable
from orthodox_baseline_fr3_common import (
    RolloutObjectiveWeights,
    build_tracker,
    differentiable_closed_loop_rollout_score,
    optimize_start_pose_multistart,
    q_from_raw,
    raw_from_q,
    repeat_vec,
)
from length_prediction.lnet_contrastive_fr3_rotation_cone_eval import batch_position_error_and_correction


DEFAULT_OUTPUT_JSON = BASE_DIR / 'baseline_eval_opt_orthodox_fr3_long_tasks_summary.json'
DEFAULT_CASES_JSONL = BASE_DIR / 'baseline_eval_opt_orthodox_fr3_long_tasks_cases.jsonl'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate an orthodox continuous-optimization FR3 baseline on the long-task benchmark.')
    parser.add_argument('--tasks-jsonl', type=Path, default=DEFAULT_TASKS_JSONL)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument('--cases-jsonl', type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-cases', type=int, default=2000)
    parser.add_argument('--num-starts', type=int, default=16)
    parser.add_argument('--ik-iters', type=int, default=80)
    parser.add_argument('--ik-lr', type=float, default=0.05)
    parser.add_argument('--opt-iters', type=int, default=80)
    parser.add_argument('--opt-lr', type=float, default=0.03)
    parser.add_argument('--rollout-horizon-steps', type=int, default=120)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--rollout-batch-size', type=int, default=256)
    return parser.parse_args()


def optimize_initial_configuration(
    tracker,
    task: dict,
    q_init_np: np.ndarray,
    opt_iters: int,
    opt_lr: float,
    rollout_horizon_steps: int,
) -> tuple[np.ndarray, dict]:
    device = tracker.robot.jnt_ranges.device
    lower = tracker.robot.jnt_ranges[:, 0].unsqueeze(0)
    upper = tracker.robot.jnt_ranges[:, 1].unsqueeze(0)
    raw_q = raw_from_q(torch.from_numpy(q_init_np.astype(np.float32)).to(device), lower, upper).requires_grad_(True)
    optimizer = torch.optim.Adam([raw_q], lr=float(opt_lr))
    weights = RolloutObjectiveWeights()
    direction_batch = repeat_vec(task['direction'], q_init_np.shape[0], device)
    normal_batch = repeat_vec(task['target_normal'], q_init_np.shape[0], device)

    for _ in range(int(opt_iters)):
        optimizer.zero_grad(set_to_none=True)
        q_batch = q_from_raw(raw_q, lower, upper)
        score, _ = differentiable_closed_loop_rollout_score(
            tracker=tracker,
            q0_batch=q_batch,
            direction_batch=direction_batch,
            target_normal_batch=normal_batch,
            horizon_steps=int(rollout_horizon_steps),
            weights=weights,
        )
        loss = -score.mean()
        loss.backward()
        optimizer.step()

    q_batch = q_from_raw(raw_q, lower, upper)
    score, stats = differentiable_closed_loop_rollout_score(
        tracker=tracker,
        q0_batch=q_batch,
        direction_batch=direction_batch,
        target_normal_batch=normal_batch,
        horizon_steps=int(rollout_horizon_steps),
        weights=weights,
    )
    best_idx = int(torch.argmax(score).item())
    return q_batch.detach().cpu().numpy().astype(np.float32), {
        'best_idx': best_idx,
        'best_score': float(score[best_idx].item()),
        'best_pred_length': float(stats['projected_length'][best_idx].item()),
        'best_total_penalty': float(stats['total_penalty'][best_idx].item()),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    tracker, tracker_device = build_tracker(device)
    tasks = load_tasks_from_jsonl(args.tasks_jsonl, args.num_cases)

    args.cases_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.cases_jsonl.write_text('', encoding='utf-8')

    selected_qs: list[np.ndarray] = []
    selected_directions: list[np.ndarray] = []
    selected_normals: list[np.ndarray] = []
    buffered_rows: list[dict] = []

    aggregate = {
        'selected_real': [],
        'gain_vs_gt': [],
        'percent_of_oracle': [],
        'pre_correction_pos_err_mm': [],
        'ik_init_s': [],
        'trajectory_opt_s': [],
        'correction_s': [],
        'rollout_eval_s': [],
        'total_case_time_s': [],
        'pred_length': [],
        'objective_score': [],
    }

    t_global0 = time.perf_counter()
    for task_no, task in enumerate(tasks, start=1):
        case_t0 = time.perf_counter()

        ik_t0 = time.perf_counter()
        q_init_batch, init_meta = optimize_start_pose_multistart(
            tracker=tracker,
            target_pos_np=task['pos'],
            target_normal_np=task['target_normal'],
            num_starts=int(args.num_starts),
            iters=int(args.ik_iters),
            lr=float(args.ik_lr),
        )
        ik_t1 = time.perf_counter()

        opt_t0 = time.perf_counter()
        q_opt_batch, opt_meta = optimize_initial_configuration(
            tracker=tracker,
            task=task,
            q_init_np=q_init_batch,
            opt_iters=int(args.opt_iters),
            opt_lr=float(args.opt_lr),
            rollout_horizon_steps=int(args.rollout_horizon_steps),
        )
        opt_t1 = time.perf_counter()

        corr_t0 = time.perf_counter()
        q_corrected_batch, raw_pos_err = batch_position_error_and_correction(
            tracker.robot,
            q_opt_batch,
            task['pos'],
            float(args.correction_damping),
            int(args.correction_iters),
            float(args.correction_tol),
            tracker_device,
        )
        corr_t1 = time.perf_counter()

        best_idx = int(opt_meta['best_idx'])
        selected_qs.append(q_corrected_batch[best_idx].astype(np.float32))
        selected_directions.append(task['direction'].astype(np.float32))
        selected_normals.append(task['target_normal'].astype(np.float32))
        buffered_rows.append({
            'task_index': int(task['task_index']),
            'oracle_rank_among_6000': int(task['oracle_rank_among_6000']),
            'task_category': task['task_category'],
            'gt_real': float(task['gt_real']),
            'direction': task['direction'].astype(np.float32),
            'target_normal': task['target_normal'].astype(np.float32),
            'selected_sample_idx': best_idx,
            'selected_from_num_starts': int(q_init_batch.shape[0]),
            'objective_score': float(opt_meta['best_score']),
            'pred_length': float(opt_meta['best_pred_length']),
            'pre_correction_pos_err_mm': float(raw_pos_err[best_idx] * 1e3),
            'q_corrected': q_corrected_batch[best_idx].astype(np.float32),
            'ik_init_s': float(ik_t1 - ik_t0),
            'trajectory_opt_s': float(opt_t1 - opt_t0),
            'correction_s': float(corr_t1 - corr_t0),
            'case_time_prefix_s': float(time.perf_counter() - case_t0),
        })

        if task_no % max(1, int(args.print_every)) == 0 or task_no == len(tasks):
            print(
                f'[opt-orthodox] task {task_no}/{len(tasks)} '
                f'gt={task["gt_real"]:.3f} pred={float(opt_meta["best_pred_length"]):.3f} '
                f'ik_t={float(ik_t1 - ik_t0):.3f}s opt_t={float(opt_t1 - opt_t0):.3f}s',
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
    rollout_share_s = float(rollout_t1 - rollout_t0) / max(len(buffered_rows), 1)

    for row, selected_real_i in zip(buffered_rows, selected_real.tolist()):
        gain_vs_gt = float(selected_real_i - row['gt_real'])
        percent_of_oracle = 100.0 * float(selected_real_i) / max(float(row['gt_real']), 1e-6)
        total_case_time_s = float(row['case_time_prefix_s']) + rollout_share_s

        aggregate['selected_real'].append(float(selected_real_i))
        aggregate['gain_vs_gt'].append(gain_vs_gt)
        aggregate['percent_of_oracle'].append(percent_of_oracle)
        aggregate['pre_correction_pos_err_mm'].append(float(row['pre_correction_pos_err_mm']))
        aggregate['ik_init_s'].append(float(row['ik_init_s']))
        aggregate['trajectory_opt_s'].append(float(row['trajectory_opt_s']))
        aggregate['correction_s'].append(float(row['correction_s']))
        aggregate['rollout_eval_s'].append(rollout_share_s)
        aggregate['total_case_time_s'].append(total_case_time_s)
        aggregate['pred_length'].append(float(row['pred_length']))
        aggregate['objective_score'].append(float(row['objective_score']))

        payload = {
            'method': 'opt_orthodox',
            'task_index': int(row['task_index']),
            'oracle_rank_among_6000': int(row['oracle_rank_among_6000']),
            'task_category': row['task_category'],
            'gt_real': float(row['gt_real']),
            'direction': row['direction'],
            'target_normal': row['target_normal'],
            'selected_sample_idx': int(row['selected_sample_idx']),
            'selected_from_num_starts': int(row['selected_from_num_starts']),
            'objective_score': float(row['objective_score']),
            'pred_length': float(row['pred_length']),
            'selected_real': float(selected_real_i),
            'gain_vs_gt': gain_vs_gt,
            'percent_of_oracle': percent_of_oracle,
            'pre_correction_pos_err_mm': float(row['pre_correction_pos_err_mm']),
            'q_corrected': row['q_corrected'],
            'ik_init_s': float(row['ik_init_s']),
            'trajectory_opt_s': float(row['trajectory_opt_s']),
            'correction_s': float(row['correction_s']),
            'rollout_eval_s': rollout_share_s,
            'total_case_time_s': total_case_time_s,
        }
        with args.cases_jsonl.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + '\n')

    summary = {
        'method': 'opt_orthodox',
        'num_cases': int(len(tasks)),
        'tasks_jsonl': str(args.tasks_jsonl),
        'cases_jsonl': str(args.cases_jsonl),
        'num_starts': int(args.num_starts),
        'ik_iters': int(args.ik_iters),
        'ik_lr': float(args.ik_lr),
        'opt_iters': int(args.opt_iters),
        'opt_lr': float(args.opt_lr),
        'rollout_horizon_steps': int(args.rollout_horizon_steps),
        'selected_real': mean_std(aggregate['selected_real']),
        'gain_vs_gt': mean_std(aggregate['gain_vs_gt']),
        'percent_of_oracle': mean_std(aggregate['percent_of_oracle']),
        'pre_correction_pos_err_mm': mean_std(aggregate['pre_correction_pos_err_mm']),
        'ik_init_s': mean_std(aggregate['ik_init_s']),
        'trajectory_opt_s': mean_std(aggregate['trajectory_opt_s']),
        'correction_s': mean_std(aggregate['correction_s']),
        'rollout_eval_s': mean_std(aggregate['rollout_eval_s']),
        'total_case_time_s': mean_std(aggregate['total_case_time_s']),
        'pred_length': mean_std(aggregate['pred_length']),
        'objective_score': mean_std(aggregate['objective_score']),
        'wall_clock_s': float(time.perf_counter() - t_global0),
    }
    args.output_json.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
