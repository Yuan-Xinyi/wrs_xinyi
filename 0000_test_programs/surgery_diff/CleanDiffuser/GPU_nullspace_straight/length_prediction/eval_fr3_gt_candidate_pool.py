from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from lnet_contrastive_fr3_rotation_cone_eval import (
    build_tracker,
    collect_candidate_qs,
    rollout_same_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate the FR3 ground-truth candidate pool by resampling many feasible joint configurations for the exact tasks used in the comprehensive contrastive evaluation.'
    )
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-candidates', type=int, default=1000)
    parser.add_argument('--oversample', type=int, default=2048)
    parser.add_argument('--task-batch-size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pos-tol-mm', type=float, default=2.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--save-best-q', action='store_true', help='Store the best candidate joint configuration for each task.')
    parser.add_argument(
        '--tasks-jsonl',
        type=Path,
        default=CURRENT_DIR / 'eval_lnet_contrastive_fr3_comprehensive_cases.jsonl',
        help='JSONL file providing the exact task list to reuse in file order.',
    )
    parser.add_argument('--output-json', type=Path, default=CURRENT_DIR / 'eval_fr3_gt_candidate_pool_summary.json')
    parser.add_argument('--cases-jsonl', type=Path, default=CURRENT_DIR / 'eval_fr3_gt_candidate_pool_cases.jsonl')
    return parser.parse_args()


def to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return (vec / max(float(np.linalg.norm(vec)), 1e-12)).astype(np.float32)


def load_tasks_from_jsonl(path: Path, num_cases: int | None = None) -> list[dict[str, np.ndarray]]:
    tasks: list[dict[str, np.ndarray]] = []
    with path.open('r', encoding='utf-8') as fh:
        for line_idx, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Failed to parse JSON on line {line_idx} of {path}: {exc}') from exc
            task_obj = row.get('task')
            if not isinstance(task_obj, dict):
                raise RuntimeError(f'Line {line_idx} of {path} does not contain a task object')
            tasks.append({
                'task_index': int(row.get('task_index', len(tasks))),
                'pos': np.asarray(task_obj['pos'], dtype=np.float32),
                'direction': normalize(np.asarray(task_obj['direction'], dtype=np.float32)),
                'normal': normalize(np.asarray(task_obj['normal'], dtype=np.float32)),
                'task_category': str(row.get('task_category', 'unknown')),
            })
            if num_cases is not None and len(tasks) >= int(num_cases):
                break
    if not tasks:
        raise RuntimeError(f'No tasks loaded from {path}')
    return tasks


def rollout_task_batch(
    tracker,
    tracker_device: torch.device,
    q_batches: list[np.ndarray],
    tasks: list[dict[str, np.ndarray]],
) -> list[np.ndarray]:
    if not q_batches:
        return []
    counts = [int(q.shape[0]) for q in q_batches]
    flat_q = np.concatenate(q_batches, axis=0).astype(np.float32)
    flat_direction = np.concatenate([
        np.repeat(task['direction'][None, :], count, axis=0).astype(np.float32)
        for task, count in zip(tasks, counts)
    ], axis=0)
    flat_normal = np.concatenate([
        np.repeat(task['normal'][None, :], count, axis=0).astype(np.float32)
        for task, count in zip(tasks, counts)
    ], axis=0)
    flat_real = rollout_same_task(tracker, tracker_device, flat_q, flat_direction, flat_normal).astype(np.float32)
    outputs: list[np.ndarray] = []
    start = 0
    for count in counts:
        outputs.append(flat_real[start:start + count].copy())
        start += count
    return outputs


def per_task_stats(
    task: dict[str, np.ndarray],
    q_np: np.ndarray,
    real_len: np.ndarray,
    sampling_time_s: float,
    rollout_time_s: float,
    total_case_time_s: float,
    save_best_q: bool,
) -> dict:
    order = np.argsort(-real_len)
    best_idx = int(order[0])
    best_len = float(real_len[best_idx])
    second_best_len = float(real_len[int(order[1])]) if order.shape[0] > 1 else best_len
    top10_count = min(10, order.shape[0])
    top10_mean = float(np.mean(real_len[order[:top10_count]]))
    q25, median, q75 = np.quantile(real_len, [0.25, 0.5, 0.75])
    zero_mask = real_len <= 1e-12
    nonzero_mask = ~zero_mask
    near90_mask = real_len >= 0.90 * max(best_len, 1e-6)
    near95_mask = real_len >= 0.95 * max(best_len, 1e-6)

    payload = {
        'task_index': int(task['task_index']),
        'task_category': task.get('task_category', 'unknown'),
        'task': {
            'pos': task['pos'].tolist(),
            'direction': task['direction'].tolist(),
            'normal': task['normal'].tolist(),
        },
        'num_candidates_gt': int(real_len.shape[0]),
        'gt_best_idx': best_idx,
        'gt_best_length': best_len,
        'gt_length_mean': float(np.mean(real_len)),
        'gt_length_median': float(median),
        'gt_length_std': float(np.std(real_len)),
        'gt_length_min': float(np.min(real_len)),
        'gt_length_max': float(np.max(real_len)),
        'gt_length_q25': float(q25),
        'gt_length_q75': float(q75),
        'gt_iqr': float(q75 - q25),
        'gt_gap_best_minus_mean': float(best_len - np.mean(real_len)),
        'gt_gap_best_minus_median': float(best_len - median),
        'gt_gap_best_minus_second': float(best_len - second_best_len),
        'gt_gap_best_minus_top10_mean': float(best_len - top10_mean),
        'gt_top10_mean': top10_mean,
        'gt_nonzero_rate': float(np.mean(nonzero_mask)),
        'gt_zero_rate': float(np.mean(zero_mask)),
        'gt_near_optimal_rate_90': float(np.mean(near90_mask)),
        'gt_near_optimal_rate_95': float(np.mean(near95_mask)),
        'gt_effective_candidate_count_90': int(np.sum(near90_mask)),
        'gt_effective_candidate_count_95': int(np.sum(near95_mask)),
        'gt_cv': float(np.std(real_len) / max(float(np.mean(real_len)), 1e-6)),
        'sampling_time_s': float(sampling_time_s),
        'rollout_time_s': float(rollout_time_s),
        'total_case_time_s': float(total_case_time_s),
    }
    if save_best_q:
        payload['gt_best_q'] = q_np[best_idx].astype(np.float32).tolist()
    return payload


def summarize_scalar_fields(cases: list[dict], fields: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for field in fields:
        values = np.asarray([float(c[field]) for c in cases], dtype=np.float32)
        summary[field] = {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
    return summary


def summarize_by_category(cases: list[dict], fields: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    grouped = {'all': cases}
    for category in sorted({str(c.get('task_category', 'unknown')) for c in cases}):
        grouped[category] = [c for c in cases if c.get('task_category') == category]
    return {
        name: summarize_scalar_fields(group_cases, fields)
        for name, group_cases in grouped.items()
        if group_cases
    }


def main() -> None:
    args = parse_args()
    np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    tracker, tracker_device = build_tracker(device)

    scalar_fields = [
        'gt_best_length',
        'gt_length_mean',
        'gt_length_median',
        'gt_length_std',
        'gt_length_min',
        'gt_length_max',
        'gt_length_q25',
        'gt_length_q75',
        'gt_iqr',
        'gt_gap_best_minus_mean',
        'gt_gap_best_minus_median',
        'gt_gap_best_minus_second',
        'gt_gap_best_minus_top10_mean',
        'gt_top10_mean',
        'gt_nonzero_rate',
        'gt_zero_rate',
        'gt_near_optimal_rate_90',
        'gt_near_optimal_rate_95',
        'gt_effective_candidate_count_90',
        'gt_effective_candidate_count_95',
        'gt_cv',
        'sampling_time_s',
        'rollout_time_s',
        'total_case_time_s',
    ]

    input_tasks = load_tasks_from_jsonl(args.tasks_jsonl)
    num_cases = len(input_tasks)
    task_batch_size = max(1, int(args.task_batch_size))
    args.cases_jsonl.write_text('')
    cases: list[dict] = []
    start_time = time.perf_counter()

    print(f'[setup] loading exact task list from {args.tasks_jsonl}')
    print(f'[setup] evaluating num_cases={num_cases} num_candidates={args.num_candidates} task_batch_size={task_batch_size}')

    for batch_start in range(0, num_cases, task_batch_size):
        batch_end = min(batch_start + task_batch_size, num_cases)
        batch_tasks = input_tasks[batch_start:batch_end]
        print(f'[batch] tasks {batch_start + 1}-{batch_end}/{num_cases}: sampling candidate pools')

        batch_q_np: list[np.ndarray] = []
        batch_sampling_times: list[float] = []
        case_start_times: list[float] = []
        for task in batch_tasks:
            case_t0 = time.perf_counter()
            sample_t0 = case_t0
            q_np, _ = collect_candidate_qs(
                tracker,
                tracker_device,
                task['pos'],
                task['direction'],
                task['normal'],
                int(args.num_candidates),
                int(args.oversample),
                float(args.pos_tol_mm),
                int(args.correction_iters),
                float(args.correction_tol),
                float(args.correction_damping),
            )
            sample_t1 = time.perf_counter()
            batch_q_np.append(q_np)
            batch_sampling_times.append(float(sample_t1 - sample_t0))
            case_start_times.append(case_t0)

        total_rollout = sum(q.shape[0] for q in batch_q_np)
        print(f'[batch] tasks {batch_start + 1}-{batch_end}/{num_cases}: rollout on {total_rollout} flattened candidates')
        rollout_t0 = time.perf_counter()
        batch_real = rollout_task_batch(tracker, tracker_device, batch_q_np, batch_tasks)
        rollout_t1 = time.perf_counter()
        batch_rollout_total = float(rollout_t1 - rollout_t0)
        counts = [max(1, int(q.shape[0])) for q in batch_q_np]
        total_count = max(1, sum(counts))
        batch_rollout_times = [batch_rollout_total * (count / total_count) for count in counts]

        for local_idx, (task, q_np, real_len, sampling_time_s, rollout_time_s, case_t0) in enumerate(
            zip(batch_tasks, batch_q_np, batch_real, batch_sampling_times, batch_rollout_times, case_start_times),
            start=1,
        ):
            task_index = int(task['task_index'])
            case_payload = per_task_stats(
                task=task,
                q_np=q_np,
                real_len=real_len,
                sampling_time_s=float(sampling_time_s),
                rollout_time_s=float(rollout_time_s),
                total_case_time_s=float(time.perf_counter() - case_t0),
                save_best_q=bool(args.save_best_q),
            )
            cases.append(case_payload)
            with args.cases_jsonl.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(to_jsonable(case_payload), ensure_ascii=False) + '\n')

            if args.print_every > 0 and (task_index % args.print_every == 0 or task_index == num_cases):
                elapsed = time.perf_counter() - start_time
                print(
                    f'[progress] {task_index}/{num_cases} elapsed={elapsed:.1f}s '
                    f'gt_best={case_payload["gt_best_length"]:.4f} '
                    f'gt_mean={case_payload["gt_length_mean"]:.4f} '
                    f'gap_best_mean={case_payload["gt_gap_best_minus_mean"]:.4f} '
                    f'near90={case_payload["gt_near_optimal_rate_90"]:.3f} '
                    f'sample_t={case_payload["sampling_time_s"]:.3f}s '
                    f'rollout_t={case_payload["rollout_time_s"]:.3f}s'
                )

    summary = {
        'args': to_jsonable(vars(args)),
        'task_source': 'random',
        'num_cases': int(len(cases)),
        'scalar_fields': scalar_fields,
        'metrics': summarize_scalar_fields(cases, scalar_fields),
        'metrics_by_category': summarize_by_category(cases, scalar_fields),
    }
    args.output_json.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
