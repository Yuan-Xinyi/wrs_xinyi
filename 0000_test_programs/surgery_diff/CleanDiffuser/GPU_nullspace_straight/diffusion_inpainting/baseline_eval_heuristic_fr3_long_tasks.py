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
from trajectory_generation.fr3_nullspace_straight import (
    GPUNullspaceStraightTracker,
    TrackerConfig,
    directional_manipulability_batch,
)


DEFAULT_OUTPUT_JSON = BASE_DIR / 'baseline_eval_heuristic_fr3_long_tasks_summary.json'
DEFAULT_CASES_JSONL = BASE_DIR / 'baseline_eval_heuristic_fr3_long_tasks_cases.jsonl'
METHOD_ORDER = (
    'manipulability',
    'joint_limit_margin',
    'joint_center',
    'short_rollout',
)
METHOD_LABELS = {
    'manipulability': 'Manipulability Selector',
    'joint_limit_margin': 'Joint-Limit-Margin Selector',
    'joint_center': 'Joint-Center Selector',
    'short_rollout': 'Short-Rollout Selector',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate four simple heuristic FR3 selectors on the long-task benchmark.')
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
    parser.add_argument('--short-horizon-steps', type=int, default=200)
    parser.add_argument('--print-every', type=int, default=1)
    parser.add_argument('--rollout-batch-size', type=int, default=256)
    parser.add_argument('--final-rollout-target', type=int, default=1000)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def load_existing_cases(path: Path) -> list[dict]:
    if not path.exists():
        return []
    cases: list[dict] = []
    with path.open('r', encoding='utf-8') as fh:
        for line_idx, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                cases.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Failed to parse existing JSON on line {line_idx} of {path}: {exc}') from exc
    return cases


def init_aggregate() -> dict[str, dict[str, list[float]]]:
    return {
        method: {
            'selected_real': [],
            'gain_vs_gt': [],
            'percent_of_oracle': [],
            'raw_pos_err_mm': [],
            'selector_time_s': [],
            'rollout_eval_s': [],
            'total_method_time_s': [],
            'selector_score': [],
            'short_horizon_length': [],
        }
        for method in METHOD_ORDER
    }


def update_aggregate(aggregate: dict[str, dict[str, list[float]]], row: dict) -> None:
    for method in METHOD_ORDER:
        result = row['results'][method]
        agg = aggregate[method]
        agg['selected_real'].append(float(result['selected_real']))
        agg['gain_vs_gt'].append(float(result['gain_vs_gt']))
        agg['percent_of_oracle'].append(float(result['percent_of_oracle']))
        agg['raw_pos_err_mm'].append(float(result['raw_pos_err_mm']))
        agg['selector_time_s'].append(float(result['selector_time_s']))
        agg['rollout_eval_s'].append(float(result['rollout_eval_s']))
        agg['total_method_time_s'].append(float(result['total_method_time_s']))
        agg['selector_score'].append(float(result['selector_score']))
        agg['short_horizon_length'].append(float(result['short_horizon_length']))


def compute_candidate_features(
    tracker,
    q_batch_np: np.ndarray,
    target_pos_np: np.ndarray,
    direction_np: np.ndarray,
    normal_np: np.ndarray,
) -> dict[str, np.ndarray]:
    device = tracker.robot.jnt_ranges.device
    q_batch = torch.from_numpy(q_batch_np.astype(np.float32)).to(device)
    q_eval = q_batch.detach().clone().requires_grad_(True)
    tcp_pos, tcp_rot = tracker.robot.fk_batch(q_eval)
    grads = []
    for dim in range(3):
        grad_dim = torch.autograd.grad(
            tcp_pos[:, dim].sum(),
            q_eval,
            retain_graph=True,
            create_graph=False,
        )[0]
        grads.append(grad_dim)
    j_pos = torch.stack(grads, dim=1)
    direction = torch.from_numpy(np.repeat(direction_np[None, :], q_batch_np.shape[0], axis=0).astype(np.float32)).to(device)
    normal = torch.from_numpy(np.repeat(normal_np[None, :], q_batch_np.shape[0], axis=0).astype(np.float32)).to(device)
    target_pos = torch.from_numpy(np.repeat(target_pos_np[None, :], q_batch_np.shape[0], axis=0).astype(np.float32)).to(device)

    mu = directional_manipulability_batch(j_pos, direction, tracker.config.damping)
    tcp_z = tcp_rot[:, :, 2]
    normal_align = torch.sum(tcp_z * normal, dim=-1)
    pos_err = torch.linalg.norm(tcp_pos - target_pos, dim=-1)

    lower = tracker.robot.jnt_ranges[:, 0].unsqueeze(0)
    upper = tracker.robot.jnt_ranges[:, 1].unsqueeze(0)
    span = (upper - lower).clamp_min(1e-6)
    lower_margin = (q_eval - lower) / span
    upper_margin = (upper - q_eval) / span
    joint_limit_margin = torch.minimum(lower_margin, upper_margin).amin(dim=-1)
    center = 0.5 * (lower + upper)
    centered = (q_eval - center) / span
    joint_center_score = -torch.mean(centered * centered, dim=-1)

    return {
        'manipulability': mu.detach().cpu().numpy().astype(np.float32),
        'normal_align': normal_align.detach().cpu().numpy().astype(np.float32),
        'pos_err': pos_err.detach().cpu().numpy().astype(np.float32),
        'joint_limit_margin': joint_limit_margin.detach().cpu().numpy().astype(np.float32),
        'joint_center': joint_center_score.detach().cpu().numpy().astype(np.float32),
    }


def select_best_index(scores: np.ndarray) -> int:
    return int(np.argmax(scores.astype(np.float64)))


def summarize_method(agg: dict[str, list[float]]) -> dict:
    return {
        'time_s': mean_std(agg['total_method_time_s']),
        'pos_mm': mean_std(agg['raw_pos_err_mm']),
        'length_m': mean_std(agg['selected_real']),
        'proximity_pct': mean_std(agg['percent_of_oracle']),
        'selector_time_s': mean_std(agg['selector_time_s']),
        'rollout_eval_s': mean_std(agg['rollout_eval_s']),
        'gain_vs_gt': mean_std(agg['gain_vs_gt']),
        'selector_score': mean_std(agg['selector_score']),
        'short_horizon_length': mean_std(agg['short_horizon_length']),
    }


def flush_pending_rollouts(
    pending_rows: list[dict],
    tracker,
    tracker_device,
    aggregate: dict[str, dict[str, list[float]]],
    cases_jsonl: Path,
) -> None:
    if not pending_rows:
        return

    q_batch = []
    direction_batch = []
    target_normal_batch = []
    row_method_offsets: list[tuple[int, int]] = []
    cursor = 0
    for row in pending_rows:
        for method in METHOD_ORDER:
            q_batch.append(row['pending_results'][method]['q_selected'])
            direction_batch.append(row['direction'])
            target_normal_batch.append(row['target_normal'])
        row_method_offsets.append((cursor, cursor + len(METHOD_ORDER)))
        cursor += len(METHOD_ORDER)

    rollout_t0 = time.perf_counter()
    selected_reals = rollout_large_batch(
        tracker=tracker,
        tracker_device=tracker_device,
        q_batch=np.stack(q_batch, axis=0).astype(np.float32),
        direction_batch=np.stack(direction_batch, axis=0).astype(np.float32),
        target_normal_batch=np.stack(target_normal_batch, axis=0).astype(np.float32),
        chunk_size=max(1, int(len(q_batch))),
    )
    shared_rollout_eval_s = float(time.perf_counter() - rollout_t0)

    with cases_jsonl.open('a', encoding='utf-8') as fh:
        for row, (start, end) in zip(pending_rows, row_method_offsets, strict=True):
            results: dict[str, dict] = {}
            for method_idx, method in enumerate(METHOD_ORDER, start=start):
                pending = row['pending_results'][method]
                selected_real = float(selected_reals[method_idx])
                selector_time_s = float(pending['selector_time_s'])
                rollout_eval_s = float(shared_rollout_eval_s)
                total_method_time_s = float(row['shared']['candidate_collection_s']) + selector_time_s + rollout_eval_s
                results[method] = {
                    'label': METHOD_LABELS[method],
                    'selected_sample_idx': int(pending['selected_sample_idx']),
                    'selected_from_num_samples': int(row['shared']['selected_from_num_samples']),
                    'selector_score': float(pending['selector_score']),
                    'short_horizon_length': float(pending['short_horizon_length']),
                    'selected_real': float(selected_real),
                    'gain_vs_gt': float(selected_real - row['gt_real']),
                    'percent_of_oracle': 100.0 * float(selected_real) / max(float(row['gt_real']), 1e-6),
                    'raw_pos_err_mm': float(pending['raw_pos_err_mm']),
                    'selector_time_s': selector_time_s,
                    'rollout_eval_s': rollout_eval_s,
                    'total_method_time_s': total_method_time_s,
                }

            payload = {
                'task_index': int(row['task_index']),
                'oracle_rank_among_6000': int(row['oracle_rank_among_6000']),
                'task_category': row['task_category'],
                'gt_real': float(row['gt_real']),
                'direction': row['direction'].astype(np.float32),
                'target_normal': row['target_normal'].astype(np.float32),
                'shared': {
                    **row['shared'],
                    'final_rollout_eval_batched_s': float(shared_rollout_eval_s),
                    'total_case_wall_clock_s': float(time.perf_counter() - row['case_t0']),
                },
                'results': results,
            }
            update_aggregate(aggregate, payload)
            fh.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + '\n')

            if row['task_idx'] % max(1, int(row['print_every'])) == 0 or row['task_idx'] == row['num_tasks']:
                print(
                    f'[heuristic-baselines] task {row["task_idx"]}/{row["num_tasks"]} '
                    f'gt={float(row["gt_real"]):.3f} '
                    f'mu={results["manipulability"]["selected_real"]:.3f} '
                    f'jl={results["joint_limit_margin"]["selected_real"]:.3f} '
                    f'jc={results["joint_center"]["selected_real"]:.3f} '
                    f'short={results["short_rollout"]["selected_real"]:.3f} '
                    f'cand_t={payload["shared"]["candidate_collection_s"]:.3f}s '
                    f'feat_t={payload["shared"]["feature_time_s"]:.3f}s '
                    f'short_all_t={payload["shared"]["short_rollout_eval_all_s"]:.3f}s '
                    f'final_batch_t={payload["shared"]["final_rollout_eval_batched_s"]:.3f}s',
                    flush=True,
                )

    pending_rows.clear()


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
    existing_cases = load_existing_cases(args.cases_jsonl) if args.resume else []
    completed_count = len(existing_cases)
    if args.resume:
        if completed_count > len(tasks):
            raise RuntimeError(f'Existing cases-jsonl has {completed_count} rows, but only {len(tasks)} tasks are requested.')
        if completed_count == len(tasks):
            print(f'[resume] all {len(tasks)} tasks are already present in {args.cases_jsonl}')
        else:
            print(f'[resume] loaded {completed_count} existing cases from {args.cases_jsonl}; continuing from task {completed_count + 1}/{len(tasks)}')
    else:
        args.cases_jsonl.write_text('', encoding='utf-8')

    aggregate = init_aggregate()
    for row in existing_cases:
        update_aggregate(aggregate, row)

    t_global0 = time.perf_counter()
    short_cap = float(args.short_horizon_steps) * float(tracker.config.dt) * float(tracker.config.task_speed)
    pending_rows: list[dict] = []
    for task_idx, task in enumerate(tasks[completed_count:], start=completed_count + 1):
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
        q_corrected, raw_pos_err = batch_position_error_and_correction(
            tracker.robot,
            q_candidates.astype(np.float32),
            task['pos'],
            float(args.correction_damping),
            int(args.correction_iters),
            float(args.correction_tol),
            tracker_device,
        )
        cand_t1 = time.perf_counter()

        feature_t0 = time.perf_counter()
        features = compute_candidate_features(
            tracker=tracker,
            q_batch_np=q_corrected,
            target_pos_np=task['pos'],
            direction_np=task['direction'],
            normal_np=task['target_normal'],
        )
        feature_t1 = time.perf_counter()

        short_t0 = time.perf_counter()
        short_lengths = rollout_large_batch(
            tracker=short_tracker,
            tracker_device=tracker_device,
            q_batch=q_corrected.astype(np.float32),
            direction_batch=np.repeat(task['direction'][None, :], q_corrected.shape[0], axis=0).astype(np.float32),
            target_normal_batch=np.repeat(task['target_normal'][None, :], q_corrected.shape[0], axis=0).astype(np.float32),
            chunk_size=min(int(args.rollout_batch_size), max(1, int(q_corrected.shape[0]))),
        )
        short_t1 = time.perf_counter()

        method_scores = {
            'manipulability': features['manipulability'],
            'joint_limit_margin': features['joint_limit_margin'],
            'joint_center': features['joint_center'],
            'short_rollout': short_lengths.astype(np.float32),
        }
        method_selector_times = {
            'manipulability': float(feature_t1 - feature_t0),
            'joint_limit_margin': float(feature_t1 - feature_t0),
            'joint_center': float(feature_t1 - feature_t0),
            'short_rollout': float((feature_t1 - feature_t0) + (short_t1 - short_t0)),
        }

        selected_indices: dict[str, int] = {}
        q_selected = []
        for method in METHOD_ORDER:
            best_idx = select_best_index(method_scores[method])
            selected_indices[method] = best_idx
            q_selected.append(q_corrected[best_idx].astype(np.float32))

        pending_results = {}
        for method_idx, method in enumerate(METHOD_ORDER):
            best_idx = selected_indices[method]
            pending_results[method] = {
                'selected_sample_idx': int(best_idx),
                'selector_score': float(method_scores[method][best_idx]),
                'short_horizon_length': float(short_lengths[best_idx]),
                'raw_pos_err_mm': float(raw_pos_err[best_idx] * 1e3),
                'selector_time_s': float(method_selector_times[method]),
                'q_selected': q_selected[method_idx],
            }

        pending_rows.append({
            'task_idx': int(task_idx),
            'num_tasks': int(len(tasks)),
            'print_every': int(args.print_every),
            'case_t0': float(case_t0),
            'task_index': int(task['task_index']),
            'oracle_rank_among_6000': int(task['oracle_rank_among_6000']),
            'task_category': task['task_category'],
            'gt_real': float(task['gt_real']),
            'direction': task['direction'].astype(np.float32),
            'target_normal': task['target_normal'].astype(np.float32),
            'shared': {
                'selected_from_num_samples': int(q_corrected.shape[0]),
                'candidate_collection_s': float(cand_t1 - cand_t0),
                'feature_time_s': float(feature_t1 - feature_t0),
                'short_rollout_eval_all_s': float(short_t1 - short_t0),
                'short_horizon_cap_m': float(short_cap),
            },
            'pending_results': pending_results,
        })

        if task_idx % max(1, int(args.print_every)) == 0 or task_idx == len(tasks):
            pending_tasks = len(pending_rows)
            pending_trajs = pending_tasks * len(METHOD_ORDER)
            print(
                f'[heuristic-baselines] queued task {task_idx}/{len(tasks)} '
                f'pending_tasks={pending_tasks} '
                f'pending_final_rollouts={pending_trajs}/{int(args.final_rollout_target)} '
                f'cand_t={float(cand_t1 - cand_t0):.3f}s '
                f'feat_t={float(feature_t1 - feature_t0):.3f}s '
                f'short_all_t={float(short_t1 - short_t0):.3f}s',
                flush=True,
            )

        if len(pending_rows) * len(METHOD_ORDER) >= int(args.final_rollout_target):
            flush_pending_rollouts(
                pending_rows=pending_rows,
                tracker=tracker,
                tracker_device=tracker_device,
                aggregate=aggregate,
                cases_jsonl=args.cases_jsonl,
            )

    flush_pending_rollouts(
        pending_rows=pending_rows,
        tracker=tracker,
        tracker_device=tracker_device,
        aggregate=aggregate,
        cases_jsonl=args.cases_jsonl,
    )

    method_summaries = {method: summarize_method(aggregate[method]) for method in METHOD_ORDER}
    table2_metrics = {
        METHOD_LABELS[method]: {
            'Time (s)': method_summaries[method]['time_s'],
            'Pos. (mm)': method_summaries[method]['pos_mm'],
            'Length (m)': method_summaries[method]['length_m'],
            'Proximity to Optimal Length (%)': method_summaries[method]['proximity_pct'],
        }
        for method in METHOD_ORDER
    }
    summary = {
        'method_family': 'simple_heuristic_selectors',
        'num_cases': int(len(existing_cases) + len(tasks[completed_count:])),
        'tasks_jsonl': str(args.tasks_jsonl),
        'cases_jsonl': str(args.cases_jsonl),
        'num_candidates': int(args.num_candidates),
        'oversample': int(args.oversample),
        'short_horizon_steps': int(args.short_horizon_steps),
        'short_horizon_cap_m': float(short_cap),
        'methods': method_summaries,
        'table2_metrics': table2_metrics,
        'wall_clock_s': float(time.perf_counter() - t_global0),
    }
    args.output_json.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
