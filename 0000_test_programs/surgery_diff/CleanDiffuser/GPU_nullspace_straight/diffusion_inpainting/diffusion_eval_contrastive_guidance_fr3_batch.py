from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from diffusion import normalize_condition
from diffusion_sample import load_model
from diffusion_eval_batch_candidates_lnet import (
    batch_position_error_and_correction,
    load_lnet_contrastive_model,
)
from diffusion_eval_contrastive_guidance import sample_with_guidance
from diffusion_vis_contrastive_guidance_cases import (
    DEFAULT_LNET_CONTRASTIVE_CKPT,
    DEFAULT_RUN_NAME,
    DEFAULT_WORKDIR,
    build_tracker,
)

BASE_DIR = Path(__file__).resolve().parent
GPU_NULLSPACE_DIR = BASE_DIR.parent
DEFAULT_TASKS_JSONL = GPU_NULLSPACE_DIR / 'length_prediction' / 'eval_lnet_contrastive_fr3_top2000_tasks_by_oracle.jsonl'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch-evaluate Franka contrastive guidance on a fixed FR3 task list with timing breakdowns.')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--lnet-contrastive-ckpt', type=Path, default=DEFAULT_LNET_CONTRASTIVE_CKPT)
    parser.add_argument('--tasks-jsonl', type=Path, default=DEFAULT_TASKS_JSONL)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num-cases', type=int, default=2000)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.0, 0.1, 0.5, 1.0, 5.0])
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--samples-per-lambda', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--fixed-guidance-step', type=float, default=1.0)
    parser.add_argument('--guidance-grad-eps', type=float, default=1e-6)
    parser.add_argument('--rollout-batch-size', type=int, default=200)
    parser.add_argument('--rollout-accum-size', type=int, default=1000)
    return parser.parse_args()


def fmt(x: float) -> str:
    return f'{float(x):.3f}'


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-12)


def load_tasks_from_jsonl(path: Path, num_cases: int | None = None) -> list[dict]:
    tasks = []
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
                'task_category': str(row.get('task_category', 'unknown')),
                'pos': np.asarray(task_obj['pos'], dtype=np.float32),
                'direction': normalize(np.asarray(task_obj['direction'], dtype=np.float32)),
                'target_normal': normalize(np.asarray(task_obj['normal'], dtype=np.float32)),
                'gt_real': float(row.get('oracle_top1_real', 0.0)),
                'oracle_rank_among_6000': int(row.get('oracle_rank_among_6000', -1)),
            })
            if num_cases is not None and len(tasks) >= int(num_cases):
                break
    if not tasks:
        raise RuntimeError(f'No tasks loaded from {path}')
    return tasks


def mean_std(values: list[float]) -> dict:
    if not values:
        return {'mean': None, 'std': None, 'min': None, 'max': None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }


def rollout_large_batch(
    tracker,
    tracker_device: torch.device,
    q_batch: np.ndarray,
    direction_batch: np.ndarray,
    target_normal_batch: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    lengths = []
    total = int(q_batch.shape[0])
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        q_chunk = torch.from_numpy(q_batch[start:end].astype(np.float32)).to(tracker_device)
        d_chunk = torch.from_numpy(direction_batch[start:end].astype(np.float32)).to(tracker_device)
        n_chunk = torch.from_numpy(target_normal_batch[start:end].astype(np.float32)).to(tracker_device)
        result = tracker.run_batch(q0_batch=q_chunk, direction_batch=d_chunk, target_normal_batch=n_chunk)
        lengths.append(result.projected_length.detach().cpu().numpy())
    return np.concatenate(lengths, axis=0).astype(np.float32)


def flush_rollout_buffer(
    tracker,
    tracker_device: torch.device,
    lambda_values: list[float],
    rollout_batch_size: int,
    aggregate: dict,
    buffered_records: list[dict],
) -> None:
    if not buffered_records:
        return

    rollout_started = time.perf_counter()
    guided_real_by_lambda = {}
    for lam in lambda_values:
        q_batch = np.stack([record['q_corr_by_lambda'][lam] for record in buffered_records], axis=0).astype(np.float32)
        direction_batch = np.stack([record['direction'] for record in buffered_records], axis=0).astype(np.float32)
        normal_batch = np.stack([record['target_normal'] for record in buffered_records], axis=0).astype(np.float32)
        guided_real_by_lambda[lam] = rollout_large_batch(
            tracker,
            tracker_device,
            q_batch,
            direction_batch,
            normal_batch,
            int(rollout_batch_size),
        )
    rollout_finished = time.perf_counter()
    rollout_time = float(rollout_finished - rollout_started)
    aggregate['timing_rollout_guided'].append(rollout_time)
    rollout_time_share = rollout_time / max(len(buffered_records), 1)

    for record_idx, record in enumerate(buffered_records):
        best_idx = None
        best_val = None
        for idx, lam in enumerate(lambda_values):
            guided_real = float(guided_real_by_lambda[lam][record_idx])
            gain = guided_real - float(record['gt_real'])
            final_score = float(record['results'][idx]['final_score'])
            diff_len = float(record['results'][idx]['final_pred_length'])
            pos_err_mm = float(record['raw_pos_err_mm'][idx])
            aggregate['lambda'][lam]['guided_real'].append(guided_real)
            aggregate['lambda'][lam]['gain'].append(gain)
            aggregate['lambda'][lam]['score'].append(final_score)
            aggregate['lambda'][lam]['diff_len'].append(diff_len)
            aggregate['lambda'][lam]['pos_err_mm'].append(pos_err_mm)
            record['guided_real_by_lambda'][lam] = guided_real
            if best_val is None or guided_real > best_val:
                best_val = guided_real
                best_idx = idx
        aggregate['lambda'][lambda_values[int(best_idx)]]['best_count'] += 1
        record['rollout_time_share'] = rollout_time_share


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed if args.seed is not None else int(np.random.SeedSequence().entropy))
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    device = torch.device(args.device)

    _, stats, model, q_dim, diffusion_steps = load_model(args.bundle, device)
    lnet_contrastive = load_lnet_contrastive_model(args.lnet_contrastive_ckpt, device)
    tracker, tracker_device = build_tracker(device)
    steps = int(args.sample_steps) if args.sample_steps is not None else int(diffusion_steps)
    lambda_values = [float(v) for v in args.lambdas]
    tasks = load_tasks_from_jsonl(args.tasks_jsonl, args.num_cases)

    aggregate = {
        'gt_real': [],
        'timing_task_load': [],
        'timing_correction': [],
        'timing_rollout_guided': [],
        'timing_total': [],
        'timing_sampling_total': [],
        'lambda': {lam: {'guided_real': [], 'gain': [], 'score': [], 'diff_len': [], 'pos_err_mm': [], 'sampling_time': [], 'best_count': 0} for lam in lambda_values},
    }

    case_records = []
    rollout_buffer = []

    print(f'[setup] loaded {len(tasks)} tasks from {args.tasks_jsonl}')

    for case_idx, anchor in enumerate(tasks):
        t_case0 = time.perf_counter()

        t0 = time.perf_counter()
        t1 = time.perf_counter()
        aggregate['timing_task_load'].append(t1 - t0)
        aggregate['gt_real'].append(float(anchor['gt_real']))

        condition_raw = np.concatenate([anchor['pos'], anchor['direction'], anchor['target_normal']], axis=0).astype(np.float32)
        condition_norm = normalize_condition(condition_raw[None, :], stats)[0]
        x_dim = q_dim + 10
        prior_np = np.zeros((int(args.samples_per_lambda), 1, x_dim), dtype=np.float32)
        prior_np[:, 0, q_dim:q_dim + 9] = np.repeat(condition_norm[None, :], int(args.samples_per_lambda), axis=0)
        prior = torch.from_numpy(prior_np).float().to(device)
        init_noise = torch.randn_like(prior)

        results = []
        sampling_total = 0.0
        for lam in lambda_values:
            ts0 = time.perf_counter()
            result = sample_with_guidance(
                model=model,
                lnet_contrastive=lnet_contrastive,
                stats=stats,
                q_dim=q_dim,
                condition_raw_np=condition_raw,
                prior=prior,
                init_noise=init_noise,
                sample_steps=steps,
                temperature=float(args.temperature),
                lambda_guidance=float(lam),
                device=device,
                fixed_guidance_step=float(args.fixed_guidance_step),
                guidance_grad_eps=float(args.guidance_grad_eps),
            )
            ts1 = time.perf_counter()
            score_batch = np.asarray(result['final_score_batch'], dtype=np.float32)
            best_sample_idx = int(np.argmax(score_batch))
            result['selected_sample_idx'] = best_sample_idx
            result['selected_from_num_samples'] = int(score_batch.shape[0])
            result['final_q'] = np.asarray(result['final_q_batch'][best_sample_idx], dtype=np.float32)
            result['final_score'] = float(score_batch[best_sample_idx])
            result['final_pred_length'] = float(np.asarray(result['final_pred_length_batch'], dtype=np.float32)[best_sample_idx])
            result['sampling_time'] = float(ts1 - ts0)
            sampling_total += result['sampling_time']
            results.append(result)
            aggregate['lambda'][lam]['sampling_time'].append(result['sampling_time'])
        aggregate['timing_sampling_total'].append(sampling_total)

        q_pred_batch = np.stack([r['final_q'] for r in results], axis=0).astype(np.float32)

        t2 = time.perf_counter()
        q_corr_batch, raw_pos_err = batch_position_error_and_correction(
            tracker.robot,
            q_pred_batch,
            anchor['pos'],
            args.correction_damping,
            args.correction_iters,
            args.correction_tol,
            tracker_device,
        )
        t3 = time.perf_counter()
        aggregate['timing_correction'].append(t3 - t2)

        record = {
            'task_index': int(anchor['task_index']),
            'oracle_rank_among_6000': int(anchor['oracle_rank_among_6000']),
            'gt_real': float(anchor['gt_real']),
            'direction': anchor['direction'],
            'target_normal': anchor['target_normal'],
            'q_corr_by_lambda': {},
            'guided_real_by_lambda': {},
            'results': results,
            'raw_pos_err_mm': (raw_pos_err * 1e3).astype(np.float32),
            'case_time_prefix': t3 - t_case0,
            'rollout_time_share': 0.0,
        }
        for idx, lam in enumerate(lambda_values):
            record['q_corr_by_lambda'][lam] = q_corr_batch[idx]
        case_records.append(record)
        rollout_buffer.append(record)

        if len(rollout_buffer) >= int(args.rollout_accum_size):
            flush_rollout_buffer(
                tracker=tracker,
                tracker_device=tracker_device,
                lambda_values=lambda_values,
                rollout_batch_size=int(args.rollout_batch_size),
                aggregate=aggregate,
                buffered_records=rollout_buffer,
            )
            rollout_buffer = []

        if args.print_every > 0 and ((case_idx + 1) % args.print_every == 0 or case_idx + 1 == args.num_cases):
            print(
                f'[prepare] {case_idx + 1}/{args.num_cases} '
                f'task={np.mean(aggregate["timing_task_load"]):.3f}s '
                f'sample_total={np.mean(aggregate["timing_sampling_total"]):.3f}s '
                f'corr={np.mean(aggregate["timing_correction"]):.3f}s'
            )

    flush_rollout_buffer(
        tracker=tracker,
        tracker_device=tracker_device,
        lambda_values=lambda_values,
        rollout_batch_size=int(args.rollout_batch_size),
        aggregate=aggregate,
        buffered_records=rollout_buffer,
    )

    for case_idx, record in enumerate(case_records):
        gt_real = float(record['gt_real'])
        parts = [
            f'case={case_idx + 1}/{args.num_cases}',
            f'task_idx={record["task_index"]}',
            f'oracle_rank={record["oracle_rank_among_6000"]}',
            f'gt={fmt(gt_real)}',
        ]
        for idx, lam in enumerate(lambda_values):
            guided_real = float(record['guided_real_by_lambda'][lam])
            gain = guided_real - gt_real
            parts.append(f'lam{lam:g}={fmt(guided_real)}')
            parts.append(f'g{lam:g}={fmt(gain)}')
            parts.append(f'sel{lam:g}={int(record["results"][idx]["selected_sample_idx"]):02d}')
        total_case_time = float(record['case_time_prefix']) + float(record['rollout_time_share'])
        aggregate['timing_total'].append(total_case_time)
        print(' '.join(parts))

    summary = {
        'num_cases': int(len(tasks)),
        'bundle': str(args.bundle),
        'lnet_contrastive_ckpt': str(args.lnet_contrastive_ckpt),
        'tasks_jsonl': str(args.tasks_jsonl),
        'sample_steps': int(steps),
        'samples_per_lambda': int(args.samples_per_lambda),
        'temperature': float(args.temperature),
        'fixed_guidance_step': float(args.fixed_guidance_step),
        'rollout_batch_size': int(args.rollout_batch_size),
        'rollout_accum_size': int(args.rollout_accum_size),
        'gt_real': mean_std(aggregate['gt_real']),
        'timing': {
            'task_loading_s': mean_std(aggregate['timing_task_load']),
            'guidance_sampling_total_s': mean_std(aggregate['timing_sampling_total']),
            'jacobian_correction_s': mean_std(aggregate['timing_correction']),
            'rollout_guided_total_s': mean_std(aggregate['timing_rollout_guided']),
            'case_total_s': mean_std(aggregate['timing_total']),
        },
        'lambdas': {},
    }
    for lam in lambda_values:
        data = aggregate['lambda'][lam]
        summary['lambdas'][str(lam)] = {
            'guided_real': mean_std(data['guided_real']),
            'gain_vs_gt': mean_std(data['gain']),
            'score': mean_std(data['score']),
            'diff_len': mean_std(data['diff_len']),
            'raw_pos_err_mm': mean_std(data['pos_err_mm']),
            'sampling_time_s': mean_std(data['sampling_time']),
            'best_count': int(data['best_count']),
        }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
