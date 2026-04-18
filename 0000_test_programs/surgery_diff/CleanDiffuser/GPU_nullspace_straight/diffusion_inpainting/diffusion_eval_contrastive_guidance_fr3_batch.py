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
DEFAULT_OUTPUT_JSON = BASE_DIR / 'diffusion_eval_contrastive_guidance_fr3_batch_summary.json'
DEFAULT_CASES_JSONL = BASE_DIR / 'diffusion_eval_contrastive_guidance_fr3_batch_cases.jsonl'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch-evaluate unguided diffusion on a fixed FR3 task list by comparing the selected seed against the mean of 32 sampled seeds.')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--lnet-contrastive-ckpt', type=Path, default=DEFAULT_LNET_CONTRASTIVE_CKPT)
    parser.add_argument('--tasks-jsonl', type=Path, default=DEFAULT_TASKS_JSONL)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument('--cases-jsonl', type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num-cases', type=int, default=2000)
    parser.add_argument('--guidance-lambda', type=float, default=0.0)
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--samples-per-lambda', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--fixed-guidance-step', type=float, default=1.0)
    parser.add_argument('--guidance-grad-eps', type=float, default=1e-6)
    parser.add_argument('--rollout-batch-size', type=int, default=5000)
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
    rollout_batch_size: int,
    aggregate: dict,
    buffered_records: list[dict],
    cases_jsonl: Path,
) -> None:
    if not buffered_records:
        return

    rollout_started = time.perf_counter()
    direction_batch = np.stack([record['direction'] for record in buffered_records], axis=0).astype(np.float32)
    normal_batch = np.stack([record['target_normal'] for record in buffered_records], axis=0).astype(np.float32)
    
    # Do rollout ONLY for selected samples (not all 64)
    selected_q_batch = np.stack([
        record['q_all_corr'][int(record['selected_sample_idx'])]
        for record in buffered_records
    ], axis=0).astype(np.float32)
    selected_count = len(buffered_records)
    
    print(f'[rollout start] {selected_count} selected samples', flush=True)
    selected_real_flat = rollout_large_batch(
        tracker,
        tracker_device,
        selected_q_batch,
        direction_batch,
        normal_batch,
        int(rollout_batch_size),
    )
    print(f'[rollout done]', flush=True)
    
    rollout_finished = time.perf_counter()
    rollout_time = float(rollout_finished - rollout_started)
    aggregate['timing_rollout_guided'].append(rollout_time)
    rollout_time_share = rollout_time / max(len(buffered_records), 1)

    for record_idx, record in enumerate(buffered_records):
        selected_real_i = float(selected_real_flat[record_idx])
        aggregate['selected']['real'].append(selected_real_i)
        aggregate['selected']['score'].append(float(record['selected_score']))
        aggregate['selected']['diff_len'].append(float(record['selected_pred_length']))
        aggregate['selected']['raw_pos_err_mm'].append(float(record['selected_raw_pos_err_mm']))
        record['selected_real'] = selected_real_i
        record['rollout_time_share'] = rollout_time_share
        
        case_payload = {
            'task_index': int(record['task_index']),
            'oracle_rank_among_6000': int(record['oracle_rank_among_6000']),
            'gt_real': float(record['gt_real']),
            'direction': record['direction'],
            'target_normal': record['target_normal'],
            'selected_sample_idx': int(record['selected_sample_idx']),
            'selected_from_num_samples': int(record['selected_from_num_samples']),
            'final_score': float(record['selected_score']),
            'final_pred_length': float(record['selected_pred_length']),
            'selected_real': float(record['selected_real']),
            'gain_vs_gt': float(record['selected_real'] - float(record['gt_real'])),
            'raw_pos_err_mm': float(record['selected_raw_pos_err_mm']),
            'q_corrected': record['q_selected_corr'],
            'sampling_total_s': float(record['sampling_total']),
            'jacobian_correction_s': float(record['jacobian_correction_s']),
            'rollout_time_share_s': float(record['rollout_time_share']),
            'total_case_time_s': float(record['case_time_prefix']) + float(record['rollout_time_share']),
            'guidance_lambda': float(record['guidance_lambda']),
        }
        with cases_jsonl.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(to_jsonable(case_payload), ensure_ascii=False) + '\n')


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed if args.seed is not None else int(np.random.SeedSequence().entropy))
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    device = torch.device(args.device)

    _, stats, model, q_dim, diffusion_steps = load_model(args.bundle, device)
    lnet_contrastive = load_lnet_contrastive_model(args.lnet_contrastive_ckpt, device)
    tracker, tracker_device = build_tracker(device)
    steps = int(args.sample_steps) if args.sample_steps is not None else int(diffusion_steps)
    tasks = load_tasks_from_jsonl(args.tasks_jsonl, args.num_cases)
    args.cases_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.cases_jsonl.write_text('', encoding='utf-8')

    aggregate = {
        'gt_real': [],
        'timing_task_load': [],
        'timing_correction': [],
        'timing_rollout_guided': [],
        'timing_total': [],
        'timing_sampling_total': [],
        'selected': {'real': [], 'score': [], 'diff_len': [], 'raw_pos_err_mm': []},
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
            lambda_guidance=float(args.guidance_lambda),
            device=device,
            fixed_guidance_step=float(args.fixed_guidance_step),
            guidance_grad_eps=float(args.guidance_grad_eps),
        )
        ts1 = time.perf_counter()
        score_batch = np.asarray(result['final_score_batch'], dtype=np.float32)
        pred_length_batch = np.asarray(result['final_pred_length_batch'], dtype=np.float32)
        q_pred_batch = np.asarray(result['final_q_batch'], dtype=np.float32)
        best_sample_idx = int(np.argmax(score_batch))
        sampling_total = float(ts1 - ts0)
        aggregate['timing_sampling_total'].append(sampling_total)
        print(f'[sampling done] case={case_idx + 1}/{args.num_cases} time={sampling_total:.2f}s', flush=True)

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
        print(f'[correction done] case={case_idx + 1}/{args.num_cases} time={t3-t2:.2f}s buffered={len(rollout_buffer)+1}', flush=True)

        # FR3 joint limits (7 DOF)
        jnt_ranges = np.array([
            [-2.8973, 2.8973],   # joint 0
            [-1.8326, 1.8326],   # joint 1
            [-2.8972, 2.8972],   # joint 2
            [-3.0718, -0.1222],  # joint 3
            [-2.8798, 2.8798],   # joint 4
            [0.4364, 4.6251],    # joint 5
            [-3.0543, 3.0543]    # joint 6
        ], dtype=np.float32)
        
        # Filter samples by joint safety (distance to limit > 3%)
        valid_mask = np.ones(int(q_corr_batch.shape[0]), dtype=bool)
        for jnt_idx in range(7):
            q_min, q_max = jnt_ranges[jnt_idx]
            range_size = q_max - q_min
            dist_to_min = q_corr_batch[:, jnt_idx] - q_min
            dist_to_max = q_max - q_corr_batch[:, jnt_idx]
            margin_pct_min = dist_to_min / range_size
            margin_pct_max = dist_to_max / range_size
            # Mark invalid if distance <= 3%
            valid_mask &= (margin_pct_min > 0.03) & (margin_pct_max > 0.03)
        
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            # From valid samples, select top-10 by position error
            valid_pos_err = raw_pos_err[valid_indices]
            top10_local_indices = np.argsort(valid_pos_err)[:min(10, len(valid_indices))]
            top10_indices = valid_indices[top10_local_indices]
            
            # From top-10, select best by LNet score
            scores_top10 = score_batch[top10_indices]
            best_in_top10 = np.argmax(scores_top10)
            best_sample_idx = int(top10_indices[best_in_top10])
            
            num_valid = len(valid_indices)
            print(f'[selection] case={case_idx + 1} valid: {num_valid}/64, selected from top-10 pos_err: idx={best_sample_idx}, pos_err={raw_pos_err[best_sample_idx]*1e3:.2f}mm, score={score_batch[best_sample_idx]:.4f}', flush=True)
        else:
            # Fallback: use best from all if none pass safety check
            best_sample_idx = int(np.argmax(score_batch))
            print(f'[warning] case={case_idx + 1} no valid samples (joint safety), using best of all', flush=True)

        record = {
            'task_index': int(anchor['task_index']),
            'oracle_rank_among_6000': int(anchor['oracle_rank_among_6000']),
            'gt_real': float(anchor['gt_real']),
            'guidance_lambda': float(args.guidance_lambda),
            'direction': anchor['direction'],
            'target_normal': anchor['target_normal'],
            'q_selected_raw': q_pred_batch[best_sample_idx],
            'q_selected_corr': q_corr_batch[best_sample_idx],
            'q_all_corr': q_corr_batch,
            'selected_sample_idx': best_sample_idx,
            'selected_from_num_samples': int(score_batch.shape[0]),
            'selected_score': float(score_batch[best_sample_idx]),
            'selected_pred_length': float(pred_length_batch[best_sample_idx]),
            'selected_raw_pos_err_mm': float(raw_pos_err[best_sample_idx] * 1e3),
            'all_score_batch': score_batch,
            'all_pred_length_batch': pred_length_batch,
            'all_raw_pos_err_mm': (raw_pos_err * 1e3).astype(np.float32),
            'case_time_prefix': t3 - t_case0,
            'sampling_total': float(sampling_total),
            'jacobian_correction_s': float(t3 - t2),
            'rollout_time_share': 0.0,
        }
        case_records.append(record)
        rollout_buffer.append(record)

        if len(rollout_buffer) >= int(args.rollout_accum_size):
            print(f'[flush start] {len(rollout_buffer)} records', flush=True)
            flush_rollout_buffer(
                tracker=tracker,
                tracker_device=tracker_device,
                rollout_batch_size=int(args.rollout_batch_size),
                aggregate=aggregate,
                buffered_records=rollout_buffer,
                cases_jsonl=args.cases_jsonl,
            )
            print(f'[flush done]', flush=True)
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
        rollout_batch_size=int(args.rollout_batch_size),
        aggregate=aggregate,
        buffered_records=rollout_buffer,
        cases_jsonl=args.cases_jsonl,
    )

    for case_idx, record in enumerate(case_records):
        gt_real = float(record['gt_real'])
        parts = [
            f'case={case_idx + 1}/{args.num_cases}',
            f'task_idx={record["task_index"]}',
            f'oracle_rank={record["oracle_rank_among_6000"]}',
            f'gt={fmt(gt_real)}',
            f'sel={fmt(record["selected_real"])}',
            f'gsel={fmt(record["selected_real"] - gt_real)}',
            f'mean={fmt(record["mean_real"])}',
            f'gmean={fmt(record["mean_real"] - gt_real)}',
            f'selidx={int(record["selected_sample_idx"]):02d}',
        ]
        total_case_time = float(record['case_time_prefix']) + float(record['rollout_time_share'])
        aggregate['timing_total'].append(total_case_time)
        print(' '.join(parts))

    summary = {
        'num_cases': int(len(tasks)),
        'bundle': str(args.bundle),
        'lnet_contrastive_ckpt': str(args.lnet_contrastive_ckpt),
        'tasks_jsonl': str(args.tasks_jsonl),
        'output_json': str(args.output_json),
        'cases_jsonl': str(args.cases_jsonl),
        'guidance_lambda': float(args.guidance_lambda),
        'sample_steps': int(steps),
        'num_samples': int(args.samples_per_lambda),
        'temperature': float(args.temperature),
        'fixed_guidance_step': float(args.fixed_guidance_step),
        'rollout_batch_size': int(args.rollout_batch_size),
        'rollout_accum_size': int(args.rollout_accum_size),
        'gt_real': mean_std(aggregate['gt_real']),
        'timing': {
            'task_loading_s': mean_std(aggregate['timing_task_load']),
            'sampling_total_s': mean_std(aggregate['timing_sampling_total']),
            'jacobian_correction_s': mean_std(aggregate['timing_correction']),
            'rollout_total_s': mean_std(aggregate['timing_rollout_guided']),
            'case_total_s': mean_std(aggregate['timing_total']),
        },
        'selected': {
            'real': mean_std(aggregate['selected']['real']),
            'score': mean_std(aggregate['selected']['score']),
            'diff_len': mean_std(aggregate['selected']['diff_len']),
            'raw_pos_err_mm': mean_std(aggregate['selected']['raw_pos_err_mm']),
        }
    }
    args.output_json.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
