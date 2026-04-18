from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from diffusion import normalize_condition
from diffusion_eval_batch_candidates_lnet import (
    batch_position_error_and_correction,
    load_lnet_contrastive_model,
)
from diffusion_eval_contrastive_guidance import sample_with_guidance
from diffusion_sample import load_model
from diffusion_vis_contrastive_guidance_cases import (
    DEFAULT_LNET_CONTRASTIVE_CKPT,
    DEFAULT_RUN_NAME,
    DEFAULT_WORKDIR,
    build_tracker,
)

BASE_DIR = Path(__file__).resolve().parent
GPU_NULLSPACE_DIR = BASE_DIR.parent
DEFAULT_TASKS_JSONL = GPU_NULLSPACE_DIR / 'length_prediction' / 'eval_lnet_contrastive_fr3_top2000_tasks_by_oracle.jsonl'
DEFAULT_BASELINE_BUNDLE = DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt'
DEFAULT_DPO_BUNDLE = (
    GPU_NULLSPACE_DIR
    / 'runs'
    / 'dit_kinematic_inpainting_runs'
    / 'ddpm32_dit_inpaint_qL_from_posdirnormal_fr3_sub10_dpo_pref_cross_cond'
    / 'bundle_best.pt'
)
DEFAULT_OUTPUT_JSON = BASE_DIR / 'diffusion_eval_dpo_vs_base_fr3_batch_summary.json'
DEFAULT_CASES_JSONL = BASE_DIR / 'diffusion_eval_dpo_vs_base_fr3_batch_cases.jsonl'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare baseline diffusion and DPO diffusion on the fixed FR3 task list.')
    parser.add_argument('--baseline-bundle', type=Path, default=DEFAULT_BASELINE_BUNDLE)
    parser.add_argument('--dpo-bundle', type=Path, default=DEFAULT_DPO_BUNDLE)
    parser.add_argument('--lnet-contrastive-ckpt', type=Path, default=DEFAULT_LNET_CONTRASTIVE_CKPT)
    parser.add_argument('--tasks-jsonl', type=Path, default=DEFAULT_TASKS_JSONL)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument('--cases-jsonl', type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num-cases', type=int, default=2000)
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--rollout-batch-size', type=int, default=200)
    parser.add_argument('--rollout-accum-size', type=int, default=50)
    parser.add_argument('--fixed-guidance-step', type=float, default=1.0)
    parser.add_argument('--guidance-grad-eps', type=float, default=1e-6)
    return parser.parse_args()


def fmt(x: float) -> str:
    return f'{float(x):.3f}'


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-12)


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


def load_tasks_from_jsonl(path: Path, num_cases: int | None = None) -> list[dict]:
    tasks = []
    with path.open('r', encoding='utf-8') as fh:
        for line_idx, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
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


def init_method_aggregate() -> dict:
    return {
        'mean_real': [],
        'mean_gain': [],
        'mean_score': [],
        'mean_pred_length': [],
        'mean_raw_pos_err_mm': [],
        'mean_sampling_time': [],
        'rollout_flush_time': [],
    }


def evaluate_one_method(
    method_name: str,
    model,
    stats: dict,
    q_dim: int,
    steps: int,
    anchor: dict,
    init_noise: torch.Tensor,
    lnet_contrastive,
    device: torch.device,
    args: argparse.Namespace,
) -> dict:
    condition_raw = np.concatenate([anchor['pos'], anchor['direction'], anchor['target_normal']], axis=0).astype(np.float32)
    condition_norm = normalize_condition(condition_raw[None, :], stats)[0]
    x_dim = q_dim + 10
    prior_np = np.zeros((int(args.num_samples), 1, x_dim), dtype=np.float32)
    prior_np[:, 0, q_dim:q_dim + 9] = np.repeat(condition_norm[None, :], int(args.num_samples), axis=0)
    prior = torch.from_numpy(prior_np).float().to(device)

    ts0 = time.perf_counter()
    result = sample_with_guidance(
        model=model,
        lnet_contrastive=lnet_contrastive,
        stats=stats,
        q_dim=q_dim,
        condition_raw_np=condition_raw,
        prior=prior,
        init_noise=init_noise.clone(),
        sample_steps=steps,
        temperature=float(args.temperature),
        lambda_guidance=0.0,
        device=device,
        fixed_guidance_step=float(args.fixed_guidance_step),
        guidance_grad_eps=float(args.guidance_grad_eps),
    )
    ts1 = time.perf_counter()

    score_batch = np.asarray(result['final_score_batch'], dtype=np.float32)
    pred_length_batch = np.asarray(result['final_pred_length_batch'], dtype=np.float32)
    q_pred_batch = np.asarray(result['final_q_batch'], dtype=np.float32)

    q_corr_batch, raw_pos_err = batch_position_error_and_correction(
        tracker.robot,
        q_pred_batch,
        anchor['pos'],
        args.correction_damping,
        args.correction_iters,
        args.correction_tol,
        tracker_device,
    )

    return {
        'method': method_name,
        'sampling_time_s': float(ts1 - ts0),
        'q_all_corr': q_corr_batch,
        'all_score_batch': score_batch,
        'all_pred_length_batch': pred_length_batch,
        'all_raw_pos_err_mm': (raw_pos_err * 1e3).astype(np.float32),
    }


def flush_rollout_buffer(
    tracker,
    tracker_device: torch.device,
    rollout_batch_size: int,
    buffered_records: list[dict],
    aggregate: dict,
    cases_jsonl: Path,
) -> None:
    if not buffered_records:
        return

    method_names = ['baseline', 'dpo']
    direction_batch = np.stack([record['direction'] for record in buffered_records], axis=0).astype(np.float32)
    normal_batch = np.stack([record['target_normal'] for record in buffered_records], axis=0).astype(np.float32)

    for method_name in method_names:
        t0 = time.perf_counter()
        all_count = len(buffered_records) * int(buffered_records[0]['methods'][method_name]['q_all_corr'].shape[0])
        all_q_batch = np.stack(
            [record['methods'][method_name]['q_all_corr'] for record in buffered_records],
            axis=0,
        ).astype(np.float32).reshape(all_count, -1)
        all_direction_batch = np.repeat(direction_batch, buffered_records[0]['methods'][method_name]['q_all_corr'].shape[0], axis=0).astype(np.float32)
        all_normal_batch = np.repeat(normal_batch, buffered_records[0]['methods'][method_name]['q_all_corr'].shape[0], axis=0).astype(np.float32)
        all_real_flat = rollout_large_batch(
            tracker,
            tracker_device,
            all_q_batch,
            all_direction_batch,
            all_normal_batch,
            int(rollout_batch_size),
        )
        all_real_by_case = all_real_flat.reshape(len(buffered_records), buffered_records[0]['methods'][method_name]['q_all_corr'].shape[0])
        t1 = time.perf_counter()
        aggregate[method_name]['rollout_flush_time'].append(float(t1 - t0))

        for record_idx, record in enumerate(buffered_records):
            gt_real = float(record['gt_real'])
            method = record['methods'][method_name]
            method['mean_real'] = float(all_real_by_case[record_idx].mean())
            aggregate[method_name]['mean_real'].append(method['mean_real'])
            aggregate[method_name]['mean_gain'].append(method['mean_real'] - gt_real)
            aggregate[method_name]['mean_score'].append(float(np.mean(method['all_score_batch'])))
            aggregate[method_name]['mean_pred_length'].append(float(np.mean(method['all_pred_length_batch'])))
            aggregate[method_name]['mean_raw_pos_err_mm'].append(float(np.mean(method['all_raw_pos_err_mm'])))
            aggregate[method_name]['mean_sampling_time'].append(float(method['sampling_time_s']))

    with cases_jsonl.open('a', encoding='utf-8') as fh:
        for record in buffered_records:
            payload = {
                'task_index': int(record['task_index']),
                'oracle_rank_among_6000': int(record['oracle_rank_among_6000']),
                'gt_real': float(record['gt_real']),
                'direction': record['direction'],
                'target_normal': record['target_normal'],
                'methods': {
                    method_name: {
                        'num_samples': int(record['methods'][method_name]['all_score_batch'].shape[0]),
                        'sampling_time_s': float(record['methods'][method_name]['sampling_time_s']),
                        'mean_score': float(np.mean(record['methods'][method_name]['all_score_batch'])),
                        'score_std': float(np.std(record['methods'][method_name]['all_score_batch'])),
                        'mean_pred_length': float(np.mean(record['methods'][method_name]['all_pred_length_batch'])),
                        'pred_length_std': float(np.std(record['methods'][method_name]['all_pred_length_batch'])),
                        'mean_real': float(record['methods'][method_name]['mean_real']),
                        'mean_gain_vs_gt': float(record['methods'][method_name]['mean_real'] - float(record['gt_real'])),
                        'mean_raw_pos_err_mm': float(np.mean(record['methods'][method_name]['all_raw_pos_err_mm'])),
                        'raw_pos_err_mm_std': float(np.std(record['methods'][method_name]['all_raw_pos_err_mm'])),
                    }
                    for method_name in method_names
                },
            }
            fh.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + '\n')


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed if args.seed is not None else int(np.random.SeedSequence().entropy))
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    device = torch.device(args.device)

    _, baseline_stats, baseline_model, baseline_q_dim, baseline_steps = load_model(args.baseline_bundle, device)
    _, dpo_stats, dpo_model, dpo_q_dim, dpo_steps = load_model(args.dpo_bundle, device)
    if int(baseline_q_dim) != int(dpo_q_dim):
        raise RuntimeError(f'q_dim mismatch: baseline={baseline_q_dim}, dpo={dpo_q_dim}')
    q_dim = int(baseline_q_dim)
    steps = int(args.sample_steps) if args.sample_steps is not None else min(int(baseline_steps), int(dpo_steps))

    lnet_contrastive = load_lnet_contrastive_model(args.lnet_contrastive_ckpt, device)
    global tracker, tracker_device
    tracker, tracker_device = build_tracker(device)
    tasks = load_tasks_from_jsonl(args.tasks_jsonl, args.num_cases)
    args.cases_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.cases_jsonl.write_text('', encoding='utf-8')

    aggregate = {
        'gt_real': [],
        'baseline': init_method_aggregate(),
        'dpo': init_method_aggregate(),
    }
    buffered_records = []

    print(f'[setup] loaded {len(tasks)} tasks from {args.tasks_jsonl}')
    print(f'[setup] baseline={args.baseline_bundle}')
    print(f'[setup] dpo={args.dpo_bundle}')

    for case_idx, anchor in enumerate(tasks):
        aggregate['gt_real'].append(float(anchor['gt_real']))
        x_dim = q_dim + 10
        init_noise = torch.randn((int(args.num_samples), 1, x_dim), device=device)

        baseline_record = evaluate_one_method(
            'baseline',
            baseline_model,
            baseline_stats,
            q_dim,
            steps,
            anchor,
            init_noise,
            lnet_contrastive,
            device,
            args,
        )
        dpo_record = evaluate_one_method(
            'dpo',
            dpo_model,
            dpo_stats,
            q_dim,
            steps,
            anchor,
            init_noise,
            lnet_contrastive,
            device,
            args,
        )

        record = {
            'task_index': int(anchor['task_index']),
            'oracle_rank_among_6000': int(anchor['oracle_rank_among_6000']),
            'gt_real': float(anchor['gt_real']),
            'direction': anchor['direction'],
            'target_normal': anchor['target_normal'],
            'methods': {
                'baseline': baseline_record,
                'dpo': dpo_record,
            },
        }
        buffered_records.append(record)

        if len(buffered_records) >= int(args.rollout_accum_size):
            flush_rollout_buffer(
                tracker=tracker,
                tracker_device=tracker_device,
                rollout_batch_size=int(args.rollout_batch_size),
                buffered_records=buffered_records,
                aggregate=aggregate,
                cases_jsonl=args.cases_jsonl,
            )
            buffered_records = []

        if args.print_every > 0 and ((case_idx + 1) % args.print_every == 0 or case_idx + 1 == len(tasks)):
            base_mean = aggregate['baseline']['mean_real']
            dpo_mean = aggregate['dpo']['mean_real']
            print(
                f'[progress] {case_idx + 1}/{len(tasks)} '
                f'base_mean={fmt(np.mean(base_mean)) if base_mean else "n/a"} '
                f'dpo_mean={fmt(np.mean(dpo_mean)) if dpo_mean else "n/a"}'
            )

    flush_rollout_buffer(
        tracker=tracker,
        tracker_device=tracker_device,
        rollout_batch_size=int(args.rollout_batch_size),
        buffered_records=buffered_records,
        aggregate=aggregate,
        cases_jsonl=args.cases_jsonl,
    )

    summary = {
        'num_cases': int(len(tasks)),
        'baseline_bundle': str(args.baseline_bundle),
        'dpo_bundle': str(args.dpo_bundle),
        'lnet_contrastive_ckpt': str(args.lnet_contrastive_ckpt),
        'tasks_jsonl': str(args.tasks_jsonl),
        'output_json': str(args.output_json),
        'cases_jsonl': str(args.cases_jsonl),
        'sample_steps': int(steps),
        'num_samples': int(args.num_samples),
        'temperature': float(args.temperature),
        'rollout_batch_size': int(args.rollout_batch_size),
        'rollout_accum_size': int(args.rollout_accum_size),
        'gt_real': mean_std(aggregate['gt_real']),
        'baseline': {
            'mean': {
                'real': mean_std(aggregate['baseline']['mean_real']),
                'gain_vs_gt': mean_std(aggregate['baseline']['mean_gain']),
                'score': mean_std(aggregate['baseline']['mean_score']),
                'pred_length': mean_std(aggregate['baseline']['mean_pred_length']),
                'raw_pos_err_mm': mean_std(aggregate['baseline']['mean_raw_pos_err_mm']),
                'sampling_time_s': mean_std(aggregate['baseline']['mean_sampling_time']),
            },
            'rollout_flush_time_s': mean_std(aggregate['baseline']['rollout_flush_time']),
        },
        'dpo': {
            'mean': {
                'real': mean_std(aggregate['dpo']['mean_real']),
                'gain_vs_gt': mean_std(aggregate['dpo']['mean_gain']),
                'score': mean_std(aggregate['dpo']['mean_score']),
                'pred_length': mean_std(aggregate['dpo']['mean_pred_length']),
                'raw_pos_err_mm': mean_std(aggregate['dpo']['mean_raw_pos_err_mm']),
                'sampling_time_s': mean_std(aggregate['dpo']['mean_sampling_time']),
            },
            'rollout_flush_time_s': mean_std(aggregate['dpo']['rollout_flush_time']),
        },
    }
    summary['delta_dpo_minus_baseline'] = {
        'mean_real_mean': (
            None
            if summary['baseline']['mean']['real']['mean'] is None or summary['dpo']['mean']['real']['mean'] is None
            else float(summary['dpo']['mean']['real']['mean'] - summary['baseline']['mean']['real']['mean'])
        ),
        'mean_gain_mean': (
            None
            if summary['baseline']['mean']['gain_vs_gt']['mean'] is None or summary['dpo']['mean']['gain_vs_gt']['mean'] is None
            else float(summary['dpo']['mean']['gain_vs_gt']['mean'] - summary['baseline']['mean']['gain_vs_gt']['mean'])
        ),
    }
    args.output_json.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
