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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch-evaluate Franka contrastive guidance on random sampled tasks with timing breakdowns.')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--lnet-contrastive-ckpt', type=Path, default=DEFAULT_LNET_CONTRASTIVE_CKPT)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num-cases', type=int, default=200)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.1, 1.0, 5.0, 10.0])
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--fixed-guidance-step', type=float, default=1.0)
    parser.add_argument('--guidance-grad-eps', type=float, default=1e-6)
    parser.add_argument('--rollout-batch-size', type=int, default=200)
    return parser.parse_args()


def fmt(x: float) -> str:
    return f'{float(x):.3f}'


def sample_random_anchor(tracker, tracker_device: torch.device) -> dict:
    q0_batch, direction_batch, target_normal_batch = tracker.sample_valid_batch(batch_size=1, device=tracker_device)
    q = q0_batch[0].detach().cpu().numpy().astype(np.float32)
    direction = direction_batch[0].detach().cpu().numpy().astype(np.float32)
    target_normal = target_normal_batch[0].detach().cpu().numpy().astype(np.float32)
    tcp_pos, _ = tracker.robot.fk_batch(q0_batch)
    pos = tcp_pos[0].detach().cpu().numpy().astype(np.float32)
    return {
        'q': q,
        'pos': pos,
        'direction': direction,
        'target_normal': target_normal,
    }


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

    aggregate = {
        'gt_real': [],
        'timing_anchor': [],
        'timing_correction': [],
        'timing_rollout_gt': [],
        'timing_rollout_guided': [],
        'timing_total': [],
        'timing_sampling_total': [],
        'lambda': {lam: {'guided_real': [], 'gain': [], 'score': [], 'diff_len': [], 'pos_err_mm': [], 'sampling_time': [], 'best_count': 0} for lam in lambda_values},
    }

    case_records = []
    gt_qs = []
    gt_dirs = []
    gt_normals = []
    guided_qs = {lam: [] for lam in lambda_values}
    guided_dirs = {lam: [] for lam in lambda_values}
    guided_normals = {lam: [] for lam in lambda_values}

    for case_idx in range(args.num_cases):
        t_case0 = time.perf_counter()

        t0 = time.perf_counter()
        anchor = sample_random_anchor(tracker, tracker_device)
        t1 = time.perf_counter()
        aggregate['timing_anchor'].append(t1 - t0)

        condition_raw = np.concatenate([anchor['pos'], anchor['direction'], anchor['target_normal']], axis=0).astype(np.float32)
        condition_norm = normalize_condition(condition_raw[None, :], stats)[0]
        x_dim = q_dim + 10
        prior_np = np.zeros((1, 1, x_dim), dtype=np.float32)
        prior_np[:, 0, q_dim:q_dim + 9] = condition_norm[None, :]
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

        gt_qs.append(anchor['q'])
        gt_dirs.append(anchor['direction'])
        gt_normals.append(anchor['target_normal'])
        record = {
            'results': results,
            'raw_pos_err_mm': (raw_pos_err * 1e3).astype(np.float32),
            'case_time_prefix': t3 - t_case0,
        }
        for idx, lam in enumerate(lambda_values):
            guided_qs[lam].append(q_corr_batch[idx])
            guided_dirs[lam].append(anchor['direction'])
            guided_normals[lam].append(anchor['target_normal'])
        case_records.append(record)

        if args.print_every > 0 and ((case_idx + 1) % args.print_every == 0 or case_idx + 1 == args.num_cases):
            print(
                f'[prepare] {case_idx + 1}/{args.num_cases} '
                f'anchor={np.mean(aggregate["timing_anchor"]):.3f}s '
                f'sample_total={np.mean(aggregate["timing_sampling_total"]):.3f}s '
                f'corr={np.mean(aggregate["timing_correction"]):.3f}s'
            )

    gt_q_batch = np.stack(gt_qs, axis=0).astype(np.float32)
    gt_dir_batch = np.stack(gt_dirs, axis=0).astype(np.float32)
    gt_normal_batch = np.stack(gt_normals, axis=0).astype(np.float32)

    t_rollout_gt0 = time.perf_counter()
    gt_real_batch = rollout_large_batch(
        tracker,
        tracker_device,
        gt_q_batch,
        gt_dir_batch,
        gt_normal_batch,
        int(args.rollout_batch_size),
    )
    t_rollout_gt1 = time.perf_counter()
    gt_rollout_time = float(t_rollout_gt1 - t_rollout_gt0)
    aggregate['timing_rollout_gt'].append(gt_rollout_time)
    aggregate['gt_real'].extend(gt_real_batch.astype(float).tolist())

    guided_real_by_lambda = {}
    guided_rollout_total = 0.0
    for lam in lambda_values:
        t_rg0 = time.perf_counter()
        guided_real = rollout_large_batch(
            tracker,
            tracker_device,
            np.stack(guided_qs[lam], axis=0).astype(np.float32),
            np.stack(guided_dirs[lam], axis=0).astype(np.float32),
            np.stack(guided_normals[lam], axis=0).astype(np.float32),
            int(args.rollout_batch_size),
        )
        t_rg1 = time.perf_counter()
        guided_real_by_lambda[lam] = guided_real
        guided_rollout_total += float(t_rg1 - t_rg0)
    aggregate['timing_rollout_guided'].append(guided_rollout_total)

    for case_idx, record in enumerate(case_records):
        gt_real = float(gt_real_batch[case_idx])
        best_idx = None
        best_val = None
        parts = [f'case={case_idx + 1}/{args.num_cases}', f'gt={fmt(gt_real)}']
        for idx, lam in enumerate(lambda_values):
            guided_real = float(guided_real_by_lambda[lam][case_idx])
            gain = guided_real - gt_real
            final_score = float(record['results'][idx]['final_score'])
            diff_len = float(record['results'][idx]['final_pred_length'])
            pos_err_mm = float(record['raw_pos_err_mm'][idx])
            aggregate['lambda'][lam]['guided_real'].append(guided_real)
            aggregate['lambda'][lam]['gain'].append(gain)
            aggregate['lambda'][lam]['score'].append(final_score)
            aggregate['lambda'][lam]['diff_len'].append(diff_len)
            aggregate['lambda'][lam]['pos_err_mm'].append(pos_err_mm)
            if best_val is None or guided_real > best_val:
                best_val = guided_real
                best_idx = idx
            parts.append(f'lam{lam:g}={fmt(guided_real)}')
            parts.append(f'g{lam:g}={fmt(gain)}')
        aggregate['lambda'][lambda_values[int(best_idx)]]['best_count'] += 1
        total_case_time = float(record['case_time_prefix']) + gt_rollout_time / args.num_cases + guided_rollout_total / args.num_cases
        aggregate['timing_total'].append(total_case_time)
        print(' '.join(parts))

    summary = {
        'num_cases': int(args.num_cases),
        'bundle': str(args.bundle),
        'lnet_contrastive_ckpt': str(args.lnet_contrastive_ckpt),
        'sample_steps': int(steps),
        'temperature': float(args.temperature),
        'fixed_guidance_step': float(args.fixed_guidance_step),
        'rollout_batch_size': int(args.rollout_batch_size),
        'gt_real': mean_std(aggregate['gt_real']),
        'timing': {
            'anchor_sampling_s': mean_std(aggregate['timing_anchor']),
            'guidance_sampling_total_s': mean_std(aggregate['timing_sampling_total']),
            'jacobian_correction_s': mean_std(aggregate['timing_correction']),
            'rollout_gt_total_s': mean_std(aggregate['timing_rollout_gt']),
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
