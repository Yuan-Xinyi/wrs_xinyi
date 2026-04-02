from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diffusion import DEFAULT_RUN_NAME, DEFAULT_WORKDIR, sample_q_length_from_condition
from diffusion_eval_batch_candidates_lnet import (
    batch_position_error_and_correction,
    build_tracker,
    load_lnet_contrastive_model,
    load_lnet_model,
    load_model,
    predict_length_models,
    sample_anchor,
)
from length_prediction.paths import LNET_CONTRASTIVE_RUNS_DIR, LNET_RUNS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate LCTSTOP against GT over many random settings.')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--lnet-ckpt', type=Path, default=LNET_RUNS_DIR / 'lnet_q_cond_to_length_sub10' / 'lnet_best.pt')
    parser.add_argument('--lnet-contrastive-ckpt', type=Path, default=LNET_CONTRASTIVE_RUNS_DIR / 'lnet_contrastive_q_cond_to_length_sub10_pref' / 'lnet_contrastive_best.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num-settings', type=int, default=2000)
    parser.add_argument('--n-samples', type=int, default=16)
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--print-every', type=int, default=50)
    return parser.parse_args()


def summarize(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {'mean': None, 'std': None, 'min': None, 'max': None, 'p50': None, 'p90': None}
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'p50': float(np.percentile(arr, 50)),
        'p90': float(np.percentile(arr, 90)),
    }


def rollout_lengths_multi_batch(tracker, q_batch_np: np.ndarray, direction_batch_np: np.ndarray, target_normal_batch_np: np.ndarray, device: torch.device) -> np.ndarray:
    q_batch = torch.from_numpy(q_batch_np.astype(np.float32)).to(device)
    direction_batch = torch.from_numpy(direction_batch_np.astype(np.float32)).to(device)
    target_normal_batch = torch.from_numpy(target_normal_batch_np.astype(np.float32)).to(device)
    result = tracker.run_batch(q0_batch=q_batch, direction_batch=direction_batch, target_normal_batch=target_normal_batch)
    return result.projected_length.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed if args.seed is not None else int(np.random.SeedSequence().entropy))
    device = torch.device(args.device)

    _, stats, model, q_dim, diffusion_steps = load_model(args.bundle, device)
    tracker, tracker_device = build_tracker(device)
    lnet = load_lnet_model(args.lnet_ckpt, device)
    lnet_contrastive = load_lnet_contrastive_model(args.lnet_contrastive_ckpt, device)

    steps = int(args.sample_steps) if args.sample_steps is not None else int(diffusion_steps)

    gt_q_list = []
    lct_qcorr_list = []
    direction_list = []
    normal_list = []
    gt_anchor_lnet_list = []
    gt_anchor_lcts_list = []
    lct_selected_score_list = []

    for setting_idx in range(args.num_settings):
        anchor = sample_anchor(tracker.robot, tracker.collision_fn, tracker_device, rng)
        condition = np.concatenate([anchor['pos'], anchor['direction'], anchor['target_normal']], axis=0).astype(np.float32)

        q_preds, _, _ = sample_q_length_from_condition(
            model=model,
            stats=stats,
            condition=condition,
            device=device,
            q_dim=q_dim,
            n_samples=int(args.n_samples),
            sample_steps=steps,
            temperature=float(args.temperature),
        )

        q_corrs, _ = batch_position_error_and_correction(
            tracker.robot,
            q_preds.astype(np.float32),
            anchor['pos'],
            args.correction_damping,
            args.correction_iters,
            args.correction_tol,
            tracker_device,
        )

        score_q = np.concatenate([anchor['q'][None, :], q_preds.astype(np.float32)], axis=0)
        lnet_pred_all, lcts_score_all = predict_length_models(
            lnet,
            lnet_contrastive,
            device,
            score_q,
            anchor['pos'],
            anchor['direction'],
            anchor['target_normal'],
        )
        candidate_scores = lcts_score_all[1:]
        lct_idx = int(np.argmax(candidate_scores))

        gt_q_list.append(anchor['q'].astype(np.float32))
        lct_qcorr_list.append(q_corrs[lct_idx].astype(np.float32))
        direction_list.append(anchor['direction'].astype(np.float32))
        normal_list.append(anchor['target_normal'].astype(np.float32))
        gt_anchor_lnet_list.append(float(lnet_pred_all[0]))
        gt_anchor_lcts_list.append(float(lcts_score_all[0]))
        lct_selected_score_list.append(float(candidate_scores[lct_idx]))

        if args.print_every > 0 and ((setting_idx + 1) % args.print_every == 0 or setting_idx + 1 == args.num_settings):
            print(f'[progress] selected {setting_idx + 1}/{args.num_settings} settings')

    gt_q = np.stack(gt_q_list, axis=0)
    lct_qcorr = np.stack(lct_qcorr_list, axis=0)
    direction_batch = np.stack(direction_list, axis=0)
    normal_batch = np.stack(normal_list, axis=0)

    gt_real = rollout_lengths_multi_batch(tracker, gt_q, direction_batch, normal_batch, tracker_device)
    lct_real = rollout_lengths_multi_batch(tracker, lct_qcorr, direction_batch, normal_batch, tracker_device)

    gap_arr = lct_real - gt_real
    rel_gain_arr = gap_arr / np.maximum(np.abs(gt_real), 1e-6)
    lct_better_count = int(np.sum(lct_real > gt_real + 1e-8))
    gt_better_count = int(np.sum(gt_real > lct_real + 1e-8))
    ties = int(args.num_settings - lct_better_count - gt_better_count)

    payload = {
        'num_settings': int(args.num_settings),
        'n_samples': int(args.n_samples),
        'sample_steps': int(steps),
        'bundle': str(args.bundle),
        'lnet_contrastive_ckpt': str(args.lnet_contrastive_ckpt),
        'selection_rule': 'LCTSTOP_by_contrastive_score_then_batch_rollout_compare_with_gt',
        'gt_real': summarize(gt_real),
        'lctstop_real': summarize(lct_real),
        'gap_lct_minus_gt': summarize(gap_arr),
        'relative_gain_lct_over_gt': summarize(rel_gain_arr),
        'head_to_head': {
            'lct_better_count': lct_better_count,
            'gt_better_count': gt_better_count,
            'tie_count': ties,
        },
        'score_summary': {
            'gt_anchor_lnet': summarize(np.asarray(gt_anchor_lnet_list, dtype=np.float32)),
            'gt_anchor_lcts': summarize(np.asarray(gt_anchor_lcts_list, dtype=np.float32)),
            'selected_lcts_score': summarize(np.asarray(lct_selected_score_list, dtype=np.float32)),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
