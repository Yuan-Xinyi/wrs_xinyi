from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from lnet_contrastive import LNetContrastive
from lnet_contrastive_fr3_rotation_cone_eval import (
    build_tracker,
    collect_candidate_qs,
    rollout_same_task,
)
from paths import LNET_CONTRASTIVE_RUNS_DIR

DEFAULT_CKPT = LNET_CONTRASTIVE_RUNS_DIR / 'lnet_contrastive_q_cond_to_length_fr3_sub10_pref' / 'lnet_contrastive_best.pt'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Comprehensive FR3 rollout evaluation for a single LNetContrastive checkpoint.')
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-cases', type=int, default=100)
    parser.add_argument('--num-candidates', type=int, default=64)
    parser.add_argument('--oversample', type=int, default=512)
    parser.add_argument('--task-batch-size', type=int, default=1)
    parser.add_argument('--topk', type=int, nargs='+', default=[1, 3, 5, 10])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pair-threshold', type=float, default=0.05)
    parser.add_argument('--pos-tol-mm', type=float, default=2.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--catastrophic-ratio', type=float, default=0.7)
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--output-json', type=Path, default=CURRENT_DIR / 'eval_lnet_contrastive_fr3_comprehensive_summary.json')
    parser.add_argument('--cases-jsonl', type=Path, default=CURRENT_DIR / 'eval_lnet_contrastive_fr3_comprehensive_cases.jsonl')
    parser.add_argument('--plot-dir', type=Path, default=CURRENT_DIR / 'eval_lnet_contrastive_fr3_comprehensive_plots')
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


def kendall_tau_from_scores(scores: np.ndarray, real_len: np.ndarray, threshold: float) -> float:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    real_len = np.asarray(real_len, dtype=np.float32).reshape(-1)
    n = int(scores.shape[0])
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = float(real_len[i] - real_len[j])
            if abs(diff) <= threshold:
                continue
            pred = float(scores[i] - scores[j])
            if pred == 0.0:
                continue
            if pred * diff > 0.0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return float((concordant - discordant) / denom)


def pair_accuracy_from_scores(scores: np.ndarray, real_len: np.ndarray, threshold: float) -> float:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    real_len = np.asarray(real_len, dtype=np.float32).reshape(-1)
    valid = 0
    correct = 0
    for i in range(scores.shape[0]):
        for j in range(i + 1, scores.shape[0]):
            diff = float(real_len[i] - real_len[j])
            if abs(diff) <= threshold:
                continue
            valid += 1
            if (scores[i] - scores[j]) * diff > 0.0:
                correct += 1
    if valid == 0:
        return 0.0
    return float(correct / valid)


def load_model(ckpt_path: Path, device: torch.device) -> LNetContrastive:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    q_dim = int(torch.as_tensor(ckpt['q_min']).numel())
    if q_dim != 7:
        raise RuntimeError(f'This evaluator is FR3-only and expects q_dim=7, got q_dim={q_dim}')
    ckpt_args = ckpt.get('args', {})
    model = LNetContrastive(
        q_min=ckpt['q_min'],
        q_max=ckpt['q_max'],
        in_min=ckpt['in_min'],
        in_max=ckpt['in_max'],
        pair_threshold=float(ckpt_args.get('pair_threshold', 0.05)),
        pair_margin=float(ckpt_args.get('pair_margin', 0.05)),
        mse_weight=float(ckpt_args.get('mse_weight', 0.2)),
        rank_weight=float(ckpt_args.get('rank_weight', 1.0)),
        max_pairs=int(ckpt_args.get('max_pairs', 4096)),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def sample_random_task(tracker, tracker_device: torch.device, task_index: int) -> dict[str, np.ndarray]:
    q_batch, direction_batch, normal_batch = tracker.sample_valid_batch(batch_size=1, device=tracker_device)
    tcp_pos, _ = tracker.robot.fk_batch(q_batch)
    return {
        'task_index': int(task_index),
        'pos': tcp_pos[0].detach().cpu().numpy().astype(np.float32),
        'direction': normalize(direction_batch[0].detach().cpu().numpy().astype(np.float32)),
        'normal': normalize(normal_batch[0].detach().cpu().numpy().astype(np.float32)),
        'task_category': 'random',
    }


def summarize_cases(cases: list[dict], topk_values: list[int], catastrophic_ratio: float) -> dict[str, float]:
    if not cases:
        return {}
    top1_real = np.asarray([c['selected_top1_real'] for c in cases], dtype=np.float32)
    oracle_real = np.asarray([c['oracle_top1_real'] for c in cases], dtype=np.float32)
    candidate_real_mean = np.asarray([c['candidate_real_mean'] for c in cases], dtype=np.float32)
    candidate_real_max = np.asarray([c['candidate_real_max'] for c in cases], dtype=np.float32)
    candidate_real_min = np.asarray([c['candidate_real_min'] for c in cases], dtype=np.float32)
    candidate_real_std = np.asarray([c['candidate_real_std'] for c in cases], dtype=np.float32)
    oracle_ratio = top1_real / np.maximum(oracle_real, 1e-6)
    gap = oracle_real - top1_real
    ranks = np.asarray([c['selected_top1_oracle_rank'] for c in cases], dtype=np.float32)
    tau = np.asarray([c['kendall_tau'] for c in cases], dtype=np.float32)
    pair_acc = np.asarray([c['pair_acc'] for c in cases], dtype=np.float32)
    sampling_time = np.asarray([c['sampling_time_s'] for c in cases], dtype=np.float32)
    rollout_time = np.asarray([c['rollout_time_s'] for c in cases], dtype=np.float32)
    inference_time = np.asarray([c['inference_time_s'] for c in cases], dtype=np.float32)
    total_time = np.asarray([c['total_case_time_s'] for c in cases], dtype=np.float32)

    summary = {
        'mean_selected_top1_real': float(np.mean(top1_real)),
        'median_selected_top1_real': float(np.median(top1_real)),
        'mean_oracle_real': float(np.mean(oracle_real)),
        'mean_candidate_real_mean': float(np.mean(candidate_real_mean)),
        'mean_candidate_real_max': float(np.mean(candidate_real_max)),
        'mean_candidate_real_min': float(np.mean(candidate_real_min)),
        'mean_candidate_real_std': float(np.mean(candidate_real_std)),
        'median_candidate_real_mean': float(np.median(candidate_real_mean)),
        'mean_oracle_gap': float(np.mean(gap)),
        'mean_oracle_ratio': float(np.mean(oracle_ratio)),
        'median_oracle_ratio': float(np.median(oracle_ratio)),
        'mean_top1_minus_candidate_mean': float(np.mean(top1_real - candidate_real_mean)),
        'mean_oracle_minus_candidate_mean': float(np.mean(oracle_real - candidate_real_mean)),
        'mean_top1_zscore_over_candidates': float(np.mean((top1_real - candidate_real_mean) / np.maximum(candidate_real_std, 1e-6))),
        'mean_oracle_zscore_over_candidates': float(np.mean((oracle_real - candidate_real_mean) / np.maximum(candidate_real_std, 1e-6))),
        'mean_selected_top1_oracle_rank': float(np.mean(ranks)),
        'median_selected_top1_oracle_rank': float(np.median(ranks)),
        'oracle_hit_rate': float(np.mean(ranks == 1.0)),
        'top3_hit_rate': float(np.mean(ranks <= 3.0)),
        'top5_hit_rate': float(np.mean(ranks <= 5.0)),
        'mean_kendall_tau': float(np.mean(tau)),
        'median_kendall_tau': float(np.median(tau)),
        'mean_pair_acc': float(np.mean(pair_acc)),
        'catastrophic_fail_rate': float(np.mean(oracle_ratio < catastrophic_ratio)),
        'mean_sampling_time_s': float(np.mean(sampling_time)),
        'mean_rollout_time_s': float(np.mean(rollout_time)),
        'mean_inference_time_s': float(np.mean(inference_time)),
        'mean_total_case_time_s': float(np.mean(total_time)),
    }
    for k in topk_values:
        topk_real = np.asarray([c[f'top{k}_best_real'] for c in cases], dtype=np.float32)
        topk_ratio = topk_real / np.maximum(oracle_real, 1e-6)
        summary[f'mean_top{k}_best_real'] = float(np.mean(topk_real))
        summary[f'mean_top{k}_oracle_ratio'] = float(np.mean(topk_ratio))
        summary[f'top{k}_oracle_hit_rate'] = float(np.mean(np.asarray([c[f'top{k}_best_oracle_rank'] for c in cases], dtype=np.float32) == 1.0))
    return summary


def summarize_by_category(
    cases: list[dict],
    topk_values: list[int],
    catastrophic_ratio: float,
) -> dict[str, dict[str, float]]:
    grouped = {'all': cases}
    for category in sorted({str(c.get('task_category', 'unknown')) for c in cases}):
        grouped[category] = [c for c in cases if c.get('task_category') == category]
    return {
        name: summarize_cases(group_cases, topk_values, catastrophic_ratio)
        for name, group_cases in grouped.items()
        if group_cases
    }


def summarize_optimal_length_percent(cases: list[dict], topk_values: list[int], num_candidates: int) -> dict[str, dict[str, float]]:
    if not cases:
        return {}

    oracle = np.asarray([c['oracle_top1_real'] for c in cases], dtype=np.float32)

    def stats_from_ratio(ratio_percent: np.ndarray) -> dict[str, float]:
        return {
            'mean': float(np.mean(ratio_percent)),
            'std': float(np.std(ratio_percent)),
            'min': float(np.min(ratio_percent)),
            'max': float(np.max(ratio_percent)),
        }

    summary = {
        f'GT-{num_candidates}samples': stats_from_ratio(np.full_like(oracle, 100.0, dtype=np.float32)),
    }
    for k in topk_values:
        topk_real = np.asarray([c[f'top{k}_best_real'] for c in cases], dtype=np.float32)
        ratio = 100.0 * topk_real / np.maximum(oracle, 1e-6)
        summary[f'Top{k}'] = stats_from_ratio(ratio.astype(np.float32))
    return summary


def make_optimal_length_latex(optimal_stats: dict[str, dict[str, float]], num_candidates: int) -> str:
    if not optimal_stats:
        return ''
    ordered_rows = [f'GT-{num_candidates}samples'] + [k for k in optimal_stats.keys() if k != f'GT-{num_candidates}samples']
    lines = [
        r'\begin{table*}[!htbp]',
        r'\centering',
        r'\setlength{\tabcolsep}{4.5pt}',
        rf'\caption{{Optimal length ratio (\%) over {num_candidates} sampled candidates.}}',
        r'\label{tab:optimal_length_ratio}',
        r'\begin{threeparttable}',
        r'\begin{tabular}{ccccc}',
        r'\toprule',
        r'& \multicolumn{4}{c}{Optimal Length (\%)} \\',
        r'\cmidrule(lr){2-5}',
        r'& {Mean} & {Std} & {Min} & {Max} \\',
        r'\midrule',
    ]
    for row_name in ordered_rows:
        vals = optimal_stats[row_name]
        lines.append(
            f'{row_name}\n& {vals["mean"]:.2f} & {vals["std"]:.2f} & {vals["min"]:.2f} & {vals["max"]:.2f} \\\\'
        )
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{threeparttable}',
        r'\end{table*}',
    ])
    return '\n'.join(lines)


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


def evaluate_case(
    case_idx: int,
    task: dict[str, np.ndarray],
    q_np: np.ndarray,
    real_len: np.ndarray,
    model: LNetContrastive,
    device: torch.device,
    topk_values: list[int],
    pair_threshold: float,
) -> dict:
    infer_t0 = time.perf_counter()
    cond_np = np.concatenate([
        np.repeat(task['pos'][None, :], q_np.shape[0], axis=0).astype(np.float32),
        np.repeat(task['direction'][None, :], q_np.shape[0], axis=0).astype(np.float32),
        np.repeat(task['normal'][None, :], q_np.shape[0], axis=0).astype(np.float32),
    ], axis=1).astype(np.float32)
    q_batch = torch.from_numpy(q_np).to(device)
    cond_batch = torch.from_numpy(cond_np).to(device)
    with torch.no_grad():
        score, pred_length = model(q_batch, cond_batch)
    infer_t1 = time.perf_counter()
    score_np = score.detach().cpu().numpy().astype(np.float32).reshape(-1)
    pred_length_np = pred_length.detach().cpu().numpy().astype(np.float32).reshape(-1)
    oracle_order = np.argsort(-real_len)
    pred_order = np.argsort(-score_np)
    pred_top1_idx = int(pred_order[0])
    candidate_mean = float(np.mean(real_len))
    candidate_min = float(np.min(real_len))
    candidate_max = float(np.max(real_len))
    candidate_std = float(np.std(real_len))

    case_payload = {
        'case_idx': int(case_idx),
        'task_index': int(task['task_index']) if 'task_index' in task else int(case_idx - 1),
        'task_category': task.get('task_category', 'unknown'),
        'task': {
            'pos': task['pos'].tolist(),
            'direction': task['direction'].tolist(),
            'normal': task['normal'].tolist(),
        },
        'oracle_top1_real': float(real_len[int(oracle_order[0])]),
        'selected_top1_idx': pred_top1_idx,
        'selected_top1_score': float(score_np[pred_top1_idx]),
        'selected_top1_pred_length': float(pred_length_np[pred_top1_idx]),
        'selected_top1_real': float(real_len[pred_top1_idx]),
        'selected_top1_oracle_rank': int(np.where(oracle_order == pred_top1_idx)[0][0]) + 1,
        'top1_percent_of_oracle': 100.0 * float(real_len[pred_top1_idx]) / max(float(real_len[int(oracle_order[0])]), 1e-6),
        'candidate_real_mean': candidate_mean,
        'candidate_real_min': candidate_min,
        'candidate_real_max': candidate_max,
        'candidate_real_std': candidate_std,
        'selected_top1_minus_candidate_mean': float(real_len[pred_top1_idx]) - candidate_mean,
        'oracle_minus_candidate_mean': float(real_len[int(oracle_order[0])]) - candidate_mean,
        'kendall_tau': kendall_tau_from_scores(score_np, real_len, pair_threshold),
        'pair_acc': pair_accuracy_from_scores(score_np, real_len, pair_threshold),
        'inference_time_s': float(infer_t1 - infer_t0),
    }
    for k in topk_values:
        topk_idx = pred_order[: min(int(k), pred_order.shape[0])]
        best_local = int(topk_idx[np.argmax(real_len[topk_idx])])
        case_payload[f'top{k}_best_idx'] = int(best_local)
        case_payload[f'top{k}_best_real'] = float(real_len[best_local])
        case_payload[f'top{k}_best_oracle_rank'] = int(np.where(oracle_order == best_local)[0][0]) + 1
        case_payload[f'top{k}_percent_of_oracle'] = 100.0 * float(real_len[best_local]) / max(float(real_len[int(oracle_order[0])]), 1e-6)
    return case_payload


def create_visualizations(cases: list[dict], summary: dict, plot_dir: Path, topk_values: list[int]) -> list[str]:
    if plt is None or not cases:
        return []

    plot_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    case_ids = np.asarray([c['case_idx'] for c in cases], dtype=np.int32)
    top1_real = np.asarray([c['selected_top1_real'] for c in cases], dtype=np.float32)
    oracle_real = np.asarray([c['oracle_top1_real'] for c in cases], dtype=np.float32)
    candidate_real_mean = np.asarray([c['candidate_real_mean'] for c in cases], dtype=np.float32)
    candidate_real_std = np.asarray([c['candidate_real_std'] for c in cases], dtype=np.float32)
    candidate_real_min = np.asarray([c['candidate_real_min'] for c in cases], dtype=np.float32)
    candidate_real_max = np.asarray([c['candidate_real_max'] for c in cases], dtype=np.float32)
    oracle_ratio = top1_real / np.maximum(oracle_real, 1e-6)
    top1_rank = np.asarray([c['selected_top1_oracle_rank'] for c in cases], dtype=np.float32)
    tau = np.asarray([c['kendall_tau'] for c in cases], dtype=np.float32)
    pair_acc = np.asarray([c['pair_acc'] for c in cases], dtype=np.float32)
    top1_score = np.asarray([c['selected_top1_score'] for c in cases], dtype=np.float32)
    top1_pred_length = np.asarray([c['selected_top1_pred_length'] for c in cases], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(case_ids, oracle_real, label='oracle_top1_real', linewidth=2.0)
    ax.plot(case_ids, top1_real, label='selected_top1_real', linewidth=2.0)
    ax.plot(case_ids, candidate_real_mean, label='candidate_real_mean', linewidth=2.0, linestyle='--')
    ax.set_title('Top1 Rollout Length vs Oracle')
    ax.set_xlabel('Case Index')
    ax.set_ylabel('Rollout Length')
    ax.legend()
    ax.grid(alpha=0.25)
    path = plot_dir / 'top1_vs_oracle.png'
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(str(path))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(case_ids, candidate_real_mean, label='candidate_real_mean', linewidth=2.0, linestyle='--')
    ax.fill_between(
        case_ids,
        candidate_real_mean - candidate_real_std,
        candidate_real_mean + candidate_real_std,
        alpha=0.2,
        label='candidate mean ± std',
    )
    ax.plot(case_ids, candidate_real_min, label='candidate_real_min', linewidth=1.5, alpha=0.8)
    ax.plot(case_ids, candidate_real_max, label='candidate_real_max', linewidth=1.5, alpha=0.8)
    ax.plot(case_ids, top1_real, label='selected_top1_real', linewidth=2.0)
    ax.plot(case_ids, oracle_real, label='oracle_top1_real', linewidth=2.0)
    ax.set_title('Candidate Pool Stats vs Selected Top1')
    ax.set_xlabel('Case Index')
    ax.set_ylabel('Rollout Length')
    ax.legend(ncol=2)
    ax.grid(alpha=0.25)
    path = plot_dir / 'candidate_pool_stats.png'
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(oracle_ratio, bins=min(20, max(5, len(cases))), color='#2e8b57', alpha=0.85, edgecolor='black')
    ax.axvline(float(np.mean(oracle_ratio)), color='red', linestyle='--', linewidth=2.0, label=f'mean={np.mean(oracle_ratio):.3f}')
    ax.set_title('Oracle Ratio Distribution')
    ax.set_xlabel('selected_top1_real / oracle_top1_real')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(alpha=0.2)
    path = plot_dir / 'oracle_ratio_hist.png'
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.arange(1, int(max(top1_rank.max(), 2)) + 2) - 0.5
    ax.hist(top1_rank, bins=bins, color='#4169e1', alpha=0.85, edgecolor='black')
    ax.axvline(float(np.mean(top1_rank)), color='red', linestyle='--', linewidth=2.0, label=f'mean={np.mean(top1_rank):.2f}')
    ax.set_title('Selected Top1 Oracle Rank Distribution')
    ax.set_xlabel('Oracle Rank of Selected Top1')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(alpha=0.2)
    path = plot_dir / 'top1_oracle_rank_hist.png'
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(top1_score, top1_real, c=oracle_ratio, cmap='viridis', s=36, alpha=0.9)
    ax.set_title('Selected Top1 Score vs Real Rollout')
    ax.set_xlabel('Selected Top1 Score')
    ax.set_ylabel('Selected Top1 Real Rollout')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Oracle Ratio')
    ax.grid(alpha=0.2)
    path = plot_dir / 'top1_score_vs_real.png'
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pair_acc, tau, s=36, alpha=0.9, color='#8b008b')
    ax.set_title('Per-case Pair Accuracy vs Kendall Tau')
    ax.set_xlabel('Pair Accuracy')
    ax.set_ylabel('Kendall Tau')
    ax.grid(alpha=0.2)
    path = plot_dir / 'pair_acc_vs_tau.png'
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(str(path))

    summary_metrics = [
        ('mean_oracle_ratio', summary['mean_oracle_ratio']),
        ('oracle_hit_rate', summary['oracle_hit_rate']),
        ('top3_hit_rate', summary['top3_hit_rate']),
        ('top5_hit_rate', summary['top5_hit_rate']),
        ('mean_kendall_tau', summary['mean_kendall_tau']),
        ('mean_pair_acc', summary['mean_pair_acc']),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [k for k, _ in summary_metrics]
    values = [v for _, v in summary_metrics]
    ax.bar(labels, values, color=['#2e8b57', '#4682b4', '#5f9ea0', '#6495ed', '#9370db', '#cd5c5c'])
    ax.set_ylim(0.0, 1.05)
    ax.set_title('Summary Metrics')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.2)
    path = plot_dir / 'summary_metrics_bar.png'
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(str(path))

    spread_metrics = [
        ('cand_mean', summary['mean_candidate_real_mean']),
        ('cand_min', summary['mean_candidate_real_min']),
        ('cand_max', summary['mean_candidate_real_max']),
        ('cand_std', summary['mean_candidate_real_std']),
        ('top1', summary['mean_selected_top1_real']),
        ('oracle', summary['mean_oracle_real']),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [k for k, _ in spread_metrics]
    values = [v for _, v in spread_metrics]
    ax.bar(labels, values, color=['#6b8e23', '#b22222', '#1e90ff', '#daa520', '#2e8b57', '#4169e1'])
    ax.set_title('Candidate Pool Mean/Min/Max/Std vs Top1/Oracle')
    ax.set_ylabel('Rollout Length')
    ax.grid(axis='y', alpha=0.2)
    path = plot_dir / 'candidate_pool_summary_bar.png'
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(str(path))

    for k in topk_values:
        topk_best = np.asarray([c[f'top{k}_best_real'] for c in cases], dtype=np.float32)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(case_ids, oracle_real, label='oracle_top1_real', linewidth=2.0)
        ax.plot(case_ids, topk_best, label=f'top{k}_best_real', linewidth=2.0)
        ax.set_title(f'Top{k} Best Rollout vs Oracle')
        ax.set_xlabel('Case Index')
        ax.set_ylabel('Rollout Length')
        ax.legend()
        ax.grid(alpha=0.25)
        path = plot_dir / f'top{k}_vs_oracle.png'
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))

    return generated


def main() -> None:
    args = parse_args()
    np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = load_model(args.ckpt, device)
    tracker, tracker_device = build_tracker(device)
    num_cases = int(args.num_cases)
    print('[setup] sampling random tasks online')
    print(f'[setup] evaluating num_cases={num_cases} num_candidates={args.num_candidates} task_batch_size={args.task_batch_size}')

    args.cases_jsonl.write_text('')
    cases: list[dict] = []
    start_time = time.perf_counter()
    task_batch_size = max(1, int(args.task_batch_size))
    for batch_start in range(0, num_cases, task_batch_size):
        batch_end = min(batch_start + task_batch_size, num_cases)
        batch_tasks = [
            sample_random_task(tracker, tracker_device, task_index=batch_start + local_idx)
            for local_idx in range(batch_end - batch_start)
        ]
        print(f'[batch] tasks {batch_start + 1}-{batch_end}/{num_cases}: sampling candidates')
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
        print(f'[batch] tasks {batch_start + 1}-{batch_end}/{num_cases}: rollout finished')
        for local_idx, (task, q_np, real_len, sampling_time_s, rollout_time_s, case_t0) in enumerate(
            zip(batch_tasks, batch_q_np, batch_real, batch_sampling_times, batch_rollout_times, case_start_times),
            start=1,
        ):
            case_idx = batch_start + local_idx
            case_payload = evaluate_case(
                case_idx=case_idx,
                task=task,
                q_np=q_np,
                real_len=real_len,
                model=model,
                device=device,
                topk_values=list(args.topk),
                pair_threshold=float(args.pair_threshold),
            )
            case_payload['sampling_time_s'] = float(sampling_time_s)
            case_payload['rollout_time_s'] = float(rollout_time_s)
            case_payload['total_case_time_s'] = float(time.perf_counter() - case_t0)
            cases.append(case_payload)
            with args.cases_jsonl.open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(to_jsonable(case_payload), ensure_ascii=False) + '\n')

            if args.print_every > 0 and (case_idx % args.print_every == 0 or case_idx == num_cases):
                elapsed = time.perf_counter() - start_time
                print(
                    f'[progress] {case_idx}/{num_cases} elapsed={elapsed:.1f}s '
                    f'top1_real={case_payload["selected_top1_real"]:.4f} '
                    f'oracle_real={case_payload["oracle_top1_real"]:.4f} '
                    f'rank={case_payload["selected_top1_oracle_rank"]} '
                    f'sample_t={case_payload["sampling_time_s"]:.3f}s '
                    f'rollout_t={case_payload["rollout_time_s"]:.3f}s'
                )

    summary = {
        'args': to_jsonable(vars(args)),
        'task_source': 'random',
        'num_cases': int(len(cases)),
        'metrics': summarize_cases(cases, list(args.topk), float(args.catastrophic_ratio)),
        'metrics_by_category': summarize_by_category(cases, list(args.topk), float(args.catastrophic_ratio)),
    }
    summary['optimal_length_percent'] = summarize_optimal_length_percent(cases, list(args.topk), int(args.num_candidates))
    summary['optimal_length_percent_latex'] = make_optimal_length_latex(summary['optimal_length_percent'], int(args.num_candidates))
    generated_plots = create_visualizations(cases, summary['metrics'], args.plot_dir, list(args.topk))
    summary['plot_dir'] = str(args.plot_dir)
    summary['plots'] = generated_plots
    args.output_json.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
