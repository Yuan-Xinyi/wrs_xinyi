from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = CURRENT_DIR / 'lnet_contrastive_fr3_rotation_cone_eval_batch.jsonl'
DEFAULT_OUTDIR = CURRENT_DIR / 'lnet_contrastive_fr3_rotation_cone_eval_batch_analysis'

RESULT_RE = re.compile(
    r'^\[(?P<case_idx>\d+)\]\s+'
    r'real_best:\s+idx=(?P<real_idx>\d+)\s+score=(?P<real_score>[-+]?\d*\.\d+)\s+real_len=(?P<real_len>[-+]?\d*\.\d+)\s+'
    r'score_best:\s+idx=(?P<score_idx>\d+)\s+score=(?P<score_score>[-+]?\d*\.\d+)\s+real_len=(?P<score_len>[-+]?\d*\.\d+)$'
)

TASK_RE = re.compile(
    r'^\[(?P<case_idx>\d+)\]\s+task:\s+pos=(?P<pos>\[[^\]]+\])\s+direction=(?P<direction>\[[^\]]+\])\s+normal=(?P<normal>\[[^\]]+\])$'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze batched Franka contrastive q-ranking log and save plots.')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    return parser.parse_args()


def parse_vector(text: str) -> list[float]:
    return [float(x.strip()) for x in text.strip()[1:-1].split(',') if x.strip()]


def mean_std(arr: np.ndarray) -> dict:
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


def parse_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    pending_task: dict[int, dict] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m_task = TASK_RE.match(line)
        if m_task:
            idx = int(m_task.group('case_idx'))
            pending_task[idx] = {
                'pos': parse_vector(m_task.group('pos')),
                'direction': parse_vector(m_task.group('direction')),
                'normal': parse_vector(m_task.group('normal')),
            }
            continue
        m = RESULT_RE.match(line)
        if not m:
            continue
        case_idx = int(m.group('case_idx'))
        row = {
            'case_idx': case_idx,
            'real_best_idx': int(m.group('real_idx')),
            'real_best_score': float(m.group('real_score')),
            'real_best_len': float(m.group('real_len')),
            'score_best_idx': int(m.group('score_idx')),
            'score_best_score': float(m.group('score_score')),
            'score_best_len': float(m.group('score_len')),
        }
        if case_idx in pending_task:
            row.update(pending_task[case_idx])
        rows.append(row)
    return rows


def save_scatter(rows: list[dict], outdir: Path) -> None:
    x = np.asarray([r['real_best_len'] for r in rows], dtype=np.float64)
    y = np.asarray([r['score_best_len'] for r in rows], dtype=np.float64)
    lim = max(float(x.max(initial=0.0)), float(y.max(initial=0.0))) * 1.03 if len(rows) else 1.0
    fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=160)
    ax.scatter(x, y, s=18, alpha=0.7, color='#2364aa', edgecolors='none')
    ax.plot([0, lim], [0, lim], '--', color='#aa3a3a', linewidth=1.2)
    ax.set_xlabel('real_best_len')
    ax.set_ylabel('score_best_len')
    ax.set_title('Score Best vs Real Best Length')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / 'scatter_score_best_vs_real_best.png')
    plt.close(fig)


def save_gap_hist(rows: list[dict], outdir: Path) -> None:
    gap = np.asarray([r['score_best_len'] - r['real_best_len'] for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    ax.hist(gap, bins=80, color='#3fa34d', alpha=0.85)
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.0)
    ax.set_xlabel('score_best_len - real_best_len')
    ax.set_ylabel('count')
    ax.set_title('Gap Distribution')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / 'hist_gap_score_best_minus_real_best.png')
    plt.close(fig)


def save_ratio_hist(rows: list[dict], outdir: Path) -> None:
    real_best = np.asarray([r['real_best_len'] for r in rows], dtype=np.float64)
    score_best = np.asarray([r['score_best_len'] for r in rows], dtype=np.float64)
    ratio = score_best / np.clip(real_best, 1e-8, None)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=160)
    ax.hist(ratio, bins=80, color='#f0a202', alpha=0.85)
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1.0)
    ax.set_xlabel('score_best_len / real_best_len')
    ax.set_ylabel('count')
    ax.set_title('Retention Ratio Distribution')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / 'hist_ratio_score_best_over_real_best.png')
    plt.close(fig)


def save_case_curve(rows: list[dict], outdir: Path) -> None:
    case_idx = np.asarray([r['case_idx'] for r in rows], dtype=np.int32)
    real_best = np.asarray([r['real_best_len'] for r in rows], dtype=np.float64)
    score_best = np.asarray([r['score_best_len'] for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=160)
    ax.plot(case_idx, real_best, label='real_best_len', color='#1b9e77', linewidth=1.2)
    ax.plot(case_idx, score_best, label='score_best_len', color='#d95f02', linewidth=1.0, alpha=0.85)
    ax.set_xlabel('case_idx')
    ax.set_ylabel('length')
    ax.set_title('Best Lengths Across Cases')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'curve_best_lengths_by_case.png')
    plt.close(fig)


def save_summary_json(rows: list[dict], outdir: Path) -> None:
    real_best = np.asarray([r['real_best_len'] for r in rows], dtype=np.float64)
    score_best = np.asarray([r['score_best_len'] for r in rows], dtype=np.float64)
    gap = score_best - real_best
    ratio = score_best / np.clip(real_best, 1e-8, None)
    exact_hit = np.asarray([r['real_best_idx'] == r['score_best_idx'] for r in rows], dtype=np.float64)
    severe_fail = np.asarray([score_best[i] <= 0.5 * real_best[i] for i in range(len(rows))], dtype=np.float64)
    zero_like_fail = np.asarray([score_best[i] <= 1e-6 for i in range(len(rows))], dtype=np.float64)
    summary = {
        'num_cases': int(len(rows)),
        'real_best_len': mean_std(real_best),
        'score_best_len': mean_std(score_best),
        'gap_score_best_minus_real_best': mean_std(gap),
        'ratio_score_best_over_real_best': mean_std(ratio),
        'exact_hit_rate': float(exact_hit.mean()) if len(rows) else None,
        'severe_fail_rate_le_50pct': float(severe_fail.mean()) if len(rows) else None,
        'zero_like_fail_rate': float(zero_like_fail.mean()) if len(rows) else None,
    }
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = parse_log(args.input)
    if not rows:
        raise RuntimeError(f'No parsable result lines found in {args.input}')
    save_summary_json(rows, args.outdir)
    save_scatter(rows, args.outdir)
    save_gap_hist(rows, args.outdir)
    save_ratio_hist(rows, args.outdir)
    save_case_curve(rows, args.outdir)
    print(f'[saved] {args.outdir}')


if __name__ == '__main__':
    main()
