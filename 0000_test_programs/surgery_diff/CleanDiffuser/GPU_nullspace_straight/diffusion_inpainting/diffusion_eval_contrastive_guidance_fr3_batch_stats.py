from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute aggregate statistics from FR3 batch evaluation JSONL files.')
    parser.add_argument('jsonl_files', type=Path, nargs='*', help='One or more case-level JSONL files.')
    parser.add_argument('--glob', dest='glob_patterns', action='append', default=[], help='Glob pattern(s) for JSONL files.')
    parser.add_argument('--indent', type=int, default=2, help='Indentation for JSON output.')
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as fh:
        for line_idx, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Failed to parse JSON on line {line_idx} of {path}: {exc}') from exc
    return rows


def stats(values: list[float]) -> dict | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }


def collect_float(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        values.append(float(value))
    return values


def summarize_rows(rows: list[dict]) -> dict:
    gt_real = collect_float(rows, 'gt_real')
    selected_real = collect_float(rows, 'selected_real')
    final_pred_length = collect_float(rows, 'final_pred_length')
    raw_pos_err_mm = collect_float(rows, 'raw_pos_err_mm')
    sampling_total_s = collect_float(rows, 'sampling_total_s')
    jacobian_correction_s = collect_float(rows, 'jacobian_correction_s')
    rollout_time_share_s = collect_float(rows, 'rollout_time_share_s')
    total_case_time_s = collect_float(rows, 'total_case_time_s')
    gain_vs_gt = collect_float(rows, 'gain_vs_gt')

    proximity_to_optimal_pct = []
    for row in rows:
        gt = row.get('gt_real')
        selected = row.get('selected_real')
        if gt is None or selected is None:
            continue
        gt = float(gt)
        selected = float(selected)
        if gt > 1e-12:
            proximity_to_optimal_pct.append(100.0 * selected / gt)

    joint_limit_gains = sorted({float(row['joint_limit_gain']) for row in rows if 'joint_limit_gain' in row})

    return {
        'num_cases': int(len(rows)),
        'joint_limit_gains': joint_limit_gains,
        'gt_real': stats(gt_real),
        'selected_real': stats(selected_real),
        'final_pred_length': stats(final_pred_length),
        'raw_pos_err_mm': stats(raw_pos_err_mm),
        'gain_vs_gt': stats(gain_vs_gt),
        'proximity_to_optimal_pct': stats(proximity_to_optimal_pct),
        'timing': {
            'sampling_total_s': stats(sampling_total_s),
            'jacobian_correction_s': stats(jacobian_correction_s),
            'rollout_time_share_s': stats(rollout_time_share_s),
            'total_case_time_s': stats(total_case_time_s),
        },
    }


def main() -> None:
    args = parse_args()
    paths: list[Path] = []
    paths.extend(args.jsonl_files)
    for pattern in args.glob_patterns:
        paths.extend(sorted(Path().glob(pattern)))
    deduped_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped_paths.append(path)
    if not deduped_paths:
        raise RuntimeError('No JSONL files provided. Use positional paths or --glob.')

    output = {}
    for path in deduped_paths:
        rows = load_jsonl(path)
        output[str(path)] = summarize_rows(rows)
    print(json.dumps(output, indent=args.indent, ensure_ascii=False))


if __name__ == '__main__':
    main()
