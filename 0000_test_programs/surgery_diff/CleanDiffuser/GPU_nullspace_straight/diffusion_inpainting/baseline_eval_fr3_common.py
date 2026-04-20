from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent
GPU_NULLSPACE_DIR = BASE_DIR.parent
DEFAULT_TASKS_JSONL = GPU_NULLSPACE_DIR / 'length_prediction' / 'eval_lnet_contrastive_fr3_top2000_tasks_by_oracle.jsonl'


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return (vec / max(float(np.linalg.norm(vec)), 1e-12)).astype(np.float32)


def load_tasks_from_jsonl(path: Path, num_cases: int | None = None) -> list[dict]:
    tasks: list[dict] = []
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


def rollout_large_batch(
    tracker,
    tracker_device: torch.device,
    q_batch: np.ndarray,
    direction_batch: np.ndarray,
    target_normal_batch: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    lengths: list[np.ndarray] = []
    total = int(q_batch.shape[0])
    for start in range(0, total, int(chunk_size)):
        end = min(start + int(chunk_size), total)
        q_chunk = torch.from_numpy(q_batch[start:end].astype(np.float32)).to(tracker_device)
        d_chunk = torch.from_numpy(direction_batch[start:end].astype(np.float32)).to(tracker_device)
        n_chunk = torch.from_numpy(target_normal_batch[start:end].astype(np.float32)).to(tracker_device)
        result = tracker.run_batch(q0_batch=q_chunk, direction_batch=d_chunk, target_normal_batch=n_chunk)
        lengths.append(result.projected_length.detach().cpu().numpy().astype(np.float32))
        del q_chunk, d_chunk, n_chunk, result
        if tracker_device.type == 'cuda':
            torch.cuda.empty_cache()
    return np.concatenate(lengths, axis=0).astype(np.float32)
