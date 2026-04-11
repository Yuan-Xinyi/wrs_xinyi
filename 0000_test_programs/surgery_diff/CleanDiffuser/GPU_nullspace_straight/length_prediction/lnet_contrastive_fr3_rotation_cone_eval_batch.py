from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from lnet_contrastive_fr3_rotation_cone_eval import (
    DEFAULT_CKPT,
    build_tracker,
    collect_candidate_qs,
    load_model,
    rollout_same_task,
    sample_task_anchor,
)

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = CURRENT_DIR / 'lnet_contrastive_fr3_rotation_cone_eval_batch.jsonl'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch-run Franka fixed-task q-ranking tests and write one-line summaries.')
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-cases', type=int, default=1000)
    parser.add_argument('--num-candidates', type=int, default=64)
    parser.add_argument('--oversample', type=int, default=512)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--pos-tol-mm', type=float, default=2.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--log-path', type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument('--print-every', type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed if args.seed is not None else int(np.random.SeedSequence().entropy))
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    device = torch.device(args.device)

    model = load_model(args.ckpt, device)
    tracker, tracker_device = build_tracker(device)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with args.log_path.open('a', encoding='utf-8') as f:
        for case_idx in range(1, int(args.num_cases) + 1):
            task_anchor = sample_task_anchor(tracker, tracker_device)
            q_batch_np, _ = collect_candidate_qs(
                tracker,
                tracker_device,
                task_anchor['pos'],
                task_anchor['direction'],
                task_anchor['normal'],
                int(args.num_candidates),
                int(args.oversample),
                float(args.pos_tol_mm),
                int(args.correction_iters),
                float(args.correction_tol),
                float(args.correction_damping),
            )
            pos_batch_np = np.repeat(task_anchor['pos'][None, :], q_batch_np.shape[0], axis=0).astype(np.float32)
            direction = task_anchor['direction']
            normal = task_anchor['normal']

            q_batch = torch.from_numpy(q_batch_np).to(device)
            cond_batch = torch.from_numpy(np.concatenate([
                pos_batch_np,
                np.repeat(direction[None, :], q_batch_np.shape[0], axis=0).astype(np.float32),
                np.repeat(normal[None, :], q_batch_np.shape[0], axis=0).astype(np.float32),
            ], axis=1)).to(device)
            with torch.no_grad():
                score_batch, _ = model(q_batch, cond_batch)
            score_np = score_batch.detach().cpu().numpy().astype(np.float32).reshape(-1)
            real_len_np = rollout_same_task(tracker, tracker_device, q_batch_np, direction, normal)

            best_real_idx = int(np.argmax(real_len_np))
            best_score_idx = int(np.argmax(score_np))
            task_line = (
                f'[{case_idx}] task: '
                f'pos={np.round(task_anchor["pos"], 4).tolist()} '
                f'direction={np.round(task_anchor["direction"], 4).tolist()} '
                f'normal={np.round(task_anchor["normal"], 4).tolist()}'
            )
            line = (
                f'[{case_idx}] '
                f'real_best: idx={best_real_idx} score={float(score_np[best_real_idx]):.4f} real_len={float(real_len_np[best_real_idx]):.4f} '
                f'score_best: idx={best_score_idx} score={float(score_np[best_score_idx]):.4f} real_len={float(real_len_np[best_score_idx]):.4f}'
            )
            print(task_line)
            print(line)
            f.write(task_line + '\n')
            f.write(line + '\n')
            f.flush()

            if args.print_every > 0 and (case_idx % args.print_every == 0 or case_idx == int(args.num_cases)):
                elapsed = time.perf_counter() - t0
                print(f'[progress] {case_idx}/{args.num_cases} elapsed={elapsed:.1f}s avg={elapsed / case_idx:.3f}s')


if __name__ == '__main__':
    main()
