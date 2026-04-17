from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd
from wrs.robot_sim.robots.franka_research_3.franka_research_3 import FrankaResearch3

from lnet_contrastive_fr3_rotation_cone_eval import (
    DEFAULT_CKPT,
    build_tracker,
    collect_candidate_qs,
    load_model,
    rollout_same_task,
    rotation_matrix_from_normal,
)

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_JSONL = CURRENT_DIR / 'eval_lnet_contrastive_fr3_comprehensive_cases.jsonl'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Filter zero-valued evaluation cases and visualize them in a WRS world.'
    )
    parser.add_argument('--cases-jsonl', type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--field',
        type=str,
        default='selected_top1_real',
        help='Case field used to define a zero case.',
    )
    parser.add_argument(
        '--zero-tol',
        type=float,
        default=1e-12,
        help='Absolute tolerance for treating a numeric value as zero.',
    )
    parser.add_argument(
        '--case-idx',
        type=int,
        default=None,
        help='Optional exact case_idx to visualize. When unset, the script visualizes the filtered zero cases.',
    )
    parser.add_argument(
        '--max-cases',
        type=int,
        default=5,
        help='Maximum number of filtered cases to visualize when --case-idx is not set.',
    )
    parser.add_argument('--num-candidates', type=int, default=32)
    parser.add_argument('--oversample', type=int, default=50)
    parser.add_argument('--pos-tol-mm', type=float, default=2.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument(
        '--show-top-n',
        type=int,
        default=5,
        help='How many score-ranked resampled candidates to print.',
    )
    parser.add_argument(
        '--faint-candidates',
        type=int,
        default=12,
        help='How many resampled candidates to show as faint gray robots in the world.',
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only print the matching zero cases without opening the world.',
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    with path.open('r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                cases.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Failed to parse JSON on line {line_idx} of {path}: {exc}') from exc
    if not cases:
        raise RuntimeError(f'No cases were found in {path}')
    return cases


def filter_zero_cases(cases: list[dict], field: str, zero_tol: float) -> list[dict]:
    matches = []
    for case in cases:
        value = case.get(field, None)
        if isinstance(value, (int, float)) and abs(float(value)) <= zero_tol:
            matches.append(case)
    return matches


def select_cases(cases: list[dict], case_idx: int | None, max_cases: int) -> list[dict]:
    if case_idx is not None:
        for case in cases:
            if int(case.get('case_idx', -1)) == int(case_idx):
                return [case]
        raise ValueError(f'case_idx={case_idx} not found in filtered cases')
    return cases[: max(0, int(max_cases))]


def describe_case(case: dict, field: str) -> str:
    return (
        f"case_idx={int(case['case_idx'])} "
        f"task_category={case.get('task_category', 'unknown')} "
        f"{field}={float(case[field]):.6f} "
        f"oracle_top1_real={float(case.get('oracle_top1_real', 0.0)):.6f} "
        f"top1_percent_of_oracle={float(case.get('top1_percent_of_oracle', 0.0)):.2f}"
    )


def task_arrays(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task = case['task']
    pos = np.asarray(task['pos'], dtype=np.float32)
    direction = np.asarray(task['direction'], dtype=np.float32)
    normal = np.asarray(task['normal'], dtype=np.float32)
    return pos, direction, normal


def replay_case(case: dict, model, tracker, tracker_device: torch.device, device: torch.device, args: argparse.Namespace) -> dict:
    pos, direction, normal = task_arrays(case)
    q_batch_np, _ = collect_candidate_qs(
        tracker,
        tracker_device,
        pos,
        direction,
        normal,
        int(args.num_candidates),
        int(args.oversample),
        float(args.pos_tol_mm),
        int(args.correction_iters),
        float(args.correction_tol),
        float(args.correction_damping),
    )
    cond_np = np.concatenate([
        np.repeat(pos[None, :], q_batch_np.shape[0], axis=0).astype(np.float32),
        np.repeat(direction[None, :], q_batch_np.shape[0], axis=0).astype(np.float32),
        np.repeat(normal[None, :], q_batch_np.shape[0], axis=0).astype(np.float32),
    ], axis=1).astype(np.float32)
    q_batch = torch.from_numpy(q_batch_np).to(device)
    cond_batch = torch.from_numpy(cond_np).to(device)
    with torch.no_grad():
        score_batch, pred_batch = model(q_batch, cond_batch)
    score_np = score_batch.detach().cpu().numpy().astype(np.float32).reshape(-1)
    pred_len_np = pred_batch.detach().cpu().numpy().astype(np.float32).reshape(-1)
    real_len_np = rollout_same_task(tracker, tracker_device, q_batch_np, direction, normal).astype(np.float32)
    score_order = np.argsort(-score_np)
    oracle_order = np.argsort(-real_len_np)
    selected_idx = int(score_order[0])
    oracle_idx = int(oracle_order[0])
    selected_rank = int(np.where(oracle_order == selected_idx)[0][0]) + 1
    selected_real = float(real_len_np[selected_idx])
    oracle_real = float(real_len_np[oracle_idx])
    zero_real_idx = np.where(real_len_np <= float(args.zero_tol))[0].astype(np.int64)
    return {
        'q_batch_np': q_batch_np,
        'score_np': score_np,
        'pred_len_np': pred_len_np,
        'real_len_np': real_len_np,
        'score_order': score_order,
        'oracle_order': oracle_order,
        'selected_idx': selected_idx,
        'oracle_idx': oracle_idx,
        'selected_rank': selected_rank,
        'selected_real': selected_real,
        'oracle_real': oracle_real,
        'selected_ratio': selected_real / max(oracle_real, 1e-6),
        'zero_real_idx': zero_real_idx,
    }


def print_replay_summary(case: dict, replay: dict, field: str, show_top_n: int) -> None:
    print('=' * 80)
    print('[stored-case] ' + describe_case(case, field))
    print(
        '[replay] '
        f"model_top1_idx={replay['selected_idx']} "
        f"model_top1_real={replay['selected_real']:.6f} "
        f"oracle_best_idx={replay['oracle_idx']} "
        f"oracle_best_real={replay['oracle_real']:.6f} "
        f"model_top1_oracle_rank={replay['selected_rank']} "
        f"model_top1_vs_oracle={100.0 * replay['selected_ratio']:.2f}% "
        f"num_zero_real={int(replay['zero_real_idx'].shape[0])}"
    )
    top_n = min(int(show_top_n), replay['score_order'].shape[0])
    print(f'[replay] model score ranking over resampled candidates (top {top_n})')
    for rank, idx in enumerate(replay['score_order'][:top_n], start=1):
        print(
            f"  model_rank={rank} candidate_idx={int(idx)} "
            f"score={float(replay['score_np'][idx]):.4f} "
            f"pred_len={float(replay['pred_len_np'][idx]):.4f} "
            f"real_len={float(replay['real_len_np'][idx]):.4f}"
        )
    zero_examples = replay['zero_real_idx'][: min(8, replay['zero_real_idx'].shape[0])]
    if zero_examples.size:
        print('[replay] zero-real candidate indices: ' + ', '.join(str(int(i)) for i in zero_examples))


def render_world(case: dict, replay: dict, faint_candidates: int) -> None:
    pos, direction, normal = task_arrays(case)
    q_batch_np = replay['q_batch_np']
    score_order = replay['score_order']
    real_len_np = replay['real_len_np']

    world = wd.World(cam_pos=[1.8, -1.6, 1.1], lookat_pos=[0.2, 0.0, 0.35])
    mgm.gen_frame().attach_to(world)
    mgm.gen_sphere(pos, radius=0.01, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)
    mgm.gen_arrow(spos=pos, epos=pos + direction * 0.20, rgb=np.array([1.0, 0.15, 0.15]), alpha=0.9).attach_to(world)
    mgm.gen_arrow(spos=pos, epos=pos + normal * 0.20, rgb=np.array([0.15, 0.45, 1.0]), alpha=0.9).attach_to(world)
    plane_rotmat = rotation_matrix_from_normal(normal)
    plane_center = pos + 0.25 * direction
    mcm.gen_box(
        xyz_lengths=[0.60, 0.60, 0.001],
        pos=plane_center,
        rotmat=plane_rotmat,
        rgb=[0.8, 0.85, 0.9],
        alpha=0.2,
    ).attach_to(world)
    mgm.gen_frame(pos=pos, rotmat=plane_rotmat, ax_length=0.09).attach_to(world)

    robot = FrankaResearch3(enable_cc=True)
    faint_count = min(int(faint_candidates), q_batch_np.shape[0])
    for idx in score_order[:faint_count]:
        robot.goto_given_conf(q_batch_np[int(idx)].astype(np.float32))
        robot.gen_meshmodel(
            rgb=np.array([0.65, 0.65, 0.65], dtype=np.float32),
            alpha=0.05,
            toggle_tcp_frame=False,
        ).attach_to(world)

    highlight_specs = [
        ('oracle_best', replay['oracle_idx'], np.array([0.10, 0.82, 0.20], dtype=np.float32), 0.45),
        ('model_top1', replay['selected_idx'], np.array([0.95, 0.18, 0.18], dtype=np.float32), 0.55),
    ]
    top3_idx = int(score_order[min(2, score_order.shape[0] - 1)])
    top5_idx = int(score_order[min(4, score_order.shape[0] - 1)])
    highlight_specs.extend([
        ('model_rank3_boundary', top3_idx, np.array([0.95, 0.62, 0.10], dtype=np.float32), 0.28),
        ('model_rank5_boundary', top5_idx, np.array([0.50, 0.25, 0.95], dtype=np.float32), 0.22),
    ])

    seen: set[int] = set()
    for label, idx, color, alpha in highlight_specs:
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        robot.goto_given_conf(q_batch_np[idx].astype(np.float32))
        robot.gen_meshmodel(rgb=color, alpha=alpha, toggle_tcp_frame=True).attach_to(world)
        end = pos + direction * float(real_len_np[idx])
        mgm.gen_stick(spos=pos, epos=end, radius=0.005, rgb=color, alpha=0.95).attach_to(world)
        mgm.gen_sphere(end, radius=0.01, rgb=color, alpha=0.95).attach_to(world)
        print(f'[vis] {label}: candidate_idx={idx} real_len={float(real_len_np[idx]):.4f}')

    world.run()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases_jsonl)
    zero_cases = filter_zero_cases(cases, args.field, float(args.zero_tol))
    print(
        f"[filter] field='{args.field}' zero_tol={float(args.zero_tol):.2e} "
        f"matches={len(zero_cases)}/{len(cases)}"
    )
    if not zero_cases:
        return
    for case in zero_cases[: min(10, len(zero_cases))]:
        print('[match] ' + describe_case(case, args.field))
    selected_cases = select_cases(zero_cases, args.case_idx, args.max_cases)
    if args.list_only:
        return

    device = torch.device(args.device)
    model = load_model(args.ckpt, device)
    tracker, tracker_device = build_tracker(device)
    for case in selected_cases:
        replay = replay_case(case, model, tracker, tracker_device, device, args)
        print_replay_summary(case, replay, args.field, args.show_top_n)
        render_world(case, replay, args.faint_candidates)


if __name__ == '__main__':
    main()
