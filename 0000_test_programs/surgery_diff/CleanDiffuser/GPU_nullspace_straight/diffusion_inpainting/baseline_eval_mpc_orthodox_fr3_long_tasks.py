from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from baseline_eval_fr3_common import DEFAULT_TASKS_JSONL, load_tasks_from_jsonl, mean_std, to_jsonable
from orthodox_baseline_fr3_common import (
    RolloutObjectiveWeights,
    build_tracker,
    differentiable_control_rollout_score,
    optimize_start_pose_multistart,
)
from trajectory_generation.fr3_nullspace_straight import directional_manipulability_batch


DEFAULT_OUTPUT_JSON = BASE_DIR / 'baseline_eval_mpc_orthodox_fr3_long_tasks_summary.json'
DEFAULT_CASES_JSONL = BASE_DIR / 'baseline_eval_mpc_orthodox_fr3_long_tasks_cases.jsonl'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate an orthodox receding-horizon MPC FR3 baseline on the long-task benchmark.')
    parser.add_argument('--tasks-jsonl', type=Path, default=DEFAULT_TASKS_JSONL)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument('--cases-jsonl', type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-cases', type=int, default=2000)
    parser.add_argument('--num-starts', type=int, default=16)
    parser.add_argument('--ik-iters', type=int, default=80)
    parser.add_argument('--ik-lr', type=float, default=0.05)
    parser.add_argument('--mpc-horizon-steps', type=int, default=40)
    parser.add_argument('--mpc-iters', type=int, default=25)
    parser.add_argument('--mpc-lr', type=float, default=0.08)
    parser.add_argument('--max-rollout-steps', type=int, default=2000)
    parser.add_argument('--print-every', type=int, default=10)
    return parser.parse_args()


def termination_name(code: int) -> str:
    return {
        0: 'low_mu',
        1: 'joint_limit',
        2: 'self_collision',
        3: 'max_steps',
        4: 'pos_tracking_error',
        5: 'orientation_violation',
    }.get(int(code), 'unknown')


def optimize_control_sequence(
    tracker,
    q_now: torch.Tensor,
    direction_t: torch.Tensor,
    normal_t: torch.Tensor,
    horizon_steps: int,
    opt_iters: int,
    opt_lr: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch_size, dof = q_now.shape
    control_seq = torch.zeros((batch_size, int(horizon_steps), dof), device=q_now.device, dtype=q_now.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([control_seq], lr=float(opt_lr))
    weights = RolloutObjectiveWeights(progress_reward=1.0, pos_error_weight=80.0, orientation_weight=20.0, collision_weight=40.0, low_mu_weight=20.0, control_weight=0.01)

    for _ in range(int(opt_iters)):
        optimizer.zero_grad(set_to_none=True)
        score, _ = differentiable_control_rollout_score(
            tracker=tracker,
            q0_batch=q_now,
            direction_batch=direction_t,
            target_normal_batch=normal_t,
            control_seq=control_seq,
            weights=weights,
        )
        loss = -score.mean()
        loss.backward()
        optimizer.step()

    score, stats = differentiable_control_rollout_score(
        tracker=tracker,
        q0_batch=q_now,
        direction_batch=direction_t,
        target_normal_batch=normal_t,
        control_seq=control_seq,
        weights=weights,
    )
    return control_seq.detach(), {
        'score': float(score[0].item()),
        'pred_length': float(stats['projected_length'][0].item()),
        'total_penalty': float(stats['total_penalty'][0].item()),
    }


def execute_mpc_rollout(
    tracker,
    q0_np: np.ndarray,
    direction_np: np.ndarray,
    target_normal_np: np.ndarray,
    horizon_steps: int,
    opt_iters: int,
    opt_lr: float,
    max_rollout_steps: int,
) -> tuple[float, dict]:
    device = tracker.robot.jnt_ranges.device
    q = torch.from_numpy(q0_np.astype(np.float32)).to(device).unsqueeze(0)
    direction_t = torch.from_numpy(direction_np.astype(np.float32)).to(device).unsqueeze(0)
    direction_t = direction_t / direction_t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    normal_t = torch.from_numpy(target_normal_np.astype(np.float32)).to(device).unsqueeze(0)
    normal_t = normal_t / normal_t.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    lower = tracker.robot.jnt_ranges[:, 0].unsqueeze(0)
    upper = tracker.robot.jnt_ranges[:, 1].unsqueeze(0)
    cos_theta_max = float(np.cos(tracker.config.theta_max))
    start_pos, _ = tracker.robot.fk_batch(q)

    q_path: list[np.ndarray] = [q[0].detach().cpu().numpy().astype(np.float32)]
    projected_lengths: list[float] = [0.0]
    plan_scores: list[float] = []
    pred_lengths: list[float] = []
    plan_times: list[float] = []
    termination_code = 3
    num_executed_steps = 0

    for step_idx in range(int(max_rollout_steps)):
        plan_t0 = time.perf_counter()
        control_seq, plan_meta = optimize_control_sequence(
            tracker=tracker,
            q_now=q,
            direction_t=direction_t,
            normal_t=normal_t,
            horizon_steps=int(horizon_steps),
            opt_iters=int(opt_iters),
            opt_lr=float(opt_lr),
        )
        plan_t1 = time.perf_counter()

        plan_scores.append(float(plan_meta['score']))
        pred_lengths.append(float(plan_meta['pred_length']))
        plan_times.append(float(plan_t1 - plan_t0))

        u0 = control_seq[:, 0, :]
        q_next = q + tracker.config.dt * u0

        tcp_pos_next, tcp_rot_next = tracker.robot.fk_batch(q_next)
        tcp_z_next = tcp_rot_next[:, :, 2]
        cos_theta_next = torch.sum(tcp_z_next * normal_t, dim=-1)

        j_cols = []
        q_eval = q_next.detach().clone().requires_grad_(True)
        tcp_pos_eval, _ = tracker.robot.fk_batch(q_eval)
        for dim in range(3):
            grad_dim = torch.autograd.grad(tcp_pos_eval[:, dim].sum(), q_eval, retain_graph=True, create_graph=False)[0]
            j_cols.append(grad_dim)
        j_pos = torch.stack(j_cols, dim=1)
        mu_val = directional_manipulability_batch(j_pos, direction_t, tracker.config.damping)

        in_range = torch.all((q_next >= lower) & (q_next <= upper), dim=1)
        coll_cost = tracker.collision_fn(q_next)
        expected_pos = start_pos + direction_t * ((step_idx + 1) * tracker.config.dt * tracker.config.task_speed)
        pos_error = torch.linalg.norm(tcp_pos_next - expected_pos, dim=-1)

        if bool((mu_val < tracker.config.mu_threshold)[0].item()):
            termination_code = 0
            break
        if not bool(in_range[0].item()):
            termination_code = 1
            break
        if bool((coll_cost > 0.0)[0].item()):
            termination_code = 2
            break
        if bool((pos_error > tracker.config.pos_error_threshold)[0].item()):
            termination_code = 4
            break
        if bool((cos_theta_next < cos_theta_max)[0].item()):
            termination_code = 5
            break

        q = q_next.detach()
        num_executed_steps += 1
        q_path.append(q[0].detach().cpu().numpy().astype(np.float32))
        delta = tcp_pos_next - start_pos
        projected_lengths.append(float(torch.sum(delta * direction_t, dim=-1)[0].item()))

    if num_executed_steps >= int(max_rollout_steps):
        termination_code = 3

    end_pos, _ = tracker.robot.fk_batch(q)
    realized_length = float(torch.sum((end_pos - start_pos) * direction_t, dim=-1)[0].item())
    return realized_length, {
        'termination_code': int(termination_code),
        'termination_reason': termination_name(int(termination_code)),
        'num_executed_steps': int(num_executed_steps),
        'q_path': q_path,
        'projected_lengths': projected_lengths,
        'mean_plan_score': float(np.mean(plan_scores)) if plan_scores else 0.0,
        'mean_pred_length': float(np.mean(pred_lengths)) if pred_lengths else 0.0,
        'last_pred_length': float(pred_lengths[-1]) if pred_lengths else 0.0,
        'mpc_plan_total_s': float(np.sum(plan_times)) if plan_times else 0.0,
        'mpc_plan_mean_s': float(np.mean(plan_times)) if plan_times else 0.0,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    tracker, _ = build_tracker(device)
    tasks = load_tasks_from_jsonl(args.tasks_jsonl, args.num_cases)

    args.cases_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.cases_jsonl.write_text('', encoding='utf-8')

    aggregate = {
        'selected_real': [],
        'gain_vs_gt': [],
        'percent_of_oracle': [],
        'ik_init_s': [],
        'mpc_plan_total_s': [],
        'mpc_plan_mean_s': [],
        'total_case_time_s': [],
        'mean_pred_length': [],
        'last_pred_length': [],
        'num_executed_steps': [],
    }

    t_global0 = time.perf_counter()
    for task_no, task in enumerate(tasks, start=1):
        case_t0 = time.perf_counter()

        ik_t0 = time.perf_counter()
        q_init_batch, init_meta = optimize_start_pose_multistart(
            tracker=tracker,
            target_pos_np=task['pos'],
            target_normal_np=task['target_normal'],
            num_starts=int(args.num_starts),
            iters=int(args.ik_iters),
            lr=float(args.ik_lr),
        )
        ik_t1 = time.perf_counter()

        best_start_idx = int(init_meta['best_idx'])
        q0 = q_init_batch[best_start_idx].astype(np.float32)

        mpc_t0 = time.perf_counter()
        selected_real, mpc_meta = execute_mpc_rollout(
            tracker=tracker,
            q0_np=q0,
            direction_np=task['direction'],
            target_normal_np=task['target_normal'],
            horizon_steps=int(args.mpc_horizon_steps),
            opt_iters=int(args.mpc_iters),
            opt_lr=float(args.mpc_lr),
            max_rollout_steps=int(args.max_rollout_steps),
        )
        mpc_t1 = time.perf_counter()

        gain_vs_gt = float(selected_real - task['gt_real'])
        percent_of_oracle = 100.0 * float(selected_real) / max(float(task['gt_real']), 1e-6)
        total_case_time_s = float(time.perf_counter() - case_t0)

        aggregate['selected_real'].append(float(selected_real))
        aggregate['gain_vs_gt'].append(gain_vs_gt)
        aggregate['percent_of_oracle'].append(percent_of_oracle)
        aggregate['ik_init_s'].append(float(ik_t1 - ik_t0))
        aggregate['mpc_plan_total_s'].append(float(mpc_meta['mpc_plan_total_s']))
        aggregate['mpc_plan_mean_s'].append(float(mpc_meta['mpc_plan_mean_s']))
        aggregate['total_case_time_s'].append(total_case_time_s)
        aggregate['mean_pred_length'].append(float(mpc_meta['mean_pred_length']))
        aggregate['last_pred_length'].append(float(mpc_meta['last_pred_length']))
        aggregate['num_executed_steps'].append(float(mpc_meta['num_executed_steps']))

        payload = {
            'method': 'mpc_orthodox',
            'task_index': int(task['task_index']),
            'oracle_rank_among_6000': int(task['oracle_rank_among_6000']),
            'task_category': task['task_category'],
            'gt_real': float(task['gt_real']),
            'direction': task['direction'],
            'target_normal': task['target_normal'],
            'selected_start_idx': int(best_start_idx),
            'selected_from_num_starts': int(q_init_batch.shape[0]),
            'selected_real': float(selected_real),
            'gain_vs_gt': gain_vs_gt,
            'percent_of_oracle': percent_of_oracle,
            'ik_init_s': float(ik_t1 - ik_t0),
            'mpc_plan_total_s': float(mpc_meta['mpc_plan_total_s']),
            'mpc_plan_mean_s': float(mpc_meta['mpc_plan_mean_s']),
            'total_case_time_s': total_case_time_s,
            'mean_pred_length': float(mpc_meta['mean_pred_length']),
            'last_pred_length': float(mpc_meta['last_pred_length']),
            'num_executed_steps': int(mpc_meta['num_executed_steps']),
            'termination_code': int(mpc_meta['termination_code']),
            'termination_reason': mpc_meta['termination_reason'],
            'q0': q0,
        }
        with args.cases_jsonl.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(to_jsonable(payload), ensure_ascii=False) + '\n')

        if task_no % max(1, int(args.print_every)) == 0 or task_no == len(tasks):
            print(
                f'[mpc-orthodox] task {task_no}/{len(tasks)} '
                f'gt={task["gt_real"]:.3f} real={selected_real:.3f} '
                f'ik_t={float(ik_t1 - ik_t0):.3f}s mpc_t={float(mpc_t1 - mpc_t0):.3f}s '
                f'steps={int(mpc_meta["num_executed_steps"])} reason={mpc_meta["termination_reason"]}',
                flush=True,
            )

    summary = {
        'method': 'mpc_orthodox',
        'num_cases': int(len(tasks)),
        'tasks_jsonl': str(args.tasks_jsonl),
        'cases_jsonl': str(args.cases_jsonl),
        'num_starts': int(args.num_starts),
        'ik_iters': int(args.ik_iters),
        'ik_lr': float(args.ik_lr),
        'mpc_horizon_steps': int(args.mpc_horizon_steps),
        'mpc_iters': int(args.mpc_iters),
        'mpc_lr': float(args.mpc_lr),
        'selected_real': mean_std(aggregate['selected_real']),
        'gain_vs_gt': mean_std(aggregate['gain_vs_gt']),
        'percent_of_oracle': mean_std(aggregate['percent_of_oracle']),
        'ik_init_s': mean_std(aggregate['ik_init_s']),
        'mpc_plan_total_s': mean_std(aggregate['mpc_plan_total_s']),
        'mpc_plan_mean_s': mean_std(aggregate['mpc_plan_mean_s']),
        'total_case_time_s': mean_std(aggregate['total_case_time_s']),
        'mean_pred_length': mean_std(aggregate['mean_pred_length']),
        'last_pred_length': mean_std(aggregate['last_pred_length']),
        'num_executed_steps': mean_std(aggregate['num_executed_steps']),
        'wall_clock_s': float(time.perf_counter() - t_global0),
    }
    args.output_json.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
