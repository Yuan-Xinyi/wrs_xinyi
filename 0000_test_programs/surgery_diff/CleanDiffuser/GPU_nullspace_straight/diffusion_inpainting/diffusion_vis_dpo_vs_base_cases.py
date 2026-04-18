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

from diffusion_vis_contrastive_guidance_cases import build_tracker

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_JSONL = BASE_DIR / 'diffusion_eval_dpo_vs_base_fr3_batch_cases.jsonl'

METHOD_COLORS = {
    'baseline': np.array([0.15, 0.45, 0.95], dtype=np.float32),
    'dpo': np.array([0.95, 0.35, 0.15], dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize one baseline-vs-DPO FR3 case together with rollout paths.')
    parser.add_argument('--cases-jsonl', type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument('--case-index', type=int, default=0, help='Zero-based line index in the cases JSONL.')
    parser.add_argument('--task-index', type=int, default=None, help='Optional task_index lookup. If set, overrides --case-index.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--show-tcp-frame', action='store_true')
    parser.add_argument('--show-ideal-gt-stick', action='store_true')
    return parser.parse_args()


def fmt(x: float) -> str:
    return f'{float(x):.3f}'


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-12)


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normalize(normal)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.cross(helper, z_axis)
    x_axis = normalize(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = normalize(y_axis)
    return np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)


def load_case(path: Path, case_index: int, task_index: int | None) -> dict:
    with path.open('r', encoding='utf-8') as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise RuntimeError(f'No cases found in {path}')
    if task_index is not None:
        for row in rows:
            if int(row.get('task_index', -1)) == int(task_index):
                return row
        raise RuntimeError(f'task_index={task_index} not found in {path}')
    if case_index < 0 or case_index >= len(rows):
        raise RuntimeError(f'case_index={case_index} is out of range for {len(rows)} rows in {path}')
    return rows[case_index]


def rollout_single(tracker, q_corr: np.ndarray, direction: np.ndarray, target_normal: np.ndarray):
    q_batch = torch.from_numpy(np.asarray(q_corr, dtype=np.float32)[None, :]).to(tracker.robot.device)
    d_batch = torch.from_numpy(np.asarray(direction, dtype=np.float32)[None, :]).to(tracker.robot.device)
    n_batch = torch.from_numpy(np.asarray(target_normal, dtype=np.float32)[None, :]).to(tracker.robot.device)
    result = tracker.run_batch(q0_batch=q_batch, direction_batch=d_batch, target_normal_batch=n_batch)
    path = np.asarray(result.tcp_path_best, dtype=np.float32)
    return {
        'projected_length': float(result.projected_length[0].detach().cpu().item()),
        'tcp_path': path,
        'start_pos': path[0].copy(),
        'end_pos': path[-1].copy(),
        'num_steps': int(path.shape[0] - 1),
    }


def add_path(world, tcp_path: np.ndarray, color: np.ndarray, radius: float) -> None:
    if tcp_path.shape[0] < 2:
        return
    for idx in range(tcp_path.shape[0] - 1):
        mgm.gen_stick(
            spos=tcp_path[idx],
            epos=tcp_path[idx + 1],
            radius=radius,
            rgb=color,
            alpha=0.95,
        ).attach_to(world)
    for idx in [0, tcp_path.shape[0] - 1]:
        mgm.gen_sphere(tcp_path[idx], radius=radius * 2.0, rgb=color, alpha=0.95).attach_to(world)


def render_case(case: dict, baseline_rollout: dict, dpo_rollout: dict, show_tcp_frame: bool, show_ideal_gt_stick: bool) -> None:
    direction = normalize(np.asarray(case['direction'], dtype=np.float32))
    target_normal = normalize(np.asarray(case['target_normal'], dtype=np.float32))
    gt_real = float(case['gt_real'])

    start_pos = baseline_rollout['start_pos']
    plane_size = 1.2
    plane_rotmat = rotation_matrix_from_normal(target_normal)
    plane_center = start_pos + 0.5 * plane_size * direction

    world = wd.World(cam_pos=[1.7, -1.5, 1.05], lookat_pos=[0.25, 0.0, 0.25])
    mgm.gen_frame().attach_to(world)
    mgm.gen_frame(pos=start_pos, rotmat=np.column_stack((direction, np.cross(target_normal, direction), target_normal)), ax_length=0.12).attach_to(world)
    mcm.gen_box(
        xyz_lengths=[plane_size, plane_size, 0.001],
        pos=plane_center,
        rotmat=plane_rotmat,
        rgb=[180 / 255, 211 / 255, 217 / 255],
        alpha=0.35,
    ).attach_to(world)

    if show_ideal_gt_stick:
        gt_color = np.array([0.15, 0.75, 0.20], dtype=np.float32)
        gt_end = start_pos + direction * gt_real
        mgm.gen_stick(spos=start_pos, epos=gt_end, radius=0.0045, rgb=gt_color, alpha=0.75).attach_to(world)
        mgm.gen_sphere(gt_end, radius=0.009, rgb=gt_color, alpha=0.9).attach_to(world)

    robot = FrankaResearch3(enable_cc=True)
    for method_name, rollout in [('baseline', baseline_rollout), ('dpo', dpo_rollout)]:
        color = METHOD_COLORS[method_name]
        q_vis = np.asarray(case['methods'][method_name]['q_corrected'], dtype=np.float32)
        robot.goto_given_conf(q_vis)
        alpha = 0.32 if method_name == 'baseline' else 0.48
        robot.gen_meshmodel(rgb=color, alpha=alpha, toggle_tcp_frame=show_tcp_frame).attach_to(world)
        add_path(world, rollout['tcp_path'], color=color, radius=0.0035 if method_name == 'baseline' else 0.0045)

    world.run()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tracker, _ = build_tracker(device)
    case = load_case(args.cases_jsonl, args.case_index, args.task_index)

    direction = normalize(np.asarray(case['direction'], dtype=np.float32))
    target_normal = normalize(np.asarray(case['target_normal'], dtype=np.float32))
    baseline_q = np.asarray(case['methods']['baseline']['q_corrected'], dtype=np.float32)
    dpo_q = np.asarray(case['methods']['dpo']['q_corrected'], dtype=np.float32)

    baseline_rollout = rollout_single(tracker, baseline_q, direction, target_normal)
    dpo_rollout = rollout_single(tracker, dpo_q, direction, target_normal)

    print(
        f"task_idx={int(case['task_index'])} oracle_rank={int(case['oracle_rank_among_6000'])} "
        f"gt_real={fmt(case['gt_real'])}"
    )
    for method_name, rollout in [('baseline', baseline_rollout), ('dpo', dpo_rollout)]:
        data = case['methods'][method_name]
        print(
            f"{method_name}: "
            f"sel_real_stored={fmt(data['selected_real'])} "
            f"sel_real_rerun={fmt(rollout['projected_length'])} "
            f"gain={fmt(data['selected_gain_vs_gt'])} "
            f"score={fmt(data['final_score'])} "
            f"pred_len={fmt(data['final_pred_length'])} "
            f"pos_err_mm={fmt(data['selected_raw_pos_err_mm'])} "
            f"mean_real={fmt(data['mean_real'])} "
            f"steps={rollout['num_steps']}"
        )

    render_case(
        case=case,
        baseline_rollout=baseline_rollout,
        dpo_rollout=dpo_rollout,
        show_tcp_frame=bool(args.show_tcp_frame),
        show_ideal_gt_stick=bool(args.show_ideal_gt_stick),
    )


if __name__ == '__main__':
    main()
