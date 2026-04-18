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

from diffusion import normalize_condition
from diffusion_eval_batch_candidates_lnet import (
    batch_position_error_and_correction,
    load_lnet_contrastive_model,
)
from diffusion_eval_contrastive_guidance import sample_with_guidance
from diffusion_sample import load_model
from diffusion_vis_contrastive_guidance_cases import (
    DEFAULT_LNET_CONTRASTIVE_CKPT,
    DEFAULT_RUN_NAME,
    DEFAULT_WORKDIR,
    build_tracker,
)

BASE_DIR = Path(__file__).resolve().parent
GPU_NULLSPACE_DIR = BASE_DIR.parent
DEFAULT_TASKS_JSONL = GPU_NULLSPACE_DIR / 'length_prediction' / 'eval_lnet_contrastive_fr3_top2000_tasks_by_oracle.jsonl'
DEFAULT_BASELINE_BUNDLE = DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt'
DEFAULT_DPO_BUNDLE = (
    GPU_NULLSPACE_DIR
    / 'runs'
    / 'dit_kinematic_inpainting_runs'
    / 'ddpm32_dit_inpaint_qL_from_posdirnormal_fr3_sub10_dpo_pref_cross_cond'
    / 'bundle_best.pt'
)

METHOD_COLORS = {
    'baseline': np.array([0.15, 0.45, 0.95], dtype=np.float32),
    'baseline_raw': np.array([0.55, 0.72, 0.98], dtype=np.float32),
    'dpo': np.array([0.95, 0.35, 0.15], dtype=np.float32),
    'dpo_raw': np.array([0.98, 0.72, 0.45], dtype=np.float32),
    'gt': np.array([0.15, 0.75, 0.20], dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize one baseline-vs-DPO FR3 task with one generated sample per model.')
    parser.add_argument('--baseline-bundle', type=Path, default=DEFAULT_BASELINE_BUNDLE)
    parser.add_argument('--dpo-bundle', type=Path, default=DEFAULT_DPO_BUNDLE)
    parser.add_argument('--lnet-contrastive-ckpt', type=Path, default=DEFAULT_LNET_CONTRASTIVE_CKPT)
    parser.add_argument('--tasks-jsonl', type=Path, default=DEFAULT_TASKS_JSONL)
    parser.add_argument('--task-index', type=int, default=0, help='Zero-based row index in tasks-jsonl.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--show-tcp-frame', action='store_true')
    parser.add_argument('--show-gt-stick', action='store_true')
    parser.add_argument('--show-rollout-meshes', action='store_true', help='Render sparse robot meshes along the rollout joint path.')
    parser.add_argument('--rollout-mesh-count', type=int, default=8, help='Number of rollout mesh snapshots per method.')
    return parser.parse_args()


def fmt(x: float) -> str:
    return f'{float(x):.3f}'


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-12)


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normalize(normal)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = normalize(np.cross(helper, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)


def rotation_matrix_from_direction_normal(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    x_axis = normalize(direction)
    z_axis = normalize(normal)
    y_axis = np.cross(z_axis, x_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        return rotation_matrix_from_normal(z_axis)
    y_axis = normalize(y_axis)
    z_axis = normalize(np.cross(x_axis, y_axis))
    return np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)


def load_tasks_from_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as fh:
        for line_idx, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            task_obj = row.get('task')
            if not isinstance(task_obj, dict):
                raise RuntimeError(f'Line {line_idx} of {path} does not contain a task object')
            rows.append({
                'task_index': int(row.get('task_index', len(rows))),
                'oracle_rank_among_6000': int(row.get('oracle_rank_among_6000', -1)),
                'pos': np.asarray(task_obj['pos'], dtype=np.float32),
                'direction': normalize(np.asarray(task_obj['direction'], dtype=np.float32)),
                'target_normal': normalize(np.asarray(task_obj['normal'], dtype=np.float32)),
                'gt_real': float(row.get('oracle_top1_real', 0.0)),
            })
    return rows


def run_single_method(
    method_name: str,
    bundle_path: Path,
    lnet_contrastive,
    tracker,
    tracker_device: torch.device,
    anchor: dict,
    init_noise: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    _, stats, model, q_dim, diffusion_steps = load_model(bundle_path, device)
    steps = int(args.sample_steps) if args.sample_steps is not None else int(diffusion_steps)
    condition_raw = np.concatenate([anchor['pos'], anchor['direction'], anchor['target_normal']], axis=0).astype(np.float32)
    condition_norm = normalize_condition(condition_raw[None, :], stats)[0]
    x_dim = q_dim + 10
    prior_np = np.zeros((1, 1, x_dim), dtype=np.float32)
    prior_np[:, 0, q_dim:q_dim + 9] = condition_norm[None, :]
    prior = torch.from_numpy(prior_np).float().to(device)

    result = sample_with_guidance(
        model=model,
        lnet_contrastive=lnet_contrastive,
        stats=stats,
        q_dim=q_dim,
        condition_raw_np=condition_raw,
        prior=prior,
        init_noise=init_noise.clone(),
        sample_steps=steps,
        temperature=float(args.temperature),
        lambda_guidance=0.0,
        device=device,
    )

    q_raw = np.asarray(result['final_q'], dtype=np.float32)
    q_corr_batch, raw_pos_err = batch_position_error_and_correction(
        tracker.robot,
        q_raw[None, :],
        anchor['pos'],
        args.correction_damping,
        args.correction_iters,
        args.correction_tol,
        tracker_device,
    )
    q_corr = q_corr_batch[0]
    raw_tcp_pos, _ = tracker.robot.fk_batch(torch.from_numpy(q_raw[None, :].astype(np.float32)).to(tracker_device))
    raw_tcp_pos = raw_tcp_pos[0].detach().cpu().numpy().astype(np.float32)
    q_batch = torch.from_numpy(q_corr[None, :].astype(np.float32)).to(tracker_device)
    d_batch = torch.from_numpy(anchor['direction'][None, :].astype(np.float32)).to(tracker_device)
    n_batch = torch.from_numpy(anchor['target_normal'][None, :].astype(np.float32)).to(tracker_device)
    rollout = tracker.run_batch(q0_batch=q_batch, direction_batch=d_batch, target_normal_batch=n_batch)
    tcp_path = np.asarray(rollout.tcp_path_best, dtype=np.float32)
    q_path = np.asarray(rollout.q_path_best, dtype=np.float32)

    return {
        'method': method_name,
        'bundle': str(bundle_path),
        'final_score': float(result['final_score']),
        'pred_length': float(result['final_pred_length']),
        'raw_pos_err_mm': float(raw_pos_err[0] * 1e3),
        'real_length': float(rollout.projected_length[0].detach().cpu().item()),
        'q_raw': q_raw,
        'q_corr': q_corr,
        'q_path': q_path,
        'raw_tcp_pos': raw_tcp_pos,
        'tcp_path': tcp_path,
        'steps': int(tcp_path.shape[0] - 1),
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
    mgm.gen_sphere(tcp_path[0], radius=radius * 2.0, rgb=color, alpha=0.95).attach_to(world)
    mgm.gen_sphere(tcp_path[-1], radius=radius * 2.2, rgb=color, alpha=0.95).attach_to(world)


def add_rollout_meshes(
    world,
    robot: FrankaResearch3,
    q_path: np.ndarray,
    color: np.ndarray,
    count: int,
    alpha: float,
    show_tcp_frame: bool,
) -> None:
    if q_path.ndim != 2 or q_path.shape[0] == 0:
        return
    count = max(1, min(int(count), int(q_path.shape[0])))
    indices = np.linspace(0, q_path.shape[0] - 1, count, dtype=int)
    used = []
    for idx in indices.tolist():
        if used and idx == used[-1]:
            continue
        used.append(idx)
    for idx in used:
        robot.goto_given_conf(np.asarray(q_path[idx], dtype=np.float32))
        robot.gen_meshmodel(rgb=color, alpha=alpha, toggle_tcp_frame=show_tcp_frame).attach_to(world)


def render_world(
    anchor: dict,
    baseline_row: dict,
    dpo_row: dict,
    show_tcp_frame: bool,
    show_gt_stick: bool,
    show_rollout_meshes: bool,
    rollout_mesh_count: int,
) -> None:
    world = wd.World(cam_pos=[1.7, -1.5, 1.05], lookat_pos=[0.25, 0.0, 0.25])
    mgm.gen_frame().attach_to(world)
    anchor_rotmat = rotation_matrix_from_direction_normal(anchor['direction'], anchor['target_normal'])
    start_pos = baseline_row['tcp_path'][0]
    mgm.gen_frame(pos=start_pos, rotmat=anchor_rotmat, ax_length=0.12).attach_to(world)

    plane_size = 1.2
    plane_rotmat = rotation_matrix_from_normal(anchor['target_normal'])
    plane_center = start_pos + 0.5 * plane_size * anchor['direction']
    mcm.gen_box(
        xyz_lengths=[plane_size, plane_size, 0.001],
        pos=plane_center,
        rotmat=plane_rotmat,
        rgb=[180 / 255, 211 / 255, 217 / 255],
        alpha=0.35,
    ).attach_to(world)

    if show_gt_stick:
        gt_color = METHOD_COLORS['gt']
        gt_end = start_pos + anchor['direction'] * float(anchor['gt_real'])
        mgm.gen_stick(spos=start_pos, epos=gt_end, radius=0.0045, rgb=gt_color, alpha=0.8).attach_to(world)
        mgm.gen_sphere(gt_end, radius=0.009, rgb=gt_color, alpha=0.9).attach_to(world)

    robot = FrankaResearch3(enable_cc=True)
    for row in [baseline_row, dpo_row]:
        color = METHOD_COLORS[row['method']]
        alpha = 0.32 if row['method'] == 'baseline' else 0.48
        robot.goto_given_conf(np.asarray(row['q_corr'], dtype=np.float32))
        robot.gen_meshmodel(rgb=color, alpha=alpha, toggle_tcp_frame=show_tcp_frame).attach_to(world)
        add_path(world, row['tcp_path'], color=color, radius=0.0035 if row['method'] == 'baseline' else 0.0045)
        if show_rollout_meshes:
            add_rollout_meshes(
                world=world,
                robot=robot,
                q_path=np.asarray(row['q_path'], dtype=np.float32),
                color=color,
                count=int(rollout_mesh_count),
                alpha=0.10 if row['method'] == 'baseline' else 0.14,
                show_tcp_frame=False,
            )

    # Show both raw poses before Jacobian correction so the correction gap is
    # visually obvious for baseline and DPO.
    for row, color_key in [(baseline_row, 'baseline_raw'), (dpo_row, 'dpo_raw')]:
        raw_color = METHOD_COLORS[color_key]
        robot.goto_given_conf(np.asarray(row['q_raw'], dtype=np.float32))
        robot.gen_meshmodel(rgb=raw_color, alpha=0.35, toggle_tcp_frame=False).attach_to(world)
        mgm.gen_sphere(row['raw_tcp_pos'], radius=0.010, rgb=raw_color, alpha=0.95).attach_to(world)
        mgm.gen_stick(
            spos=row['raw_tcp_pos'],
            epos=row['tcp_path'][0],
            radius=0.0025,
            rgb=raw_color,
            alpha=0.75,
        ).attach_to(world)

    world.run()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    tracker, tracker_device = build_tracker(device)
    tasks = load_tasks_from_jsonl(args.tasks_jsonl)
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise RuntimeError(f'task_index={args.task_index} is out of range for {len(tasks)} tasks in {args.tasks_jsonl}')
    anchor = tasks[args.task_index]

    rng = np.random.default_rng(args.seed if args.seed is not None else int(np.random.SeedSequence().entropy))
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    q_dim = 7
    x_dim = q_dim + 10
    init_noise = torch.randn((1, 1, x_dim), device=device)

    lnet_contrastive = load_lnet_contrastive_model(args.lnet_contrastive_ckpt, device)
    baseline_row = run_single_method(
        method_name='baseline',
        bundle_path=args.baseline_bundle,
        lnet_contrastive=lnet_contrastive,
        tracker=tracker,
        tracker_device=tracker_device,
        anchor=anchor,
        init_noise=init_noise,
        args=args,
        device=device,
    )
    dpo_row = run_single_method(
        method_name='dpo',
        bundle_path=args.dpo_bundle,
        lnet_contrastive=lnet_contrastive,
        tracker=tracker,
        tracker_device=tracker_device,
        anchor=anchor,
        init_noise=init_noise,
        args=args,
        device=device,
    )

    print(
        f"task_idx={int(anchor['task_index'])} oracle_rank={int(anchor['oracle_rank_among_6000'])} "
        f"gt_real={fmt(anchor['gt_real'])}"
    )
    for row in [baseline_row, dpo_row]:
        print(
            f"{row['method']}: real={fmt(row['real_length'])} "
            f"gain={fmt(row['real_length'] - float(anchor['gt_real']))} "
            f"score={fmt(row['final_score'])} "
            f"pred_len={fmt(row['pred_length'])} "
            f"pos_err_mm={fmt(row['raw_pos_err_mm'])} "
            f"steps={row['steps']}"
        )

    render_world(
        anchor=anchor,
        baseline_row=baseline_row,
        dpo_row=dpo_row,
        show_tcp_frame=bool(args.show_tcp_frame),
        show_gt_stick=bool(args.show_gt_stick),
        show_rollout_meshes=bool(args.show_rollout_meshes),
        rollout_mesh_count=int(args.rollout_mesh_count),
    )


if __name__ == '__main__':
    main()
