#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from diffusion import sample_q_from_condition, sample_q_length_from_condition
from diffusion_sample import load_model
LENGTH_PRED_DIR = PARENT_DIR / 'length_prediction'
if str(LENGTH_PRED_DIR) not in sys.path:
    sys.path.insert(0, str(LENGTH_PRED_DIR))
from length_prediction.lnet_contrastive_fr3_rotation_cone_eval import (
    batch_position_error_and_correction,
    build_tracker,
    rollout_same_task,
)
from trajectory_generation.fr3_nullspace_straight import (
    directional_manipulability_batch,
    joints_in_range_mask,
    position_jacobian_batch,
    project_direction_to_plane_batch,
    random_unit_vectors_batch,
)

try:
    import wrs.modeling.collision_model as mcm
    import wrs.modeling.geometric_model as mgm
    import wrs.visualization.panda.world as wd
    from wrs.robot_sim.robots.franka_research_3.franka_research_3 import FrankaResearch3

    WRS_AVAILABLE = True
except ImportError:
    WRS_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Random-task FR3 diffusion candidate evaluation and 3D visualization.')
    parser.add_argument(
        '--bundle',
        type=Path,
        default=BASE_DIR.parent / 'runs' / 'dit_kinematic_inpainting_runs' / 'ddpm32_dit_inpaint_q_only_long' / 'bundle_latest.pt',
    )
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n-samples', type=int, default=1, help='How many random tasks to generate.')
    parser.add_argument('--n-preds', type=int, default=8, help='How many diffusion candidates per task.')
    parser.add_argument('--steps', type=int, default=32, help='Diffusion sampling steps.')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed.')
    parser.add_argument('--plane-size', type=float, default=1.0)
    parser.add_argument('--arrow-length', type=float, default=0.18)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--anchor-oversample', type=int, default=64, help='Per-try batch size for random task generation.')
    parser.add_argument('--anchor-max-tries', type=int, default=100, help='Maximum tries for random task generation.')
    parser.add_argument('--show-all-preds', action='store_true', help='Render every corrected prediction instead of only top candidates.')
    parser.add_argument('--no-vis', action='store_true')
    return parser.parse_args()


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normal / max(np.linalg.norm(normal), 1e-12)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-12)
    return np.column_stack((x_axis, y_axis, z_axis))


def rotation_error_deg_batch(rot_batch: np.ndarray, rot_ref: np.ndarray) -> np.ndarray:
    errs = []
    for rot in rot_batch:
        cos_theta = np.clip((np.trace(rot.T @ rot_ref) - 1.0) * 0.5, -1.0, 1.0)
        errs.append(float(np.degrees(np.arccos(cos_theta))))
    return np.asarray(errs, dtype=np.float32)


def add_task_geometry(world, pos: np.ndarray, direction: np.ndarray, normal: np.ndarray, plane_size: float, arrow_length: float) -> None:
    plane_rotmat = rotation_matrix_from_normal(normal)
    plane_center = pos + 0.5 * plane_size * direction
    mcm.gen_box(
        xyz_lengths=[plane_size, plane_size, 0.001],
        pos=plane_center,
        rotmat=plane_rotmat,
        rgb=[180 / 255, 211 / 255, 217 / 255],
        alpha=0.35,
    ).attach_to(world)
    mgm.gen_frame(pos=pos, rotmat=plane_rotmat, ax_length=0.08).attach_to(world)
    mgm.gen_sphere(pos, radius=0.010, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)
    mgm.gen_arrow(spos=pos, epos=pos + direction * arrow_length, rgb=np.array([1.0, 0.12, 0.12]), alpha=0.95).attach_to(world)
    mgm.gen_arrow(spos=pos, epos=pos + normal * arrow_length, rgb=np.array([0.10, 0.45, 1.0]), alpha=0.95).attach_to(world)


def sample_random_task_anchor(tracker, tracker_device: torch.device, oversample: int, max_tries: int) -> dict:
    oversample = max(1, int(oversample))
    cos_theta_max = float(np.cos(tracker.config.theta_max))
    for try_idx in range(int(max_tries)):
        q_cand = tracker.robot.rand_conf_batch(oversample).to(tracker_device)
        n_cand = random_unit_vectors_batch(oversample, device=tracker_device)
        d_cand = project_direction_to_plane_batch(torch.randn(oversample, 3, device=tracker_device), n_cand)
        j_pos, _ = position_jacobian_batch(tracker.robot, q_cand, create_graph=False)
        mu = directional_manipulability_batch(j_pos, d_cand, tracker.config.damping)
        tcp_pos, tcp_rot = tracker.robot.fk_batch(q_cand)
        tcp_z = tcp_rot[:, :, 2]
        cos_theta = torch.sum(tcp_z * n_cand, dim=-1)
        coll_cost = tracker.collision_fn(q_cand)
        valid_mask = (
            joints_in_range_mask(tracker.robot, q_cand)
            & (coll_cost <= 0.0)
            & (mu > tracker.config.mu_threshold)
            & (cos_theta > cos_theta_max)
        )
        valid_idx = torch.where(valid_mask)[0]
        if len(valid_idx) > 0:
            idx = int(valid_idx[0].item())
            print(f'[sample] try={try_idx + 1} valid=1/{oversample}')
            return {
                'q': q_cand[idx].detach().cpu().numpy().astype(np.float32),
                'pos': tcp_pos[idx].detach().cpu().numpy().astype(np.float32),
                'direction': d_cand[idx].detach().cpu().numpy().astype(np.float32),
                'normal': n_cand[idx].detach().cpu().numpy().astype(np.float32),
            }
        if tracker_device.type == 'cuda':
            torch.cuda.empty_cache()
    raise RuntimeError(f'Failed to sample a valid random FR3 task after {max_tries} tries with oversample={oversample}.')


def build_case(anchor: dict, q_pred: np.ndarray, q_corr: np.ndarray, pred_lengths: np.ndarray, gt_real: float, real_lengths: np.ndarray, tracker_device: torch.device, tracker) -> dict:
    q_stack_np = np.vstack([anchor['q'][None, :], q_corr]).astype(np.float32)
    q_stack = torch.from_numpy(q_stack_np).to(device=tracker_device, dtype=torch.float32)
    tcp_pos_t, tcp_rot_t = tracker.robot.fk_batch(q_stack)
    tcp_pos = tcp_pos_t.detach().cpu().numpy().astype(np.float32)
    tcp_rot = tcp_rot_t.detach().cpu().numpy().astype(np.float32)

    gt_pos = tcp_pos[0]
    gt_rot = tcp_rot[0]
    pred_pos = tcp_pos[1:]
    pred_rot = tcp_rot[1:]

    pos_ref = anchor['pos']
    rot_ref = gt_rot
    case = {
        'anchor': anchor,
        'q_pred': q_pred.astype(np.float32),
        'q_corr': q_corr.astype(np.float32),
        'pred_lengths': pred_lengths.astype(np.float32),
        'real_lengths': real_lengths.astype(np.float32),
        'gt_real': float(gt_real),
        'gt_fk_pos': gt_pos,
        'gt_fk_rotmat': gt_rot,
        'gt_pos_err_mm': float(np.linalg.norm(gt_pos - pos_ref) * 1000.0),
        'gt_rot_err_deg': float(rotation_error_deg_batch(gt_rot[None, ...], rot_ref)[0]),
        'pred_fk_pos': pred_pos,
        'pred_fk_rotmat': pred_rot,
        'pred_pos_err_mm': np.linalg.norm(pred_pos - pos_ref[None, :], axis=1).astype(np.float32) * 1000.0,
        'pred_rot_err_deg': rotation_error_deg_batch(pred_rot, rot_ref),
    }
    case['best_real_idx'] = int(np.argmax(case['real_lengths']))
    case['best_pred_idx'] = int(np.argmax(case['pred_lengths']))
    return case


def print_case_summary(case_idx: int, case: dict) -> None:
    anchor = case['anchor']
    print(
        f"--- Random task {case_idx} | "
        f"gt_real={case['gt_real']:.3f} m | "
        f"gt_pos_err={case['gt_pos_err_mm']:.3f} mm | "
        f"gt_rot_err={case['gt_rot_err_deg']:.3f} deg ---"
    )
    print(
        "anchor pos = "
        f"{np.array2string(anchor['pos'], precision=4, suppress_small=True)} | "
        "direction = "
        f"{np.array2string(anchor['direction'], precision=4, suppress_small=True)} | "
        "normal = "
        f"{np.array2string(anchor['normal'], precision=4, suppress_small=True)}"
    )
    for idx, (pred_len, real_len, pos_err, rot_err) in enumerate(
        zip(case['pred_lengths'], case['real_lengths'], case['pred_pos_err_mm'], case['pred_rot_err_deg'])
    ):
        print(
            f"  pred[{idx:02d}] pred_len={float(pred_len):.3f} m | "
            f"real_len={float(real_len):.3f} m | "
            f"pos_err={float(pos_err):.3f} mm | "
            f"rot_err={float(rot_err):.3f} deg"
        )
        print(
            f"           q_raw = {np.array2string(case['q_pred'][idx], precision=4, suppress_small=True)}"
        )
        print(
            f"           q_corr= {np.array2string(case['q_corr'][idx], precision=4, suppress_small=True)}"
        )


def visualize_case(case: dict, plane_size: float, arrow_length: float, show_all_preds: bool) -> None:
    if not WRS_AVAILABLE:
        print('⚠️ WRS environment not available; visualization skipped.')
        return

    anchor = case['anchor']
    start = anchor['pos']
    direction = anchor['direction']
    normal = anchor['normal']

    world = wd.World(cam_pos=[2.0, 2.0, 0.8], lookat_pos=[0.2, 0.0, 0.4])
    mgm.gen_frame().attach_to(world)
    add_task_geometry(world, start, direction, normal, plane_size=plane_size, arrow_length=arrow_length)

    gt_color = np.array([0.05, 0.75, 0.20], dtype=np.float32)
    best_real_color = np.array([1.0, 0.25, 0.0], dtype=np.float32)
    best_pred_color = np.array([0.15, 0.45, 0.95], dtype=np.float32)

    robot_gt = FrankaResearch3(enable_cc=True)
    robot_gt.goto_given_conf(anchor['q'].astype(np.float32))
    robot_gt.gen_meshmodel(rgb=gt_color, alpha=0.85, toggle_tcp_frame=True).attach_to(world)
    mgm.gen_sphere(case['gt_fk_pos'], radius=0.006, rgb=gt_color, alpha=0.95).attach_to(world)
    gt_end = start + direction * float(case['gt_real'])
    mgm.gen_stick(spos=start, epos=gt_end, radius=0.0045, rgb=gt_color, alpha=0.95).attach_to(world)
    mgm.gen_sphere(gt_end, radius=0.009, rgb=gt_color, alpha=0.95).attach_to(world)

    indices = range(len(case['q_corr'])) if show_all_preds else sorted({case['best_real_idx'], case['best_pred_idx']})
    for idx in indices:
        q_corr = case['q_corr'][idx]
        real_len = float(case['real_lengths'][idx])
        is_best_real = idx == case['best_real_idx']
        is_best_pred = idx == case['best_pred_idx']
        color = best_real_color if is_best_real else best_pred_color if is_best_pred else np.array([1.0, 0.45, 0.1], dtype=np.float32)
        alpha = 0.80 if (is_best_real or is_best_pred) else 0.18
        robot_pred = FrankaResearch3(enable_cc=False)
        robot_pred.goto_given_conf(q_corr.astype(np.float32))
        robot_pred.gen_meshmodel(rgb=color, alpha=alpha, toggle_tcp_frame=False).attach_to(world)
        mgm.gen_sphere(case['pred_fk_pos'][idx], radius=0.0045, rgb=color, alpha=min(0.95, alpha + 0.15)).attach_to(world)
        pred_end = start + direction * real_len
        mgm.gen_stick(spos=start, epos=pred_end, radius=0.0038, rgb=color, alpha=min(0.95, alpha + 0.15)).attach_to(world)

    world.run()


def main() -> None:
    args = parse_args()
    if not args.bundle.exists():
        print(f'❌ bundle not found: {args.bundle}')
        return

    seed = args.seed if args.seed is not None else int(np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    device = torch.device(args.device)

    _, stats, model, q_dim, diffusion_steps = load_model(args.bundle, device)
    tracker, tracker_device = build_tracker(device)
    steps = int(args.steps) if args.steps is not None else int(diffusion_steps)

    for sample_idx in range(int(args.n_samples)):
        t0 = time.perf_counter()
        anchor = sample_random_task_anchor(
            tracker,
            tracker_device,
            oversample=int(args.anchor_oversample),
            max_tries=int(args.anchor_max_tries),
        )
        t1 = time.perf_counter()

        condition = np.concatenate([anchor['pos'], anchor['direction'], anchor['normal']], axis=0).astype(np.float32)
        token_dim = int(stats.get('token_dim', q_dim + 9))
        if token_dim == q_dim + 10:
            q_pred, pred_lengths, _ = sample_q_length_from_condition(
                model=model,
                stats=stats,
                condition=condition,
                device=device,
                q_dim=q_dim,
                n_samples=int(args.n_preds),
                sample_steps=steps,
                temperature=1.0,
            )
        else:
            q_pred = sample_q_from_condition(
                model=model,
                stats=stats,
                condition=condition,
                device=device,
                q_dim=q_dim,
                n_samples=int(args.n_preds),
                sample_steps=steps,
                temperature=1.0,
            )
            pred_lengths = np.full((int(args.n_preds),), np.nan, dtype=np.float32)
        t2 = time.perf_counter()

        q_corr, raw_pos_err = batch_position_error_and_correction(
            tracker.robot,
            q_pred.astype(np.float32),
            anchor['pos'],
            float(args.correction_damping),
            int(args.correction_iters),
            float(args.correction_tol),
            tracker_device,
        )
        t3 = time.perf_counter()

        rollout_q = np.vstack([anchor['q'][None, :], q_corr]).astype(np.float32)
        rollout_lengths = rollout_same_task(tracker, tracker_device, rollout_q, anchor['direction'], anchor['normal'])
        t4 = time.perf_counter()

        case = build_case(anchor, q_pred, q_corr, pred_lengths, float(rollout_lengths[0]), rollout_lengths[1:], tracker_device, tracker)
        case['raw_pos_err_mm'] = raw_pos_err.astype(np.float32) * 1000.0

        print(
            f"[time] sample={sample_idx + 1}/{args.n_samples} "
            f"task={t1 - t0:.3f}s sample={t2 - t1:.3f}s correction={t3 - t2:.3f}s rollout={t4 - t3:.3f}s"
        )
        print_case_summary(sample_idx, case)

        if not args.no_vis:
            visualize_case(case, plane_size=float(args.plane_size), arrow_length=float(args.arrow_length), show_all_preds=bool(args.show_all_preds))


if __name__ == '__main__':
    main()
