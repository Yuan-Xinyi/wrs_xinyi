from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax2torch
import numpy as np
import torch
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
from wrs.robot_sim.robots.franka_research_3.franka_research_3 import FrankaResearch3
from wrs.robot_sim.robots.franka_research_3.sphere_collision_checker import SphereCollisionChecker

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[4]
TRAJ_DIR = CURRENT_DIR.parent / 'trajectory_generation'
if str(TRAJ_DIR) not in sys.path:
    sys.path.insert(0, str(TRAJ_DIR))

from lnet_contrastive import LNetContrastive
from fr3_nullspace_straight import (
    FrankaResearch3GPU,
    GPUNullspaceStraightTracker,
    TrackerConfig,
    directional_manipulability_batch,
    joints_in_range_mask,
    position_jacobian_batch,
)

DEFAULT_CKPT = CURRENT_DIR.parent / 'runs' / 'lnet_contrastive_runs' / 'lnet_contrastive_q_cond_to_length_fr3_sub10_pref' / 'lnet_contrastive_best.pt'
DEFAULT_URDF = PROJECT_ROOT / 'wrs' / 'robot_sim' / 'robots' / 'franka_research_3' / 'franka_research_3_ccsphere.urdf'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate Franka contrastive ranking for many different start joint angles under one fixed task condition.')
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-candidates', type=int, default=16)
    parser.add_argument('--oversample', type=int, default=512)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--pos-tol-mm', type=float, default=2.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--vis', dest='vis', action='store_true')
    parser.add_argument('--no-vis', dest='vis', action='store_false')
    parser.set_defaults(vis=True)
    return parser.parse_args()


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def descending_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind='stable')
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float('nan')
    x = x.astype(np.float64) - float(np.mean(x))
    y = y.astype(np.float64) - float(np.mean(y))
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom < 1e-12:
        return float('nan')
    return float(np.dot(x, y) / denom)


def topk_overlap(rank_a: np.ndarray, rank_b: np.ndarray, k: int) -> int:
    top_a = set(np.where(rank_a <= k)[0].tolist())
    top_b = set(np.where(rank_b <= k)[0].tolist())
    return int(len(top_a & top_b))


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normalize(normal)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(z_axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = normalize(np.cross(helper, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)


def load_model(ckpt_path: Path, device: torch.device) -> LNetContrastive:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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


def build_tracker(device: torch.device) -> tuple[GPUNullspaceStraightTracker, torch.device]:
    franka = FrankaResearch3GPU(device=device)
    cc_model = SphereCollisionChecker(str(DEFAULT_URDF))
    vmap_jax_cost = jax.jit(jax.vmap(cc_model.self_collision_cost, in_axes=(0, None, None)))
    collision_fn = jax2torch.jax2torch(lambda q_batch: vmap_jax_cost(q_batch, 1.0, -0.005))
    tracker = GPUNullspaceStraightTracker(
        robot=franka.robot,
        collision_fn=collision_fn,
        config=TrackerConfig(),
        print_every=0,
    )
    return tracker, device


def sample_task_anchor(tracker: GPUNullspaceStraightTracker, tracker_device: torch.device) -> dict:
    q_batch, direction_batch, normal_batch = tracker.sample_valid_batch(batch_size=1, device=tracker_device)
    tcp_pos, _ = tracker.robot.fk_batch(q_batch)
    return {
        'q': q_batch[0].detach().cpu().numpy().astype(np.float32),
        'pos': tcp_pos[0].detach().cpu().numpy().astype(np.float32),
        'direction': direction_batch[0].detach().cpu().numpy().astype(np.float32),
        'normal': normal_batch[0].detach().cpu().numpy().astype(np.float32),
    }


def batch_position_error_and_correction(robot, q_batch_np: np.ndarray, target_pos_np: np.ndarray, damping: float, max_iters: int, tol: float, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    q_batch = torch.from_numpy(q_batch_np.astype(np.float32)).to(device)
    target_pos = torch.from_numpy(np.repeat(target_pos_np[None, :].astype(np.float32), q_batch_np.shape[0], axis=0)).to(device)
    tcp_pos_raw, _ = robot.fk_batch(q_batch)
    raw_pos_err = torch.linalg.norm(target_pos - tcp_pos_raw, dim=1)

    q_corr = q_batch.clone()
    lower = robot.jnt_ranges[:, 0].unsqueeze(0)
    upper = robot.jnt_ranges[:, 1].unsqueeze(0)
    for _ in range(max_iters):
        tcp_pos, _ = robot.fk_batch(q_corr)
        err = target_pos - tcp_pos
        err_norm = torch.linalg.norm(err, dim=1)
        if bool(torch.all(err_norm < tol)):
            break
        j_pos, _ = position_jacobian_batch(robot, q_corr, create_graph=False)
        eye = torch.eye(3, device=device, dtype=q_corr.dtype).unsqueeze(0).expand(j_pos.shape[0], -1, -1)
        metric = j_pos @ j_pos.transpose(1, 2) + (damping ** 2) * eye
        dq = (j_pos.transpose(1, 2) @ torch.linalg.solve(metric, err.unsqueeze(-1))).squeeze(-1)
        active = (err_norm >= tol).float().unsqueeze(1)
        q_corr = q_corr + active * dq
        q_corr = torch.max(torch.min(q_corr, upper), lower)
    return q_corr.detach().cpu().numpy().astype(np.float32), raw_pos_err.detach().cpu().numpy().astype(np.float32)


def collect_candidate_qs(tracker: GPUNullspaceStraightTracker, tracker_device: torch.device, target_pos: np.ndarray, direction: np.ndarray, normal: np.ndarray, num_candidates: int, oversample: int, pos_tol_mm: float, correction_iters: int, correction_tol: float, correction_damping: float) -> tuple[np.ndarray, np.ndarray]:
    q_list = []
    err_list = []
    pos_tol_m = float(pos_tol_mm) * 1e-3
    cos_theta_max = float(np.cos(tracker.config.theta_max))
    direction_t = torch.from_numpy(np.repeat(direction[None, :].astype(np.float32), oversample, axis=0)).to(tracker_device)
    normal_t = torch.from_numpy(np.repeat(normal[None, :].astype(np.float32), oversample, axis=0)).to(tracker_device)

    while sum(arr.shape[0] for arr in q_list) < num_candidates:
        q_rand = tracker.robot.rand_conf_batch(int(oversample)).detach().cpu().numpy().astype(np.float32)
        q_corr_np, raw_err_np = batch_position_error_and_correction(
            tracker.robot,
            q_rand,
            target_pos,
            correction_damping,
            correction_iters,
            correction_tol,
            tracker_device,
        )
        q_corr = torch.from_numpy(q_corr_np).to(tracker_device)
        tcp_pos, tcp_rot = tracker.robot.fk_batch(q_corr)
        pos_err = torch.linalg.norm(tcp_pos - torch.from_numpy(np.repeat(target_pos[None, :].astype(np.float32), q_corr_np.shape[0], axis=0)).to(tracker_device), dim=1)
        coll_cost = tracker.collision_fn(q_corr)
        j_pos, _ = position_jacobian_batch(tracker.robot, q_corr, create_graph=False)
        mu = directional_manipulability_batch(j_pos, direction_t[: q_corr.shape[0]], tracker.config.damping)
        tcp_z = tcp_rot[:, :, 2]
        cos_theta = torch.sum(tcp_z * normal_t[: q_corr.shape[0]], dim=-1)
        valid = (
            joints_in_range_mask(tracker.robot, q_corr)
            & (coll_cost <= 0.0)
            & (mu > tracker.config.mu_threshold)
            & (cos_theta > cos_theta_max)
            & (pos_err <= pos_tol_m)
        )
        if bool(valid.any()):
            q_keep = q_corr_np[valid.detach().cpu().numpy()]
            err_keep = raw_err_np[valid.detach().cpu().numpy()]
            q_list.append(q_keep)
            err_list.append(err_keep)
        collected = sum(arr.shape[0] for arr in q_list)
        print(f'[collect] {collected}/{num_candidates} candidate qs')

    q_all = np.concatenate(q_list, axis=0)[:num_candidates].astype(np.float32)
    err_all = np.concatenate(err_list, axis=0)[:num_candidates].astype(np.float32)
    return q_all, err_all


def rollout_same_task(tracker: GPUNullspaceStraightTracker, tracker_device: torch.device, q_batch_np: np.ndarray, direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    q_batch = torch.from_numpy(q_batch_np.astype(np.float32)).to(tracker_device)
    direction_batch = torch.from_numpy(np.repeat(direction[None, :].astype(np.float32), q_batch_np.shape[0], axis=0)).to(tracker_device)
    normal_batch = torch.from_numpy(np.repeat(normal[None, :].astype(np.float32), q_batch_np.shape[0], axis=0)).to(tracker_device)
    result = tracker.run_batch(q0_batch=q_batch, direction_batch=direction_batch, target_normal_batch=normal_batch)
    return result.projected_length.detach().cpu().numpy().astype(np.float32)


def render_best_q_comparison(anchor: dict, q_batch_np: np.ndarray, score_np: np.ndarray, real_len_np: np.ndarray, direction: np.ndarray, normal: np.ndarray) -> None:
    best_real_idx = int(np.argmax(real_len_np))
    best_score_idx = int(np.argmax(score_np))
    start = anchor['pos']
    world = wd.World(cam_pos=[1.8, -1.6, 1.1], lookat_pos=[0.2, 0.0, 0.35])
    robot = FrankaResearch3(enable_cc=True)
    mgm.gen_sphere(start, radius=0.01, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)
    plane_rotmat = rotation_matrix_from_normal(normal)
    plane_center = start + 0.25 * direction
    mcm.gen_box(xyz_lengths=[0.6, 0.6, 0.001], pos=plane_center, rotmat=plane_rotmat, rgb=[0.8, 0.85, 0.9], alpha=0.3).attach_to(world)
    mgm.gen_frame(pos=start, rotmat=plane_rotmat, ax_length=0.09).attach_to(world)

    cases = [
        ('real_best', best_real_idx, np.array([0.10, 0.85, 0.20], dtype=np.float32)),
        ('score_best', best_score_idx, np.array([0.95, 0.20, 0.20], dtype=np.float32)),
    ]
    for label, idx, color in cases:
        robot.goto_given_conf(q_batch_np[idx].astype(np.float32))
        robot.gen_meshmodel(rgb=color, alpha=0.35, toggle_tcp_frame=True).attach_to(world)
        end = start + direction * float(real_len_np[idx])
        mgm.gen_stick(spos=start, epos=end, radius=0.005, rgb=color, alpha=0.95).attach_to(world)
        mgm.gen_sphere(end, radius=0.01, rgb=color, alpha=0.95).attach_to(world)
        print(f'[vis] {label}: idx={idx} score={float(score_np[idx]):.4f} real_len={float(real_len_np[idx]):.4f}')
    world.run()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed if args.seed is not None else int(np.random.SeedSequence().entropy))
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    device = torch.device(args.device)

    t0 = time.perf_counter()
    model = load_model(args.ckpt, device)
    tracker, tracker_device = build_tracker(device)
    task_anchor = sample_task_anchor(tracker, tracker_device)
    t1 = time.perf_counter()

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

    t2 = time.perf_counter()
    q_batch = torch.from_numpy(q_batch_np).to(device)
    cond_batch = torch.from_numpy(np.concatenate([
        pos_batch_np,
        np.repeat(direction[None, :], q_batch_np.shape[0], axis=0).astype(np.float32),
        np.repeat(normal[None, :], q_batch_np.shape[0], axis=0).astype(np.float32),
    ], axis=1)).to(device)
    with torch.no_grad():
        score_batch, length_batch = model(q_batch, cond_batch)
    score_np = score_batch.detach().cpu().numpy().astype(np.float32).reshape(-1)
    pred_len_np = length_batch.detach().cpu().numpy().astype(np.float32).reshape(-1)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    real_len_np = rollout_same_task(tracker, tracker_device, q_batch_np, direction, normal)
    t5 = time.perf_counter()

    score_rank = descending_rank(score_np)
    pred_len_rank = descending_rank(pred_len_np)
    real_rank = descending_rank(real_len_np)
    topk = min(int(args.top_k), int(q_batch_np.shape[0]))

    summary = {
        'num_candidates': int(q_batch_np.shape[0]),
        'task': {
            'pos': np.round(task_anchor['pos'], 4).tolist(),
            'direction': np.round(direction, 4).tolist(),
            'normal': np.round(normal, 4).tolist(),
        },
        'timing_s': {
            'setup_and_task_sampling': float(t1 - t0),
            'candidate_collection': float(t2 - t1),
            'score_batch': float(t3 - t2),
            'rollout_batch': float(t5 - t4),
            'total': float(t5 - t0),
        },
        'ranking': {
            'spearman_score_vs_real': spearman_corr(score_rank, real_rank),
            'spearman_pred_len_vs_real': spearman_corr(pred_len_rank, real_rank),
            'topk_overlap_score_vs_real': topk_overlap(score_rank, real_rank, topk),
            'topk_overlap_pred_len_vs_real': topk_overlap(pred_len_rank, real_rank, topk),
            'topk': topk,
            'best_real_idx': int(np.argmax(real_len_np)),
            'best_score_idx': int(np.argmax(score_np)),
            'best_pred_len_idx': int(np.argmax(pred_len_np)),
        },
    }
    print(json.dumps(summary, indent=2))
    print('')
    cols = [
        ('idx', 5),
        ('score', 10),
        ('score_rank', 12),
        ('pred_len', 10),
        ('pred_rank', 11),
        ('real_len', 10),
        ('real_rank', 11),
    ]
    header = ' '.join(f'{name:<{width}}' for name, width in cols)
    print(header)
    print('-' * len(header))
    order = np.argsort(real_rank)
    for idx in order:
        row = [
            f'{int(idx):<{cols[0][1]}d}',
            f'{float(score_np[idx]):<{cols[1][1]}.4f}',
            f'{int(score_rank[idx]):<{cols[2][1]}d}',
            f'{float(pred_len_np[idx]):<{cols[3][1]}.4f}',
            f'{int(pred_len_rank[idx]):<{cols[4][1]}d}',
            f'{float(real_len_np[idx]):<{cols[5][1]}.4f}',
            f'{int(real_rank[idx]):<{cols[6][1]}d}',
        ]
        print(' '.join(row))

    if args.vis:
        render_best_q_comparison(task_anchor, q_batch_np, score_np, real_len_np, direction, normal)

if __name__ == '__main__':
    main()
