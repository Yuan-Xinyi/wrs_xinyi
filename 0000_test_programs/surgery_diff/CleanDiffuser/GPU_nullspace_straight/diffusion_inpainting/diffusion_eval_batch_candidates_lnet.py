import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import numpy as np
import torch

import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd

from diffusion import DEFAULT_RUN_NAME, DEFAULT_WORKDIR, sample_q_length_from_condition
from diffusion_sample import load_model, normalize_direction
import jax2torch
from wrs.robot_sim.robots.xarmlite6_wg.sphere_collision_checker import SphereCollisionChecker
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
from trajectory_generation.xarm_nullspave_straight_gpu import GPUNullspaceStraightTracker, TrackerConfig, position_jacobian_batch, directional_manipulability_batch, damped_pseudoinverse_batch
from length_prediction.lnet import LNet
from length_prediction.lnet_contrastive import LNetContrastive
from length_prediction.paths import LNET_RUNS_DIR, LNET_CONTRASTIVE_RUNS_DIR
import jax
import time


DIRECTION_AXIS = 0
NORMAL_AXIS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize diffusion candidates for a random GT joint configuration.')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--lnet-ckpt', type=Path, default=LNET_RUNS_DIR / 'lnet_q_cond_to_length_sub10' / 'lnet_best.pt')
    parser.add_argument('--lnet-contrastive-ckpt', type=Path, default=LNET_CONTRASTIVE_RUNS_DIR / 'lnet_contrastive_q_cond_to_length_sub10' / 'lnet_contrastive_best.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n-samples', type=int, default=16)
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--no-vis', action='store_true')
    return parser.parse_args()


def sample_anchor(robot, collision_fn, device: torch.device, rng: np.random.Generator, batch_size: int = 512, max_tries: int = 4096) -> dict:
    tried = 0
    lower = robot.jnt_ranges[:, 0].detach().cpu().numpy()
    upper = robot.jnt_ranges[:, 1].detach().cpu().numpy()
    while tried < max_tries:
        cur_batch = min(batch_size, max_tries - tried)
        q_batch_np = rng.uniform(lower, upper, size=(cur_batch, lower.shape[0])).astype(np.float32)
        q_batch = torch.from_numpy(q_batch_np).to(device)
        coll_cost = collision_fn(q_batch)
        valid_mask = coll_cost <= 0.0
        if bool(valid_mask.any()):
            q_valid = q_batch[valid_mask]
            pos_valid, rot_valid = robot.fk_batch(q_valid)
            q = q_valid[0].detach().cpu().numpy().astype(np.float32)
            pos = pos_valid[0].detach().cpu().numpy().astype(np.float32)
            rotmat = rot_valid[0].detach().cpu().numpy().astype(np.float32)
            direction = normalize_direction(rotmat[:, DIRECTION_AXIS])
            target_normal = normalize_direction(rotmat[:, NORMAL_AXIS])
            return {
                'q': q,
                'pos': pos,
                'rotmat': rotmat,
                'direction': direction,
                'target_normal': target_normal,
            }
        tried += cur_batch
    raise RuntimeError('Failed to sample a valid random GT configuration.')


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normal / max(np.linalg.norm(normal), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-12)
    return np.column_stack((x_axis, y_axis, z_axis))


def build_tracker(device: torch.device) -> tuple[GPUNullspaceStraightTracker, torch.device]:
    xarm = xarm6_gpu.XArmLite6GPU(device=device)
    cc_model = SphereCollisionChecker('wrs/robot_sim/robots/xarmlite6_wg/xarm6_sphere_visuals.urdf')
    vmap_jax_cost = jax.jit(jax.vmap(cc_model.self_collision_cost, in_axes=(0, None, None)))
    collision_fn = jax2torch.jax2torch(lambda q_batch: vmap_jax_cost(q_batch, 1.0, -0.005))
    tracker = GPUNullspaceStraightTracker(
        robot=xarm.robot,
        collision_fn=collision_fn,
        config=TrackerConfig(),
        print_every=0,
    )
    return tracker, device


def batch_position_error_and_correction(
    robot,
    q_batch_np: np.ndarray,
    target_pos_np: np.ndarray,
    damping: float,
    max_iters: int,
    tol: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
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
        dq = (damped_pseudoinverse_batch(j_pos, damping) @ err.unsqueeze(-1)).squeeze(-1)
        active = (err_norm >= tol).float().unsqueeze(1)
        q_corr = q_corr + active * dq
        q_corr = torch.max(torch.min(q_corr, upper), lower)

    return q_corr.detach().cpu().numpy().astype(np.float32), raw_pos_err.detach().cpu().numpy().astype(np.float32)



def load_lnet_model(ckpt_path: Path, device: torch.device) -> LNet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = LNet(
        q_min=ckpt['q_min'],
        q_max=ckpt['q_max'],
        in_min=ckpt['in_min'],
        in_max=ckpt['in_max'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def load_lnet_contrastive_model(ckpt_path: Path, device: torch.device) -> LNetContrastive:
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


def predict_length_models(
    lnet: LNet,
    lnet_contrastive: LNetContrastive,
    device: torch.device,
    q_batch_np: np.ndarray,
    pos_np: np.ndarray,
    direction_np: np.ndarray,
    normal_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cond_np = np.concatenate([pos_np, direction_np, normal_np], axis=0).astype(np.float32)
    cond_batch_np = np.repeat(cond_np[None, :], q_batch_np.shape[0], axis=0)
    q_batch = torch.from_numpy(q_batch_np.astype(np.float32)).to(device)
    cond_batch = torch.from_numpy(cond_batch_np).to(device)
    with torch.no_grad():
        lnet_pred = lnet(q_batch, cond_batch).detach().cpu().numpy()
        contrastive_score, _ = lnet_contrastive(q_batch, cond_batch)
        contrastive_pred = contrastive_score.detach().cpu().numpy()
    return lnet_pred.astype(np.float32), contrastive_pred.astype(np.float32)

def directional_mu_batch(
    tracker: GPUNullspaceStraightTracker,
    tracker_device: torch.device,
    q_batch_np: np.ndarray,
    direction_np: np.ndarray,
) -> np.ndarray:
    q_batch = torch.from_numpy(q_batch_np.astype(np.float32)).to(tracker_device)
    direction_batch = torch.from_numpy(np.repeat(direction_np[None, :].astype(np.float32), q_batch_np.shape[0], axis=0)).to(tracker_device)
    j_pos, _ = position_jacobian_batch(tracker.robot, q_batch, create_graph=False)
    mu = directional_manipulability_batch(j_pos, direction_batch, tracker.config.damping)
    return mu.detach().cpu().numpy()


def rollout_lengths_batch(
    tracker: GPUNullspaceStraightTracker,
    tracker_device: torch.device,
    q_batch_np: np.ndarray,
    direction_np: np.ndarray,
    target_normal_np: np.ndarray,
) -> np.ndarray:
    q_batch = torch.from_numpy(q_batch_np.astype(np.float32)).to(tracker_device)
    direction_batch = torch.from_numpy(np.repeat(direction_np[None, :].astype(np.float32), q_batch_np.shape[0], axis=0)).to(tracker_device)
    target_normal_batch = torch.from_numpy(np.repeat(target_normal_np[None, :].astype(np.float32), q_batch_np.shape[0], axis=0)).to(tracker_device)
    result = tracker.run_batch(q0_batch=q_batch, direction_batch=direction_batch, target_normal_batch=target_normal_batch)
    return result.projected_length.detach().cpu().numpy()



def descending_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind='stable')
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks

def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed if args.seed is not None else int(np.random.SeedSequence().entropy))
    device = torch.device(args.device)
    _, stats, model, q_dim, diffusion_steps = load_model(args.bundle, device)
    tracker, tracker_device = build_tracker(device)

    t_anchor0 = time.perf_counter()
    anchor = sample_anchor(tracker.robot, tracker.collision_fn, tracker_device, rng)
    t_anchor1 = time.perf_counter()
    print(f'[time] anchor_sampling={t_anchor1 - t_anchor0:.3f}s')

    condition = np.concatenate([anchor['pos'], anchor['direction'], anchor['target_normal']], axis=0).astype(np.float32)
    t_sample0 = time.perf_counter()
    q_preds, pred_lengths, _ = sample_q_length_from_condition(
        model=model,
        stats=stats,
        condition=condition,
        device=device,
        q_dim=q_dim,
        n_samples=int(args.n_samples),
        sample_steps=int(args.sample_steps) if args.sample_steps is not None else int(diffusion_steps),
        temperature=float(args.temperature),
    )
    t_sample1 = time.perf_counter()
    print(f'[time] diffusion_sampling={t_sample1 - t_sample0:.3f}s for {len(q_preds)} candidates')

    lnet = load_lnet_model(args.lnet_ckpt, device)
    lnet_contrastive = load_lnet_contrastive_model(args.lnet_contrastive_ckpt, device)
    t_corr0 = time.perf_counter()
    q_corrs, pos_errs = batch_position_error_and_correction(
        tracker.robot,
        q_preds.astype(np.float32),
        anchor['pos'],
        args.correction_damping,
        args.correction_iters,
        args.correction_tol,
        tracker_device,
    )
    t_corr1 = time.perf_counter()
    print(f'[time] jacobian_correction={t_corr1 - t_corr0:.3f}s')

    rollout_q = np.concatenate([anchor['q'][None, :], q_corrs], axis=0)
    score_q = np.concatenate([anchor['q'][None, :], q_preds.astype(np.float32)], axis=0)
    t_roll0 = time.perf_counter()
    all_real_lengths = rollout_lengths_batch(tracker, tracker_device, rollout_q, anchor['direction'], anchor['target_normal'])
    all_mu = directional_mu_batch(tracker, tracker_device, rollout_q, anchor['direction'])
    t_roll1 = time.perf_counter()
    print(f'[time] rollout_plus_mu={t_roll1 - t_roll0:.3f}s')

    t_score0 = time.perf_counter()
    all_lnet_pred, all_lnet_contrastive_pred = predict_length_models(
        lnet, lnet_contrastive, device, score_q, anchor['pos'], anchor['direction'], anchor['target_normal']
    )
    t_score1 = time.perf_counter()
    print(f'[time] model_scoring={t_score1 - t_score0:.3f}s')

    gt_real_length = float(all_real_lengths[0])
    gt_mu = float(all_mu[0])
    gt_lnet_pred = float(all_lnet_pred[0])
    gt_lnet_contrastive_score = float(all_lnet_contrastive_pred[0])
    candidate_real_lengths = all_real_lengths[1:]
    candidate_mu = all_mu[1:]
    candidate_lnet_pred = all_lnet_pred[1:]
    candidate_lnet_contrastive_score = all_lnet_contrastive_pred[1:]
    candidate_real_rank = descending_rank(candidate_real_lengths) if len(candidate_real_lengths) > 0 else np.asarray([], dtype=int)
    candidate_lnet_rank = descending_rank(candidate_lnet_pred) if len(candidate_lnet_pred) > 0 else np.asarray([], dtype=int)
    candidate_lcts_rank = descending_rank(candidate_lnet_contrastive_score) if len(candidate_lnet_contrastive_score) > 0 else np.asarray([], dtype=int)
    min_pos_idx = int(np.argmin(pos_errs)) if len(pos_errs) > 0 else -1
    max_lnet_idx = int(np.argmax(candidate_lnet_pred)) if len(candidate_lnet_pred) > 0 else -1
    max_lcts_idx = int(np.argmax(candidate_lnet_contrastive_score)) if len(candidate_lnet_contrastive_score) > 0 else -1

    row_fmt = '{label:<8} {idx:>4} {diff:>10.6f} {lnet:>10.6f} {clen:>10.6f} {real:>10.6f} {rrank:>6} {lrank:>6} {crank:>6} {mu:>10.6f} {pos:>12}'
    print('---------------------------------------------------------------------------------------------------------------------')
    print(f"{'label':<8} {'idx':>4} {'diff_len':>10} {'lnet_len':>10} {'lcts_score':>10} {'real_len':>10} {'r_real':>6} {'r_lnet':>6} {'r_lcts':>6} {'mu':>10} {'pos_err_mm':>12}")
    print('---------------------------------------------------------------------------------------------------------------------')
    print(row_fmt.format(label='GT', idx='--', diff=float('nan'), lnet=gt_lnet_pred, clen=gt_lnet_contrastive_score, real=gt_real_length, rrank='--', lrank='--', crank='--', mu=gt_mu, pos='--'))
    if min_pos_idx >= 0:
        print(row_fmt.format(
            label='MINPOS',
            idx=f'{min_pos_idx:02d}',
            diff=float(pred_lengths[min_pos_idx]),
            lnet=float(candidate_lnet_pred[min_pos_idx]),
            clen=float(candidate_lnet_contrastive_score[min_pos_idx]),
            real=float(candidate_real_lengths[min_pos_idx]),
            rrank=int(candidate_real_rank[min_pos_idx]),
            lrank=int(candidate_lnet_rank[min_pos_idx]),
            crank=int(candidate_lcts_rank[min_pos_idx]),
            mu=float(candidate_mu[min_pos_idx]),
            pos=f'{float(pos_errs[min_pos_idx]) * 1e3:.3f}',
        ))
    if max_lnet_idx >= 0:
        print(row_fmt.format(
            label='LNETTOP',
            idx=f'{max_lnet_idx:02d}',
            diff=float(pred_lengths[max_lnet_idx]),
            lnet=float(candidate_lnet_pred[max_lnet_idx]),
            clen=float(candidate_lnet_contrastive_score[max_lnet_idx]),
            real=float(candidate_real_lengths[max_lnet_idx]),
            rrank=int(candidate_real_rank[max_lnet_idx]),
            lrank=int(candidate_lnet_rank[max_lnet_idx]),
            crank=int(candidate_lcts_rank[max_lnet_idx]),
            mu=float(candidate_mu[max_lnet_idx]),
            pos=f'{float(pos_errs[max_lnet_idx]) * 1e3:.3f}',
        ))
    if max_lcts_idx >= 0:
        print(row_fmt.format(
            label='LCTSTOP',
            idx=f'{max_lcts_idx:02d}',
            diff=float(pred_lengths[max_lcts_idx]),
            lnet=float(candidate_lnet_pred[max_lcts_idx]),
            clen=float(candidate_lnet_contrastive_score[max_lcts_idx]),
            real=float(candidate_real_lengths[max_lcts_idx]),
            rrank=int(candidate_real_rank[max_lcts_idx]),
            lrank=int(candidate_lnet_rank[max_lcts_idx]),
            crank=int(candidate_lcts_rank[max_lcts_idx]),
            mu=float(candidate_mu[max_lcts_idx]),
            pos=f'{float(pos_errs[max_lcts_idx]) * 1e3:.3f}',
        ))
    print('---------------------------------------------------------------------------------------------------------------------')
    sorted_idx = np.argsort(-candidate_real_lengths, kind='stable')
    for idx in sorted_idx:
        print(row_fmt.format(
            label='CAND',
            idx=f'{int(idx):02d}',
            diff=float(pred_lengths[idx]),
            lnet=float(candidate_lnet_pred[idx]),
            clen=float(candidate_lnet_contrastive_score[idx]),
            real=float(candidate_real_lengths[idx]),
            rrank=int(candidate_real_rank[idx]),
            lrank=int(candidate_lnet_rank[idx]),
            crank=int(candidate_lcts_rank[idx]),
            mu=float(candidate_mu[idx]),
            pos=f'{float(pos_errs[idx]) * 1e3:.3f}',
        ))
    print('---------------------------------------------------------------------------------------------------------------------')
    t_print1 = time.perf_counter()
    print(f'[time] table_print={(t_print1 - t_score1):.3f}s')
    if args.no_vis:
        return

    world = wd.World(cam_pos=[1.7, -1.5, 1.05], lookat_pos=[0.25, 0.0, 0.25])
    mgm.gen_frame().attach_to(world)
    mgm.gen_frame(pos=anchor['pos'], rotmat=anchor['rotmat'], ax_length=0.12).attach_to(world)
    start = anchor['pos']
    direction = anchor['direction']
    plane_size = 1.2
    plane_rotmat = rotation_matrix_from_normal(anchor['target_normal'])
    plane_center = start + 0.5 * plane_size * direction
    mcm.gen_box(
        xyz_lengths=[plane_size, plane_size, 0.001],
        pos=plane_center,
        rotmat=plane_rotmat,
        rgb=[180/255, 211/255, 217/255],
        alpha=0.5,
    ).attach_to(world)
    mgm.gen_sphere(start, radius=0.010, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)
    gt_end = start + direction * gt_real_length
    gt_color = np.array([0.1, 0.75, 0.2], dtype=np.float32)
    mgm.gen_stick(spos=start, epos=gt_end, radius=0.0045, rgb=gt_color, alpha=0.90).attach_to(world)
    mgm.gen_sphere(gt_end, radius=0.009, rgb=gt_color, alpha=0.95).attach_to(world)
    
    import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
    robot = xarm6_sim.XArmLite6Miller(enable_cc=True)
    robot.goto_given_conf(anchor['q'])
    robot.gen_meshmodel(rgb=gt_color, alpha=0.55, toggle_tcp_frame=True).attach_to(world)

    if max_lnet_idx >= 0:
        raw_color = np.array([0.15, 0.45, 0.95], dtype=np.float32)
        corr_color = np.array([0.15, 0.45, 0.95], dtype=np.float32)
        robot.goto_given_conf(q_preds[max_lnet_idx].astype(np.float32))
        robot.gen_meshmodel(rgb=raw_color, alpha=0.18, toggle_tcp_frame=False).attach_to(world)
        robot.goto_given_conf(q_corrs[max_lnet_idx].astype(np.float32))
        robot.gen_meshmodel(rgb=corr_color, alpha=0.75, toggle_tcp_frame=False).attach_to(world)
        pred_end = start + direction * float(candidate_lnet_pred[max_lnet_idx])
        mgm.gen_stick(spos=start, epos=pred_end, radius=0.004, rgb=corr_color, alpha=0.95).attach_to(world)
        mgm.gen_sphere(pred_end, radius=0.009, rgb=corr_color, alpha=0.95).attach_to(world)

    t_vis0 = time.perf_counter()
    world.run()
    t_vis1 = time.perf_counter()
    print(f'[time] visualization={(t_vis1 - t_vis0):.3f}s')


if __name__ == '__main__':
    main()
