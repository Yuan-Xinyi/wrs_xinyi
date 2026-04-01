from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd

from lnet_contrastive import LNetContrastive
from paths import DEFAULT_H5, LNET_CONTRASTIVE_RUNS_DIR
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = LNET_CONTRASTIVE_RUNS_DIR / 'lnet_contrastive_q_cond_to_length_sub10' / 'lnet_contrastive_best.pt'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a trained contrastive LNet on one dataset sample.')
    parser.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--traj-id', type=str, default=None)
    parser.add_argument('--point-idx', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-vis', action='store_true')
    return parser.parse_args()


def load_model(ckpt_path: Path, device: torch.device) -> LNetContrastive:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get('args', {})
    model = LNetContrastive(
        q_min=ckpt['q_min'],
        q_max=ckpt['q_max'],
        in_min=ckpt['in_min'],
        in_max=ckpt['in_max'],
        pair_threshold=float(args.get('pair_threshold', 0.05)),
        pair_margin=float(args.get('pair_margin', 0.05)),
        mse_weight=float(args.get('mse_weight', 0.2)),
        rank_weight=float(args.get('rank_weight', 1.0)),
        max_pairs=int(args.get('max_pairs', 4096)),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model




def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normal / max(np.linalg.norm(normal), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-12)
    return np.column_stack((x_axis, y_axis, z_axis))

def load_sample(h5_path: Path, traj_id: str | None, point_idx: int | None, seed: int | None) -> dict:
    rng = np.random.default_rng(seed if seed is not None else int(np.random.SeedSequence().entropy))
    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f['trajectories'].keys())
        key = traj_id if traj_id is not None else keys[int(rng.integers(len(keys)))]
        grp = f['trajectories'][key]
        n = int(grp.attrs['num_points'])
        idx = int(point_idx) if point_idx is not None else int(rng.integers(n))
        return {
            'traj_key': key,
            'point_idx': idx,
            'q': np.asarray(grp['q'][idx], dtype=np.float32),
            'pos': np.asarray(grp['tcp_pos'][idx], dtype=np.float32),
            'direction': np.asarray(grp.attrs['direction'], dtype=np.float32),
            'normal': np.asarray(grp.attrs['target_normal'], dtype=np.float32),
            'target_length': float(np.asarray(grp['remaining_length'][idx], dtype=np.float32)),
        }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.ckpt, device)
    sample = load_sample(args.h5_path, args.traj_id, args.point_idx, args.seed)

    q = torch.from_numpy(sample['q']).float().unsqueeze(0).to(device)
    cond = torch.from_numpy(np.concatenate([sample['pos'], sample['direction'], sample['normal']], axis=0)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        score, pred_length = model(q, cond)
        score = float(score.item())
        pred_length = float(pred_length.item())
    grad = model.get_guidance_gradient(q, cond).detach().cpu().numpy()[0]

    abs_error = abs(pred_length - sample['target_length'])
    print(f"pred_length={pred_length:.2f} target_length={sample['target_length']:.2f} abs_error={abs_error:.2f}")
    print("dq_grad=" + np.array2string(grad, precision=2, suppress_small=False))

    if args.no_vis:
        return

    pos = sample['pos']
    direction = sample['direction']
    normal = sample['normal']
    robot = XArmLite6Miller(enable_cc=True)
    robot.goto_given_conf(sample['q'])

    world = wd.World(cam_pos=[1.7, -1.5, 1.05], lookat_pos=[0.25, 0.0, 0.25])
    mgm.gen_frame().attach_to(world)
    _, rotmat = robot.fk(sample['q'], update=False)
    mgm.gen_frame(pos=pos, rotmat=np.asarray(rotmat), ax_length=0.12).attach_to(world)

    plane_size = 1.2
    plane_rotmat = rotation_matrix_from_normal(normal)
    plane_center = pos + 0.5 * plane_size * direction
    mcm.gen_box(
        xyz_lengths=[plane_size, plane_size, 0.001],
        pos=plane_center,
        rotmat=plane_rotmat,
        rgb=[180 / 255, 211 / 255, 217 / 255],
        alpha=0.5,
    ).attach_to(world)

    mgm.gen_sphere(pos, radius=0.010, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)
    mgm.gen_arrow(spos=pos, epos=pos + direction * 0.18, rgb=np.array([1.0, 0.1, 0.1]), alpha=0.9).attach_to(world)
    mgm.gen_arrow(spos=pos, epos=pos + normal * 0.18, rgb=np.array([0.1, 0.5, 1.0]), alpha=0.9).attach_to(world)

    gt_color = np.array([0.1, 0.75, 0.2], dtype=np.float32)
    pred_color = np.array([0.15, 0.45, 0.95], dtype=np.float32)
    gt_end = pos + direction * float(sample['target_length'])
    pred_end = pos + direction * float(pred_length)

    robot.gen_meshmodel(rgb=gt_color, alpha=0.55, toggle_tcp_frame=True).attach_to(world)
    mgm.gen_stick(spos=pos, epos=gt_end, radius=0.0045, rgb=gt_color, alpha=0.90).attach_to(world)
    mgm.gen_sphere(gt_end, radius=0.009, rgb=gt_color, alpha=0.95).attach_to(world)

    mgm.gen_stick(spos=pos, epos=pred_end, radius=0.0045, rgb=pred_color, alpha=0.95).attach_to(world)
    mgm.gen_sphere(pred_end, radius=0.009, rgb=pred_color, alpha=0.95).attach_to(world)

    world.run()


if __name__ == '__main__':
    main()
