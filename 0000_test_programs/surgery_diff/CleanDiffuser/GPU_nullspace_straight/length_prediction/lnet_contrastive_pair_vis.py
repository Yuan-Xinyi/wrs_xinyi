from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd

from paths import DEFAULT_H5_PREF
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize one preference pair with similar cond and different length.')
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5_PREF)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--mode', choices=['best_gap', 'random'], default='random')
    parser.add_argument('--no-vis', action='store_true')
    return parser.parse_args()


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normal / max(np.linalg.norm(normal), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-12)
    return np.column_stack((x_axis, y_axis, z_axis))


def load_pair(h5_path: Path, mode: str, seed: int | None) -> dict:
    rng = np.random.default_rng(seed if seed is not None else int(np.random.SeedSequence().entropy))
    with h5py.File(h5_path, 'r') as f:
        pref = f['contrastive_pref']
        q = np.asarray(pref['q'][:], dtype=np.float32)
        pos = np.asarray(pref['pos'][:], dtype=np.float32)
        direction = np.asarray(pref['direction'][:], dtype=np.float32)
        normal = np.asarray(pref['normal'][:], dtype=np.float32)
        length = np.asarray(pref['length'][:], dtype=np.float32)
        mu = np.asarray(pref['mu'][:], dtype=np.float32)
        traj_idx = np.asarray(pref['traj_idx'][:], dtype=np.int32)
        point_idx = np.asarray(pref['point_idx'][:], dtype=np.int32)
        traj_keys = np.asarray(pref['traj_key'][:]).astype(str)
        offsets = np.asarray(pref['neighbor_offsets'][:], dtype=np.int64)
        neighbors = np.asarray(pref['neighbor_index'][:], dtype=np.int32)
        dists = np.asarray(pref['neighbor_distance'][:], dtype=np.float32)
        valid_anchors = np.asarray(pref['valid_anchor_index'][:], dtype=np.int32)

    if valid_anchors.size == 0:
        raise RuntimeError('No valid preference pairs found in contrastive_pref.')

    if mode == 'random':
        anchor = int(valid_anchors[int(rng.integers(valid_anchors.size))])
        start = int(offsets[anchor])
        end = int(offsets[anchor + 1])
        sel = int(rng.integers(start, end))
        partner = int(neighbors[sel])
        cond_dist = float(dists[sel])
    else:
        best_anchor = -1
        best_partner = -1
        best_dist = 0.0
        best_gap = -np.inf
        for anchor in valid_anchors:
            start = int(offsets[anchor])
            end = int(offsets[anchor + 1])
            if end <= start:
                continue
            local_neighbors = neighbors[start:end]
            local_dists = dists[start:end]
            local_gaps = np.abs(length[anchor] - length[local_neighbors])
            local_idx = int(np.argmax(local_gaps))
            gap = float(local_gaps[local_idx])
            if gap > best_gap:
                best_gap = gap
                best_anchor = int(anchor)
                best_partner = int(local_neighbors[local_idx])
                best_dist = float(local_dists[local_idx])
        anchor = best_anchor
        partner = best_partner
        cond_dist = best_dist

    return {
        'a_idx': anchor,
        'b_idx': partner,
        'cond_dist': cond_dist,
        'a': {
            'q': q[anchor],
            'pos': pos[anchor],
            'direction': direction[anchor],
            'normal': normal[anchor],
            'length': float(length[anchor]),
            'mu': float(mu[anchor]),
            'traj_key': traj_keys[traj_idx[anchor]],
            'point_idx': int(point_idx[anchor]),
        },
        'b': {
            'q': q[partner],
            'pos': pos[partner],
            'direction': direction[partner],
            'normal': normal[partner],
            'length': float(length[partner]),
            'mu': float(mu[partner]),
            'traj_key': traj_keys[traj_idx[partner]],
            'point_idx': int(point_idx[partner]),
        },
    }


def print_pair(pair: dict) -> None:
    a = pair['a']
    b = pair['b']
    print(f"pair=({pair['a_idx']},{pair['b_idx']}) cond_dist={pair['cond_dist']:.4f} length_gap={abs(a['length'] - b['length']):.4f}")
    print(
        f"A traj={a['traj_key']} point={a['point_idx']} len={a['length']:.4f} mu={a['mu']:.4f} pos={np.array2string(a['pos'], precision=4)}"
    )
    print(
        f"B traj={b['traj_key']} point={b['point_idx']} len={b['length']:.4f} mu={b['mu']:.4f} pos={np.array2string(b['pos'], precision=4)}"
    )


def visualize_pair(pair: dict) -> None:
    a = pair['a']
    b = pair['b']
    world = wd.World(cam_pos=[1.7, -1.5, 1.05], lookat_pos=[0.25, 0.0, 0.25])
    mgm.gen_frame().attach_to(world)

    plane_size = 1.0
    plane_rotmat = rotation_matrix_from_normal(a['normal'])
    plane_center = 0.5 * (a['pos'] + b['pos']) + 0.25 * plane_size * a['direction']
    mcm.gen_box(
        xyz_lengths=[plane_size, plane_size, 0.001],
        pos=plane_center,
        rotmat=plane_rotmat,
        rgb=[180 / 255, 211 / 255, 217 / 255],
        alpha=0.35,
    ).attach_to(world)

    robot = XArmLite6Miller(enable_cc=True)
    a_color = np.array([0.10, 0.75, 0.20], dtype=np.float32)
    b_color = np.array([0.95, 0.45, 0.10], dtype=np.float32)

    for sample, color in ((a, a_color), (b, b_color)):
        robot.goto_given_conf(sample['q'])
        _, rotmat = robot.fk(sample['q'], update=False)
        robot.gen_meshmodel(rgb=color, alpha=0.50, toggle_tcp_frame=True).attach_to(world)
        mgm.gen_frame(pos=sample['pos'], rotmat=np.asarray(rotmat), ax_length=0.08).attach_to(world)
        mgm.gen_sphere(sample['pos'], radius=0.008, rgb=color, alpha=0.95).attach_to(world)
        mgm.gen_arrow(spos=sample['pos'], epos=sample['pos'] + sample['direction'] * 0.16, rgb=color, alpha=0.90).attach_to(world)
        end = sample['pos'] + sample['direction'] * sample['length']
        mgm.gen_stick(spos=sample['pos'], epos=end, radius=0.0045, rgb=color, alpha=0.90).attach_to(world)
        mgm.gen_sphere(end, radius=0.008, rgb=color, alpha=0.95).attach_to(world)

    world.run()


def main() -> None:
    args = parse_args()
    pair = load_pair(args.h5_path, args.mode, args.seed)
    print_pair(pair)
    if not args.no_vis:
        visualize_pair(pair)


if __name__ == '__main__':
    main()
