import argparse
import random
import sys
from pathlib import Path

import h5py
import numpy as np

from wrs import wd
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as fr3_sim
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim


PARENT_DIR = Path(__file__).resolve().parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
from utils import helper_functions as helpers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize one trajectory from an HDF5 trajectory dataset.')
    parser.add_argument('h5_path', type=Path)
    parser.add_argument('--traj-key', type=str, default=None, help='Trajectory group key like traj_000123. Default: random.')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--vis-mode', choices=['anime', 'static'], default='anime')
    return parser.parse_args()


def make_robot(robot_name: str):
    if robot_name == 'franka_research_3':
        return fr3_sim.FrankaResearch3(enable_cc=True)
    if robot_name in {'xarm_lite6', 'xarmlite6', 'xarmlite6_miller'}:
        return xarm6_sim.XArmLite6Miller(enable_cc=True)
    raise ValueError(f'Unsupported robot type in HDF5: {robot_name}')


def pick_trajectory_key(root: h5py.Group, traj_key: str | None, seed: int | None) -> str:
    keys = sorted(root.keys())
    if not keys:
        raise RuntimeError('No trajectories found in HDF5.')
    if traj_key is not None:
        if traj_key not in root:
            raise KeyError(f'Trajectory key not found: {traj_key}')
        return traj_key
    rng = random.Random(seed)
    return rng.choice(keys)


def main() -> None:
    args = parse_args()
    h5_path = args.h5_path.resolve()
    with h5py.File(h5_path, 'r') as h5f:
        robot_name = str(h5f.attrs.get('robot', 'franka_research_3'))
        root = h5f['trajectories']
        traj_key = pick_trajectory_key(root, args.traj_key, args.seed)
        grp = root[traj_key]
        q_path = np.asarray(grp['q'][:], dtype=np.float32)
        tcp_pos = np.asarray(grp['tcp_pos'][:], dtype=np.float32)
        direction = np.asarray(grp.attrs['direction'], dtype=np.float32)
        target_normal = np.asarray(grp.attrs['target_normal'], dtype=np.float32)
        print(f'h5={h5_path}')
        print(f'robot={robot_name}')
        print(f'traj_key={traj_key}')
        print(f'num_points={int(grp.attrs.get("num_points", q_path.shape[0]))}')
        print(f'termination={grp.attrs.get("termination_reason", "unknown")}')
        print(f'total_projected_length={float(grp.attrs.get("total_projected_length", 0.0)):.4f} m')
        print(f'total_euclidean_length={float(grp.attrs.get("total_euclidean_length", 0.0)):.4f} m')
        print('direction=', np.array2string(direction, precision=4, separator=', '))
        print('target_normal=', np.array2string(target_normal, precision=4, separator=', '))

    robot = make_robot(robot_name)
    world = wd.World(cam_pos=[2.6, 2.2, 1.4], lookat_pos=[0.0, 0.0, 0.5])
    mgm.gen_frame().attach_to(world)
    if tcp_pos.shape[0] >= 2:
        line_segs = [[tcp_pos[i], tcp_pos[i + 1]] for i in range(tcp_pos.shape[0] - 1)]
        mgm.gen_linesegs(line_segs, thickness=0.003, rgb=np.array([0.1, 0.8, 0.2]), alpha=1.0).attach_to(world)
    mgm.gen_arrow(
        spos=tcp_pos[0],
        epos=tcp_pos[0] + 0.25 * direction,
        stick_radius=0.005,
        rgb=np.array([1.0, 0.2, 0.2]),
    ).attach_to(world)
    mgm.gen_sphere(tcp_pos[0], radius=0.01, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)
    mgm.gen_sphere(tcp_pos[-1], radius=0.012, rgb=np.array([1.0, 0.5, 0.0]), alpha=1.0).attach_to(world)

    if args.vis_mode == 'anime':
        helpers.visualize_anime_path(world, robot, q_path)
    else:
        helpers.visualize_static_path(world, robot, q_path)


if __name__ == '__main__':
    main()
