import argparse
from pathlib import Path

import numpy as np

from wrs import wd, mgm, mcm
from wrs.robot_sim.robots.franka_research_3.franka_research_3 import FrankaResearch3
from wrs.robot_sim.robots.franka_research_3.sphere_collision_checker import SphereCollisionChecker

DEFAULT_URDF = Path('wrs/robot_sim/robots/franka_research_3/franka_research_3_ccsphere.urdf')
LINK_COLORS = [
    np.array([0.95, 0.25, 0.25]),
    np.array([0.95, 0.55, 0.20]),
    np.array([0.95, 0.80, 0.20]),
    np.array([0.45, 0.80, 0.25]),
    np.array([0.20, 0.75, 0.75]),
    np.array([0.20, 0.45, 0.95]),
    np.array([0.55, 0.30, 0.95]),
    np.array([0.95, 0.20, 0.75]),
    np.array([0.60, 0.60, 0.60]),
    np.array([0.50, 0.80, 0.50]),
    np.array([0.80, 0.50, 0.80]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare URDF sphere local frame against WRS link local frame for FrankaResearch3.')
    parser.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
    parser.add_argument('--link', type=int, default=1, help='0..10, where 8 is hand, 9 is left_finger, 10 is right_finger')
    parser.add_argument('--q', type=float, nargs=7, default=None, help='Joint configuration; default is robot.rand_conf().')
    parser.add_argument('--show-world', action='store_true', help='Also show the full robot and world-space spheres for reference.')
    return parser.parse_args()


def get_link_obj(robot: FrankaResearch3, link_idx: int):
    if link_idx == 0:
        return robot.manipulator.jlc.anchor.lnk_list[0]
    if 1 <= link_idx <= 7:
        return robot.manipulator.jlc.jnts[link_idx - 1].lnk
    if link_idx == 8:
        return robot.end_effector.jlc.anchor.lnk_list[0]
    if link_idx == 9:
        return robot.end_effector.jlc.jnts[0].lnk
    if link_idx == 10:
        return robot.end_effector.jlc.jnts[1].lnk
    raise ValueError(f'Unsupported link index: {link_idx}')


def get_mesh_path(link_obj) -> str:
    cmodel = getattr(link_obj, 'cmodel', None)
    if cmodel is None:
        raise RuntimeError('Selected link has no collision model / mesh to render.')
    for attr in ['_file_path', 'file_path', '_initor', 'initor']:
        if hasattr(cmodel, attr):
            value = getattr(cmodel, attr)
            if isinstance(value, str) and value:
                return value
    raise RuntimeError('Could not determine mesh path from selected link collision model.')


def to_local_points(world_points: np.ndarray, pos: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
    return (rotmat.T @ (world_points - pos).T).T


def main() -> None:
    args = parse_args()

    robot = FrankaResearch3(enable_cc=False)
    q = robot.rand_conf() if args.q is None else np.asarray(args.q, dtype=np.float64)
    robot.goto_given_conf(jnt_values=q)

    checker = SphereCollisionChecker(str(args.urdf))
    sphere_positions = np.asarray(checker.update(q))
    sphere_radii = np.asarray(checker.sphere_radii)
    sphere_link_indices = np.asarray(checker.sphere_link_indices)

    mask = sphere_link_indices == args.link
    if not np.any(mask):
        raise RuntimeError(f'Link {args.link} has no spheres in {args.urdf}.')

    link_obj = get_link_obj(robot, args.link)
    link_pos = np.asarray(link_obj.gl_pos, dtype=np.float64)
    link_rotmat = np.asarray(link_obj.gl_rotmat, dtype=np.float64)
    local_points = to_local_points(sphere_positions[mask], link_pos, link_rotmat)
    local_radii = sphere_radii[mask]
    mesh_path = get_mesh_path(link_obj)

    print(f'[info] link={args.link} q={np.array2string(q, precision=5, separator=", ")}')
    print(f'[info] mesh={mesh_path}')
    print(f'[info] wrs_link_pos={np.array2string(link_pos, precision=5, separator=", ")}')
    print(f'[info] local_center={np.array2string(local_points.mean(axis=0), precision=5, separator=", ")}')
    print(f'[info] local_aabb_min={np.array2string(local_points.min(axis=0), precision=5, separator=", ")}')
    print(f'[info] local_aabb_max={np.array2string(local_points.max(axis=0), precision=5, separator=", ")}')

    world = wd.World(cam_pos=[1.6, 1.4, 0.9], lookat_pos=[0.0, 0.0, 0.0])
    mgm.gen_frame(ax_length=0.08).attach_to(world)
    mgm.gen_frame(pos=np.zeros(3), rotmat=np.eye(3), ax_length=0.12).attach_to(world)

    mesh_model = mcm.CollisionModel(initor=mesh_path)
    mesh_model.rgba = np.array([0.7, 0.7, 0.7, 0.7])
    mesh_model.attach_to(world)

    color = LINK_COLORS[args.link % len(LINK_COLORS)]
    for pos, radius in zip(local_points, local_radii):
        mcm.gen_sphere(radius=float(radius), pos=np.asarray(pos), rgb=color, alpha=0.25).attach_to(world)

    if args.show_world:
        robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(world)
        robot.gen_meshmodel(alpha=0.2, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(world)
        for pos, radius in zip(sphere_positions[mask], local_radii):
            mcm.gen_sphere(radius=float(radius), pos=np.asarray(pos), rgb=color, alpha=0.1).attach_to(world)

    print('[legend] Grey mesh is the WRS link mesh in its own local frame; colored spheres are URDF spheres transformed back into that same local frame using the WRS link pose.')
    world.run()


if __name__ == '__main__':
    main()
