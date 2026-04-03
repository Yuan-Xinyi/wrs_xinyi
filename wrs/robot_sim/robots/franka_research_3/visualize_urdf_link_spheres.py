import argparse
from pathlib import Path

import numpy as np

from wrs import wd, mgm, mcm
from wrs.robot_sim.robots.franka_research_3.franka_research_3_sphere_collcheck import FrankaResearch3SphereCollCheck
from wrs.robot_sim.robots.franka_research_3.sphere_collision_checker import SphereCollisionChecker
from wrs.robot_sim.robots.franka_research_3.franka_research_3 import FrankaResearch3

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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize spheres from a single link in the Franka sphere URDF.')
    parser.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
    parser.add_argument('--link', type=int, default=8, help='0..10, where 8 is hand, 9 is left_finger, 10 is right_finger; or -1 for all links')
    parser.add_argument('--q', type=float, nargs=7, default=None, help='Joint configuration. Default is zeros.')
    parser.add_argument('--no-mesh', action='store_true', help='Hide link mesh.')
    parser.add_argument('--show-full-stick', action='store_true', help='Show full robot stick model for reference.')
    parser.add_argument('--show-frames', action='store_true', help='Show joint/TCP frames on robot.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # robot = FrankaResearch3SphereCollCheck(enable_cc=False)
    robot = FrankaResearch3(enable_cc=True)
    q = np.zeros(7, dtype=np.float64) if args.q is None else np.asarray(args.q, dtype=np.float64)
    # q = robot.rand_conf() if args.q is None else np.asarray(args.q, dtype=np.float64)
    # q[6] = 3.14159265359
    robot.goto_given_conf(jnt_values=q)
    checker = SphereCollisionChecker(str(args.urdf))
    sphere_positions = np.asarray(checker.update(q))
    sphere_radii = np.asarray(checker.sphere_radii)
    sphere_link_indices = np.asarray(checker.sphere_link_indices)
    link_names = checker.link_order

    show_links = set(range(len(link_names))) if args.link < 0 else {int(args.link)}

    print(f'[info] urdf={args.urdf}')
    print(f'[info] q={np.array2string(q, precision=5, separator=", ")}')
    for link_idx in sorted(show_links):
        mask = sphere_link_indices == link_idx
        pts = sphere_positions[mask]
        rs = sphere_radii[mask]
        if len(pts) == 0:
            print(f'[link {link_idx}] {link_names[link_idx]}: no spheres')
            continue
        print(
            f'[link {link_idx}] {link_names[link_idx]} count={len(pts)} '
            f'center={np.array2string(pts.mean(axis=0), precision=5, separator=", ")} '
            f'aabb_min={np.array2string(pts.min(axis=0), precision=5, separator=", ")} '
            f'aabb_max={np.array2string(pts.max(axis=0), precision=5, separator=", ")} '
            f'radius_range=({rs.min():.5f}, {rs.max():.5f})'
        )

    world = wd.World(cam_pos=[2.2, 2.0, 1.2], lookat_pos=[0.0, 0.0, 0.45])
    mgm.gen_frame().attach_to(world)
    if args.show_full_stick:
        robot.gen_stickmodel(toggle_tcp_frame=args.show_frames, toggle_jnt_frames=args.show_frames).attach_to(world)

    if not args.no_mesh:
        for link_idx in sorted(show_links):
            if link_idx == 0:
                robot.manipulator.jlc.anchor.lnk_list[0].gen_meshmodel(alpha=0.9).attach_to(world)
            elif 1 <= link_idx <= 7:
                robot.manipulator.jlc.jnts[link_idx - 1].lnk.gen_meshmodel(alpha=0.9).attach_to(world)
            elif link_idx == 8:
                robot.end_effector.jlc.anchor.lnk_list[0].gen_meshmodel(alpha=0.9).attach_to(world)
            elif link_idx == 9:
                robot.end_effector.jlc.jnts[0].lnk.gen_meshmodel(alpha=0.9).attach_to(world)
            elif link_idx == 10:
                robot.end_effector.jlc.jnts[1].lnk.gen_meshmodel(alpha=0.9).attach_to(world)

    for link_idx in sorted(show_links):
        color = LINK_COLORS[link_idx % len(LINK_COLORS)]
        mask = sphere_link_indices == link_idx
        pts = sphere_positions[mask]
        rs = sphere_radii[mask]
        for pos, radius in zip(pts, rs):
            mcm.gen_sphere(radius=float(radius), pos=np.asarray(pos), rgb=color, alpha=0.2).attach_to(world)
        if len(pts) > 0:
            mgm.gen_frame(pos=pts.mean(axis=0), rotmat=np.eye(3), ax_length=0.08).attach_to(world)

    print('[legend] colored spheres come directly from the current URDF through SphereCollisionChecker.update(q)')
    world.run()


if __name__ == '__main__':
    main()
