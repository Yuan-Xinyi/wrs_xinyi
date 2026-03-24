import argparse
from pathlib import Path

import numpy as np

import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka_sim
from wrs import wd
from analyze_nullspace_validation_point import (
    compute_nullspace_svd,
    farthest_point_sampling_jointspace,
    unique_projected_candidates,
)
from helper_functions import visualize_anime_path
from xarm_trail1 import WorkspaceContour, is_pose_inside_workspace


CONTOUR_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl")
MAX_START_ATTEMPTS = 5000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize feasible nullspace solutions only, with TCP z-axis constrained to lie inside a cone around a plane normal."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--svd-tol", type=float, default=1e-6)
    parser.add_argument("--amplitude", type=float, default=0.35)
    parser.add_argument("--grid-resolution", type=int, default=8)
    parser.add_argument("--projection-iters", type=int, default=60)
    parser.add_argument("--projection-damping", type=float, default=1e-4)
    parser.add_argument("--position-tol", type=float, default=1e-5)
    parser.add_argument("--cone-angle-deg", type=float, default=30.0)
    parser.add_argument("--normal-x", type=float, default=None)
    parser.add_argument("--normal-y", type=float, default=None)
    parser.add_argument("--normal-z", type=float, default=None)
    parser.add_argument("--dedup-tol", type=float, default=1e-3)
    parser.add_argument("--keep-candidates", type=int, default=4000)
    return parser.parse_args()


def normalize_vec(vec: np.ndarray):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Vector norm too small.")
    return vec / norm


def sample_valid_start_conf(robot, contour, rng: np.random.Generator):
    jnt_ranges = np.asarray(robot.jnt_ranges, dtype=np.float64)
    for attempt_idx in range(1, MAX_START_ATTEMPTS + 1):
        start_q = rng.uniform(jnt_ranges[:, 0], jnt_ranges[:, 1])
        start_pos, start_rot = robot.fk(start_q)
        if not is_pose_inside_workspace(contour, start_pos):
            continue
        robot.goto_given_conf(start_q)
        if robot.is_collided():
            continue
        print(f"[StartQ] found valid q_start after {attempt_idx} attempts", flush=True)
        return {
            "start_q": np.asarray(start_q, dtype=np.float64),
            "start_pos": np.asarray(start_pos, dtype=np.float64),
            "start_rot": np.asarray(start_rot, dtype=np.float64),
            "attempts": int(attempt_idx),
        }
    raise RuntimeError(f"Failed to sample a valid q_start after {MAX_START_ATTEMPTS} attempts.")


def make_orthonormal_basis(normal: np.ndarray):
    normal = normalize_vec(normal)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tangent1 = ref - np.dot(ref, normal) * normal
    tangent1 = normalize_vec(tangent1)
    tangent2 = normalize_vec(np.cross(normal, tangent1))
    return normal, tangent1, tangent2


def axis_angle_error_deg(axis_a: np.ndarray, axis_b: np.ndarray):
    axis_a = normalize_vec(axis_a)
    axis_b = normalize_vec(axis_b)
    dot_val = float(np.clip(np.dot(axis_a, axis_b), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot_val)))


def angle_to_normal_rad(axis: np.ndarray, normal: np.ndarray):
    axis = normalize_vec(axis)
    normal = normalize_vec(normal)
    dot_val = float(np.clip(np.dot(axis, normal), -1.0, 1.0))
    return float(np.arccos(dot_val))


def damped_pseudoinverse_step(j_mat: np.ndarray, err: np.ndarray, damping: float):
    jj_t = j_mat @ j_mat.T
    damped = jj_t + (damping ** 2) * np.eye(jj_t.shape[0], dtype=np.float64)
    return j_mat.T @ np.linalg.solve(damped, err)


def finite_difference_angle_gradient(robot, q: np.ndarray, plane_normal: np.ndarray, eps: float = 1e-5):
    q = np.asarray(q, dtype=np.float64)
    grad = np.zeros_like(q)
    for i in range(len(q)):
        dq = np.zeros_like(q)
        dq[i] = eps
        _, rot_p = robot.fk(q + dq, update=False)
        _, rot_m = robot.fk(q - dq, update=False)
        phi_p = angle_to_normal_rad(np.asarray(rot_p, dtype=np.float64)[:, 2], plane_normal)
        phi_m = angle_to_normal_rad(np.asarray(rot_m, dtype=np.float64)[:, 2], plane_normal)
        grad[i] = (phi_p - phi_m) / (2.0 * eps)
    return grad


def project_to_target_position(
    robot,
    q_init: np.ndarray,
    target_pos: np.ndarray,
    plane_normal: np.ndarray,
    target_angle_rad: float | None,
    position_tol: float,
    max_iters: int,
    damping: float,
    angle_tol_rad: float = 1e-3,
):
    q = np.asarray(q_init, dtype=np.float64).copy()

    for iter_idx in range(max_iters):
        if not robot.are_jnts_in_ranges(q):
            return None
        pos, rot = robot.fk(q, update=True)
        pos = np.asarray(pos, dtype=np.float64)
        rot = np.asarray(rot, dtype=np.float64)
        axis = normalize_vec(rot[:, 2])
        pos_err = np.asarray(target_pos - pos, dtype=np.float64)
        if target_angle_rad is None:
            if np.linalg.norm(pos_err) <= position_tol:
                return {
                    "q": q.copy(),
                    "pos": pos.copy(),
                    "axis": axis.copy(),
                    "iterations": int(iter_idx),
                    "position_error": float(np.linalg.norm(pos_err)),
                    "on_boundary": False,
                }
            j_full = np.asarray(robot.jacobian(), dtype=np.float64)
            j_pos = j_full[:3, :]
            q = q + damped_pseudoinverse_step(j_pos, pos_err, damping=damping)
            continue

        phi = angle_to_normal_rad(axis, plane_normal)
        angle_err = np.array([target_angle_rad - phi], dtype=np.float64)
        if np.linalg.norm(pos_err) <= position_tol and abs(angle_err[0]) <= angle_tol_rad:
            return {
                "q": q.copy(),
                "pos": pos.copy(),
                "axis": axis.copy(),
                "iterations": int(iter_idx),
                "position_error": float(np.linalg.norm(pos_err)),
                "on_boundary": True,
            }
        j_full = np.asarray(robot.jacobian(), dtype=np.float64)
        j_pos = j_full[:3, :]
        j_phi = finite_difference_angle_gradient(robot, q, plane_normal).reshape(1, -1)
        j_aug = np.vstack([j_pos, j_phi])
        err_aug = np.concatenate([pos_err, angle_err], axis=0)
        q = q + damped_pseudoinverse_step(j_aug, err_aug, damping=damping)
    return None


def resolve_plane_normal(start_rot: np.ndarray, args):
    if args.normal_x is not None and args.normal_y is not None and args.normal_z is not None:
        return normalize_vec(np.array([args.normal_x, args.normal_y, args.normal_z], dtype=np.float64))
    return normalize_vec(np.asarray(start_rot[:, 2], dtype=np.float64))


def solve_nullspace_candidates(
    robot,
    start_q: np.ndarray,
    start_pos: np.ndarray,
    plane_normal: np.ndarray,
    svd_tol: float,
    amplitude: float,
    grid_resolution: int,
    projection_iters: int,
    projection_damping: float,
    position_tol: float,
    cone_angle_deg: float,
    dedup_tol: float,
    keep_candidates: int,
):
    svd_info = compute_nullspace_svd(robot, start_q, tol=svd_tol)
    null_basis = svd_info["null_basis"]
    print(
        f"[Nullspace] rank={svd_info['rank']} | nullspace_dim={null_basis.shape[1]} | singular_values={np.array2string(svd_info['s'], precision=5, separator=', ')}",
        flush=True,
    )
    if null_basis.size == 0:
        return svd_info, []

    coeff_axis = np.linspace(-amplitude, amplitude, grid_resolution, dtype=np.float64)
    mesh = np.meshgrid(*([coeff_axis] * null_basis.shape[1]), indexing="ij")
    coeff_mat = np.stack([m.reshape(-1) for m in mesh], axis=1)
    print(f"[Nullspace] projecting {len(coeff_mat)} local seeds", flush=True)

    projected = []
    cone_angle_rad = np.deg2rad(cone_angle_deg)
    for coeff_idx, coeff in enumerate(coeff_mat, start=1):
        q_seed = start_q + null_basis @ coeff
        _, seed_rot = robot.fk(q_seed, update=False)
        seed_axis = normalize_vec(np.asarray(seed_rot, dtype=np.float64)[:, 2])
        seed_phi = angle_to_normal_rad(seed_axis, plane_normal)
        target_angle_rad = None if seed_phi <= cone_angle_rad else cone_angle_rad
        projected_item = project_to_target_position(
            robot=robot,
            q_init=q_seed,
            target_pos=start_pos,
            plane_normal=plane_normal,
            target_angle_rad=target_angle_rad,
            position_tol=position_tol,
            max_iters=projection_iters,
            damping=projection_damping,
        )
        if projected_item is not None:
            projected.append(projected_item)
        if coeff_idx % 50 == 0 or coeff_idx == len(coeff_mat):
            print(f"[Nullspace] projected={coeff_idx}/{len(coeff_mat)} | valid_so_far={len(projected)}", flush=True)

    projected = unique_projected_candidates(projected, tol=dedup_tol)
    print(f"[Nullspace] unique_projected={len(projected)}", flush=True)

    candidates = []
    for proj_idx, projected_item in enumerate(projected, start=1):
        q_candidate = np.asarray(projected_item["q"], dtype=np.float64)
        if not robot.are_jnts_in_ranges(q_candidate):
            continue
        robot.goto_given_conf(q_candidate)
        if robot.is_collided():
            continue
        candidate_pos, candidate_rot = robot.fk(q_candidate)
        candidate_pos = np.asarray(candidate_pos, dtype=np.float64)
        candidate_rot = np.asarray(candidate_rot, dtype=np.float64)
        if np.linalg.norm(candidate_pos - start_pos) > position_tol:
            continue
        tcp_z_axis = normalize_vec(candidate_rot[:, 2])
        axis_error_deg = axis_angle_error_deg(tcp_z_axis, plane_normal)
        if axis_error_deg > cone_angle_deg:
            continue
        candidates.append(
            {
                "q": q_candidate,
                "start_rot": candidate_rot,
                "tcp_z_axis": tcp_z_axis,
                "axis_error_deg": axis_error_deg,
                "on_boundary": bool(projected_item.get("on_boundary", False)),
            }
        )
        if proj_idx % 50 == 0 or proj_idx == len(projected):
            print(f"[Nullspace] feasible_checked={proj_idx}/{len(projected)} | feasible={len(candidates)}", flush=True)

    representative = farthest_point_sampling_jointspace(
        [{"q": item["q"], "line_length": 1.0} for item in candidates],
        keep_count=keep_candidates,
    )
    rep_qs = [item["q"] for item in representative]
    final_candidates = []
    for q in rep_qs:
        for item in candidates:
            if np.allclose(item["q"], q):
                final_candidates.append(item)
                break

    print(f"[Nullspace] representative_candidates={len(final_candidates)} | keep={keep_candidates}", flush=True)
    return svd_info, final_candidates


def visualize_candidates(start_record: dict, candidates: list[dict]):
    base = wd.World(cam_pos=[1.15, 0.45, 0.5], lookat_pos=[0.25, 0.0, 0.0])
    mgm.gen_frame().attach_to(base)

    preview_robot = franka_sim.FrankaResearch3(enable_cc=False)
    preview_robot.goto_given_conf(start_record["start_q"])
    preview_robot.gen_meshmodel(rgb=[0.20, 0.70, 0.95], alpha=0.55).attach_to(base)

    start_pos = start_record["start_pos"]
    plane_normal_fixed = normalize_vec(start_record["plane_normal"])
    mgm.gen_sphere(start_pos, radius=0.005, rgb=[0.95, 0.15, 0.15]).attach_to(base)
    mgm.gen_arrow(start_pos, start_pos + plane_normal_fixed * 0.10, stick_radius=0.0022, rgb=[0.95, 0.15, 0.15]).attach_to(base)

    for item in candidates:
        axis_end = start_pos + item["tcp_z_axis"] * 0.08
        axis_rgb = [1.00, 0.72, 0.35] if item["on_boundary"] else [0.60, 0.80, 1.00]
        mgm.gen_stick(start_pos, axis_end, radius=0.0008, rgb=axis_rgb, alpha=0.18).attach_to(base)
        preview_robot.goto_given_conf(item["q"])
        # preview_robot.gen_meshmodel(rgb=[0.96, 0.96, 0.96], alpha=0.12).attach_to(base)

    print(f"sampling_attempts: {start_record['attempts']}")
    print(f"start_q: {np.array2string(start_record['start_q'], precision=4, separator=', ')}")
    print(f"start_pos: {np.array2string(start_pos, precision=4, separator=', ')}")
    print(f"plane_normal_fixed: {np.array2string(plane_normal_fixed, precision=4, separator=', ')}")
    print(f"num_nullspace_candidates: {len(candidates)}")
    print(f"max_axis_error_deg: {max(item['axis_error_deg'] for item in candidates):.4f}")
    print(f"boundary_candidates: {sum(int(item['on_boundary']) for item in candidates)}")

    anime_robot = franka_sim.FrankaResearch3(enable_cc=False)
    path = np.asarray([item["q"] for item in candidates], dtype=np.float64)
    base.run()  # Run the visualizer first to avoid potential OpenGL context issues with jax2torch in collision checking during projection. Then start the animation after the visualizer window is open.
    # visualize_anime_path(base, anime_robot, path)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    contour = WorkspaceContour(contour_path=str(CONTOUR_PATH), z_value=0.0)
    robot = franka_sim.FrankaResearch3(enable_cc=True)

    start_record = sample_valid_start_conf(robot=robot, contour=contour, rng=rng)
    plane_normal = resolve_plane_normal(start_record["start_rot"], args)
    start_record["plane_normal"] = plane_normal
    _, candidates = solve_nullspace_candidates(
        robot=robot,
        start_q=start_record["start_q"],
        start_pos=start_record["start_pos"],
        plane_normal=plane_normal,
        svd_tol=args.svd_tol,
        amplitude=args.amplitude,
        grid_resolution=args.grid_resolution,
        projection_iters=args.projection_iters,
        projection_damping=args.projection_damping,
        position_tol=args.position_tol,
        cone_angle_deg=args.cone_angle_deg,
        dedup_tol=args.dedup_tol,
        keep_candidates=args.keep_candidates,
    )
    if len(candidates) == 0:
        raise RuntimeError("No feasible nullspace candidates found for this sample.")

    visualize_candidates(start_record, candidates)


if __name__ == "__main__":
    main()
