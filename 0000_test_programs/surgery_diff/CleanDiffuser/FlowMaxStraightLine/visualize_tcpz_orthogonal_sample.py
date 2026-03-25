import argparse
from pathlib import Path

import numpy as np

import wrs.modeling.geometric_model as mgm
from wrs import wd
from helper_functions import visualize_anime_path
from visualize_nullspace_feasible_anime import (
    create_robot,
    normalize_vec,
    resolve_plane_normal,
    sample_valid_start_conf,
    solve_nullspace_candidates,
)
from xarm_trail1 import MAX_STEPS, STEP_SIZE, WorkspaceContour, trace_line_by_ik


CONTOUR_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample feasible TCP-z directions with the nullspace defaults, then search the best straight-line trajectory in the fixed plane."
    )
    parser.add_argument("--robot", type=str, default="xarmlite6", choices=["franka", "xarmlite6"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--svd-tol", type=float, default=1e-6)
    parser.add_argument("--amplitude", type=float, default=0.35)
    parser.add_argument("--grid-resolution", type=int, default=2)
    parser.add_argument("--projection-iters", type=int, default=60)
    parser.add_argument("--projection-damping", type=float, default=1e-4)
    parser.add_argument("--position-tol", type=float, default=1e-5)
    parser.add_argument("--cone-angle-deg", type=float, default=60.0)
    parser.add_argument("--normal-x", type=float, default=None)
    parser.add_argument("--normal-y", type=float, default=None)
    parser.add_argument("--normal-z", type=float, default=None)
    parser.add_argument("--dedup-tol", type=float, default=1e-3)
    parser.add_argument("--keep-candidates", type=int, default=4000)
    parser.add_argument("--num-direction-samples", type=int, default=36)
    parser.add_argument("--min-best-length", type=float, default=0.10)
    parser.add_argument("--max-sample-retries", type=int, default=20)
    parser.add_argument("--max-joint-step-norm", type=float, default=1)
    parser.add_argument("--collision-check-substeps", type=int, default=10)
    parser.add_argument("--show-all-directions", action="store_true")
    parser.add_argument("--show-all-trajectories", action="store_true")
    return parser.parse_args()


def make_orthonormal_basis(axis: np.ndarray):
    axis = normalize_vec(axis)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(ref, axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tangent_1 = normalize_vec(ref - np.dot(ref, axis) * axis)
    tangent_2 = normalize_vec(np.cross(axis, tangent_1))
    return axis, tangent_1, tangent_2


def sample_plane_directions(plane_normal: np.ndarray, num_samples: int, rng: np.random.Generator):
    _, tangent_1, tangent_2 = make_orthonormal_basis(plane_normal)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    angles = phase + np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False, dtype=np.float64)
    directions = [normalize_vec(np.cos(theta) * tangent_1 + np.sin(theta) * tangent_2) for theta in angles]
    return np.asarray(directions, dtype=np.float64)


def evaluate_best_direction(robot, contour, start_q: np.ndarray, directions: np.ndarray, candidate_idx: int, max_joint_step_norm: float, collision_check_substeps: int):
    best_direction = None
    best_result = None
    all_results = []
    for direction in directions:
        result = trace_line_by_ik(
            robot=robot,
            contour=contour,
            start_q=np.asarray(start_q, dtype=np.float64),
            direction=np.asarray(direction, dtype=np.float64),
            step_size=STEP_SIZE,
            max_steps=MAX_STEPS,
            max_joint_step_norm=max_joint_step_norm,
        )
        all_results.append(
            {
                "direction": np.asarray(direction, dtype=np.float64),
                "candidate_idx": int(candidate_idx),
                "result": result,
            }
        )
        if best_result is None or result["line_length"] > best_result["line_length"]:
            best_result = result
            best_direction = np.asarray(direction, dtype=np.float64)
    return best_direction, best_result, all_results


def sample_visualizable_case(
    robot,
    contour,
    rng: np.random.Generator,
    plane_normal_args,
    svd_tol: float,
    amplitude: float,
    grid_resolution: int,
    projection_iters: int,
    projection_damping: float,
    position_tol: float,
    cone_angle_deg: float,
    dedup_tol: float,
    keep_candidates: int,
    num_direction_samples: int,
    min_best_length: float,
    max_sample_retries: int,
    max_joint_step_norm: float,
    collision_check_substeps: int,
):
    last_payload = None
    for retry_idx in range(1, max_sample_retries + 1):
        print(f"[Sample] retry={retry_idx}/{max_sample_retries}", flush=True)
        sampled = sample_valid_start_conf(robot=robot, contour=contour, rng=rng)
        plane_normal_fixed = resolve_plane_normal(sampled["start_rot"], plane_normal_args)
        sampled["plane_normal"] = plane_normal_fixed

        line_directions = sample_plane_directions(plane_normal_fixed, num_direction_samples, rng=rng)
        print(f"[Sample] plane fixed | num_line_directions={len(line_directions)}", flush=True)

        _, orientation_candidates = solve_nullspace_candidates(
            robot=robot,
            start_q=sampled["start_q"],
            start_pos=sampled["start_pos"],
            plane_normal=plane_normal_fixed,
            svd_tol=svd_tol,
            amplitude=amplitude,
            grid_resolution=grid_resolution,
            projection_iters=projection_iters,
            projection_damping=projection_damping,
            position_tol=position_tol,
            cone_angle_deg=cone_angle_deg,
            dedup_tol=dedup_tol,
            keep_candidates=keep_candidates,
        )
        if len(orientation_candidates) == 0:
            print("[Sample] rejected sample | no feasible nullspace candidates", flush=True)
            continue

        best_candidate = None
        best_direction = None
        best_result = None
        all_results = []
        print(
            f"[Evaluate] evaluating line directions | orientation_candidates={len(orientation_candidates)} | directions_per_candidate={len(line_directions)}",
            flush=True,
        )
        for candidate_idx, orientation_candidate in enumerate(orientation_candidates, start=1):
            candidate_best_direction, candidate_best_result, candidate_results = evaluate_best_direction(
                robot=robot,
                contour=contour,
                start_q=orientation_candidate["q"],
                directions=line_directions,
                candidate_idx=candidate_idx - 1,
                max_joint_step_norm=max_joint_step_norm,
                collision_check_substeps=collision_check_substeps,
            )
            for item in candidate_results:
                item["candidate_q"] = orientation_candidate["q"]
                item["candidate_tcp_z_axis"] = orientation_candidate["tcp_z_axis"]
                item["candidate_on_boundary"] = bool(orientation_candidate.get("on_boundary", False))
            all_results.extend(candidate_results)
            if candidate_best_result is None:
                continue
            if best_result is None or candidate_best_result["line_length"] > best_result["line_length"]:
                best_candidate = orientation_candidate
                best_direction = candidate_best_direction
                best_result = candidate_best_result
            if candidate_idx % 5 == 0 or candidate_idx == len(orientation_candidates):
                current_best = 0.0 if best_result is None else float(best_result["line_length"])
                print(f"[Evaluate] processed_candidates={candidate_idx}/{len(orientation_candidates)} | current_best_L={current_best:.4f}", flush=True)

        last_payload = (
            sampled,
            plane_normal_fixed,
            line_directions,
            orientation_candidates,
            best_candidate,
            best_direction,
            best_result,
            all_results,
        )
        if best_result is None:
            print("[Sample] rejected sample | no valid straight-line prefix candidate", flush=True)
            continue
        if float(best_result["line_length"]) >= float(min_best_length):
            print(f"[Sample] accepted sample with best_line_length={best_result['line_length']:.4f} m", flush=True)
            return last_payload
        current_best = float(best_result["line_length"])
        print(f"[Sample] rejected sample | best_line_length={current_best:.4f} < min_best_length={min_best_length:.4f}", flush=True)
    return last_payload


def color_from_length(length: float, max_length: float):
    score = 0.0 if max_length <= 1e-8 else float(np.clip(length / max_length, 0.0, 1.0))
    return [0.20 + 0.76 * (1.0 - score), 0.22 + 0.62 * score, 0.18]


def visualize_sample(
    robot_name: str,
    record: dict,
    line_directions: np.ndarray,
    orientation_candidates: list[dict],
    best_candidate: dict,
    best_direction: np.ndarray,
    best_result: dict,
    all_results: list[dict],
    show_all_directions: bool,
    show_all_trajectories: bool,
):
    base = wd.World(cam_pos=[1.15, 0.45, 0.5], lookat_pos=[0.25, 0.0, 0.0])
    mgm.gen_frame().attach_to(base)

    robot = create_robot(robot_name, enable_cc=False)
    robot.goto_given_conf(record["start_q"])
    robot.gen_meshmodel(rgb=[0.20, 0.70, 0.95], alpha=0.55).attach_to(base)

    start_pos = record["start_pos"]
    plane_normal_fixed = record["plane_normal_fixed"]

    mgm.gen_sphere(start_pos, radius=0.005, rgb=[0.95, 0.15, 0.15]).attach_to(base)
    mgm.gen_arrow(start_pos, start_pos + plane_normal_fixed * 0.10, stick_radius=0.0022, rgb=[0.95, 0.15, 0.15]).attach_to(base)

    for candidate in orientation_candidates:
        axis_rgb = [1.00, 0.72, 0.35] if candidate.get("on_boundary", False) else [0.60, 0.80, 1.00]
        axis_end = start_pos + candidate["tcp_z_axis"] * 0.08
        mgm.gen_stick(start_pos, axis_end, radius=0.0008, rgb=axis_rgb, alpha=0.2).attach_to(base)
        robot.goto_given_conf(candidate["q"])
        robot.gen_meshmodel(rgb=[0.96, 0.96, 0.96], alpha=0.2).attach_to(base)

    max_length = max(float(item["result"]["line_length"]) for item in all_results) if all_results else 0.0

    if show_all_directions:
        for direction in line_directions:
            ray_end = start_pos + direction * 0.08
            mgm.gen_stick(start_pos, ray_end, radius=0.0010, rgb=[0.85, 0.70, 0.15], alpha=0.08).attach_to(base)

    if show_all_trajectories:
        for item in all_results:
            result = item["result"]
            traj = np.asarray(result["traj_pos"], dtype=np.float64)
            if len(traj) < 2:
                continue
            rgb = color_from_length(float(result["line_length"]), max_length)
            alpha = 0.22 + 0.32 * (0.0 if max_length <= 1e-8 else float(np.clip(result["line_length"] / max_length, 0.0, 1.0)))
            mgm.gen_stick(traj[0], traj[-1], radius=0.0012, rgb=rgb, alpha=alpha).attach_to(base)
            mgm.gen_sphere(traj[-1], radius=0.0018, rgb=rgb).attach_to(base)

    if best_candidate is not None:
        robot.goto_given_conf(best_candidate["q"])
        robot.gen_meshmodel(rgb=[0.10, 0.85, 0.28], alpha=0.42).attach_to(base)
        mgm.gen_arrow(start_pos, start_pos + best_candidate["tcp_z_axis"] * 0.09, stick_radius=0.0015, rgb=[0.90, 0.35, 0.85]).attach_to(base)

    mgm.gen_arrow(start_pos, start_pos + best_direction * 0.09, stick_radius=0.0018, rgb=[0.96, 0.60, 0.08]).attach_to(base)
    traj = np.asarray(best_result["traj_pos"], dtype=np.float64)
    if len(traj) >= 2:
        mgm.gen_stick(traj[0], traj[-1], radius=0.0018, rgb=[0.3, 0.90, 0.28]).attach_to(base)

    print(f"sampling_attempts: {record['attempts']}")
    print(f"robot: {robot_name}")
    print(f"start_q: {np.array2string(record['start_q'], precision=4, separator=', ')}")
    print(f"start_pos: {np.array2string(start_pos, precision=4, separator=', ')}")
    print(f"plane_normal_fixed: {np.array2string(plane_normal_fixed, precision=4, separator=', ')}")
    print(f"num_nullspace_candidates: {len(orientation_candidates)}")
    print(f"cone_angle_deg: {record['cone_angle_deg']:.1f}")
    if best_candidate is not None:
        print(f"best_tcp_z_axis: {np.array2string(best_candidate['tcp_z_axis'], precision=4, separator=', ')}")
        print(f"best_axis_error_deg: {float(best_candidate['axis_error_deg']):.4f}")
        print(f"best_on_boundary: {bool(best_candidate.get('on_boundary', False))}")
    print(f"best_direction: {np.array2string(best_direction, precision=4, separator=', ')}")
    print(f"dot(plane_normal_fixed, best_direction): {float(np.dot(plane_normal_fixed, best_direction)):.8f}")
    print(f"best_line_length: {best_result['line_length']:.4f} m")
    print(f"continuity_ok: {bool(best_result.get('continuity_ok', False))}")
    print(f"max_joint_step_norm: {float(best_result.get('max_joint_step_norm', 0.0)):.4f}")
    print(f"termination_reason: {best_result['termination_reason']}")
    if show_all_trajectories:
        print("all trajectories are visualized as faint line segments; greener/longer means farther reach.")

    anime_robot = create_robot(robot_name, enable_cc=False)
    visualize_anime_path(base, anime_robot, np.asarray(best_result["traj_q"], dtype=np.float64))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    contour = WorkspaceContour(contour_path=str(CONTOUR_PATH), z_value=0.0)
    robot = create_robot(args.robot, enable_cc=True)

    sampled, plane_normal_fixed, line_directions, orientation_candidates, best_candidate, best_direction, best_result, all_results = sample_visualizable_case(
        robot=robot,
        contour=contour,
        rng=rng,
        plane_normal_args=args,
        svd_tol=args.svd_tol,
        amplitude=args.amplitude,
        grid_resolution=args.grid_resolution,
        projection_iters=args.projection_iters,
        projection_damping=args.projection_damping,
        position_tol=args.position_tol,
        cone_angle_deg=args.cone_angle_deg,
        dedup_tol=args.dedup_tol,
        keep_candidates=args.keep_candidates,
        num_direction_samples=args.num_direction_samples,
        min_best_length=args.min_best_length,
        max_sample_retries=args.max_sample_retries,
        max_joint_step_norm=args.max_joint_step_norm,
        collision_check_substeps=args.collision_check_substeps,
    )
    if sampled is None or best_candidate is None or best_direction is None or best_result is None:
        raise RuntimeError("Failed to find a visualizable nullspace sample with a valid straight-line trajectory.")

    record = {
        "start_q": sampled["start_q"],
        "start_pos": sampled["start_pos"],
        "plane_normal_fixed": plane_normal_fixed,
        "attempts": sampled["attempts"],
        "cone_angle_deg": float(args.cone_angle_deg),
    }
    visualize_sample(
        robot_name=args.robot,
        record=record,
        line_directions=line_directions,
        orientation_candidates=orientation_candidates,
        best_candidate=best_candidate,
        best_direction=best_direction,
        best_result=best_result,
        all_results=all_results,
        show_all_directions=args.show_all_directions,
        show_all_trajectories=args.show_all_trajectories,
    )


if __name__ == "__main__":
    main()
