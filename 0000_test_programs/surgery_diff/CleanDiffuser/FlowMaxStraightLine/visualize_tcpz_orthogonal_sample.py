import argparse
from pathlib import Path

import numpy as np

import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka_sim
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
from wrs import wd
from helper_functions import visualize_anime_path
from xarm_trail1 import (
    MAX_STEPS,
    STEP_SIZE,
    WorkspaceContour,
    is_pose_inside_workspace,
    trace_line_by_ik,
)


CONTOUR_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl")
MAX_START_ATTEMPTS = 5000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize one random q_start sample with a fixed drawing plane, while allowing end-effector orientation to tilt slightly."
    )
    parser.add_argument("--robot", type=str, default="xarm", choices=["xarm", "franka"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sampling-mode", type=str, default="orientation_cone", choices=["orthogonal_circle", "orientation_cone"])
    parser.add_argument("--num-direction-samples", type=int, default=12, help="How many straight-line directions to search in the fixed plane.")
    parser.add_argument("--num-orientation-samples", type=int, default=12, help="How many tilted end-effector orientations to sample in the cone.")
    parser.add_argument("--cone-angle-deg", type=float, default=60.0, help="Max orientation tilt angle relative to the original end-effector orientation.")
    parser.add_argument("--position-tol", type=float, default=1e-5, help="Exact TCP position tolerance for orientation-perturbed IK candidates.")
    parser.add_argument("--min-best-length", type=float, default=0.10)
    parser.add_argument("--max-sample-retries", type=int, default=200)
    parser.add_argument("--show-all-directions", type=bool, default=True)
    parser.add_argument("--show-all-trajectories", type=bool, default=True)
    return parser.parse_args()


def make_robot(robot_name: str, enable_cc: bool):
    if robot_name == "xarm":
        return xarm6_sim.XArmLite6Miller(enable_cc=enable_cc)
    if robot_name == "franka":
        return franka_sim.FrankaResearch3(enable_cc=enable_cc)
    raise ValueError(f"Unsupported robot: {robot_name}")


def normalize_vec(vec: np.ndarray):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Vector norm too small.")
    return vec / norm


def make_orthonormal_basis(axis: np.ndarray):
    axis = normalize_vec(axis)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(ref, axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tangent_1 = ref - np.dot(ref, axis) * axis
    tangent_1 = normalize_vec(tangent_1)
    tangent_2 = normalize_vec(np.cross(axis, tangent_1))
    return axis, tangent_1, tangent_2


def axis_angle_rotmat(axis: np.ndarray, angle_rad: float):
    axis = normalize_vec(axis)
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def sample_plane_directions(plane_normal: np.ndarray, num_samples: int, rng: np.random.Generator):
    _, tangent_1, tangent_2 = make_orthonormal_basis(plane_normal)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    angles = phase + np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False, dtype=np.float64)
    directions = [normalize_vec(np.cos(theta) * tangent_1 + np.sin(theta) * tangent_2) for theta in angles]
    return np.asarray(directions, dtype=np.float64)


def sample_orientation_rotmats(start_rot: np.ndarray, num_samples: int, cone_angle_deg: float, rng: np.random.Generator):
    start_rot = np.asarray(start_rot, dtype=np.float64)
    if num_samples <= 1 or cone_angle_deg <= 1e-8:
        return np.asarray([start_rot], dtype=np.float64)

    x_axis = normalize_vec(start_rot[:, 0])
    y_axis = normalize_vec(start_rot[:, 1])
    tilt_angle = np.deg2rad(cone_angle_deg)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    azimuths = phase + np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False, dtype=np.float64)

    rotmats = [start_rot.copy()]
    for az in azimuths:
        tilt_axis = normalize_vec(np.cos(az) * x_axis + np.sin(az) * y_axis)
        rotmats.append(axis_angle_rotmat(tilt_axis, tilt_angle) @ start_rot)
    return np.asarray(rotmats, dtype=np.float64)


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


def solve_orientation_candidates(robot, start_pos: np.ndarray, orientation_rotmats: np.ndarray, position_tol: float):
    candidates = []
    for orientation_idx, rotmat in enumerate(orientation_rotmats):
        ik_solutions = robot.ik(
            tgt_pos=start_pos,
            tgt_rotmat=rotmat,
            option="multiple",
        )
        if ik_solutions is None or len(ik_solutions) == 0:
            continue
        for q_candidate in ik_solutions:
            q_candidate = np.asarray(q_candidate, dtype=np.float64)
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
            candidates.append(
                {
                    "orientation_idx": int(orientation_idx),
                    "q": q_candidate,
                    "start_rot": candidate_rot,
                    "tcp_z_axis": normalize_vec(candidate_rot[:, 2]),
                }
            )
        if (orientation_idx + 1) % 10 == 0 or (orientation_idx + 1) == len(orientation_rotmats):
            print(
                f"[OrientationIK] processed={orientation_idx + 1}/{len(orientation_rotmats)} | valid={len(candidates)}",
                flush=True,
            )
    return candidates


def evaluate_best_direction(robot, contour, start_q: np.ndarray, directions: np.ndarray, orientation_idx: int):
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
        )
        all_results.append(
            {
                "direction": np.asarray(direction, dtype=np.float64),
                "orientation_idx": int(orientation_idx),
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
    sampling_mode: str,
    num_direction_samples: int,
    num_orientation_samples: int,
    cone_angle_deg: float,
    position_tol: float,
    min_best_length: float,
    max_sample_retries: int,
):
    last_payload = None
    for _ in range(max_sample_retries):
        retry_idx = _ + 1
        print(f"[Sample] retry={retry_idx}/{max_sample_retries}", flush=True)
        sampled = sample_valid_start_conf(robot=robot, contour=contour, rng=rng)
        plane_normal_fixed = normalize_vec(sampled["start_rot"][:, 2])
        line_directions = sample_plane_directions(plane_normal_fixed, num_direction_samples, rng=rng)
        print(
            f"[Sample] plane fixed | num_line_directions={len(line_directions)} | mode={sampling_mode}",
            flush=True,
        )

        if sampling_mode == "orientation_cone":
            orientation_rotmats = sample_orientation_rotmats(
                start_rot=sampled["start_rot"],
                num_samples=num_orientation_samples,
                cone_angle_deg=cone_angle_deg,
                rng=rng,
            )
            print(
                f"[OrientationIK] solving orientation cone IK | rotmats={len(orientation_rotmats)} | cone_angle_deg={cone_angle_deg:.1f}",
                flush=True,
            )
            orientation_candidates = solve_orientation_candidates(
                robot=robot,
                start_pos=sampled["start_pos"],
                orientation_rotmats=orientation_rotmats,
                position_tol=position_tol,
            )
            print(f"[OrientationIK] valid orientation candidates={len(orientation_candidates)}", flush=True)
        else:
            orientation_candidates = [
                {
                    "orientation_idx": 0,
                    "q": sampled["start_q"].copy(),
                    "start_rot": sampled["start_rot"].copy(),
                    "tcp_z_axis": plane_normal_fixed.copy(),
                }
            ]

        best_candidate = None
        best_direction = None
        best_result = None
        all_results = []
        print(
            f"[Evaluate] evaluating line directions | orientation_candidates={len(orientation_candidates)} | directions_per_candidate={len(line_directions)}",
            flush=True,
        )
        for orientation_candidate in orientation_candidates:
            candidate_best_direction, candidate_best_result, candidate_results = evaluate_best_direction(
                robot=robot,
                contour=contour,
                start_q=orientation_candidate["q"],
                directions=line_directions,
                orientation_idx=orientation_candidate["orientation_idx"],
            )
            for item in candidate_results:
                item["candidate_q"] = orientation_candidate["q"]
                item["candidate_tcp_z_axis"] = orientation_candidate["tcp_z_axis"]
            all_results.extend(candidate_results)
            if best_result is None or candidate_best_result["line_length"] > best_result["line_length"]:
                best_candidate = orientation_candidate
                best_direction = candidate_best_direction
                best_result = candidate_best_result
            if (orientation_candidate["orientation_idx"] + 1) % 5 == 0 or (orientation_candidate["orientation_idx"] + 1) == len(orientation_candidates):
                current_best = 0.0 if best_result is None else float(best_result["line_length"])
                print(
                    f"[Evaluate] processed_orientation_idx={orientation_candidate['orientation_idx'] + 1} | current_best_L={current_best:.4f}",
                    flush=True,
                )

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
        if best_result is not None and float(best_result["line_length"]) >= float(min_best_length):
            print(f"[Sample] accepted sample with best_line_length={best_result['line_length']:.4f} m", flush=True)
            return last_payload
        current_best = 0.0 if best_result is None else float(best_result["line_length"])
        print(
            f"[Sample] rejected sample | best_line_length={current_best:.4f} < min_best_length={min_best_length:.4f}",
            flush=True,
        )
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

    robot = make_robot(robot_name, enable_cc=False)
    robot.goto_given_conf(record["start_q"])
    robot.gen_meshmodel(rgb=[0.20, 0.70, 0.95], alpha=0.55).attach_to(base)

    start_pos = record["start_pos"]
    plane_normal_fixed = record["plane_normal_fixed"]

    mgm.gen_sphere(start_pos, radius=0.005, rgb=[0.95, 0.15, 0.15]).attach_to(base)
    mgm.gen_arrow(start_pos, start_pos + plane_normal_fixed * 0.10, stick_radius=0.0022, rgb=[0.95, 0.15, 0.15]).attach_to(base)

    if record["sampling_mode"] == "orientation_cone":
        for candidate in orientation_candidates:
            axis_end = start_pos + candidate["tcp_z_axis"] * 0.08
            mgm.gen_stick(start_pos, axis_end, radius=0.0008, rgb=[0.95, 0.50, 0.50], alpha=0.2).attach_to(base)
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

    print(f"robot: {robot_name}")
    print(f"sampling_attempts: {record['attempts']}")
    print(f"base_seed_q: {np.array2string(record['start_q'], precision=4, separator=', ')}")
    print(f"start_q: {np.array2string(record['start_q'], precision=4, separator=', ')}")
    print(f"start_pos: {np.array2string(start_pos, precision=4, separator=', ')}")
    print(f"plane_normal_fixed: {np.array2string(plane_normal_fixed, precision=4, separator=', ')}")
    print(f"sampling_mode: {record['sampling_mode']}")
    if record["sampling_mode"] == "orientation_cone":
        print(f"cone_angle_deg: {record['cone_angle_deg']:.1f}")
        print(f"num_orientation_candidates: {len(orientation_candidates)}")
        if best_candidate is not None:
            print(f"best_tcp_z_axis: {np.array2string(best_candidate['tcp_z_axis'], precision=4, separator=', ')}")
    print(f"best_direction: {np.array2string(best_direction, precision=4, separator=', ')}")
    print(f"dot(plane_normal_fixed, best_direction): {float(np.dot(plane_normal_fixed, best_direction)):.8f}")
    print(f"best_line_length: {best_result['line_length']:.4f} m")
    print(f"termination_reason: {best_result['termination_reason']}")
    if show_all_trajectories:
        print("all trajectories are visualized as faint line segments; greener/longer means farther reach.")
    anime_robot = make_robot(robot_name, enable_cc=False)
    visualize_anime_path(base, anime_robot, np.asarray(best_result["traj_q"], dtype=np.float64))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    contour = WorkspaceContour(contour_path=str(CONTOUR_PATH), z_value=0.0)
    robot = make_robot(args.robot, enable_cc=True)

    sampled, plane_normal_fixed, line_directions, orientation_candidates, best_candidate, best_direction, best_result, all_results = sample_visualizable_case(
        robot=robot,
        contour=contour,
        rng=rng,
        sampling_mode=args.sampling_mode,
        num_direction_samples=args.num_direction_samples,
        num_orientation_samples=args.num_orientation_samples,
        cone_angle_deg=args.cone_angle_deg,
        position_tol=args.position_tol,
        min_best_length=args.min_best_length,
        max_sample_retries=args.max_sample_retries,
    )

    record = {
        "start_q": sampled["start_q"],
        "start_pos": sampled["start_pos"],
        "plane_normal_fixed": plane_normal_fixed,
        "attempts": sampled["attempts"],
        "sampling_mode": args.sampling_mode,
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
