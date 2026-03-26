import argparse
from pathlib import Path

import numpy as np

import wrs.modeling.geometric_model as mgm
from wrs import wd
from helper_functions import visualize_anime_path
from visualize_tcpz_orthogonal_sample import (
    CONTOUR_PATH,
    MAX_START_ATTEMPTS,
    make_robot,
    normalize_vec,
    sample_orientation_rotmats,
    sample_valid_start_conf,
    solve_orientation_candidates,
)
from xarm_trail1 import MAX_STEPS, STEP_SIZE, WorkspaceContour, trace_line_by_ik


FIXED_DIRECTION = np.array([1.0, 0.0, 0.0], dtype=np.float64)
NUM_ORIENTATION_SAMPLES = 200
CONE_ANGLE_DEG = 60.0
POSITION_TOL = 1e-5


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze fixed-direction mobility at one position with varying end-effector orientations and joint solutions.")
    parser.add_argument("--robot", type=str, default="xarm", choices=["xarm", "franka"])
    return parser.parse_args()


def compute_position_jacobian(robot, q: np.ndarray):
    robot.fk(np.asarray(q, dtype=np.float64), update=True)
    return np.asarray(robot.jacobian(), dtype=np.float64)[:3, :]


def directional_manipulability(j_pos: np.ndarray, direction: np.ndarray):
    direction = np.asarray(direction, dtype=np.float64)
    return float(np.sqrt(direction @ (j_pos @ j_pos.T) @ direction))


def accumulated_directional_manipulability(robot, traj_q: np.ndarray, direction: np.ndarray):
    total = 0.0
    for q in np.asarray(traj_q, dtype=np.float64):
        total += directional_manipulability(compute_position_jacobian(robot, q), direction)
    return float(total)


def color_from_score(score: float, rgb_low: np.ndarray, rgb_high: np.ndarray):
    score = float(np.clip(score, 0.0, 1.0))
    rgb_low = np.asarray(rgb_low, dtype=np.float64)
    rgb_high = np.asarray(rgb_high, dtype=np.float64)
    return (1.0 - score) * rgb_low + score * rgb_high


def angle_between_dirs(a: np.ndarray, b: np.ndarray):
    c = float(np.clip(np.dot(normalize_vec(a), normalize_vec(b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def pearson_corr(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def main():
    args = parse_args()
    rng = np.random.default_rng()
    contour = WorkspaceContour(contour_path=str(CONTOUR_PATH), z_value=0.0)
    robot = make_robot(args.robot, enable_cc=True)
    fixed_direction = FIXED_DIRECTION.copy()

    sample_attempt = 0
    while True:
        sample_attempt += 1
        sampled = sample_valid_start_conf(robot=robot, contour=contour, rng=rng)
        orientation_rotmats = sample_orientation_rotmats(
            start_rot=sampled["start_rot"],
            num_samples=NUM_ORIENTATION_SAMPLES,
            cone_angle_deg=CONE_ANGLE_DEG,
            rng=rng,
        )
        candidates = solve_orientation_candidates(
            robot=robot,
            start_pos=sampled["start_pos"],
            orientation_rotmats=orientation_rotmats,
            position_tol=POSITION_TOL,
        )
        if len(candidates) == 0:
            continue

        entries = []
        for candidate in candidates:
            j_pos = compute_position_jacobian(robot, candidate["q"])
            inst_manip = directional_manipulability(j_pos, fixed_direction)
            trace = trace_line_by_ik(
                robot=robot,
                contour=contour,
                start_q=np.asarray(candidate["q"], dtype=np.float64),
                direction=fixed_direction,
                step_size=STEP_SIZE,
                max_steps=MAX_STEPS,
            )
            accum_manip = accumulated_directional_manipulability(robot, trace["traj_q"], fixed_direction)
            entries.append({
                "candidate_q": np.asarray(candidate["q"], dtype=np.float64),
                "tcp_z_axis": np.asarray(candidate["tcp_z_axis"], dtype=np.float64),
                "orientation_idx": int(candidate["orientation_idx"]),
                "inst_manip": float(inst_manip),
                "accum_manip": float(accum_manip),
                "line_length": float(trace["line_length"]),
                "trace": trace,
            })

        best_actual = max(entries, key=lambda item: item["line_length"])
        if best_actual["line_length"] > 0.0:
            break

    start_pos = np.asarray(sampled["start_pos"], dtype=np.float64)
    start_rot = np.asarray(sampled["start_rot"], dtype=np.float64)
    plane_normal = normalize_vec(start_rot[:, 2])
    start_j = compute_position_jacobian(robot, sampled["start_q"])
    eigvals = np.linalg.eigvalsh(start_j @ start_j.T)

    inst_scores = np.asarray([item["inst_manip"] for item in entries], dtype=np.float64)
    accum_scores = np.asarray([item["accum_manip"] for item in entries], dtype=np.float64)
    line_lengths = np.asarray([item["line_length"] for item in entries], dtype=np.float64)
    best_inst = max(entries, key=lambda item: item["inst_manip"])
    best_accum = max(entries, key=lambda item: item["accum_manip"])

    actual_order = np.argsort(-line_lengths)
    inst_order = np.argsort(-inst_scores)
    accum_order = np.argsort(-accum_scores)
    actual_rank_of_inst_best = int(np.where(inst_order == np.argmax(line_lengths))[0][0] + 1)
    actual_rank_of_accum_best = int(np.where(accum_order == np.argmax(line_lengths))[0][0] + 1)
    inst_rank_of_actual_best = int(np.where(actual_order == np.argmax(inst_scores))[0][0] + 1)
    accum_rank_of_actual_best = int(np.where(actual_order == np.argmax(accum_scores))[0][0] + 1)

    base = wd.World(cam_pos=[1.15, 0.45, 0.5], lookat_pos=[0.25, 0.0, 0.0])
    mgm.gen_frame().attach_to(base)

    preview_robot = make_robot(args.robot, enable_cc=False)
    preview_robot.goto_given_conf(sampled["start_q"])
    preview_robot.gen_meshmodel(rgb=[0.20, 0.70, 0.95], alpha=0.25).attach_to(base)
    mgm.gen_sphere(start_pos, radius=0.005, rgb=[0.95, 0.15, 0.15]).attach_to(base)
    mgm.gen_arrow(start_pos, start_pos + fixed_direction * 0.12, stick_radius=0.0022, rgb=[0.96, 0.60, 0.08]).attach_to(base)
    mgm.gen_arrow(start_pos, start_pos + plane_normal * 0.08, stick_radius=0.0018, rgb=[0.95, 0.15, 0.15]).attach_to(base)

    inst_max = float(np.max(inst_scores)) if len(inst_scores) > 0 else 0.0
    accum_max = float(np.max(accum_scores)) if len(accum_scores) > 0 else 0.0
    length_max = float(np.max(line_lengths)) if len(line_lengths) > 0 else 0.0
    inst_vis_scale = 0.16
    accum_vis_scale = 0.24

    for item in entries:
        axis = normalize_vec(item["tcp_z_axis"])
        inst_score = 0.0 if inst_max <= 1e-12 else float(item["inst_manip"] / inst_max)
        inst_end = start_pos + axis * (0.03 + inst_vis_scale * inst_score)
        inst_rgb = color_from_score(inst_score, [0.12, 0.45, 0.18], [0.20, 0.98, 0.28])
        mgm.gen_sphere(inst_end, radius=0.0020 + 0.0016 * inst_score, rgb=inst_rgb, alpha=0.65).attach_to(base)

        accum_score = 0.0 if accum_max <= 1e-12 else float(item["accum_manip"] / accum_max)
        accum_end = start_pos + axis * (0.05 + accum_vis_scale * accum_score)
        accum_rgb = color_from_score(accum_score, [0.45, 0.12, 0.48], [0.98, 0.20, 0.92])
        mgm.gen_sphere(accum_end, radius=0.0017 + 0.0015 * accum_score, rgb=accum_rgb, alpha=0.55).attach_to(base)

        traj = np.asarray(item["trace"]["traj_pos"], dtype=np.float64)
        if len(traj) < 2:
            continue
        length_score = 0.0 if length_max <= 1e-12 else float(item["line_length"] / length_max)
        traj_rgb = color_from_score(length_score, [0.10, 0.25, 0.55], [0.20, 0.95, 0.25])
        mgm.gen_stick(traj[0], traj[-1], radius=0.0014, rgb=traj_rgb, alpha=0.28).attach_to(base)
        mgm.gen_sphere(traj[-1], radius=0.0018, rgb=traj_rgb, alpha=0.35).attach_to(base)

    mgm.gen_arrow(start_pos, start_pos + best_inst["tcp_z_axis"] * 0.09, stick_radius=0.0015, rgb=[0.20, 0.98, 0.28]).attach_to(base)
    mgm.gen_arrow(start_pos, start_pos + best_accum["tcp_z_axis"] * 0.10, stick_radius=0.0015, rgb=[0.92, 0.18, 0.88]).attach_to(base)

    preview_robot.goto_given_conf(best_actual["candidate_q"])
    preview_robot.gen_meshmodel(rgb=[0.20, 0.95, 0.25], alpha=0.40).attach_to(base)
    best_traj = np.asarray(best_actual["trace"]["traj_pos"], dtype=np.float64)
    if len(best_traj) >= 2:
        mgm.gen_stick(best_traj[0], best_traj[-1], radius=0.0024, rgb=[0.20, 0.95, 0.25]).attach_to(base)
        mgm.gen_sphere(best_traj[-1], radius=0.0026, rgb=[0.20, 0.95, 0.25]).attach_to(base)

    print(f"robot: {args.robot}")
    print(f"sample_attempt: {sample_attempt}")
    print(f"base_start_q: {np.array2string(sampled['start_q'], precision=4, separator=', ')}")
    print(f"fixed_start_pos: {np.array2string(start_pos, precision=4, separator=', ')}")
    print(f"fixed_direction: {np.array2string(fixed_direction, precision=4, separator=', ')}")
    print(f"num_orientation_candidates: {len(entries)}")
    print(f"base_jacobian_rank: {int(np.linalg.matrix_rank(start_j))}")
    print(f"base_position_singular_values: {np.array2string(np.sqrt(np.clip(eigvals, 0.0, None))[::-1], precision=5, separator=', ')}")
    print(f"best_inst_tcp_z_axis: {np.array2string(best_inst['tcp_z_axis'], precision=4, separator=', ')}")
    print(f"best_inst_manipulability: {best_inst['inst_manip']:.6f}")
    print(f"best_inst_actual_length: {best_inst['line_length']:.4f} m")
    print(f"best_accum_tcp_z_axis: {np.array2string(best_accum['tcp_z_axis'], precision=4, separator=', ')}")
    print(f"best_accum_manipulability: {best_accum['accum_manip']:.6f}")
    print(f"best_accum_actual_length: {best_accum['line_length']:.4f} m")
    print(f"best_actual_tcp_z_axis: {np.array2string(best_actual['tcp_z_axis'], precision=4, separator=', ')}")
    print(f"best_actual_length: {best_actual['line_length']:.4f} m")
    print(f"best_actual_inst_manipulability: {best_actual['inst_manip']:.6f}")
    print(f"best_actual_accum_manipulability: {best_actual['accum_manip']:.6f}")
    print(f"inst_vs_actual_axis_gap_deg: {angle_between_dirs(best_inst['tcp_z_axis'], best_actual['tcp_z_axis']):.3f}")
    print(f"accum_vs_actual_axis_gap_deg: {angle_between_dirs(best_accum['tcp_z_axis'], best_actual['tcp_z_axis']):.3f}")
    print(f"pearson_corr(inst_manipulability, line_length): {pearson_corr(inst_scores, line_lengths):.4f}")
    print(f"pearson_corr(accum_manipulability, line_length): {pearson_corr(accum_scores, line_lengths):.4f}")
    print(f"actual_rank_of_inst_best: {actual_rank_of_inst_best}/{len(entries)}")
    print(f"inst_rank_of_actual_best: {inst_rank_of_actual_best}/{len(entries)}")
    print(f"actual_rank_of_accum_best: {actual_rank_of_accum_best}/{len(entries)}")
    print(f"accum_rank_of_actual_best: {accum_rank_of_actual_best}/{len(entries)}")
    print(f"best_actual_termination_reason: {best_actual['trace']['termination_reason']}")
    print("visualization: green spheres = instantaneous Jacobian score, purple spheres = accumulated Jacobian score, blue-to-green segments = actual reachable length along the fixed direction")

    anime_robot = make_robot(args.robot, enable_cc=False)
    visualize_anime_path(base, anime_robot, np.asarray(best_actual['trace']['traj_q'], dtype=np.float64))


if __name__ == "__main__":
    main()
