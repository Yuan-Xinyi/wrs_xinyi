import argparse
from pathlib import Path

import numpy as np

import wrs.modeling.geometric_model as mgm
from wrs import wd
from helper_functions import visualize_anime_path
from visualize_tcpz_orthogonal_sample import make_robot, make_orthonormal_basis, normalize_vec
from xarm_trail1 import MAX_STEPS, STEP_SIZE, WorkspaceContour, is_pose_inside_workspace, trace_line_by_ik


CONTOUR_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl")
NUM_DIRECTIONS = 256
MAX_START_ATTEMPTS = 5000


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze planar mobility from Jacobian manipulability and compare it with actual straight-line feasibility.")
    parser.add_argument("--robot", type=str, default="franka", choices=["xarm", "franka"])
    return parser.parse_args()


def compute_position_jacobian(robot, q: np.ndarray):
    robot.fk(np.asarray(q, dtype=np.float64), update=True)
    return np.asarray(robot.jacobian(), dtype=np.float64)[:3, :]


def sample_valid_start_conf(robot, contour, rng: np.random.Generator):
    jnt_ranges = np.asarray(robot.jnt_ranges, dtype=np.float64)
    for _ in range(MAX_START_ATTEMPTS):
        q = rng.uniform(jnt_ranges[:, 0], jnt_ranges[:, 1])
        pos, rot = robot.fk(q)
        if not is_pose_inside_workspace(contour, pos):
            continue
        robot.goto_given_conf(q)
        if robot.is_collided():
            continue
        return np.asarray(q, dtype=np.float64), np.asarray(pos, dtype=np.float64), np.asarray(rot, dtype=np.float64)
    raise RuntimeError("Failed to find a valid start configuration.")


def sample_plane_directions(plane_normal: np.ndarray, num_samples: int):
    _, tangent_1, tangent_2 = make_orthonormal_basis(plane_normal)
    angles = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False, dtype=np.float64)
    return np.asarray([normalize_vec(np.cos(a) * tangent_1 + np.sin(a) * tangent_2) for a in angles], dtype=np.float64)


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

    sample_attempt = 0
    while True:
        sample_attempt += 1
        start_q, start_pos, start_rot = sample_valid_start_conf(robot, contour, rng)
        plane_normal = normalize_vec(start_rot[:, 2])
        directions = sample_plane_directions(plane_normal, NUM_DIRECTIONS)
        j_pos = compute_position_jacobian(robot, start_q)
        jj_t = j_pos @ j_pos.T
        eigvals = np.linalg.eigvalsh(jj_t)

        entries = []
        for direction in directions:
            mob = directional_manipulability(j_pos, direction)
            trace = trace_line_by_ik(
                robot=robot,
                contour=contour,
                start_q=start_q,
                direction=direction,
                step_size=STEP_SIZE,
                max_steps=MAX_STEPS,
            )
            accum_manip = accumulated_directional_manipulability(robot, trace["traj_q"], direction)
            entries.append({
                "direction": np.asarray(direction, dtype=np.float64),
                "manip": float(mob),
                "accum_manip": float(accum_manip),
                "line_length": float(trace["line_length"]),
                "trace": trace,
            })

        best_actual = max(entries, key=lambda item: item["line_length"])
        if best_actual["line_length"] > 0.0:
            break

    manip_scores = np.asarray([item["manip"] for item in entries], dtype=np.float64)
    accum_scores = np.asarray([item["accum_manip"] for item in entries], dtype=np.float64)
    line_lengths = np.asarray([item["line_length"] for item in entries], dtype=np.float64)
    best_pred = max(entries, key=lambda item: item["manip"])
    best_accum = max(entries, key=lambda item: item["accum_manip"])
    actual_order = np.argsort(-line_lengths)
    pred_rank_of_actual_best = int(np.where(actual_order == np.argmax(manip_scores))[0][0] + 1)
    accum_rank_of_actual_best = int(np.where(actual_order == np.argmax(accum_scores))[0][0] + 1)
    manip_order = np.argsort(-manip_scores)
    actual_rank_of_pred_best = int(np.where(manip_order == np.argmax(line_lengths))[0][0] + 1)
    accum_order = np.argsort(-accum_scores)
    actual_rank_of_accum_best = int(np.where(accum_order == np.argmax(line_lengths))[0][0] + 1)

    base = wd.World(cam_pos=[1.15, 0.45, 0.5], lookat_pos=[0.25, 0.0, 0.0])
    mgm.gen_frame().attach_to(base)

    preview_robot = make_robot(args.robot, enable_cc=False)
    preview_robot.goto_given_conf(start_q)
    preview_robot.gen_meshmodel(rgb=[0.20, 0.70, 0.95], alpha=0.50).attach_to(base)
    mgm.gen_sphere(start_pos, radius=0.005, rgb=[0.95, 0.15, 0.15]).attach_to(base)
    mgm.gen_arrow(start_pos, start_pos + plane_normal * 0.10, stick_radius=0.0022, rgb=[0.95, 0.15, 0.15]).attach_to(base)

    manip_max = float(np.max(manip_scores)) if len(manip_scores) > 0 else 0.0
    accum_max = float(np.max(accum_scores)) if len(accum_scores) > 0 else 0.0
    length_max = float(np.max(line_lengths)) if len(line_lengths) > 0 else 0.0
    manip_vis_scale = 0.18
    accum_vis_scale = 0.26

    for item in entries:
        manip_score = 0.0 if manip_max <= 1e-12 else float(item["manip"] / manip_max)
        manip_end = start_pos + item["direction"] * (0.04 + manip_vis_scale * manip_score)
        manip_rgb = color_from_score(manip_score, [0.12, 0.45, 0.18], [0.20, 0.98, 0.28])
        mgm.gen_sphere(manip_end, radius=0.0020 + 0.0018 * manip_score, rgb=manip_rgb, alpha=0.65).attach_to(base)

        accum_score = 0.0 if accum_max <= 1e-12 else float(item["accum_manip"] / accum_max)
        accum_end = start_pos + item["direction"] * (0.06 + accum_vis_scale * accum_score)
        accum_rgb = color_from_score(accum_score, [0.45, 0.12, 0.48], [0.98, 0.20, 0.92])
        mgm.gen_sphere(accum_end, radius=0.0017 + 0.0015 * accum_score, rgb=accum_rgb, alpha=0.55).attach_to(base)

        traj = np.asarray(item["trace"]["traj_pos"], dtype=np.float64)
        if len(traj) < 2:
            continue
        length_score = 0.0 if length_max <= 1e-12 else float(item["line_length"] / length_max)
        actual_rgb = color_from_score(length_score, [0.10, 0.25, 0.55], [0.20, 0.95, 0.25])
        mgm.gen_stick(traj[0], traj[-1], radius=0.0014, rgb=actual_rgb, alpha=0.28).attach_to(base)
        mgm.gen_sphere(traj[-1], radius=0.0018, rgb=actual_rgb, alpha=0.35).attach_to(base)

    mgm.gen_arrow(start_pos, start_pos + best_pred["direction"] * 0.09, stick_radius=0.0016, rgb=[0.98, 0.70, 0.10]).attach_to(base)
    mgm.gen_arrow(start_pos, start_pos + best_accum["direction"] * 0.10, stick_radius=0.0016, rgb=[0.92, 0.18, 0.88]).attach_to(base)
    mgm.gen_arrow(start_pos, start_pos + best_actual["direction"] * 0.09, stick_radius=0.0018, rgb=[0.20, 0.90, 0.25]).attach_to(base)

    best_traj = np.asarray(best_actual["trace"]["traj_pos"], dtype=np.float64)
    if len(best_traj) >= 2:
        mgm.gen_stick(best_traj[0], best_traj[-1], radius=0.0024, rgb=[0.20, 0.95, 0.25]).attach_to(base)
        mgm.gen_sphere(best_traj[-1], radius=0.0026, rgb=[0.20, 0.95, 0.25]).attach_to(base)

    print(f"robot: {args.robot}")
    print(f"sample_attempt: {sample_attempt}")
    print(f"start_q: {np.array2string(start_q, precision=4, separator=', ')}")
    print(f"start_pos: {np.array2string(start_pos, precision=4, separator=', ')}")
    print(f"plane_normal: {np.array2string(plane_normal, precision=4, separator=', ')}")
    print(f"jacobian_rank: {int(np.linalg.matrix_rank(j_pos))}")
    print(f"position_singular_values: {np.array2string(np.sqrt(np.clip(eigvals, 0.0, None))[::-1], precision=5, separator=', ')}")
    print(f"best_pred_direction: {np.array2string(best_pred['direction'], precision=4, separator=', ')}")
    print(f"best_pred_manipulability: {best_pred['manip']:.6f}")
    print(f"best_pred_actual_length: {best_pred['line_length']:.4f} m")
    print(f"best_accum_direction: {np.array2string(best_accum['direction'], precision=4, separator=', ')}")
    print(f"best_accum_manipulability: {best_accum['accum_manip']:.6f}")
    print(f"best_accum_actual_length: {best_accum['line_length']:.4f} m")
    print(f"best_actual_direction: {np.array2string(best_actual['direction'], precision=4, separator=', ')}")
    print(f"best_actual_length: {best_actual['line_length']:.4f} m")
    print(f"best_actual_manipulability: {best_actual['manip']:.6f}")
    print(f"best_actual_accum_manipulability: {best_actual['accum_manip']:.6f}")
    print(f"pred_vs_actual_angle_gap_deg: {angle_between_dirs(best_pred['direction'], best_actual['direction']):.3f}")
    print(f"accum_vs_actual_angle_gap_deg: {angle_between_dirs(best_accum['direction'], best_actual['direction']):.3f}")
    print(f"pearson_corr(manipulability, line_length): {pearson_corr(manip_scores, line_lengths):.4f}")
    print(f"pearson_corr(accum_manipulability, line_length): {pearson_corr(accum_scores, line_lengths):.4f}")
    print(f"actual_rank_of_pred_best: {actual_rank_of_pred_best}/{len(entries)}")
    print(f"pred_rank_of_actual_best: {pred_rank_of_actual_best}/{len(entries)}")
    print(f"actual_rank_of_accum_best: {actual_rank_of_accum_best}/{len(entries)}")
    print(f"accum_rank_of_actual_best: {accum_rank_of_actual_best}/{len(entries)}")
    print(f"best_actual_termination_reason: {best_actual['trace']['termination_reason']}")
    print("visualization: green spheres = instantaneous Jacobian manipulability, purple spheres = accumulated Jacobian manipulability, blue-to-green segments = actual reachable straight-line length")

    anime_robot = make_robot(args.robot, enable_cc=False)
    visualize_anime_path(base, anime_robot, np.asarray(best_actual['trace']['traj_q'], dtype=np.float64))


if __name__ == "__main__":
    main()
