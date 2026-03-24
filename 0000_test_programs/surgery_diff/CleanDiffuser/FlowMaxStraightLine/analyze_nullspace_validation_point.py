import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
from wrs import wd
from validate_topstart_common import (
    BASE_DIR,
    DEFAULT_H5_PATH,
    create_timestamped_result_dir,
    load_flat_entries_from_h5,
    load_validation_entry_indices,
    to_jsonable,
)
from xarm_trail1 import MAX_STEPS, STEP_SIZE, WorkspaceContour, trace_line_by_ik


DEFAULT_BUNDLE_PATH = BASE_DIR / "flow_matching_topstart_runs" / "dit_rectifiedflow_q_from_posdir" / "bundle_best.pt"
DEFAULT_RESULTS_DIR = BASE_DIR / "nullspace_analysis_results"
CONTOUR_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze representative nullspace candidates for one fixed validation workspace position."
    )
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--val-rank", type=int, default=0, help="Index inside saved validation indices.")
    parser.add_argument("--random-sample", action="store_true", help="Randomly choose one validation entry instead of using --val-rank.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for random sample / direction generation.")
    parser.add_argument("--axis-x", type=float, default=None, help="Optional override for cone axis x. If omitted, use TCP pointing axis.")
    parser.add_argument("--axis-y", type=float, default=None, help="Optional override for cone axis y. If omitted, use TCP pointing axis.")
    parser.add_argument("--axis-z", type=float, default=None, help="Optional override for cone axis z. If omitted, use TCP pointing axis.")
    parser.add_argument("--normal-x", type=float, default=None, help="Deprecated alias for --axis-x.")
    parser.add_argument("--normal-y", type=float, default=None, help="Deprecated alias for --axis-y.")
    parser.add_argument("--normal-z", type=float, default=None, help="Deprecated alias for --axis-z.")
    parser.add_argument("--angle-deviation-deg", type=float, default=30.0, help="Allowed angular deviation around the cone axis.")
    parser.add_argument("--num-direction-samples", type=int, default=13, help="How many directions to sample inside the cone.")
    parser.add_argument("--svd-tol", type=float, default=1e-6)
    parser.add_argument("--amplitude", type=float, default=0.35, help="Max coefficient magnitude in nullspace coordinates.")
    parser.add_argument("--grid-resolution", type=int, default=9, help="Per-dimension grid size in nullspace coordinates.")
    parser.add_argument("--num-random-seeds", type=int, default=3000, help="Number of global random joint seeds for projection.")
    parser.add_argument("--keep-candidates", type=int, default=200, help="Number of representative candidates to keep after dedup.")
    parser.add_argument("--position-tol", type=float, default=1e-5, help="Meters. Strict target position tolerance after projection.")
    parser.add_argument("--projection-iters", type=int, default=60)
    parser.add_argument("--projection-damping", type=float, default=1e-4)
    parser.add_argument("--dedup-tol", type=float, default=1e-3)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize-geometry-only", action="store_true", help="Only visualize GT line, normal, and allowed angular sector. Skip nullspace analysis.")
    return parser.parse_args()


def wrap_angle_difference(delta):
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def rotmat_angle_error_deg(rot_a: np.ndarray, rot_b: np.ndarray):
    rel = rot_a.T @ rot_b
    trace_val = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(trace_val)))


def build_record(entries: dict, entry_idx: int):
    return {
        "entry_idx": int(entry_idx),
        "kernel_idx": int(entries["kernel_idx"][entry_idx]),
        "slot_idx": int(entries["slot_idx"][entry_idx]),
        "start_q": np.asarray(entries["start_q"][entry_idx], dtype=np.float64),
        "start_pos": np.asarray(entries["start_pos"][entry_idx], dtype=np.float64),
        "direction_vec": np.asarray(entries["direction_vec"][entry_idx], dtype=np.float64),
        "stored_line_length": float(entries["line_length"][entry_idx]),
    }


def normalize_vec(vec: np.ndarray):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Direction/normal vector has near-zero norm.")
    return vec / norm


def choose_entry_index(val_entry_indices: np.ndarray, val_rank: int, random_sample: bool, seed: int | None):
    if random_sample:
        rng = np.random.default_rng(seed)
        return int(rng.choice(val_entry_indices))
    val_rank = int(np.clip(val_rank, 0, len(val_entry_indices) - 1))
    return int(val_entry_indices[val_rank])


def make_orthonormal_basis(normal: np.ndarray):
    normal = normalize_vec(normal)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tangent1 = ref - np.dot(ref, normal) * normal
    tangent1 = normalize_vec(tangent1)
    tangent2 = np.cross(normal, tangent1)
    tangent2 = normalize_vec(tangent2)
    return normal, tangent1, tangent2


def resolve_cone_axis(args, tcp_pointing_axis: np.ndarray):
    if args.axis_x is not None and args.axis_y is not None and args.axis_z is not None:
        return normalize_vec(np.array([args.axis_x, args.axis_y, args.axis_z], dtype=np.float64))
    if args.normal_x is not None and args.normal_y is not None and args.normal_z is not None:
        return normalize_vec(np.array([args.normal_x, args.normal_y, args.normal_z], dtype=np.float64))
    return normalize_vec(np.asarray(tcp_pointing_axis, dtype=np.float64))


def sample_directions_in_cone(axis: np.ndarray, reference_direction: np.ndarray, angle_deviation_deg: float, num_samples: int):
    axis, tangent1, tangent2 = make_orthonormal_basis(axis)
    reference_direction = normalize_vec(reference_direction)
    ref_proj = reference_direction - np.dot(reference_direction, axis) * axis
    if np.linalg.norm(ref_proj) < 1e-12:
        ref_proj = tangent1
    else:
        ref_proj = normalize_vec(ref_proj)
    ref_x = np.dot(ref_proj, tangent1)
    ref_y = np.dot(ref_proj, tangent2)
    ref_angle = np.arctan2(ref_y, ref_x)

    if angle_deviation_deg <= 1e-8 or num_samples <= 1:
        return np.asarray([axis], dtype=np.float64)

    cone_angle = np.deg2rad(angle_deviation_deg)
    directions = [axis]
    ring_count = max(num_samples - 1, 1)
    for idx in range(ring_count):
        az = ref_angle + 2.0 * np.pi * idx / ring_count
        direction = (
            np.cos(cone_angle) * axis
            + np.sin(cone_angle) * (np.cos(az) * tangent1 + np.sin(az) * tangent2)
        )
        directions.append(normalize_vec(direction))
    return np.asarray(directions, dtype=np.float64)


def iter_with_progress(iterable, total: int | None, desc: str):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=False)
    return iterable


def compute_position_jacobian(robot, q: np.ndarray):
    robot.fk(np.asarray(q, dtype=np.float64), update=True)
    j_full = np.asarray(robot.jacobian(), dtype=np.float64)
    return j_full[:3, :]


def compute_nullspace_svd(robot, q: np.ndarray, tol: float):
    j_pos = compute_position_jacobian(robot, q)
    u, s, vh = np.linalg.svd(j_pos, full_matrices=True)
    rank = int(np.sum(s > tol))
    null_basis = vh.T[:, rank:]
    return {
        "jacobian": j_pos,
        "u": u,
        "s": s,
        "vh": vh,
        "rank": rank,
        "null_basis": null_basis,
    }


def damped_pseudoinverse_step(j_mat: np.ndarray, err: np.ndarray, damping: float):
    jj_t = j_mat @ j_mat.T
    damped = jj_t + (damping ** 2) * np.eye(jj_t.shape[0], dtype=np.float64)
    return j_mat.T @ np.linalg.solve(damped, err)


def project_to_target_position(
    robot,
    q_init: np.ndarray,
    target_pos: np.ndarray,
    position_tol: float,
    max_iters: int,
    damping: float,
):
    q = np.asarray(q_init, dtype=np.float64).copy()
    for iter_idx in range(max_iters):
        if not robot.are_jnts_in_ranges(q):
            return None
        pos, _ = robot.fk(q, update=True)
        pos = np.asarray(pos, dtype=np.float64)
        err = np.asarray(target_pos - pos, dtype=np.float64)
        if np.linalg.norm(err) <= position_tol:
            return {
                "q": q.copy(),
                "pos": pos.copy(),
                "iterations": int(iter_idx),
                "position_error": float(np.linalg.norm(err)),
            }
        j_pos = compute_position_jacobian(robot, q)
        dq = damped_pseudoinverse_step(j_pos, err, damping=damping)
        q = q + dq
    return None


def unique_projected_candidates(projected: list[dict], tol: float):
    unique = []
    for item in projected:
        q = np.asarray(item["q"], dtype=np.float64)
        duplicated = False
        for existing in unique:
            if np.max(np.abs(wrap_angle_difference(q - existing["q"]))) < tol:
                duplicated = True
                break
        if not duplicated:
            unique.append(item)
    return unique


def farthest_point_sampling_jointspace(candidates: list[dict], keep_count: int):
    if len(candidates) <= keep_count:
        return candidates
    q_mat = np.stack([np.asarray(item["q"], dtype=np.float64) for item in candidates], axis=0)
    lengths = np.asarray([item["line_length"] for item in candidates], dtype=np.float64)
    selected = [int(np.argmax(lengths))]
    min_dist = np.full(len(candidates), np.inf, dtype=np.float64)

    while len(selected) < keep_count:
        last_q = q_mat[selected[-1]]
        dist = np.linalg.norm(wrap_angle_difference(q_mat - last_q), axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected] = -np.inf
        next_idx = int(np.argmax(min_dist))
        if next_idx in selected or not np.isfinite(min_dist[next_idx]):
            break
        selected.append(next_idx)

    kept = [candidates[idx] for idx in selected]
    kept.sort(key=lambda item: item["line_length"], reverse=True)
    return kept


def select_common_direction(robot, contour, start_q: np.ndarray, direction_candidates: np.ndarray):
    best_result = None
    best_direction = None
    for direction in direction_candidates:
        result = trace_line_by_ik(
            robot=robot,
            contour=contour,
            start_q=np.asarray(start_q, dtype=np.float64),
            direction=np.asarray(direction, dtype=np.float64),
            step_size=STEP_SIZE,
            max_steps=MAX_STEPS,
        )
        if best_result is None or result["line_length"] > best_result["line_length"]:
            best_result = result
            best_direction = np.asarray(direction, dtype=np.float64)
    return best_direction, best_result


def evaluate_candidate(robot, contour, record: dict, q_candidate: np.ndarray):
    q_candidate = np.asarray(q_candidate, dtype=np.float64)
    pos, rot = robot.fk(q_candidate)
    gt_pos, gt_rot = record["fk_start_pos"], record["fk_start_rot"]
    best_direction = np.asarray(record["evaluation_direction_vec"], dtype=np.float64)
    best_result = trace_line_by_ik(
        robot=robot,
        contour=contour,
        start_q=q_candidate,
        direction=best_direction,
        step_size=STEP_SIZE,
        max_steps=MAX_STEPS,
    )
    return {
        "q": q_candidate,
        "start_pos": np.asarray(pos, dtype=np.float64),
        "start_rot": np.asarray(rot, dtype=np.float64),
        "position_error": float(np.linalg.norm(pos - gt_pos)),
        "rotation_error_deg": rotmat_angle_error_deg(gt_rot, rot),
        "joint_distance_to_gt": float(np.linalg.norm(wrap_angle_difference(q_candidate - record["start_q"]))),
        "line_length": float(best_result["line_length"]),
        "termination_reason": best_result["termination_reason"],
        "traj_pos": np.asarray(best_result["traj_pos"], dtype=np.float64),
        "num_success_steps": int(best_result["num_success_steps"]),
        "best_direction_vec": best_direction,
    }


def scan_nullspace_candidates(
    record: dict,
    amplitude: float,
    grid_resolution: int,
    pos_tol: float,
    svd_tol: float,
    projection_iters: int,
    projection_damping: float,
    dedup_tol: float,
    num_random_seeds: int,
    keep_candidates: int,
):
    contour = WorkspaceContour(contour_path=str(CONTOUR_PATH), z_value=0.0)
    robot = xarm6_sim.XArmLite6Miller(enable_cc=True)

    gt_pos, gt_rot = robot.fk(record["start_q"])
    record["fk_start_pos"] = np.asarray(gt_pos, dtype=np.float64)
    record["fk_start_rot"] = np.asarray(gt_rot, dtype=np.float64)

    svd_info = compute_nullspace_svd(robot, record["start_q"], tol=svd_tol)
    null_basis = svd_info["null_basis"]

    candidates = []
    projected = []
    if null_basis.shape[1] > 0:
        coeff_axis = np.linspace(-amplitude, amplitude, grid_resolution, dtype=np.float64)
        mesh = np.meshgrid(*([coeff_axis] * null_basis.shape[1]), indexing="ij")
        coeff_mat = np.stack([m.reshape(-1) for m in mesh], axis=1)
        for coeff in iter_with_progress(coeff_mat, total=len(coeff_mat), desc="Project local nullspace seeds"):
            q_seed = record["start_q"] + null_basis @ coeff
            projected_item = project_to_target_position(
                robot=robot,
                q_init=q_seed,
                target_pos=record["fk_start_pos"],
                position_tol=pos_tol,
                max_iters=projection_iters,
                damping=projection_damping,
            )
            if projected_item is None:
                continue
            projected_item["coeff"] = coeff.copy()
            projected_item["seed_type"] = "local_nullspace"
            projected.append(projected_item)

    jnt_ranges = np.asarray(robot.jnt_ranges, dtype=np.float64)
    for _ in iter_with_progress(range(num_random_seeds), total=num_random_seeds, desc="Project random joint seeds"):
        q_seed = np.random.uniform(jnt_ranges[:, 0], jnt_ranges[:, 1])
        projected_item = project_to_target_position(
            robot=robot,
            q_init=q_seed,
            target_pos=record["fk_start_pos"],
            position_tol=pos_tol,
            max_iters=projection_iters,
            damping=projection_damping,
        )
        if projected_item is None:
            continue
        projected_item["coeff"] = None
        projected_item["seed_type"] = "random_joint_seed"
        projected.append(projected_item)

    projected = unique_projected_candidates(projected, tol=dedup_tol)
    all_valid_projected = len(projected)
    for item in iter_with_progress(projected, total=len(projected), desc="Evaluate line lengths"):
        candidate = evaluate_candidate(robot, contour, record, item["q"])
        candidate["coeff"] = None if item["coeff"] is None else np.asarray(item["coeff"], dtype=np.float64)
        candidate["projection_iterations"] = int(item["iterations"])
        candidate["seed_type"] = item["seed_type"]
        if candidate["position_error"] <= pos_tol:
            candidates.append(candidate)
    candidates.sort(key=lambda item: item["line_length"], reverse=True)
    total_valid_candidates = len(candidates)
    candidates = farthest_point_sampling_jointspace(candidates, keep_count=keep_candidates)
    svd_info["all_valid_projected"] = int(all_valid_projected)
    svd_info["total_valid_candidates"] = int(total_valid_candidates)
    svd_info["kept_candidates"] = int(len(candidates))
    return svd_info, candidates


def color_from_score(score: float):
    score = float(np.clip(score, 0.0, 1.0))
    return [0.18 + 0.70 * (1.0 - score), 0.18 + 0.62 * score, 0.88 - 0.52 * score]


def save_stats_figure(record: dict, svd_info: dict, candidates: list[dict], fig_path: Path):
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    singular_values = svd_info["s"]
    null_basis = svd_info["null_basis"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.bar(np.arange(len(singular_values)), singular_values, color="#4c78a8", alpha=0.85)
    ax.set_title("Position Jacobian Singular Values")
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Magnitude")
    ax.grid(True, axis="y", alpha=0.22)

    ax = axes[0, 1]
    if null_basis.size > 0:
        im = ax.imshow(null_basis.T, aspect="auto", cmap="coolwarm")
        ax.set_title("Nullspace Basis Vectors")
        ax.set_xlabel("Joint index")
        ax.set_ylabel("Basis index")
        ax.set_xticks(np.arange(null_basis.shape[0]))
        ax.set_xticklabels([f"q{i}" for i in range(null_basis.shape[0])])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "No nullspace basis", ha="center", va="center", fontsize=14)
        ax.set_axis_off()

    ax = axes[1, 0]
    if candidates:
        coeff_norm = np.asarray(
            [np.linalg.norm(item["coeff"]) if item["coeff"] is not None else np.nan for item in candidates],
            dtype=np.float64,
        )
        coeff_norm = np.where(np.isfinite(coeff_norm), coeff_norm, 0.0)
        ys = np.asarray([item["line_length"] for item in candidates], dtype=np.float64)
        order = np.argsort(coeff_norm)
        ax.plot(coeff_norm[order], ys[order], linewidth=2.2, marker="o", markersize=3.0, color="#1b9e77")
        ax.axhline(record["stored_line_length"], color="#d95f02", linestyle="--", linewidth=2.0, label="dataset GT")
        ax.legend(frameon=True)
    ax.set_title("Line Length Across Projected Nullspace Candidates")
    ax.set_xlabel("||coeff|| in nullspace coordinates")
    ax.set_ylabel("Line length (m)")
    ax.grid(True, alpha=0.22)

    ax = axes[1, 1]
    if candidates:
        position_err = np.asarray([item["position_error"] for item in candidates], dtype=np.float64)
        line_length = np.asarray([item["line_length"] for item in candidates], dtype=np.float64)
        proj_iters = np.asarray([item["projection_iterations"] for item in candidates], dtype=np.float64)
        sc = ax.scatter(position_err, line_length, c=proj_iters, cmap="viridis", s=46)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="projection iters")
    ax.set_title("Projected Position Error vs Line Length")
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Line length (m)")
    ax.grid(True, alpha=0.22)

    fig.suptitle("Representative Nullspace Candidates Around Validation Workspace Point", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_json(record: dict, svd_info: dict, candidates: list[dict], json_path: Path):
    payload = {
        "record": to_jsonable(record),
        "rank": int(svd_info["rank"]),
        "singular_values": to_jsonable(svd_info["s"]),
        "nullspace_dim": int(svd_info["null_basis"].shape[1]),
        "all_valid_projected": int(svd_info.get("all_valid_projected", 0)),
        "total_valid_candidates": int(svd_info.get("total_valid_candidates", len(candidates))),
        "kept_candidates": int(svd_info.get("kept_candidates", len(candidates))),
        "num_valid_candidates": len(candidates),
        "best_candidate": to_jsonable(candidates[0]) if candidates else None,
        "candidates": to_jsonable(candidates),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def visualize_geometry_only(record: dict, title: str):
    base = wd.World(cam_pos=[1.15, 0.45, 0.5], lookat_pos=[0.25, 0.0, 0.0])
    mgm.gen_frame().attach_to(base)
    robot = xarm6_sim.XArmLite6Miller(enable_cc=False)
    contour_robot = xarm6_sim.XArmLite6Miller(enable_cc=True)
    contour = WorkspaceContour(contour_path=str(CONTOUR_PATH), z_value=0.0)

    robot.goto_given_conf(record["start_q"])
    robot.gen_meshmodel(rgb=[0.96, 0.48, 0.12], alpha=0.55).attach_to(base)
    mgm.gen_sphere(record["fk_start_pos"], radius=0.005, rgb=[0.95, 0.08, 0.08]).attach_to(base)

    axis_end = record["fk_start_pos"] + record["cone_axis_vec"] * 0.10
    mgm.gen_arrow(record["fk_start_pos"], axis_end, stick_radius=0.0022, rgb=[0.95, 0.15, 0.15]).attach_to(base)

    ref_end = record["fk_start_pos"] + record["reference_direction_vec"] * 0.09
    mgm.gen_arrow(record["fk_start_pos"], ref_end, stick_radius=0.0018, rgb=[0.98, 0.75, 0.12]).attach_to(base)
    for boundary_dir in record["cone_boundary_dirs"]:
        boundary_end = record["fk_start_pos"] + boundary_dir * 0.085
        mgm.gen_stick(record["fk_start_pos"], boundary_end, radius=0.0012, rgb=[0.95, 0.45, 0.45], alpha=0.35).attach_to(base)

    gt_direction = np.asarray(record["evaluation_direction_vec"], dtype=np.float64)
    gt_result = trace_line_by_ik(
        robot=contour_robot,
        contour=contour,
        start_q=record["start_q"],
        direction=gt_direction,
        step_size=STEP_SIZE,
        max_steps=MAX_STEPS,
    )
    gt_traj = np.asarray(gt_result["traj_pos"], dtype=np.float64)
    if len(gt_traj) >= 2:
        mgm.gen_stick(gt_traj[0], gt_traj[-1], radius=0.0028, rgb=[0.96, 0.48, 0.12]).attach_to(base)
    mgm.gen_arrow(
        record["fk_start_pos"],
        record["fk_start_pos"] + gt_direction * 0.075,
        stick_radius=0.0016,
        rgb=[0.96, 0.48, 0.12],
    ).attach_to(base)

    print(title)
    print(f"entry_idx={record['entry_idx']} | kernel_idx={record['kernel_idx']} | slot_idx={record['slot_idx']}")
    print(f"cone_axis_vec={np.array2string(record['cone_axis_vec'], precision=4, separator=', ')}")
    print(f"reference_direction_vec={np.array2string(record['reference_direction_vec'], precision=4, separator=', ')}")
    print(f"evaluation_direction_vec={np.array2string(record['evaluation_direction_vec'], precision=4, separator=', ')}")
    print(f"angle_deviation_deg={record['angle_deviation_deg']:.1f}")
    print(f"gt_line_length={gt_result['line_length']:.4f} m")
    base.run()


def visualize(record: dict, svd_info: dict, candidates: list[dict], title: str):
    base = wd.World(cam_pos=[1.15, 0.45, 0.5], lookat_pos=[0.25, 0.0, 0.0])
    mgm.gen_frame().attach_to(base)
    robot = xarm6_sim.XArmLite6Miller(enable_cc=False)

    robot.goto_given_conf(record["start_q"])
    robot.gen_meshmodel(rgb=[0.96, 0.48, 0.12], alpha=0.48).attach_to(base)
    mgm.gen_sphere(record["fk_start_pos"], radius=0.005, rgb=[0.95, 0.08, 0.08]).attach_to(base)
    axis_end = record["fk_start_pos"] + record["cone_axis_vec"] * 0.10
    mgm.gen_arrow(record["fk_start_pos"], axis_end, stick_radius=0.0022, rgb=[0.95, 0.15, 0.15]).attach_to(base)
    ref_end = record["fk_start_pos"] + record["reference_direction_vec"] * 0.09
    mgm.gen_arrow(record["fk_start_pos"], ref_end, stick_radius=0.0018, rgb=[0.98, 0.75, 0.12]).attach_to(base)
    for boundary_dir in record["cone_boundary_dirs"]:
        boundary_end = record["fk_start_pos"] + boundary_dir * 0.085
        mgm.gen_stick(record["fk_start_pos"], boundary_end, radius=0.0012, rgb=[0.95, 0.45, 0.45], alpha=0.35).attach_to(base)
    gt_robot = xarm6_sim.XArmLite6Miller(enable_cc=True)
    gt_contour = WorkspaceContour(contour_path=str(CONTOUR_PATH), z_value=0.0)
    gt_eval = evaluate_candidate(gt_robot, gt_contour, record, record["start_q"])
    gt_traj = np.asarray(gt_eval["traj_pos"], dtype=np.float64)
    if len(gt_traj) >= 2:
        mgm.gen_stick(gt_traj[0], gt_traj[-1], radius=0.0028, rgb=[0.96, 0.48, 0.12]).attach_to(base)
    mgm.gen_arrow(
        record["fk_start_pos"],
        record["fk_start_pos"] + gt_eval["best_direction_vec"] * 0.07,
        stick_radius=0.0015,
        rgb=[0.96, 0.48, 0.12],
    ).attach_to(base)

    if svd_info["null_basis"].size > 0:
        for item in candidates:
            q_vis = item["q"]
            tcp_pos = item["start_pos"]
            rgb = color_from_score(item["line_length"] / max(candidates[0]["line_length"], 1e-8))
            robot.goto_given_conf(q_vis)
            robot.gen_meshmodel(rgb=rgb, alpha=0.025).attach_to(base)
            mgm.gen_sphere(tcp_pos, radius=0.0018, rgb=rgb).attach_to(base)

    if candidates:
        best = candidates[0]
        robot.goto_given_conf(best["q"])
        robot.gen_meshmodel(rgb=[0.10, 0.78, 0.28], alpha=0.50).attach_to(base)
        traj = best["traj_pos"]
        if len(traj) >= 2:
            mgm.gen_stick(traj[0], traj[-1], radius=0.0028, rgb=[0.10, 0.78, 0.28]).attach_to(base)
        mgm.gen_arrow(
            record["fk_start_pos"],
            record["fk_start_pos"] + best["best_direction_vec"] * 0.075,
            stick_radius=0.0016,
            rgb=[0.10, 0.78, 0.28],
        ).attach_to(base)

    print(title)
    print(f"nullspace_dim={svd_info['null_basis'].shape[1]}")
    print(f"GT stored length: {record['stored_line_length']:.4f} m")
    print(
        f"cone_axis_vec={np.array2string(record['cone_axis_vec'], precision=4, separator=', ')} "
        f"| reference_dir={np.array2string(record['reference_direction_vec'], precision=4, separator=', ')} "
        f"| angle_deviation_deg={record['angle_deviation_deg']:.1f}"
    )
    if candidates:
        coeff_str = "None" if candidates[0]["coeff"] is None else np.array2string(candidates[0]["coeff"], precision=4, separator=', ')
        print(
            f"Best candidate coeff={coeff_str} "
            f"| line_length={candidates[0]['line_length']:.4f} m | pos_err={candidates[0]['position_error']:.5f} m"
        )
        print(f"common evaluation direction={np.array2string(record['evaluation_direction_vec'], precision=4, separator=', ')}")
    base.run()


def main():
    args = parse_args()
    results_dir = create_timestamped_result_dir(args.results_dir, prefix="nullspace_validation_analysis")

    entries = load_flat_entries_from_h5(args.h5_path)
    val_entry_indices = load_validation_entry_indices(args.bundle_path)
    entry_idx = choose_entry_index(val_entry_indices, args.val_rank, args.random_sample, args.seed)
    val_rank_matches = np.where(val_entry_indices == entry_idx)[0]
    val_rank = int(val_rank_matches[0]) if len(val_rank_matches) > 0 else -1
    record = build_record(entries, entry_idx)
    pose_probe_robot = xarm6_sim.XArmLite6Miller(enable_cc=False)
    fk_start_pos, fk_start_rot = pose_probe_robot.fk(record["start_q"])
    record["fk_start_pos"] = np.asarray(fk_start_pos, dtype=np.float64)
    record["fk_start_rot"] = np.asarray(fk_start_rot, dtype=np.float64)
    tcp_pointing_axis = np.asarray(record["fk_start_rot"][:, 2], dtype=np.float64)
    cone_axis_vec = resolve_cone_axis(args, tcp_pointing_axis)
    reference_direction = np.asarray(record["direction_vec"], dtype=np.float64)
    direction_candidates = sample_directions_in_cone(
        axis=cone_axis_vec,
        reference_direction=reference_direction,
        angle_deviation_deg=args.angle_deviation_deg,
        num_samples=args.num_direction_samples,
    )
    direction_robot = xarm6_sim.XArmLite6Miller(enable_cc=True)
    direction_contour = WorkspaceContour(contour_path=str(CONTOUR_PATH), z_value=0.0)
    evaluation_direction_vec, _ = select_common_direction(
        robot=direction_robot,
        contour=direction_contour,
        start_q=record["start_q"],
        direction_candidates=direction_candidates,
    )
    record["cone_axis_vec"] = cone_axis_vec
    ref_proj = reference_direction - np.dot(reference_direction, cone_axis_vec) * cone_axis_vec
    if np.linalg.norm(ref_proj) < 1e-12:
        _, tangent1, _ = make_orthonormal_basis(cone_axis_vec)
        ref_proj = tangent1
    record["reference_direction_vec"] = normalize_vec(ref_proj)
    record["angle_deviation_deg"] = float(args.angle_deviation_deg)
    record["direction_candidates"] = direction_candidates
    record["evaluation_direction_vec"] = evaluation_direction_vec
    if len(direction_candidates) >= 2:
        record["cone_boundary_dirs"] = np.asarray([direction_candidates[1], direction_candidates[-1]], dtype=np.float64)
    else:
        record["cone_boundary_dirs"] = direction_candidates

    if args.visualize_geometry_only:
        visualize_geometry_only(record, title="GT Geometry Visualization")
        return

    started = time.time()
    svd_info, candidates = scan_nullspace_candidates(
        record=record,
        amplitude=args.amplitude,
        grid_resolution=args.grid_resolution,
        pos_tol=args.position_tol,
        svd_tol=args.svd_tol,
        projection_iters=args.projection_iters,
        projection_damping=args.projection_damping,
        dedup_tol=args.dedup_tol,
        num_random_seeds=args.num_random_seeds,
        keep_candidates=args.keep_candidates,
    )
    elapsed = time.time() - started

    record["direction_name"] = ["x", "y", "z"][int(np.argmax(np.abs(record["direction_vec"])))]

    fig_path = results_dir / "nullspace_candidate_stats.png"
    json_path = results_dir / "nullspace_candidate_analysis.json"
    save_stats_figure(record, svd_info, candidates, fig_path)
    save_json(record, svd_info, candidates, json_path)

    print(f"results_dir: {results_dir}")
    print(f"entry_idx: {entry_idx} | val_rank: {val_rank}")
    print(f"kernel_idx: {record['kernel_idx']} | slot_idx: {record['slot_idx']}")
    print(f"start_pos: {np.array2string(record['start_pos'], precision=4, separator=', ')}")
    print(f"dataset_direction: {record['direction_name']} -> {np.array2string(record['direction_vec'], precision=4, separator=', ')}")
    print(
        f"cone_axis_vec: {np.array2string(record['cone_axis_vec'], precision=4, separator=', ')} "
        f"| reference_dir: {np.array2string(record['reference_direction_vec'], precision=4, separator=', ')} "
        f"| eval_dir: {np.array2string(record['evaluation_direction_vec'], precision=4, separator=', ')} "
        f"| angle_deviation_deg={record['angle_deviation_deg']:.1f} "
        f"| num_direction_samples={len(record['direction_candidates'])}"
    )
    print(f"stored_gt_length: {record['stored_line_length']:.4f} m")
    print(f"jacobian_rank: {svd_info['rank']} | nullspace_dim: {svd_info['null_basis'].shape[1]}")
    print(f"singular_values: {np.array2string(svd_info['s'], precision=5, separator=', ')}")
    print(
        f"all_valid_projected: {svd_info.get('all_valid_projected', 0)} | "
        f"total_valid_candidates: {svd_info.get('total_valid_candidates', len(candidates))} | "
        f"kept_candidates: {len(candidates)}"
    )
    if candidates:
        coeff_str = "None" if candidates[0]["coeff"] is None else np.array2string(candidates[0]["coeff"], precision=4, separator=', ')
        print(
            f"best_candidate_coeff={coeff_str} "
            f"| line_length={candidates[0]['line_length']:.4f} m "
            f"| pos_err={candidates[0]['position_error']:.5f} m "
            f"| rot_err_deg={candidates[0]['rotation_error_deg']:.3f} "
            f"| termination={candidates[0]['termination_reason']} "
            f"| seed_type={candidates[0]['seed_type']}"
        )
    print(f"analysis_time_s: {elapsed:.3f}")
    print(f"stats_figure: {fig_path}")
    print(f"json_path: {json_path}")

    if args.visualize:
        visualize(record, svd_info, candidates, title="Validation-Point SVD Nullspace Analysis")


if __name__ == "__main__":
    main()
