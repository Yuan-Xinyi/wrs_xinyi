import argparse
import io
import os
import shutil
import time
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np

import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
from visualize_nullspace_feasible_anime import normalize_vec, solve_nullspace_candidates
from xarm_trail1 import (
    MAX_STEPS,
    STEP_SIZE,
    WorkspaceContour,
    is_pose_inside_workspace,
    trace_line_by_ik,
)

BASE_DIR = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine")
DATASET_DIR = BASE_DIR / "datasets"
CHECKPOINT_DIR = DATASET_DIR / "checkpoints_nullspace_straight"
INPUT_KERNEL_PATH = DATASET_DIR / "cvt_kernels_collision_free.npy"
OUTPUT_H5_PATH = DATASET_DIR / "xarm_nullspace_straight_top10.h5"
CONTOUR_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl")

TOP_K = 10
MIN_LINE_LENGTH = 0.1
NUM_DIRECTION_SAMPLES = 12
CHECKPOINT_INTERVAL = 100
SEED = 20260324

SVD_TOL = 1e-6
AMPLITUDE = 0.35
GRID_RESOLUTION = 2
PROJECTION_ITERS = 60
PROJECTION_DAMPING = 1e-4
POSITION_TOL = 1e-5
CONE_ANGLE_DEG = 60.0
DEDUP_TOL = 1e-3
KEEP_CANDIDATES = 4000
MAX_JOINT_STEP_NORM = 1.0
def parse_args():
    parser = argparse.ArgumentParser(description="Collect nullspace-conditioned straight-line dataset from collision-free XArm kernels.")
    parser.add_argument("--input-npy", type=Path, default=INPUT_KERNEL_PATH)
    parser.add_argument("--output-h5", type=Path, default=OUTPUT_H5_PATH)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--start-idx", type=int, default=None, help="Inclusive kernel index lower bound.")
    parser.add_argument("--end-idx", type=int, default=None, help="Exclusive kernel index upper bound.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing HDF5 if present.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--min-line-length", type=float, default=MIN_LINE_LENGTH)
    parser.add_argument("--num-direction-samples", type=int, default=NUM_DIRECTION_SAMPLES)
    parser.add_argument("--svd-tol", type=float, default=SVD_TOL)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE)
    parser.add_argument("--grid-resolution", type=int, default=GRID_RESOLUTION)
    parser.add_argument("--projection-iters", type=int, default=PROJECTION_ITERS)
    parser.add_argument("--projection-damping", type=float, default=PROJECTION_DAMPING)
    parser.add_argument("--position-tol", type=float, default=POSITION_TOL)
    parser.add_argument("--cone-angle-deg", type=float, default=CONE_ANGLE_DEG)
    parser.add_argument("--dedup-tol", type=float, default=DEDUP_TOL)
    parser.add_argument("--keep-candidates", type=int, default=KEEP_CANDIDATES)
    parser.add_argument("--max-joint-step-norm", type=float, default=MAX_JOINT_STEP_NORM)
    parser.add_argument("--collision-check-substeps", type=int, default=10)
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


def evaluate_best_direction(robot, contour, start_q: np.ndarray, directions: np.ndarray, max_joint_step_norm: float, collision_check_substeps: int):
    best_result = None
    best_direction = None
    reason_counts = {}
    for direction in directions:
        result = trace_line_by_ik(
            robot=robot,
            contour=contour,
            start_q=np.asarray(start_q, dtype=np.float64),
            direction=np.asarray(direction, dtype=np.float64),
            step_size=STEP_SIZE,
            max_steps=MAX_STEPS,
            max_joint_step_norm=max_joint_step_norm,
            collision_check_substeps=collision_check_substeps,
        )
        reason = str(result["termination_reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if best_result is None or result["line_length"] > best_result["line_length"]:
            best_result = result
            best_direction = np.asarray(direction, dtype=np.float64)
    return best_direction, best_result, reason_counts


def build_empty_kernel_result(kernel_idx: int, base_start_q: np.ndarray, start_pos: np.ndarray, reason_counts: dict):
    base_start_qs = np.full((TOP_K, 6), np.nan, dtype=np.float32)
    start_qs = np.full((TOP_K, 6), np.nan, dtype=np.float32)
    start_poss = np.full((TOP_K, 3), np.nan, dtype=np.float32)
    plane_normals = np.full((TOP_K, 3), np.nan, dtype=np.float32)
    tcp_z_axes = np.full((TOP_K, 3), np.nan, dtype=np.float32)
    direction_vecs = np.full((TOP_K, 3), np.nan, dtype=np.float32)
    line_lengths = np.full((TOP_K,), np.nan, dtype=np.float32)
    num_success_steps = np.zeros((TOP_K,), dtype=np.int32)
    termination_reasons = np.full((TOP_K,), b"", dtype="S32")
    max_joint_step_norms = np.full((TOP_K,), np.nan, dtype=np.float32)
    axis_error_deg = np.full((TOP_K,), np.nan, dtype=np.float32)
    on_boundary = np.zeros((TOP_K,), dtype=np.bool_)
    valid_mask = np.zeros((TOP_K,), dtype=np.bool_)
    return {
        "kernel_idx": int(kernel_idx),
        "valid_count": 0,
        "feasible_candidate_count": 0,
        "termination_reason_counts": reason_counts,
        "base_start_qs": base_start_qs,
        "start_qs": start_qs,
        "start_poss": start_poss,
        "plane_normals": plane_normals,
        "tcp_z_axes": tcp_z_axes,
        "direction_vecs": direction_vecs,
        "line_lengths": line_lengths,
        "num_success_steps": num_success_steps,
        "termination_reasons": termination_reasons,
        "max_joint_step_norms": max_joint_step_norms,
        "axis_error_deg": axis_error_deg,
        "on_boundary": on_boundary,
        "valid_mask": valid_mask,
    }


def collect_kernel_records(kernel_idx: int, kernel_q: np.ndarray, robot, contour, args):
    rng = np.random.default_rng(args.seed + int(kernel_idx))
    base_start_q = np.asarray(kernel_q, dtype=np.float64)
    start_pos, start_rot = robot.fk(base_start_q)
    start_pos = np.asarray(start_pos, dtype=np.float64)
    start_rot = np.asarray(start_rot, dtype=np.float64)

    candidate_records = []
    reason_counts = {}
    if not is_pose_inside_workspace(contour, start_pos):
        return build_empty_kernel_result(kernel_idx, base_start_q, start_pos, reason_counts)

    robot.goto_given_conf(base_start_q)
    if robot.is_collided():
        return build_empty_kernel_result(kernel_idx, base_start_q, start_pos, reason_counts)

    plane_normal = normalize_vec(start_rot[:, 2])
    line_directions = sample_plane_directions(plane_normal, args.num_direction_samples, rng=rng)
    with redirect_stdout(io.StringIO()):
        _, orientation_candidates = solve_nullspace_candidates(
            robot=robot,
            start_q=base_start_q,
            start_pos=start_pos,
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
    feasible_candidate_count = len(orientation_candidates)

    for orientation_candidate in orientation_candidates:
        best_direction, best_result, candidate_reason_counts = evaluate_best_direction(
            robot=robot,
            contour=contour,
            start_q=orientation_candidate["q"],
            directions=line_directions,
            max_joint_step_norm=args.max_joint_step_norm,
            collision_check_substeps=args.collision_check_substeps,
        )
        for reason, count in candidate_reason_counts.items():
            reason_counts[reason] = reason_counts.get(reason, 0) + int(count)
        if best_result is None:
            continue
        if float(best_result["line_length"]) < float(args.min_line_length):
            continue
        candidate_records.append(
            {
                "base_start_q": base_start_q.copy(),
                "start_q": np.asarray(orientation_candidate["q"], dtype=np.float64),
                "start_pos": start_pos.copy(),
                "plane_normal": plane_normal.copy(),
                "tcp_z_axis": np.asarray(orientation_candidate["tcp_z_axis"], dtype=np.float64),
                "best_direction_vec": best_direction.copy(),
                "best_line_length": float(best_result["line_length"]),
                "best_num_success_steps": int(best_result["num_success_steps"]),
                "best_termination_reason": str(best_result["termination_reason"]),
                "best_max_joint_step_norm": float(best_result.get("max_joint_step_norm", 0.0)),
                "axis_error_deg": float(orientation_candidate.get("axis_error_deg", 0.0)),
                "on_boundary": bool(orientation_candidate.get("on_boundary", False)),
            }
        )

    candidate_records.sort(key=lambda item: item["best_line_length"], reverse=True)
    top_records = candidate_records[: args.top_k]

    base_start_qs = np.full((args.top_k, 6), np.nan, dtype=np.float32)
    start_qs = np.full((args.top_k, 6), np.nan, dtype=np.float32)
    start_poss = np.full((args.top_k, 3), np.nan, dtype=np.float32)
    plane_normals = np.full((args.top_k, 3), np.nan, dtype=np.float32)
    tcp_z_axes = np.full((args.top_k, 3), np.nan, dtype=np.float32)
    direction_vecs = np.full((args.top_k, 3), np.nan, dtype=np.float32)
    line_lengths = np.full((args.top_k,), np.nan, dtype=np.float32)
    num_success_steps = np.zeros((args.top_k,), dtype=np.int32)
    termination_reasons = np.full((args.top_k,), b"", dtype="S32")
    max_joint_step_norms = np.full((args.top_k,), np.nan, dtype=np.float32)
    axis_error_deg = np.full((args.top_k,), np.nan, dtype=np.float32)
    on_boundary = np.zeros((args.top_k,), dtype=np.bool_)
    valid_mask = np.zeros((args.top_k,), dtype=np.bool_)

    for write_idx, record in enumerate(top_records):
        base_start_qs[write_idx] = record["base_start_q"].astype(np.float32)
        start_qs[write_idx] = record["start_q"].astype(np.float32)
        start_poss[write_idx] = record["start_pos"].astype(np.float32)
        plane_normals[write_idx] = record["plane_normal"].astype(np.float32)
        tcp_z_axes[write_idx] = record["tcp_z_axis"].astype(np.float32)
        direction_vecs[write_idx] = record["best_direction_vec"].astype(np.float32)
        line_lengths[write_idx] = float(record["best_line_length"])
        num_success_steps[write_idx] = int(record["best_num_success_steps"])
        termination_reasons[write_idx] = record["best_termination_reason"].encode("ascii", errors="ignore")
        max_joint_step_norms[write_idx] = float(record["best_max_joint_step_norm"])
        axis_error_deg[write_idx] = float(record["axis_error_deg"])
        on_boundary[write_idx] = bool(record["on_boundary"])
        valid_mask[write_idx] = True

    return {
        "kernel_idx": int(kernel_idx),
        "valid_count": int(len(top_records)),
        "feasible_candidate_count": int(feasible_candidate_count),
        "termination_reason_counts": reason_counts,
        "base_start_qs": base_start_qs,
        "start_qs": start_qs,
        "start_poss": start_poss,
        "plane_normals": plane_normals,
        "tcp_z_axes": tcp_z_axes,
        "direction_vecs": direction_vecs,
        "line_lengths": line_lengths,
        "num_success_steps": num_success_steps,
        "termination_reasons": termination_reasons,
        "max_joint_step_norms": max_joint_step_norms,
        "axis_error_deg": axis_error_deg,
        "on_boundary": on_boundary,
        "valid_mask": valid_mask,
    }


def create_or_open_h5(output_path: Path, kernel_qs: np.ndarray, args):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and output_path.exists() else "w"
    h5_file = h5py.File(output_path, mode)

    num_kernels = int(kernel_qs.shape[0])
    top_k = int(args.top_k)
    if "kernel_qs" not in h5_file:
        h5_file.create_dataset("kernel_qs", data=kernel_qs.astype(np.float32), compression="gzip", shuffle=True)
        h5_file.create_dataset("top_base_start_q", shape=(num_kernels, top_k, 6), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("top_start_q", shape=(num_kernels, top_k, 6), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("top_start_pos", shape=(num_kernels, top_k, 3), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("top_plane_normal", shape=(num_kernels, top_k, 3), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("top_tcp_z_axis", shape=(num_kernels, top_k, 3), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("top_direction_vec", shape=(num_kernels, top_k, 3), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("top_line_length", shape=(num_kernels, top_k), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("top_num_success_steps", shape=(num_kernels, top_k), dtype=np.int32, compression="gzip", shuffle=True)
        h5_file.create_dataset("top_termination_reason", shape=(num_kernels, top_k), dtype="S32", compression="gzip", shuffle=True)
        h5_file.create_dataset("top_max_joint_step_norm", shape=(num_kernels, top_k), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("top_axis_error_deg", shape=(num_kernels, top_k), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("top_on_boundary", shape=(num_kernels, top_k), dtype=np.bool_, compression="gzip", shuffle=True)
        h5_file.create_dataset("top_valid_mask", shape=(num_kernels, top_k), dtype=np.bool_, compression="gzip", shuffle=True)
        h5_file.create_dataset("top_valid_count", shape=(num_kernels,), dtype=np.int32, compression="gzip", shuffle=True)
        h5_file.create_dataset("done_mask", shape=(num_kernels,), dtype=np.bool_, compression="gzip", shuffle=True)
        h5_file["done_mask"][:] = False
        h5_file.attrs["created_unix_time"] = time.time()

    h5_file.attrs["input_kernel_path"] = str(args.input_npy)
    h5_file.attrs["contour_path"] = str(CONTOUR_PATH)
    h5_file.attrs["checkpoint_interval"] = int(args.checkpoint_interval)
    h5_file.attrs["top_k"] = int(args.top_k)
    h5_file.attrs["min_line_length"] = float(args.min_line_length)
    h5_file.attrs["num_direction_samples"] = int(args.num_direction_samples)
    h5_file.attrs["step_size"] = float(STEP_SIZE)
    h5_file.attrs["max_steps"] = int(MAX_STEPS)
    h5_file.attrs["svd_tol"] = float(args.svd_tol)
    h5_file.attrs["amplitude"] = float(args.amplitude)
    h5_file.attrs["grid_resolution"] = int(args.grid_resolution)
    h5_file.attrs["projection_iters"] = int(args.projection_iters)
    h5_file.attrs["projection_damping"] = float(args.projection_damping)
    h5_file.attrs["position_tol"] = float(args.position_tol)
    h5_file.attrs["cone_angle_deg"] = float(args.cone_angle_deg)
    h5_file.attrs["dedup_tol"] = float(args.dedup_tol)
    h5_file.attrs["keep_candidates"] = int(args.keep_candidates)
    h5_file.attrs["max_joint_step_norm"] = float(args.max_joint_step_norm)
    h5_file.attrs["collision_check_substeps"] = int(args.collision_check_substeps)
    h5_file.attrs["execution_path"] = "collision_free_kernels_to_nullspace_straight_topk"
    return h5_file


def write_kernel_result(h5_file, result):
    idx = int(result["kernel_idx"])
    h5_file["top_base_start_q"][idx] = result["base_start_qs"]
    h5_file["top_start_q"][idx] = result["start_qs"]
    h5_file["top_start_pos"][idx] = result["start_poss"]
    h5_file["top_plane_normal"][idx] = result["plane_normals"]
    h5_file["top_tcp_z_axis"][idx] = result["tcp_z_axes"]
    h5_file["top_direction_vec"][idx] = result["direction_vecs"]
    h5_file["top_line_length"][idx] = result["line_lengths"]
    h5_file["top_num_success_steps"][idx] = result["num_success_steps"]
    h5_file["top_termination_reason"][idx] = result["termination_reasons"]
    h5_file["top_max_joint_step_norm"][idx] = result["max_joint_step_norms"]
    h5_file["top_axis_error_deg"][idx] = result["axis_error_deg"]
    h5_file["top_on_boundary"][idx] = result["on_boundary"]
    h5_file["top_valid_mask"][idx] = result["valid_mask"]
    h5_file["top_valid_count"][idx] = int(result["valid_count"])
    h5_file["done_mask"][idx] = True


def save_checkpoint_snapshots(output_path: Path, checkpoint_dir: Path, done_count: int):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.h5"
    indexed_path = checkpoint_dir / f"checkpoint_{done_count:05d}.h5"
    shutil.copy2(output_path, latest_path)
    if not indexed_path.exists():
        shutil.copy2(output_path, indexed_path)
    return latest_path, indexed_path


def flush_checkpoint(h5_file, output_path: Path, checkpoint_dir: Path, started_at: float):
    done_count = int(np.count_nonzero(h5_file["done_mask"][:]))
    h5_file.attrs["completed_kernels"] = done_count
    h5_file.attrs["last_flush_unix_time"] = time.time()
    h5_file.flush()
    latest_path, indexed_path = save_checkpoint_snapshots(output_path, checkpoint_dir, done_count)
    elapsed = time.time() - started_at
    print(
        f"[Checkpoint] completed={done_count}/{h5_file['done_mask'].shape[0]} | elapsed={elapsed / 60.0:.1f} min | "
        f"file={output_path} | latest={latest_path} | snapshot={indexed_path}",
        flush=True,
    )


def main():
    args = parse_args()
    if not args.input_npy.exists():
        raise FileNotFoundError(f"Input kernel file not found: {args.input_npy}")

    kernel_qs = np.load(args.input_npy).astype(np.float32)
    if kernel_qs.ndim != 2 or kernel_qs.shape[1] != 6:
        raise ValueError(f"Expected input kernel array shape [N, 6], got {kernel_qs.shape}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    h5_file = create_or_open_h5(args.output_h5, kernel_qs, args)
    done_mask = h5_file["done_mask"][:]
    pending_indices = np.flatnonzero(~done_mask)
    if args.start_idx is not None:
        pending_indices = pending_indices[pending_indices >= int(args.start_idx)]
    if args.end_idx is not None:
        pending_indices = pending_indices[pending_indices < int(args.end_idx)]

    print(f"[Config] input={args.input_npy}", flush=True)
    print(f"[Config] output={args.output_h5}", flush=True)
    print(f"[Config] kernels={len(kernel_qs)} | pending={len(pending_indices)}", flush=True)
    print(f"[Config] start_idx={args.start_idx} | end_idx={args.end_idx}", flush=True)
    print(f"[Config] execution=single_process | aligned_with=nullspace_straight_mode", flush=True)
    print(
        f"[Config] top_k={args.top_k} | num_direction_samples={args.num_direction_samples} | "
        f"step={STEP_SIZE} | max_steps={MAX_STEPS} | min_length={args.min_line_length} | "
        f"max_joint_step_norm={args.max_joint_step_norm} | collision_check_substeps={args.collision_check_substeps}",
        flush=True,
    )

    if len(pending_indices) == 0:
        print("[INFO] No pending kernels. The HDF5 dataset is already complete.", flush=True)
        h5_file.close()
        return

    started_at = time.time()
    processed_since_flush = 0
    total_done_before = int(np.count_nonzero(done_mask))

    contour = WorkspaceContour(contour_path=str(CONTOUR_PATH), z_value=0.0)
    robot = xarm6_sim.XArmLite6Miller(enable_cc=True)

    try:
        for completed_idx, kernel_idx in enumerate(pending_indices, start=1):
            kernel_started = time.time()
            result = collect_kernel_records(
                kernel_idx=int(kernel_idx),
                kernel_q=kernel_qs[int(kernel_idx)],
                robot=robot,
                contour=contour,
                args=args,
            )
            write_kernel_result(h5_file, result)
            processed_since_flush += 1

            total_done_now = total_done_before + completed_idx
            elapsed = time.time() - started_at
            kernel_dt = time.time() - kernel_started
            if result["termination_reason_counts"]:
                reason_summary = ", ".join(
                    f"{reason}:{count}" for reason, count in sorted(result["termination_reason_counts"].items())
                )
            else:
                reason_summary = "none"
            best_length = 0.0 if result["valid_count"] == 0 else float(np.nanmax(result["line_lengths"]))
            print(
                f"[Progress] processed={completed_idx}/{len(pending_indices)} | done_total={total_done_now}/{len(kernel_qs)} | "
                f"kernel_dt={kernel_dt:.1f}s | elapsed={elapsed / 60.0:.1f} min | valid={result['valid_count']} | "
                f"feasible_candidates={result['feasible_candidate_count']} | "
                f"best={best_length:.3f} | reasons={reason_summary}",
                flush=True,
            )

            if processed_since_flush >= args.checkpoint_interval:
                flush_checkpoint(h5_file, args.output_h5, CHECKPOINT_DIR, started_at)
                processed_since_flush = 0
    finally:
        flush_checkpoint(h5_file, args.output_h5, CHECKPOINT_DIR, started_at)
        h5_file.close()


if __name__ == "__main__":
    main()
