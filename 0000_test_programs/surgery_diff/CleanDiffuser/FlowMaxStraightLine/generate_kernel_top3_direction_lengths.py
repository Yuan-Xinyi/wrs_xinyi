import argparse
import shutil
import time
from pathlib import Path

import h5py
import numpy as np

from visualize_tcpz_orthogonal_sample import (
    CONTOUR_PATH,
    make_robot,
    normalize_vec,
    sample_orientation_rotmats,
    sample_plane_directions,
    solve_orientation_candidates,
)
from xarm_trail1 import MAX_STEPS, STEP_SIZE, WorkspaceContour, is_pose_inside_workspace, trace_line_by_ik


BASE_DIR = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine")
DATASET_DIR = BASE_DIR / "datasets"
CHECKPOINT_DIR = DATASET_DIR / "checkpoints_kernel_top3_direction_lengths"
INPUT_NPY_PATH = DATASET_DIR / "cvt_kernels_collision_free_10000.npy"
OUTPUT_H5_PATH = DATASET_DIR / "xarm_tcpz_orthogonal_kernel_top3_10000.h5"
TOP_K = 3
CHECKPOINT_INTERVAL = 100


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate each kernel with the current tcpz-orthogonal script logic and save top-3 direction lengths."
    )
    parser.add_argument("--input-npy", type=Path, default=INPUT_NPY_PATH)
    parser.add_argument("--output-h5", type=Path, default=OUTPUT_H5_PATH)
    parser.add_argument("--robot", type=str, default="xarm", choices=["xarm", "franka"])
    parser.add_argument("--sampling-mode", type=str, default="orientation_cone", choices=["orthogonal_circle", "orientation_cone"])
    parser.add_argument("--num-direction-samples", type=int, default=12)
    parser.add_argument("--num-orientation-samples", type=int, default=12)
    parser.add_argument("--cone-angle-deg", type=float, default=60.0)
    parser.add_argument("--position-tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260325)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def create_or_open_h5(output_path: Path, num_kernels: int, resume: bool, checkpoint_interval: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and output_path.exists() else "w"
    h5_file = h5py.File(output_path, mode)

    if "kernel_q" not in h5_file:
        h5_file.create_dataset("kernel_q", shape=(num_kernels, 6), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("kernel_start_pos", shape=(num_kernels, 3), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("plane_normal_fixed", shape=(num_kernels, 3), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("top_direction_vec", shape=(num_kernels, TOP_K, 3), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("top_line_length", shape=(num_kernels, TOP_K), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("top_num_success_steps", shape=(num_kernels, TOP_K), dtype=np.int32, compression="gzip", shuffle=True)
        h5_file.create_dataset("top_termination_reason", shape=(num_kernels, TOP_K), dtype="S32", compression="gzip", shuffle=True)
        h5_file.create_dataset("top_candidate_q", shape=(num_kernels, TOP_K, 6), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("top_candidate_tcp_z_axis", shape=(num_kernels, TOP_K, 3), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("top_orientation_idx", shape=(num_kernels, TOP_K), dtype=np.int32, compression="gzip", shuffle=True)
        h5_file.create_dataset("top_valid_mask", shape=(num_kernels, TOP_K), dtype=np.bool_, compression="gzip", shuffle=True)
        h5_file.create_dataset("top_valid_count", shape=(num_kernels,), dtype=np.int32, compression="gzip", shuffle=True)
        h5_file.create_dataset("feasible_orientation_candidates", shape=(num_kernels,), dtype=np.int32, compression="gzip", shuffle=True)
        h5_file.create_dataset("kernel_status", shape=(num_kernels,), dtype="S32", compression="gzip", shuffle=True)
        h5_file.create_dataset("done_mask", shape=(num_kernels,), dtype=np.bool_, compression="gzip", shuffle=True)
        h5_file["done_mask"][:] = False
        h5_file["top_valid_mask"][:] = False
        h5_file["top_valid_count"][:] = 0
        h5_file["feasible_orientation_candidates"][:] = 0
        h5_file["kernel_status"][:] = b"pending"
        h5_file.attrs["created_unix_time"] = time.time()

    h5_file.attrs["step_size"] = STEP_SIZE
    h5_file.attrs["max_steps"] = MAX_STEPS
    h5_file.attrs["top_k"] = TOP_K
    h5_file.attrs["contour_path"] = str(CONTOUR_PATH)
    h5_file.attrs["checkpoint_interval"] = int(checkpoint_interval)
    return h5_file


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
        f"[Checkpoint] completed={done_count}/{h5_file['done_mask'].shape[0]} | "
        f"elapsed={elapsed / 60.0:.1f} min | file={output_path} | latest={latest_path} | snapshot={indexed_path}",
        flush=True,
    )


def build_orientation_candidates(robot, start_q, start_pos, start_rot, rng, sampling_mode, num_orientation_samples, cone_angle_deg, position_tol):
    plane_normal_fixed = normalize_vec(start_rot[:, 2])
    if sampling_mode == "orientation_cone":
        orientation_rotmats = sample_orientation_rotmats(
            start_rot=np.asarray(start_rot, dtype=np.float64),
            num_samples=num_orientation_samples,
            cone_angle_deg=cone_angle_deg,
            rng=rng,
        )
        candidates = solve_orientation_candidates(
            robot=robot,
            start_pos=np.asarray(start_pos, dtype=np.float64),
            orientation_rotmats=np.asarray(orientation_rotmats, dtype=np.float64),
            position_tol=position_tol,
        )
    else:
        candidates = [
            {
                "orientation_idx": 0,
                "q": np.asarray(start_q, dtype=np.float64).copy(),
                "start_rot": np.asarray(start_rot, dtype=np.float64).copy(),
                "tcp_z_axis": plane_normal_fixed.copy(),
            }
        ]
    return plane_normal_fixed, candidates


def collect_one_kernel(kernel_idx: int, kernel_q: np.ndarray, robot, contour, rng, sampling_mode, num_direction_samples, num_orientation_samples, cone_angle_deg, position_tol):
    kernel_q = np.asarray(kernel_q, dtype=np.float64)
    start_pos, start_rot = robot.fk(kernel_q)
    start_pos = np.asarray(start_pos, dtype=np.float64)
    start_rot = np.asarray(start_rot, dtype=np.float64)

    result = {
        "kernel_idx": int(kernel_idx),
        "kernel_q": kernel_q,
        "kernel_start_pos": start_pos,
        "plane_normal_fixed": normalize_vec(start_rot[:, 2]),
        "feasible_orientation_candidates": 0,
        "status": "ok",
        "top_entries": [],
    }

    if not is_pose_inside_workspace(contour, start_pos):
        result["status"] = "outside_workspace"
        return result

    robot.goto_given_conf(kernel_q)
    if robot.is_collided():
        result["status"] = "kernel_collision"
        return result

    line_directions = sample_plane_directions(result["plane_normal_fixed"], num_direction_samples, rng=rng)
    plane_normal_fixed, orientation_candidates = build_orientation_candidates(
        robot=robot,
        start_q=kernel_q,
        start_pos=start_pos,
        start_rot=start_rot,
        rng=rng,
        sampling_mode=sampling_mode,
        num_orientation_samples=num_orientation_samples,
        cone_angle_deg=cone_angle_deg,
        position_tol=position_tol,
    )
    result["plane_normal_fixed"] = plane_normal_fixed
    result["feasible_orientation_candidates"] = len(orientation_candidates)
    if len(orientation_candidates) == 0:
        result["status"] = "no_orientation_candidate"
        return result

    scored_entries = []
    for orientation_candidate in orientation_candidates:
        for direction in line_directions:
            trace = trace_line_by_ik(
                robot=robot,
                contour=contour,
                start_q=np.asarray(orientation_candidate["q"], dtype=np.float64),
                direction=np.asarray(direction, dtype=np.float64),
                step_size=STEP_SIZE,
                max_steps=MAX_STEPS,
            )
            scored_entries.append(
                {
                    "direction_vec": np.asarray(direction, dtype=np.float64),
                    "line_length": float(trace["line_length"]),
                    "num_success_steps": int(trace["num_success_steps"]),
                    "termination_reason": str(trace["termination_reason"]),
                    "candidate_q": np.asarray(orientation_candidate["q"], dtype=np.float64),
                    "candidate_tcp_z_axis": np.asarray(orientation_candidate["tcp_z_axis"], dtype=np.float64),
                    "orientation_idx": int(orientation_candidate["orientation_idx"]),
                }
            )

    scored_entries.sort(
        key=lambda item: (
            float(item["line_length"]),
            -int(item["orientation_idx"]),
        ),
        reverse=True,
    )
    result["top_entries"] = scored_entries[:TOP_K]
    if len(result["top_entries"]) == 0:
        result["status"] = "no_direction_evaluated"
    return result


def write_kernel_result(h5_file, result: dict):
    idx = int(result["kernel_idx"])
    h5_file["kernel_q"][idx] = np.asarray(result["kernel_q"], dtype=np.float32)
    h5_file["kernel_start_pos"][idx] = np.asarray(result["kernel_start_pos"], dtype=np.float32)
    h5_file["plane_normal_fixed"][idx] = np.asarray(result["plane_normal_fixed"], dtype=np.float32)
    h5_file["feasible_orientation_candidates"][idx] = int(result["feasible_orientation_candidates"])
    h5_file["kernel_status"][idx] = str(result["status"]).encode("ascii", errors="ignore")

    valid_count = min(len(result["top_entries"]), TOP_K)
    h5_file["top_valid_count"][idx] = int(valid_count)
    h5_file["top_valid_mask"][idx] = False

    direction_block = np.full((TOP_K, 3), np.nan, dtype=np.float32)
    length_block = np.full((TOP_K,), np.nan, dtype=np.float32)
    steps_block = np.zeros((TOP_K,), dtype=np.int32)
    reason_block = np.asarray([b""] * TOP_K, dtype="S32")
    candidate_q_block = np.full((TOP_K, 6), np.nan, dtype=np.float32)
    candidate_axis_block = np.full((TOP_K, 3), np.nan, dtype=np.float32)
    orientation_idx_block = np.full((TOP_K,), -1, dtype=np.int32)

    for slot_idx, entry in enumerate(result["top_entries"][:TOP_K]):
        direction_block[slot_idx] = np.asarray(entry["direction_vec"], dtype=np.float32)
        length_block[slot_idx] = float(entry["line_length"])
        steps_block[slot_idx] = int(entry["num_success_steps"])
        reason_block[slot_idx] = str(entry["termination_reason"]).encode("ascii", errors="ignore")
        candidate_q_block[slot_idx] = np.asarray(entry["candidate_q"], dtype=np.float32)
        candidate_axis_block[slot_idx] = np.asarray(entry["candidate_tcp_z_axis"], dtype=np.float32)
        orientation_idx_block[slot_idx] = int(entry["orientation_idx"])

    h5_file["top_direction_vec"][idx] = direction_block
    h5_file["top_line_length"][idx] = length_block
    h5_file["top_num_success_steps"][idx] = steps_block
    h5_file["top_termination_reason"][idx] = reason_block
    h5_file["top_candidate_q"][idx] = candidate_q_block
    h5_file["top_candidate_tcp_z_axis"][idx] = candidate_axis_block
    h5_file["top_orientation_idx"][idx] = orientation_idx_block
    h5_file["top_valid_mask"][idx, :valid_count] = True
    h5_file["done_mask"][idx] = True


def main():
    args = parse_args()
    if not args.input_npy.exists():
        raise FileNotFoundError(f"Input kernel file not found: {args.input_npy}")

    kernel_qs = np.asarray(np.load(args.input_npy), dtype=np.float64)
    num_kernels = int(kernel_qs.shape[0])
    start_idx = max(0, int(args.start_idx))
    end_idx = num_kernels if args.end_idx is None else min(int(args.end_idx), num_kernels)
    if start_idx >= end_idx:
        raise ValueError(f"Invalid range: start_idx={start_idx}, end_idx={end_idx}, num_kernels={num_kernels}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    h5_file = create_or_open_h5(
        output_path=args.output_h5,
        num_kernels=num_kernels,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
    )
    h5_file.attrs["input_npy"] = str(args.input_npy)
    h5_file.attrs["robot"] = args.robot
    h5_file.attrs["sampling_mode"] = args.sampling_mode
    h5_file.attrs["num_direction_samples"] = int(args.num_direction_samples)
    h5_file.attrs["num_orientation_samples"] = int(args.num_orientation_samples)
    h5_file.attrs["cone_angle_deg"] = float(args.cone_angle_deg)
    h5_file.attrs["position_tol"] = float(args.position_tol)
    h5_file.attrs["seed"] = int(args.seed)

    done_mask = h5_file["done_mask"][:]
    target_indices = np.arange(start_idx, end_idx, dtype=np.int64)
    pending_indices = target_indices[~done_mask[target_indices]]

    print(f"[Config] input={args.input_npy}", flush=True)
    print(f"[Config] output={args.output_h5}", flush=True)
    print(f"[Config] robot={args.robot} | sampling_mode={args.sampling_mode}", flush=True)
    print(f"[Config] kernels={num_kernels} | pending={len(pending_indices)}", flush=True)
    print(f"[Config] start_idx={start_idx} | end_idx={end_idx}", flush=True)
    print(
        f"[Config] top_k={TOP_K} | num_direction_samples={args.num_direction_samples} | "
        f"num_orientation_samples={args.num_orientation_samples} | cone_angle_deg={args.cone_angle_deg:.1f}",
        flush=True,
    )

    if len(pending_indices) == 0:
        print("[INFO] No pending kernels in the requested range.", flush=True)
        h5_file.close()
        return

    contour = WorkspaceContour(contour_path=str(CONTOUR_PATH), z_value=0.0)
    robot = make_robot(args.robot, enable_cc=True)
    started_at = time.time()
    processed_since_flush = 0
    done_before = int(np.count_nonzero(done_mask))
    rng = np.random.default_rng(args.seed + start_idx)

    try:
        for processed_idx, kernel_idx in enumerate(pending_indices, start=1):
            kernel_started = time.time()
            result = collect_one_kernel(
                kernel_idx=int(kernel_idx),
                kernel_q=kernel_qs[int(kernel_idx)],
                robot=robot,
                contour=contour,
                rng=rng,
                sampling_mode=args.sampling_mode,
                num_direction_samples=args.num_direction_samples,
                num_orientation_samples=args.num_orientation_samples,
                cone_angle_deg=args.cone_angle_deg,
                position_tol=args.position_tol,
            )
            write_kernel_result(h5_file, result)
            processed_since_flush += 1

            kernel_dt = time.time() - kernel_started
            elapsed = time.time() - started_at
            total_done = done_before + processed_idx
            best_length = 0.0 if len(result["top_entries"]) == 0 else float(result["top_entries"][0]["line_length"])
            print(
                f"[Progress] processed={processed_idx}/{len(pending_indices)} | "
                f"done_total={total_done}/{num_kernels} | kernel_dt={kernel_dt:.1f}s | "
                f"elapsed={elapsed / 60.0:.1f} min | valid={len(result['top_entries'])} | "
                f"feasible_candidates={result['feasible_orientation_candidates']} | "
                f"best={best_length:.3f} | status={result['status']}",
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
