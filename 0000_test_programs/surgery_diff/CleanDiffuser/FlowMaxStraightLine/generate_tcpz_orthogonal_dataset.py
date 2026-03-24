import argparse
import shutil
import time
from pathlib import Path

import h5py
import numpy as np

import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
from xarm_trail1 import MAX_STEPS, STEP_SIZE, WorkspaceContour, is_pose_inside_workspace, trace_line_by_ik


BASE_DIR = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine")
DATASET_DIR = BASE_DIR / "datasets"
CHECKPOINT_DIR = DATASET_DIR / "checkpoints_tcpz_orthogonal"
CONTOUR_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl")
OUTPUT_H5_PATH = DATASET_DIR / "xarm_tcpz_orthogonal_best_direction_dataset.h5"

NUM_SAMPLES = 10000
NUM_DIRECTION_SAMPLES = 72
CHECKPOINT_INTERVAL = 100
MAX_START_ATTEMPTS = 5000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample q_start, search the longest straight-line direction d with a_z · d = 0, and save [q_start, d, L]."
    )
    parser.add_argument("--output-h5", type=Path, default=OUTPUT_H5_PATH)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--num-direction-samples", type=int, default=NUM_DIRECTION_SAMPLES)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--seed", type=int, default=20260324)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def normalize_vec(vec: np.ndarray):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Vector norm is too small.")
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


def sample_orthogonal_directions(axis: np.ndarray, num_samples: int, rng: np.random.Generator):
    axis, tangent_1, tangent_2 = make_orthonormal_basis(axis)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    angles = phase + np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False, dtype=np.float64)
    directions = [
        normalize_vec(np.cos(theta) * tangent_1 + np.sin(theta) * tangent_2)
        for theta in angles
    ]
    return np.asarray(directions, dtype=np.float64)


def sample_valid_start_conf(robot, contour, rng: np.random.Generator, max_attempts: int):
    jnt_ranges = np.asarray(robot.jnt_ranges, dtype=np.float64)
    for attempt_idx in range(1, max_attempts + 1):
        start_q = rng.uniform(jnt_ranges[:, 0], jnt_ranges[:, 1])
        start_pos, start_rot = robot.fk(start_q)
        if not is_pose_inside_workspace(contour, start_pos):
            continue
        robot.goto_given_conf(start_q)
        if robot.is_collided():
            continue
        return {
            "start_q": np.asarray(start_q, dtype=np.float64),
            "start_pos": np.asarray(start_pos, dtype=np.float64),
            "start_rot": np.asarray(start_rot, dtype=np.float64),
            "attempts": int(attempt_idx),
        }
    raise RuntimeError(f"Failed to sample a valid q_start after {max_attempts} attempts.")


def evaluate_best_direction(robot, contour, start_q: np.ndarray, directions: np.ndarray):
    best_result = None
    best_direction = None
    for direction in directions:
        result = trace_line_by_ik(
            robot=robot,
            contour=contour,
            start_q=start_q,
            direction=np.asarray(direction, dtype=np.float64),
            step_size=STEP_SIZE,
            max_steps=MAX_STEPS,
        )
        if best_result is None or result["line_length"] > best_result["line_length"]:
            best_result = result
            best_direction = np.asarray(direction, dtype=np.float64)
    return best_direction, best_result


def create_or_open_h5(output_path: Path, num_samples: int, resume: bool, checkpoint_interval: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and output_path.exists() else "w"
    h5_file = h5py.File(output_path, mode)

    if "start_q" not in h5_file:
        h5_file.create_dataset("start_q", shape=(num_samples, 6), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("start_pos", shape=(num_samples, 3), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("tcp_z_axis", shape=(num_samples, 3), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("best_direction_vec", shape=(num_samples, 3), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("best_line_length", shape=(num_samples,), dtype=np.float32, compression="gzip", shuffle=True, fillvalue=np.nan)
        h5_file.create_dataset("best_num_success_steps", shape=(num_samples,), dtype=np.int32, compression="gzip", shuffle=True)
        h5_file.create_dataset("best_termination_reason", shape=(num_samples,), dtype="S32", compression="gzip", shuffle=True)
        h5_file.create_dataset("sampling_attempts", shape=(num_samples,), dtype=np.int32, compression="gzip", shuffle=True)
        h5_file.create_dataset("done_mask", shape=(num_samples,), dtype=np.bool_, compression="gzip", shuffle=True)
        h5_file["done_mask"][:] = False
        h5_file.attrs["created_unix_time"] = time.time()

    h5_file.attrs["num_samples"] = int(num_samples)
    h5_file.attrs["step_size"] = STEP_SIZE
    h5_file.attrs["max_steps"] = MAX_STEPS
    h5_file.attrs["max_start_attempts"] = int(MAX_START_ATTEMPTS)
    h5_file.attrs["contour_path"] = str(CONTOUR_PATH)
    h5_file.attrs["checkpoint_interval"] = int(checkpoint_interval)
    h5_file.attrs["search_constraint"] = "tcp_z_axis_dot_direction_equals_zero"
    return h5_file


def write_sample_result(h5_file, sample_idx: int, result: dict):
    h5_file["start_q"][sample_idx] = result["start_q"].astype(np.float32)
    h5_file["start_pos"][sample_idx] = result["start_pos"].astype(np.float32)
    h5_file["tcp_z_axis"][sample_idx] = result["tcp_z_axis"].astype(np.float32)
    h5_file["best_direction_vec"][sample_idx] = result["best_direction_vec"].astype(np.float32)
    h5_file["best_line_length"][sample_idx] = float(result["best_line_length"])
    h5_file["best_num_success_steps"][sample_idx] = int(result["best_num_success_steps"])
    h5_file["best_termination_reason"][sample_idx] = str(result["best_termination_reason"]).encode("ascii", errors="ignore")
    h5_file["sampling_attempts"][sample_idx] = int(result["sampling_attempts"])
    h5_file["done_mask"][sample_idx] = True


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
    h5_file.attrs["completed_samples"] = done_count
    h5_file.attrs["last_flush_unix_time"] = time.time()
    h5_file.flush()
    latest_path, indexed_path = save_checkpoint_snapshots(output_path, checkpoint_dir, done_count)
    elapsed = time.time() - started_at
    print(
        f"[Checkpoint] completed={done_count}/{h5_file['done_mask'].shape[0]} | "
        f"elapsed={elapsed / 60.0:.1f} min | file={output_path} | latest={latest_path} | snapshot={indexed_path}",
        flush=True,
    )


def collect_one_sample(sample_idx: int, robot, contour, rng: np.random.Generator, num_direction_samples: int):
    sampled = sample_valid_start_conf(robot=robot, contour=contour, rng=rng, max_attempts=MAX_START_ATTEMPTS)
    tcp_z_axis = normalize_vec(sampled["start_rot"][:, 2])
    directions = sample_orthogonal_directions(axis=tcp_z_axis, num_samples=num_direction_samples, rng=rng)
    best_direction_vec, best_result = evaluate_best_direction(
        robot=robot,
        contour=contour,
        start_q=sampled["start_q"],
        directions=directions,
    )
    return {
        "sample_idx": int(sample_idx),
        "start_q": sampled["start_q"],
        "start_pos": sampled["start_pos"],
        "tcp_z_axis": tcp_z_axis,
        "best_direction_vec": best_direction_vec,
        "best_line_length": float(best_result["line_length"]),
        "best_num_success_steps": int(best_result["num_success_steps"]),
        "best_termination_reason": best_result["termination_reason"],
        "sampling_attempts": int(sampled["attempts"]),
    }


def main():
    args = parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    h5_file = create_or_open_h5(
        output_path=args.output_h5,
        num_samples=args.num_samples,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
    )
    h5_file.attrs["num_direction_samples"] = int(args.num_direction_samples)
    h5_file.attrs["seed"] = int(args.seed)

    done_mask = h5_file["done_mask"][:]
    pending_indices = np.flatnonzero(~done_mask)

    print(f"[Config] output={args.output_h5}", flush=True)
    print(f"[Config] num_samples={args.num_samples} | pending={len(pending_indices)}", flush=True)
    print(
        f"[Config] step={STEP_SIZE} | max_steps={MAX_STEPS} | num_direction_samples={args.num_direction_samples} | seed={args.seed}",
        flush=True,
    )

    if len(pending_indices) == 0:
        print("[INFO] No pending samples. Dataset is already complete.", flush=True)
        h5_file.close()
        return

    contour = WorkspaceContour(contour_path=str(CONTOUR_PATH), z_value=0.0)
    robot = xarm6_sim.XArmLite6Miller(enable_cc=True)
    started_at = time.time()
    processed_since_flush = 0
    done_before = int(np.count_nonzero(done_mask))
    rng = np.random.default_rng(args.seed + done_before)

    try:
        for processed_idx, sample_idx in enumerate(pending_indices, start=1):
            sample_started = time.time()
            result = collect_one_sample(
                sample_idx=int(sample_idx),
                robot=robot,
                contour=contour,
                rng=rng,
                num_direction_samples=args.num_direction_samples,
            )
            write_sample_result(h5_file, int(sample_idx), result)
            processed_since_flush += 1

            sample_dt = time.time() - sample_started
            elapsed = time.time() - started_at
            total_done = done_before + processed_idx
            print(
                f"[Progress] processed={processed_idx}/{len(pending_indices)} | "
                f"done_total={total_done}/{args.num_samples} | "
                f"sample_dt={sample_dt:.2f}s | elapsed={elapsed / 60.0:.1f} min | "
                f"L={result['best_line_length']:.3f} | "
                f"attempts={result['sampling_attempts']} | "
                f"termination={result['best_termination_reason']}",
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
