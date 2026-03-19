import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np

import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
from xarm_trail1 import (
    DIRECTION_CONFIGS,
    LOCAL_RANGE_RADIUS,
    LOCAL_RANGE_SCALE,
    MAX_STEPS,
    STEP_SIZE,
    WorkspaceContour,
    sample_local_start_q,
    trace_line_by_ik,
)

BASE_DIR = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine")
DATASET_DIR = BASE_DIR / "datasets"
CHECKPOINT_DIR = DATASET_DIR / "checkpoints"
INPUT_KERNEL_PATH = DATASET_DIR / "cvt_kernels_collision_free.npy"
OUTPUT_H5_PATH = DATASET_DIR / "xarm_trail1_large_scale_top10.h5"
CONTOUR_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl")

N_RAW = 100
MIN_LINE_LENGTH = 0.1
TOP_K = 5
CHECKPOINT_INTERVAL = 1000
NUM_DIRECTIONS = len(DIRECTION_CONFIGS)
TOTAL_TOP_K = TOP_K * NUM_DIRECTIONS
KERNEL_PROGRESS_EVERY = 50


def collect_kernel_records(kernel_idx, kernel_q, robot, contour):
    np.random.seed(20260319 + int(kernel_idx))
    sampled_qs, _, _ = sample_local_start_q(
        robot=robot,
        num_samples=N_RAW,
        center_q=kernel_q,
        range_scale=LOCAL_RANGE_SCALE,
    )

    direction_records = {direction_name: [] for direction_name in DIRECTION_CONFIGS}
    for sample_idx, start_q in enumerate(sampled_qs, start=1):
        for direction_name, cfg in DIRECTION_CONFIGS.items():
            result = trace_line_by_ik(
                robot=robot,
                contour=contour,
                start_q=start_q,
                direction=cfg["vec"],
                step_size=STEP_SIZE,
                max_steps=MAX_STEPS,
            )
            if result["line_length"] < MIN_LINE_LENGTH:
                continue
            direction_records[direction_name].append(
                (
                    float(result["line_length"]),
                    np.asarray(result["start_q"], dtype=np.float32),
                    np.asarray(cfg["vec"], dtype=np.float32),
                    np.asarray(result["start_pos"], dtype=np.float32),
                    direction_name,
                )
            )

        if sample_idx % KERNEL_PROGRESS_EVERY == 0 or sample_idx == len(sampled_qs):
            progress_parts = []
            for direction_name in DIRECTION_CONFIGS:
                success_count = len(direction_records[direction_name])
                best_length = 0.0 if success_count == 0 else direction_records[direction_name][0][0]
                progress_parts.append(f"{direction_name.upper()}:succ={success_count},best={best_length:.3f}")
            print(
                f"[Kernel {kernel_idx:05d}] sampled={sample_idx}/{len(sampled_qs)} | "
                + " | ".join(progress_parts),
                flush=True,
            )

    start_qs = np.full((TOTAL_TOP_K, 6), np.nan, dtype=np.float32)
    direction_vecs = np.full((TOTAL_TOP_K, 3), np.nan, dtype=np.float32)
    start_poss = np.full((TOTAL_TOP_K, 3), np.nan, dtype=np.float32)
    line_lengths = np.full((TOTAL_TOP_K,), np.nan, dtype=np.float32)
    direction_names = np.full((TOTAL_TOP_K,), "", dtype="S1")
    valid_mask = np.zeros((TOTAL_TOP_K,), dtype=bool)

    total_valid_count = 0
    write_idx = 0
    for direction_name in DIRECTION_CONFIGS:
        direction_records[direction_name].sort(key=lambda item: item[0], reverse=True)
        top_records = direction_records[direction_name][:TOP_K]
        total_valid_count += len(top_records)
        for record in top_records:
            line_length, start_q, direction_vec, start_pos, encoded_direction_name = record
            start_qs[write_idx] = start_q
            direction_vecs[write_idx] = direction_vec
            start_poss[write_idx] = start_pos
            line_lengths[write_idx] = line_length
            direction_names[write_idx] = encoded_direction_name.encode("ascii")
            valid_mask[write_idx] = True
            write_idx += 1

    return {
        "kernel_idx": int(kernel_idx),
        "valid_count": int(total_valid_count),
        "direction_success_counts": {
            direction_name: int(len(direction_records[direction_name])) for direction_name in DIRECTION_CONFIGS
        },
        "direction_best_lengths": {
            direction_name: (
                0.0
                if len(direction_records[direction_name]) == 0
                else float(direction_records[direction_name][0][0])
            )
            for direction_name in DIRECTION_CONFIGS
        },
        "start_qs": start_qs,
        "direction_vecs": direction_vecs,
        "start_poss": start_poss,
        "line_lengths": line_lengths,
        "direction_names": direction_names,
        "valid_mask": valid_mask,
    }


def create_or_open_h5(output_path, kernel_qs, resume, input_kernel_path, checkpoint_interval):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and output_path.exists() else "w"
    h5_file = h5py.File(output_path, mode)

    num_kernels = kernel_qs.shape[0]
    if "kernel_qs" not in h5_file:
        h5_file.create_dataset("kernel_qs", data=kernel_qs.astype(np.float32), compression="gzip", shuffle=True)
        h5_file.create_dataset(
            "top_start_q",
            shape=(num_kernels, TOTAL_TOP_K, 6),
            dtype=np.float32,
            compression="gzip",
            shuffle=True,
            fillvalue=np.nan,
        )
        h5_file.create_dataset(
            "top_direction_vec",
            shape=(num_kernels, TOTAL_TOP_K, 3),
            dtype=np.float32,
            compression="gzip",
            shuffle=True,
            fillvalue=np.nan,
        )
        h5_file.create_dataset(
            "top_start_pos",
            shape=(num_kernels, TOTAL_TOP_K, 3),
            dtype=np.float32,
            compression="gzip",
            shuffle=True,
            fillvalue=np.nan,
        )
        h5_file.create_dataset(
            "top_line_length",
            shape=(num_kernels, TOTAL_TOP_K),
            dtype=np.float32,
            compression="gzip",
            shuffle=True,
            fillvalue=np.nan,
        )
        h5_file.create_dataset(
            "top_direction_name",
            shape=(num_kernels, TOTAL_TOP_K),
            dtype="S1",
            compression="gzip",
            shuffle=True,
        )
        h5_file.create_dataset(
            "top_valid_mask",
            shape=(num_kernels, TOTAL_TOP_K),
            dtype=np.bool_,
            compression="gzip",
            shuffle=True,
        )
        h5_file.create_dataset("top_valid_count", shape=(num_kernels,), dtype=np.int32, compression="gzip", shuffle=True)
        h5_file.create_dataset("done_mask", shape=(num_kernels,), dtype=np.bool_, compression="gzip", shuffle=True)
        h5_file["done_mask"][:] = False
        h5_file.attrs["created_unix_time"] = time.time()

    h5_file.attrs["n_raw"] = N_RAW
    h5_file.attrs["local_range_radius"] = LOCAL_RANGE_RADIUS
    h5_file.attrs["local_range_scale"] = LOCAL_RANGE_SCALE
    h5_file.attrs["step_size"] = STEP_SIZE
    h5_file.attrs["max_steps"] = MAX_STEPS
    h5_file.attrs["min_line_length"] = MIN_LINE_LENGTH
    h5_file.attrs["top_k"] = TOP_K
    h5_file.attrs["checkpoint_interval"] = checkpoint_interval
    h5_file.attrs["contour_path"] = str(CONTOUR_PATH)
    h5_file.attrs["input_kernel_path"] = str(input_kernel_path)
    h5_file.attrs["execution_path"] = "aligned_with_xarm_trail1"
    return h5_file


def write_kernel_result(h5_file, result):
    idx = result["kernel_idx"]
    h5_file["top_start_q"][idx] = result["start_qs"]
    h5_file["top_direction_vec"][idx] = result["direction_vecs"]
    h5_file["top_start_pos"][idx] = result["start_poss"]
    h5_file["top_line_length"][idx] = result["line_lengths"]
    h5_file["top_direction_name"][idx] = result["direction_names"]
    h5_file["top_valid_mask"][idx] = result["valid_mask"]
    h5_file["top_valid_count"][idx] = result["valid_count"]
    h5_file["done_mask"][idx] = True


def flush_checkpoint(h5_file, output_path, started_at):
    done_count = int(np.count_nonzero(h5_file["done_mask"][:]))
    h5_file.attrs["completed_kernels"] = done_count
    h5_file.attrs["last_flush_unix_time"] = time.time()
    h5_file.flush()
    elapsed = time.time() - started_at
    print(
        f"[Checkpoint] completed={done_count}/{h5_file['done_mask'].shape[0]} | "
        f"elapsed={elapsed / 60.0:.1f} min | file={output_path}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset collection strictly aligned with xarm_trail1.py.")
    parser.add_argument("--input-npy", type=Path, default=INPUT_KERNEL_PATH)
    parser.add_argument("--output-h5", type=Path, default=OUTPUT_H5_PATH)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--resume", action="store_true", help="Resume from existing HDF5 if present.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_npy.exists():
        raise FileNotFoundError(f"Input kernel file not found: {args.input_npy}")

    kernel_qs = np.load(args.input_npy).astype(np.float32)
    if kernel_qs.ndim != 2 or kernel_qs.shape[1] != 6:
        raise ValueError(f"Expected input kernel array shape [N, 6], got {kernel_qs.shape}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    h5_file = create_or_open_h5(
        args.output_h5,
        kernel_qs,
        resume=args.resume,
        input_kernel_path=args.input_npy,
        checkpoint_interval=args.checkpoint_interval,
    )
    done_mask = h5_file["done_mask"][:]
    pending_indices = np.flatnonzero(~done_mask)

    print(f"[Config] input={args.input_npy}", flush=True)
    print(f"[Config] output={args.output_h5}", flush=True)
    print(f"[Config] kernels={len(kernel_qs)} | pending={len(pending_indices)}", flush=True)
    print(f"[Config] execution=single_process | aligned_with=xarm_trail1.py", flush=True)
    print(
        f"[Config] local_range_radius={LOCAL_RANGE_RADIUS} | "
        f"local_range_scale={LOCAL_RANGE_SCALE} | "
        f"step={STEP_SIZE} | max_steps={MAX_STEPS} | min_length={MIN_LINE_LENGTH}",
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
            kernel_start_time = time.time()
            result = collect_kernel_records(
                kernel_idx=int(kernel_idx),
                kernel_q=kernel_qs[int(kernel_idx)],
                robot=robot,
                contour=contour,
            )
            write_kernel_result(h5_file, result)
            processed_since_flush += 1

            total_done_now = total_done_before + completed_idx
            elapsed = time.time() - started_at
            kernel_elapsed = time.time() - kernel_start_time
            direction_summary = " | ".join(
                [
                    f"{direction_name.upper()}:succ={result['direction_success_counts'][direction_name]},"
                    f"best={result['direction_best_lengths'][direction_name]:.3f}"
                    for direction_name in DIRECTION_CONFIGS
                ]
            )
            print(
                f"[Progress] processed={completed_idx}/{len(pending_indices)} | "
                f"done_total={total_done_now}/{len(kernel_qs)} | "
                f"kernel_dt={kernel_elapsed:.1f}s | "
                f"elapsed={elapsed / 60.0:.1f} min | "
                f"{direction_summary}",
                flush=True,
            )

            if processed_since_flush >= args.checkpoint_interval:
                flush_checkpoint(h5_file, args.output_h5, started_at)
                processed_since_flush = 0
    finally:
        flush_checkpoint(h5_file, args.output_h5, started_at)
        h5_file.close()


if __name__ == "__main__":
    main()
