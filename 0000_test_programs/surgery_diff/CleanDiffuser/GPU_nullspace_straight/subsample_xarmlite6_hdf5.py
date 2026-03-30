import argparse
import time
from pathlib import Path

import h5py
import numpy as np


DATASET_KEYS = [
    "q",
    "tcp_pos",
    "tcp_rotmat",
    "mu",
    "remaining_length",
    "remaining_euclidean_length",
    "progress_length",
    "pos_error",
    "cos_theta",
    "boundary_active",
    "step_index",
    "is_terminal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subsample XArmLite6 trajectory HDF5 by a fixed stride and save a new HDF5.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("0000_test_programs/surgery_diff/CleanDiffuser/GPU_nullspace_straight/xarmlite6_gpu_trajectories_100000.hdf5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("0000_test_programs/surgery_diff/CleanDiffuser/GPU_nullspace_straight/xarmlite6_gpu_trajectories_100000_sub10.hdf5"),
    )
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-every", type=int, default=1000)
    return parser.parse_args()



def compute_keep_indices(num_points: int, stride: int) -> np.ndarray:
    keep = np.arange(0, num_points, stride, dtype=np.int64)
    if keep.size == 0 or keep[-1] != num_points - 1:
        keep = np.concatenate([keep, np.array([num_points - 1], dtype=np.int64)])
    return np.unique(keep)



def copy_root_attrs(src: h5py.File, dst: h5py.File, stride: int) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    dst.attrs["subsample_stride"] = int(stride)
    dst.attrs["subsample_terminal_kept"] = True



def recompute_traj_attrs(grp_out: h5py.Group, src_attrs: h5py.AttributeManager, keep_idx: np.ndarray) -> None:
    total_projected_length = float(grp_out["progress_length"][-1])
    total_euclidean_length = float(grp_out["remaining_euclidean_length"][0])
    mu = np.asarray(grp_out["mu"][:], dtype=np.float32)
    boundary_active = np.asarray(grp_out["boundary_active"][:], dtype=np.uint8)
    pos_error = np.asarray(grp_out["pos_error"][:], dtype=np.float32)

    grp_out.attrs["trajectory_id"] = int(src_attrs["trajectory_id"])
    grp_out.attrs["termination_code"] = int(src_attrs["termination_code"])
    grp_out.attrs["termination_reason"] = src_attrs["termination_reason"]
    grp_out.attrs["num_points"] = int(keep_idx.shape[0])
    grp_out.attrs["total_projected_length"] = total_projected_length
    grp_out.attrs["total_euclidean_length"] = total_euclidean_length
    grp_out.attrs["mean_mu"] = float(mu.mean())
    grp_out.attrs["min_mu"] = float(mu.min())
    grp_out.attrs["max_mu"] = float(mu.max())
    grp_out.attrs["boundary_hit_count"] = int(boundary_active.sum())
    grp_out.attrs["max_pos_error"] = float(pos_error.max())
    grp_out.attrs["start_q"] = np.asarray(src_attrs["start_q"], dtype=np.float32)
    grp_out.attrs["start_pos"] = np.asarray(src_attrs["start_pos"], dtype=np.float32)
    grp_out.attrs["direction"] = np.asarray(src_attrs["direction"], dtype=np.float32)
    grp_out.attrs["target_normal"] = np.asarray(src_attrs["target_normal"], dtype=np.float32)



def write_subsampled_group(src_grp: h5py.Group, dst_grp: h5py.Group, stride: int) -> tuple[int, int]:
    num_points = int(src_grp.attrs["num_points"])
    keep_idx = compute_keep_indices(num_points=num_points, stride=stride)

    for key in DATASET_KEYS:
        data = src_grp[key][keep_idx]
        dst_grp.create_dataset(key, data=data, compression="gzip")

    new_count = int(keep_idx.shape[0])
    dst_grp["step_index"][:] = np.arange(new_count, dtype=np.int32)
    is_terminal = np.zeros(new_count, dtype=np.uint8)
    is_terminal[-1] = 1
    dst_grp["is_terminal"][:] = is_terminal

    recompute_traj_attrs(dst_grp, src_grp.attrs, keep_idx)
    return num_points, new_count



def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    start_time = time.perf_counter()
    mode = "w" if args.overwrite else "x"
    with h5py.File(input_path, "r") as src, h5py.File(output_path, mode) as dst:
        copy_root_attrs(src, dst, args.stride)
        src_root = src["trajectories"]
        dst_root = dst.create_group("trajectories")
        keys = sorted(src_root.keys())
        total_traj = len(keys)
        total_points_before = 0
        total_points_after = 0

        for idx, key in enumerate(keys, start=1):
            src_grp = src_root[key]
            dst_grp = dst_root.create_group(key)
            before, after = write_subsampled_group(src_grp, dst_grp, args.stride)
            total_points_before += before
            total_points_after += after

            if args.print_every > 0 and (idx % args.print_every == 0 or idx == total_traj):
                elapsed = time.perf_counter() - start_time
                keep_ratio = total_points_after / max(total_points_before, 1)
                print(
                    f"[subsample] trajectories={idx}/{total_traj} "
                    f"points={total_points_after}/{total_points_before} keep_ratio={keep_ratio:.4f} "
                    f"elapsed={elapsed:.2f}s"
                )

        dst.attrs["num_trajectories_collected"] = total_traj
        dst.attrs["num_points_before"] = int(total_points_before)
        dst.attrs["num_points_after"] = int(total_points_after)
        dst.attrs["subsample_keep_ratio"] = float(total_points_after / max(total_points_before, 1))

    total_time = time.perf_counter() - start_time
    print(f"[done] output={output_path} total_time={total_time:.2f}s")


if __name__ == "__main__":
    main()
