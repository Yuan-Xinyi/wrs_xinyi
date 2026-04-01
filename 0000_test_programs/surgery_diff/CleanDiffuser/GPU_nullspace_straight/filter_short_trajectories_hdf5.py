import argparse
import time
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Filter out short trajectories from an HDF5 dataset and save a new HDF5.')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('0000_test_programs/surgery_diff/CleanDiffuser/GPU_nullspace_straight/xarmlite6_gpu_trajectories_100000_sub10.hdf5'),
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('0000_test_programs/surgery_diff/CleanDiffuser/GPU_nullspace_straight/xarmlite6_gpu_trajectories_100000_sub10_minlen10cm.hdf5'),
    )
    parser.add_argument('--min-length', type=float, default=0.10, help='Minimum total_projected_length in meters to keep a trajectory.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--print-every', type=int, default=2000)
    return parser.parse_args()


def copy_attrs(src_attrs: h5py.AttributeManager, dst_attrs: h5py.AttributeManager) -> None:
    for key, value in src_attrs.items():
        dst_attrs[key] = value


def copy_group_contents(src_grp: h5py.Group, dst_grp: h5py.Group) -> None:
    copy_attrs(src_grp.attrs, dst_grp.attrs)
    for key in src_grp.keys():
        src_ds = src_grp[key]
        dst_grp.create_dataset(key, data=src_ds[...], compression='gzip')


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f'Output already exists: {output_path}. Use --overwrite to replace it.')
    if args.min_length < 0.0:
        raise ValueError('--min-length must be >= 0')

    start_time = time.perf_counter()
    mode = 'w' if args.overwrite else 'x'
    with h5py.File(input_path, 'r') as src, h5py.File(output_path, mode) as dst:
        copy_attrs(src.attrs, dst.attrs)
        src_root = src['trajectories']
        dst_root = dst.create_group('trajectories')
        keys = sorted(src_root.keys())

        kept = 0
        removed = 0
        kept_points = 0
        removed_points = 0

        for idx, key in enumerate(keys, start=1):
            src_grp = src_root[key]
            total_projected_length = float(src_grp.attrs['total_projected_length'])
            num_points = int(src_grp.attrs['num_points'])
            if total_projected_length < args.min_length:
                removed += 1
                removed_points += num_points
            else:
                dst_grp = dst_root.create_group(key)
                copy_group_contents(src_grp, dst_grp)
                kept += 1
                kept_points += num_points

            if args.print_every > 0 and (idx % args.print_every == 0 or idx == len(keys)):
                elapsed = time.perf_counter() - start_time
                print(
                    f'[filter] trajectories={idx}/{len(keys)} kept={kept} removed={removed} '
                    f'kept_points={kept_points} removed_points={removed_points} elapsed={elapsed:.2f}s'
                )

        dst.attrs['num_trajectories_target'] = kept
        dst.attrs['num_trajectories_collected'] = kept
        dst.attrs['filtered_min_total_projected_length'] = float(args.min_length)
        dst.attrs['filtered_source_dataset'] = str(input_path)
        dst.attrs['filtered_removed_trajectories'] = int(removed)
        dst.attrs['filtered_kept_trajectories'] = int(kept)
        dst.attrs['filtered_removed_points'] = int(removed_points)
        dst.attrs['filtered_kept_points'] = int(kept_points)

    total_time = time.perf_counter() - start_time
    print(f'[done] output={output_path} kept={kept} removed={removed} total_time={total_time:.2f}s')


if __name__ == '__main__':
    main()
