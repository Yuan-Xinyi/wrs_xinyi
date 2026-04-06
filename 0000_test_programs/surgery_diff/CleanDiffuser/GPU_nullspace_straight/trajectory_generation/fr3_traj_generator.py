import argparse
import importlib.util
from pathlib import Path
import time

import h5py
import jax
import jax2torch
import numpy as np
import torch

from wrs.robot_sim.robots.franka_research_3.sphere_collision_checker import SphereCollisionChecker


SCRIPT_PATH = Path(__file__).resolve().parent / 'fr3_nullspace_straight.py'
DATASET_KEYS = [
    'q',
    'tcp_pos',
    'tcp_rotmat',
    'mu',
    'remaining_length',
    'remaining_euclidean_length',
    'progress_length',
    'pos_error',
    'cos_theta',
    'boundary_active',
    'step_index',
    'is_terminal',
]


def load_gpu_demo_module():
    spec = importlib.util.spec_from_file_location('franka_gpu_nullspace_demo', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect Franka Research 3 GPU null-space trajectories into HDF5.')
    parser.add_argument('--output', type=str, default=str(Path(__file__).resolve().parent.parent / 'datasets' / 'franka_research_3_gpu_trajectories_sub10.hdf5'))
    parser.add_argument('--num-trajectories', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--subsample-stride', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--speed', type=float, default=0.10)
    parser.add_argument('--damping', type=float, default=1e-3)
    parser.add_argument('--null-gain', type=float, default=0.6)
    parser.add_argument('--theta-max-deg', type=float, default=30.0)
    parser.add_argument('--boundary-gain', type=float, default=10.0)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--mu-threshold', type=float, default=0.01)
    parser.add_argument('--pos-error-threshold', type=float, default=0.01)
    parser.add_argument('--print-every', type=int, default=50)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def compute_keep_indices(num_points: int, stride: int) -> np.ndarray:
    keep = np.arange(0, num_points, stride, dtype=np.int64)
    if keep.size == 0 or keep[-1] != num_points - 1:
        keep = np.concatenate([keep, np.array([num_points - 1], dtype=np.int64)])
    return np.unique(keep)


def subsample_trajectory(traj: dict, stride: int) -> dict:
    if stride <= 1:
        return traj

    keep_idx = compute_keep_indices(num_points=int(traj['num_points']), stride=stride)
    out = dict(traj)
    for key in DATASET_KEYS:
        out[key] = np.asarray(traj[key])[keep_idx]

    new_count = int(keep_idx.shape[0])
    out['step_index'] = np.arange(new_count, dtype=np.int32)
    out['is_terminal'] = np.zeros(new_count, dtype=np.uint8)
    out['is_terminal'][-1] = 1
    mu = np.asarray(out['mu'], dtype=np.float32)
    boundary_active = np.asarray(out['boundary_active'], dtype=np.uint8)
    pos_error = np.asarray(out['pos_error'], dtype=np.float32)
    out['num_points'] = new_count
    out['total_projected_length'] = float(np.asarray(out['progress_length'], dtype=np.float32)[-1])
    out['total_euclidean_length'] = float(np.asarray(out['remaining_euclidean_length'], dtype=np.float32)[0])
    out['mean_mu'] = float(mu.mean())
    out['min_mu'] = float(mu.min())
    out['max_mu'] = float(mu.max())
    out['boundary_hit_count'] = int(boundary_active.sum())
    out['max_pos_error'] = float(pos_error.max())
    return out


def write_trajectory(group: h5py.Group, traj: dict) -> None:
    scalar_attrs = [
        'trajectory_id',
        'termination_code',
        'termination_reason',
        'num_points',
        'total_projected_length',
        'total_euclidean_length',
        'mean_mu',
        'min_mu',
        'max_mu',
        'boundary_hit_count',
        'max_pos_error',
    ]
    vector_attrs = ['start_q', 'start_pos', 'direction', 'target_normal']
    for key in scalar_attrs:
        group.attrs[key] = traj[key]
    for key in vector_attrs:
        group.attrs[key] = traj[key]

    for key in DATASET_KEYS:
        group.create_dataset(key, data=traj[key], compression='gzip')


def main() -> None:
    args = parse_args()
    if args.subsample_stride < 1:
        raise ValueError('--subsample-stride must be >= 1')
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f'Output file already exists: {output_path}. Use --overwrite to replace it.')
    if args.device == 'cpu':
        raise RuntimeError('This runner requires CUDA because the Franka GPU demo uses wrs.neuro JLChain on CUDA.')

    gpu_demo = load_gpu_demo_module()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('CUDA is not available, but this Franka GPU runner requires a CUDA device.')

    franka = gpu_demo.FrankaResearch3GPU(device=device)
    robot = franka.robot

    cc_model = SphereCollisionChecker('wrs/robot_sim/robots/franka_research_3/franka_research_3_ccsphere.urdf')
    vmap_jax_cost = jax.jit(jax.vmap(cc_model.self_collision_cost, in_axes=(0, None, None)))
    collision_fn = jax2torch.jax2torch(lambda q_batch: vmap_jax_cost(q_batch, 1.0, -0.005))

    tracker = gpu_demo.GPUNullspaceStraightTracker(
        robot=robot,
        collision_fn=collision_fn,
        print_every=args.print_every,
        config=gpu_demo.TrackerConfig(
            dt=args.dt,
            task_speed=args.speed,
            damping=args.damping,
            null_gain=args.null_gain,
            theta_max=np.deg2rad(args.theta_max_deg),
            boundary_gain=args.boundary_gain,
            max_steps=args.max_steps,
            mu_threshold=args.mu_threshold,
            pos_error_threshold=args.pos_error_threshold,
        ),
    )

    mode = 'w' if args.overwrite else 'x'
    start_time = time.perf_counter()
    with h5py.File(output_path, mode) as h5f:
        h5f.attrs['num_trajectories_target'] = args.num_trajectories
        h5f.attrs['batch_size'] = args.batch_size
        h5f.attrs['dt'] = args.dt
        h5f.attrs['task_speed'] = args.speed
        h5f.attrs['damping'] = args.damping
        h5f.attrs['null_gain'] = args.null_gain
        h5f.attrs['theta_max_deg'] = args.theta_max_deg
        h5f.attrs['boundary_gain'] = args.boundary_gain
        h5f.attrs['mu_threshold'] = args.mu_threshold
        h5f.attrs['pos_error_threshold'] = args.pos_error_threshold
        h5f.attrs['target_normal_mode'] = 'random_per_sample'
        h5f.attrs['robot'] = 'franka_research_3'
        h5f.attrs['subsample_stride'] = args.subsample_stride
        h5f.attrs['subsample_terminal_kept'] = True

        root = h5f.create_group('trajectories')
        collected = 0
        traj_id = 0
        batch_idx = 0
        total_points_before = 0
        total_points_after = 0

        while collected < args.num_trajectories:
            batch_idx += 1
            batch_start = time.perf_counter()
            current_batch = min(args.batch_size, args.num_trajectories - collected)
            q0_batch, direction_batch, target_normal_batch = tracker.sample_valid_batch(
                batch_size=current_batch,
                device=device,
            )
            trajectories = tracker.collect_batch_trajectories(
                q0_batch=q0_batch,
                direction_batch=direction_batch,
                target_normal_batch=target_normal_batch,
            )

            for traj in trajectories:
                traj['trajectory_id'] = traj_id
                total_points_before += int(traj['num_points'])
                traj = subsample_trajectory(traj, args.subsample_stride)
                total_points_after += int(traj['num_points'])
                grp = root.create_group(f'traj_{traj_id:06d}')
                write_trajectory(grp, traj)
                traj_id += 1
                collected += 1

            h5f.attrs['num_trajectories_collected'] = collected
            h5f.attrs['num_points_before'] = total_points_before
            h5f.attrs['num_points_after'] = total_points_after
            h5f.attrs['subsample_keep_ratio'] = float(total_points_after / max(total_points_before, 1))
            h5f.flush()
            batch_elapsed = time.perf_counter() - batch_start
            total_elapsed = time.perf_counter() - start_time
            progress = 100.0 * collected / max(args.num_trajectories, 1)
            print(
                f'[runner] batch={batch_idx} collected={collected}/{args.num_trajectories} '
                f'({progress:.1f}%) keep_ratio={total_points_after / max(total_points_before, 1):.4f} '
                f'batch_time={batch_elapsed:.2f}s total_time={total_elapsed:.2f}s'
            )

    total_elapsed = time.perf_counter() - start_time
    print(f'[runner] finished: {output_path} total_time={total_elapsed:.2f}s')


if __name__ == '__main__':
    main()
