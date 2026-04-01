import argparse
import importlib.util
from pathlib import Path
import time

import h5py
import jax
import jax2torch
import numpy as np
import torch

from wrs.robot_sim.robots.xarmlite6_wg.sphere_collision_checker import SphereCollisionChecker


SCRIPT_PATH = Path(__file__).resolve().parent / "xarm_nullspave_straight_gpu.py"


def load_gpu_demo_module():
    spec = importlib.util.spec_from_file_location("gpu_nullspace_demo", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect XArmLite6 GPU null-space trajectories into HDF5.")
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parent / "xarmlite6_gpu_trajectories.hdf5"))
    parser.add_argument("--num-trajectories", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--speed", type=float, default=0.10)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--null-gain", type=float, default=0.6)
    parser.add_argument("--theta-max-deg", type=float, default=5.0)
    parser.add_argument("--boundary-gain", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--mu-threshold", type=float, default=0.01)
    parser.add_argument("--pos-error-threshold", type=float, default=0.01)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_trajectory(group: h5py.Group, traj: dict) -> None:
    scalar_attrs = [
        "trajectory_id",
        "termination_code",
        "termination_reason",
        "num_points",
        "total_projected_length",
        "total_euclidean_length",
        "mean_mu",
        "min_mu",
        "max_mu",
        "boundary_hit_count",
        "max_pos_error",
    ]
    vector_attrs = ["start_q", "start_pos", "direction", "target_normal"]
    for key in scalar_attrs:
        group.attrs[key] = traj[key]
    for key in vector_attrs:
        group.attrs[key] = traj[key]

    dataset_keys = [
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
    for key in dataset_keys:
        group.create_dataset(key, data=traj[key], compression="gzip")



def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")

    gpu_demo = load_gpu_demo_module()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    xarm = gpu_demo.xarm6_gpu.XArmLite6GPU(device=device)
    robot = xarm.robot

    cc_model = SphereCollisionChecker('wrs/robot_sim/robots/xarmlite6_wg/xarm6_sphere_visuals.urdf')
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


    mode = "w" if args.overwrite else "x"
    start_time = time.perf_counter()
    with h5py.File(output_path, mode) as h5f:
        h5f.attrs["num_trajectories_target"] = args.num_trajectories
        h5f.attrs["batch_size"] = args.batch_size
        h5f.attrs["dt"] = args.dt
        h5f.attrs["task_speed"] = args.speed
        h5f.attrs["damping"] = args.damping
        h5f.attrs["null_gain"] = args.null_gain
        h5f.attrs["theta_max_deg"] = args.theta_max_deg
        h5f.attrs["boundary_gain"] = args.boundary_gain
        h5f.attrs["mu_threshold"] = args.mu_threshold
        h5f.attrs["pos_error_threshold"] = args.pos_error_threshold
        h5f.attrs["target_normal_mode"] = "random_per_sample"

        root = h5f.create_group("trajectories")
        collected = 0
        traj_id = 0
        batch_idx = 0

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
                traj["trajectory_id"] = traj_id
                grp = root.create_group(f"traj_{traj_id:06d}")
                write_trajectory(grp, traj)
                traj_id += 1
                collected += 1

            h5f.attrs["num_trajectories_collected"] = collected
            h5f.flush()
            batch_elapsed = time.perf_counter() - batch_start
            total_elapsed = time.perf_counter() - start_time
            progress = 100.0 * collected / max(args.num_trajectories, 1)
            print(
                f"[runner] batch={batch_idx} collected={collected}/{args.num_trajectories} "
                f"({progress:.1f}%) batch_time={batch_elapsed:.2f}s total_time={total_elapsed:.2f}s"
            )

    total_elapsed = time.perf_counter() - start_time
    print(f"[runner] finished: {output_path} total_time={total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
