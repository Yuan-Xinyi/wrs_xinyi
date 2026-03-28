import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one trajectory in XArmLite6 GPU HDF5 dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "xarmlite6_gpu_trajectories.hdf5"),
        help="Path to HDF5 dataset.",
    )
    parser.add_argument(
        "--traj-id",
        type=int,
        default=0,
        help="Trajectory id to inspect, mapped to group name traj_XXXXXX.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional output image path. If omitted, show interactively.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    h5_path = Path(args.input).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    group_name = f"traj_{args.traj_id:06d}"
    with h5py.File(h5_path, "r") as h5f:
        traj_root = h5f["trajectories"]
        if group_name not in traj_root:
            raise KeyError(f"Trajectory group not found: {group_name}")
        grp = traj_root[group_name]

        mu = grp["mu"][:]
        remaining_length = grp["remaining_length"][:]
        progress_length = grp["progress_length"][:]
        pos_error = grp["pos_error"][:]
        cos_theta = grp["cos_theta"][:]
        step_index = grp["step_index"][:]
        attrs = dict(grp.attrs)

    print(f"input={h5_path}")
    print(f"trajectory_group={group_name}")
    print(f"termination_reason={attrs['termination_reason']}")
    print(f"num_points={int(attrs['num_points'])}")
    print(f"total_projected_length={float(attrs['total_projected_length']):.6f} m")
    print(f"total_euclidean_length={float(attrs['total_euclidean_length']):.6f} m")
    print(f"mean_mu={float(attrs['mean_mu']):.6f}")
    print(f"min_mu={float(attrs['min_mu']):.6f}")
    print(f"max_mu={float(attrs['max_mu']):.6f}")
    print(f"boundary_hit_count={int(attrs['boundary_hit_count'])}")
    print(f"max_pos_error={float(attrs['max_pos_error']):.6f} m")
    print(f"remaining_length_last={float(remaining_length[-1]):.6f} m")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    axes[0, 0].plot(step_index, remaining_length, color="tab:blue", linewidth=2)
    axes[0, 0].set_title("remaining_length")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("m")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(step_index, mu, color="tab:green", linewidth=2)
    axes[0, 1].set_title("mu")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylabel("directional manipulability")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(step_index, progress_length, color="tab:orange", linewidth=2)
    axes[1, 0].set_title("progress_length")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("m")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(step_index, pos_error, color="tab:red", linewidth=2, label="pos_error")
    axes[1, 1].plot(step_index, cos_theta, color="tab:purple", linewidth=2, label="cos_theta")
    axes[1, 1].set_title("pos_error / cos_theta")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle(
        f"{group_name} | term={attrs['termination_reason']} | total_L={float(attrs['total_projected_length']):.4f}m",
        fontsize=12,
    )

    if args.save:
        save_path = Path(args.save).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
        print(f"saved_plot={save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
