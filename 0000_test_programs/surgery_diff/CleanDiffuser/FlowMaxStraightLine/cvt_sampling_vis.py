from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import samply

import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim

BASE_DIR = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine")
RESULTS_DIR = BASE_DIR / "Results"
DATASET_DIR = BASE_DIR / "dataset"


def generate_cvt_kernels(robot, n_kernels=10000):
    print(f"Generating {n_kernels} CVT kernels in {robot.n_dof}D...")
    normalized_kernels = samply.hypercube.cvt(n_kernels, robot.n_dof)
    jnt_mins = robot.jnt_ranges[:, 0]
    jnt_maxs = robot.jnt_ranges[:, 1]
    kernels_q = jnt_mins + normalized_kernels * (jnt_maxs - jnt_mins)
    print(f"Kernel shape: {kernels_q.shape}")
    return kernels_q


def plot_joint_space_distribution(kernels_q, fig_path):
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, kernels_q.shape[1]))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for joint_idx in range(kernels_q.shape[1]):
        hist, bin_edges = np.histogram(kernels_q[:, joint_idx], bins=80, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.plot(
            bin_centers,
            hist,
            color=colors[joint_idx],
            linewidth=1.8,
            label=f"q{joint_idx + 1}",
        )

    ax.set_xlabel("Joint value (rad)")
    ax.set_ylabel("Frequency density")
    ax.set_title("Joint-space frequency distribution of CVT kernels")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True, ncol=3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    robot = xarm6_sim.XArmLite6Miller()

    n_kernels = 50000
    kernels = generate_cvt_kernels(robot, n_kernels=n_kernels)

    save_path = DATASET_DIR / "cvt_kernels_raw.npy"
    fig_path = RESULTS_DIR / "cvt_kernels_joint_distribution.png"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(save_path, kernels)
    plot_joint_space_distribution(kernels, fig_path)

    print(f"Saved kernels to: {save_path}")
    print(f"Saved joint distribution figure to: {fig_path}")
