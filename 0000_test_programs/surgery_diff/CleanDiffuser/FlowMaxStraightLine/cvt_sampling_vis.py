from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import samply

import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
from collision_checker import XarmCollisionChecker

# Notes:
# 1. This script generates CVT kernels in the joint space of the XArmLite
# 2. It visualizes the joint-space distribution of the generated kernels.
# 3. It performs collision checking on the kernels and 
# splits them into collision-free and in-collision sets.

BASE_DIR = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine")
RESULTS_DIR = BASE_DIR / "results"
DATASET_DIR = BASE_DIR / "datasets"


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


def split_kernels_by_collision(
    raw_kernels_path,
    collision_free_kernels_path,
    collision_kernels_path,
    batch_size=4096,
):
    kernels_q = np.load(raw_kernels_path)
    cc_checker = XarmCollisionChecker()

    collision_free_masks = []
    total = len(kernels_q)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_q = kernels_q[start:end]
        batch_in_collision = cc_checker.check_collision(batch_q)
        if hasattr(batch_in_collision, "detach"):
            batch_in_collision = batch_in_collision.detach().cpu().numpy()
        batch_in_collision = np.asarray(batch_in_collision, dtype=bool)
        collision_free_masks.append(~batch_in_collision)
        print(
            f"Collision checking progress: {end}/{total} | "
            f"collision-free in batch: {(~batch_in_collision).sum()}/{len(batch_q)}"
        )

    collision_free_mask = np.concatenate(collision_free_masks, axis=0)
    collision_free_q = kernels_q[collision_free_mask]
    collision_q = kernels_q[~collision_free_mask]
    np.save(collision_free_kernels_path, collision_free_q)
    np.save(collision_kernels_path, collision_q)

    print(f"Collision-free kernels: {len(collision_free_q)}/{total}")
    print(f"Collision kernels: {len(collision_q)}/{total}")
    print(f"Saved collision-free kernels to: {collision_free_kernels_path}")
    print(f"Saved collision kernels to: {collision_kernels_path}")
    return collision_free_q, collision_q


if __name__ == "__main__":
    save_path = DATASET_DIR / "cvt_kernels_raw_10000.npy"
    collision_free_save_path = DATASET_DIR / "cvt_kernels_collision_free_10000.npy"
    collision_save_path = DATASET_DIR / "cvt_kernels_in_collision_10000.npy"
    fig_path = RESULTS_DIR / "cvt_kernels_joint_distribution_10000.png"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        kernels = np.load(save_path)
        print(f"Loaded existing kernels from: {save_path}")
    else:
        robot = xarm6_sim.XArmLite6Miller()
        kernels = generate_cvt_kernels(robot, n_kernels=10000)
        np.save(save_path, kernels)
        print(f"Saved kernels to: {save_path}")

    plot_joint_space_distribution(kernels, fig_path)
    split_kernels_by_collision(
        save_path,
        collision_free_save_path,
        collision_save_path,
    )
    print(f"Saved joint distribution figure to: {fig_path}")
