import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath

from wrs import mcm, wd
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim

warnings.filterwarnings("ignore")

NUM_RANDOM_STARTS = 5000
STEP_SIZE = 0.005
MAX_STEPS = 240
CLUSTER_THRESHOLD = 0.18
TOP_K = 15
LOCAL_SAMPLING = True
LOCAL_RANGE_SCALE = 0.35
RESULT_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine/xarm_trail1_results.pkl")
JOINT_TREND_FIG_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine/xarm_trail1_joint_trends.png")


class WorkspaceContour:
    def __init__(self, contour_path, z_value=0.0):
        with open(contour_path, "rb") as f:
            self.contour = pickle.load(f)
        self.path = MplPath(self.contour)
        self.z_value = z_value

    def contains_xy(self, xy_points):
        return self.path.contains_points(xy_points)


def is_pose_inside_workspace(contour, pos):
    return bool(contour.contains_xy(np.asarray(pos[:2]).reshape(1, 2))[0])


def wrap_angle_difference(delta):
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def sample_local_start_q(robot, num_samples, center_q=None, range_scale=LOCAL_RANGE_SCALE):
    jnt_ranges = np.asarray(robot.jnt_ranges, dtype=float)
    if center_q is None:
        center_q = robot.rand_conf()
    else:
        center_q = np.asarray(center_q, dtype=float)

    span = (jnt_ranges[:, 1] - jnt_ranges[:, 0]) * range_scale
    low = np.maximum(jnt_ranges[:, 0], center_q - 0.5 * span)
    high = np.minimum(jnt_ranges[:, 1], center_q + 0.5 * span)
    return np.random.uniform(low=low, high=high, size=(num_samples, robot.n_dof))


def select_closest_solution(solutions, seed_q):
    if not solutions:
        return None
    best_q = None
    best_dist = None
    for q in solutions:
        q = np.asarray(q, dtype=float)
        dist = np.linalg.norm(wrap_angle_difference(q - seed_q))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_q = q
    return best_q


def trace_line_by_ik(robot, contour, start_q, direction, step_size, max_steps):
    start_q = np.asarray(start_q, dtype=float)
    start_pos, start_rot = robot.fk(start_q)
    if not is_pose_inside_workspace(contour, start_pos):
        return {
            "start_q": start_q,
            "start_pos": start_pos,
            "start_rot": start_rot,
            "traj_q": np.asarray([start_q]),
            "traj_pos": np.asarray([start_pos]),
            "line_length": 0.0,
            "num_success_steps": 0,
            "success": False,
            "termination_reason": "start_outside_workspace",
        }

    robot.goto_given_conf(start_q)
    if robot.is_collided():
        return {
            "start_q": start_q,
            "start_pos": start_pos,
            "start_rot": start_rot,
            "traj_q": np.asarray([start_q]),
            "traj_pos": np.asarray([start_pos]),
            "line_length": 0.0,
            "num_success_steps": 0,
            "success": False,
            "termination_reason": "start_in_collision",
        }

    traj_q = [start_q.copy()]
    traj_pos = [np.asarray(start_pos, dtype=float).copy()]
    current_q = start_q.copy()
    termination_reason = "max_steps_reached"

    for step_idx in range(1, max_steps + 1):
        tgt_pos = start_pos + direction * (step_idx * step_size)
        if not is_pose_inside_workspace(contour, tgt_pos):
            termination_reason = "out_of_workspace"
            break

        ik_solutions = robot.ik(
            tgt_pos=tgt_pos,
            tgt_rotmat=start_rot,
            seed_jnt_values=current_q,
            option="multiple",
        )
        if ik_solutions is None or len(ik_solutions) == 0:
            termination_reason = "ik_failed"
            break

        next_q = select_closest_solution(ik_solutions, current_q)
        if next_q is None or not robot.are_jnts_in_ranges(next_q):
            termination_reason = "joint_limit"
            break

        robot.goto_given_conf(next_q)
        if robot.is_collided():
            termination_reason = "self_collision"
            break

        tcp_pos, _ = robot.fk(next_q)
        traj_q.append(next_q.copy())
        traj_pos.append(np.asarray(tcp_pos, dtype=float).copy())
        current_q = next_q.copy()

    num_success_steps = len(traj_q) - 1
    return {
        "start_q": start_q,
        "start_pos": start_pos,
        "start_rot": start_rot,
        "traj_q": np.asarray(traj_q),
        "traj_pos": np.asarray(traj_pos),
        "line_length": num_success_steps * step_size,
        "num_success_steps": num_success_steps,
        "success": num_success_steps > 0,
        "termination_reason": termination_reason,
    }


def analyze_top_configs(sorted_results, all_results, jnt_ranges, top_k=TOP_K):
    if not sorted_results:
        return {
            "topk_size": 0,
            "has_cluster": False,
            "cluster_count": 0,
            "largest_cluster_size": 0,
            "mean_pairwise_joint_distance": None,
            "mean_pairwise_normalized_distance": None,
            "baseline_pairwise_normalized_distance": None,
            "pairwise_normalized_distance_matrix": None,
            "cluster_labels": [],
        }

    actual_top_k = min(top_k, len(sorted_results))
    q_top = np.stack([sorted_results[i]["start_q"] for i in range(actual_top_k)], axis=0)
    q_ranges = jnt_ranges[:, 1] - jnt_ranges[:, 0]
    q_norm = q_top / q_ranges[None, :]

    diff = wrap_angle_difference(q_top[:, None, :] - q_top[None, :, :])
    dist = np.linalg.norm(diff, axis=-1)
    diff_norm = wrap_angle_difference(q_norm[:, None, :] - q_norm[None, :, :])
    dist_norm = np.linalg.norm(diff_norm, axis=-1)

    upper_mask = np.triu(np.ones((actual_top_k, actual_top_k), dtype=bool), k=1)
    mean_pairwise_joint_distance = float(dist[upper_mask].mean()) if upper_mask.any() else 0.0
    mean_pairwise_normalized_distance = float(dist_norm[upper_mask].mean()) if upper_mask.any() else 0.0

    all_q = np.stack([item["start_q"] for item in all_results], axis=0)
    if len(all_q) > 1:
        rand_idx = np.random.randint(0, len(all_q), size=(min(200, len(all_q) * 2), 2))
        rand_idx = rand_idx[rand_idx[:, 0] != rand_idx[:, 1]]
        base_diff = wrap_angle_difference(
            all_q[rand_idx[:, 0]] / q_ranges[None, :] - all_q[rand_idx[:, 1]] / q_ranges[None, :]
        )
        baseline_pairwise_normalized_distance = float(np.linalg.norm(base_diff, axis=-1).mean())
    else:
        baseline_pairwise_normalized_distance = None

    adjacency = dist_norm <= CLUSTER_THRESHOLD
    visited = np.zeros(actual_top_k, dtype=bool)
    cluster_labels = [-1] * actual_top_k
    cluster_sizes = []
    cluster_id = 0
    for i in range(actual_top_k):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            cluster_labels[node] = cluster_id
            neighbors = np.where(adjacency[node])[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        cluster_sizes.append(size)
        cluster_id += 1

    largest_cluster_size = max(cluster_sizes) if cluster_sizes else 0
    return {
        "topk_size": actual_top_k,
        "has_cluster": largest_cluster_size >= 3,
        "cluster_count": len(cluster_sizes),
        "largest_cluster_size": largest_cluster_size,
        "mean_pairwise_joint_distance": mean_pairwise_joint_distance,
        "mean_pairwise_normalized_distance": mean_pairwise_normalized_distance,
        "baseline_pairwise_normalized_distance": baseline_pairwise_normalized_distance,
        "pairwise_normalized_distance_matrix": dist_norm,
        "cluster_labels": cluster_labels,
    }


def print_summary(total_starts, sorted_results, top_analysis):
    print("\n" + "=" * 72)
    print("Fixed world-x straight-line IK experiment finished")
    print("=" * 72)
    print(f"Total random starts: {total_starts}")
    print(f"Successful starts   : {len(sorted_results)}")
    print(f"Local sampling mode : {LOCAL_SAMPLING}")
    if LOCAL_SAMPLING:
        print(f"Local range scale   : {LOCAL_RANGE_SCALE:.2f} of each joint range")
        if "sampling_center_q" in top_analysis:
            print(f"Sampling center q   : {np.array2string(top_analysis['sampling_center_q'], precision=3, separator=', ')}")

    if not sorted_results:
        print("No start configuration can advance along +x by IK.")
        return

    reason_counts = {}
    for item in sorted_results:
        reason = item["termination_reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print(f"\nTop {min(TOP_K, len(sorted_results))} ranked starts:")
    for rank, item in enumerate(sorted_results[:TOP_K], start=1):
        print(
            f"#{rank:02d} | "
            f"L={item['line_length']:.4f} m | "
            f"steps={item['num_success_steps']:3d} | "
            f"reason={item['termination_reason']} | "
            f"cluster={top_analysis['cluster_labels'][rank - 1]} | "
            f"q0={np.array2string(item['start_q'], precision=3, separator=', ')}"
        )

    print(f"\nTop-{top_analysis['topk_size']} joint-space similarity:")
    print(f"Mean pairwise joint distance          : {top_analysis['mean_pairwise_joint_distance']:.4f} rad")
    print(f"Mean pairwise normalized distance     : {top_analysis['mean_pairwise_normalized_distance']:.4f}")
    if top_analysis["baseline_pairwise_normalized_distance"] is not None:
        print(f"Random-start baseline normalized dist : {top_analysis['baseline_pairwise_normalized_distance']:.4f}")
    print(f"Cluster threshold                     : {CLUSTER_THRESHOLD:.2f}")
    print(f"Cluster count                         : {top_analysis['cluster_count']}")
    print(f"Largest cluster size                  : {top_analysis['largest_cluster_size']}")
    print(f"Top-{top_analysis['topk_size']} appears clustered        : {top_analysis['has_cluster']}")


def visualize_top_candidates(base, sorted_results, robot):
    for rank, item in enumerate(sorted_results[:TOP_K], start=1):
        traj_pos = item["traj_pos"]
        start_q = item["start_q"]
        robot.goto_given_conf(start_q)
        rgb = [max(0.15, 1.0 - 0.012 * rank), min(0.9, 0.1 + 0.014 * rank), 0.25]
        alpha = 0.18 if rank > 5 else 0.32
        robot.gen_meshmodel(rgb=rgb, alpha=alpha).attach_to(base)

        start_p = traj_pos[0]
        end_p = traj_pos[-1] if len(traj_pos) >= 2 else traj_pos[0]
        color = [max(0.1, 1.0 - 0.015 * rank), min(0.95, 0.15 + 0.012 * rank), 0.2]
        mgm.gen_stick(start_p, end_p, radius=0.003, rgb=color).attach_to(base)
        mgm.gen_sphere(start_p, radius=0.004, rgb=[0.1, 0.8, 0.1]).attach_to(base)


def plot_joint_trends(sorted_results, jnt_ranges, fig_path, top_k=TOP_K, sample_low=None, sample_high=None):
    if not sorted_results:
        return

    actual_top_k = min(top_k, len(sorted_results))
    q_top = np.stack([item["start_q"] for item in sorted_results[:actual_top_k]], axis=0)
    joint_ids = np.arange(1, q_top.shape[1] + 1)
    colors = plt.cm.viridis(np.linspace(0.08, 0.95, actual_top_k))
    y_min = float(np.min(jnt_ranges[:, 0]))
    y_max = float(np.max(jnt_ranges[:, 1]))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for joint_idx, joint_id in enumerate(joint_ids):
        ax.vlines(
            joint_id,
            jnt_ranges[joint_idx, 0],
            jnt_ranges[joint_idx, 1],
            color="lightgray",
            linewidth=10,
            alpha=0.8,
            zorder=0,
        )
        ax.text(
            joint_id,
            jnt_ranges[joint_idx, 1] + 0.03 * (y_max - y_min),
            f"[{jnt_ranges[joint_idx, 0]:.2f}, {jnt_ranges[joint_idx, 1]:.2f}]",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=20,
            color="dimgray",
        )
        if sample_low is not None and sample_high is not None:
            half_width = 0.14
            ax.hlines(
                sample_low[joint_idx],
                joint_id - half_width,
                joint_id + half_width,
                color="black",
                linewidth=2.0,
                alpha=0.95,
                zorder=3,
            )
            ax.hlines(
                sample_high[joint_idx],
                joint_id - half_width,
                joint_id + half_width,
                color="black",
                linewidth=2.0,
                alpha=0.95,
                zorder=3,
            )

    for rank in range(actual_top_k):
        ax.plot(
            joint_ids,
            q_top[rank],
            color=colors[rank],
            alpha=0.22 if rank >= 5 else 0.75,
            linewidth=1.1 if rank >= 5 else 2.0,
            marker="o",
            markersize=3.8,
            label=f"top-{rank + 1}" if rank < 8 else None,
            zorder=2,
        )

    ax.set_xlim(0.7, q_top.shape[1] + 0.3)
    ax.set_ylim(y_min - 0.08 * (y_max - y_min), y_max + 0.14 * (y_max - y_min))
    ax.set_xticks(joint_ids)
    ax.set_xticklabels([f"q{i}" for i in joint_ids])
    ax.set_xlabel("Joint dimension")
    ax.set_ylabel("Joint angle (rad)")
    ax.set_title(f"Top-{actual_top_k} start joint-angle trends")
    ax.grid(True, alpha=0.25)
    if actual_top_k > 0:
        ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    base = wd.World(cam_pos=[1.2, 0.5, 0.5], lookat_pos=[0.3, 0.0, 0.0])
    mgm.gen_frame().attach_to(base)
    
    contour = WorkspaceContour(
        contour_path="0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl",
        z_value=0.0,
    )
    robot = xarm6_sim.XArmLite6Miller(enable_cc=True)
    direction = np.array([1.0, 0.0, 0.0], dtype=float)
    center_q = None
    if LOCAL_SAMPLING:
        jnt_ranges = np.asarray(robot.jnt_ranges, dtype=float)
        center_q = np.asarray(robot.rand_conf(), dtype=float)
        span = (jnt_ranges[:, 1] - jnt_ranges[:, 0]) * LOCAL_RANGE_SCALE
        sample_low = np.maximum(jnt_ranges[:, 0], center_q - 0.5 * span)
        sample_high = np.minimum(jnt_ranges[:, 1], center_q + 0.5 * span)
        all_start_q = sample_local_start_q(
            robot=robot,
            num_samples=NUM_RANDOM_STARTS,
            center_q=center_q,
            range_scale=LOCAL_RANGE_SCALE,
        )
        print(f"[INFO] Generated {NUM_RANDOM_STARTS} local random starts around home configuration.")
        print(f"[INFO] Local sampling center q: {np.array2string(center_q, precision=3, separator=', ')}")
        print(f"[INFO] Local sampling low     : {np.array2string(sample_low, precision=3, separator=', ')}")
        print(f"[INFO] Local sampling high    : {np.array2string(sample_high, precision=3, separator=', ')}")
    else:
        all_start_q = np.asarray([robot.rand_conf() for _ in range(NUM_RANDOM_STARTS)])
        print(f"[INFO] Generated {NUM_RANDOM_STARTS} global random starts.")

    all_results = []
    for idx in range(NUM_RANDOM_STARTS):
        start_q = all_start_q[idx]
        result = trace_line_by_ik(
            robot=robot,
            contour=contour,
            start_q=start_q,
            direction=direction,
            step_size=STEP_SIZE,
            max_steps=MAX_STEPS,
        )
        result["global_index"] = idx
        all_results.append(result)
        if (idx + 1) % 50 == 0:
            print(f"[INFO] Processed {idx + 1}/{NUM_RANDOM_STARTS}")

    successful_results = [item for item in all_results if item["success"]]
    successful_results.sort(key=lambda x: x["line_length"], reverse=True)
    top_analysis = analyze_top_configs(successful_results, all_results, robot.jnt_ranges, top_k=TOP_K)
    top_analysis["sampling_center_q"] = center_q.copy() if LOCAL_SAMPLING else None
    print_summary(NUM_RANDOM_STARTS, successful_results, top_analysis)

    plot_joint_trends(
        sorted_results=successful_results,
        jnt_ranges=np.asarray(robot.jnt_ranges, dtype=float),
        fig_path=JOINT_TREND_FIG_PATH,
        top_k=TOP_K,
        sample_low=sample_low if LOCAL_SAMPLING else None,
        sample_high=sample_high if LOCAL_SAMPLING else None,
    )
    print(f"[INFO] Saved joint trend figure to: {JOINT_TREND_FIG_PATH}")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "wb") as f:
        pickle.dump(
            {
                "config": {
                "num_random_starts": NUM_RANDOM_STARTS,
                "step_size": STEP_SIZE,
                "max_steps": MAX_STEPS,
                "local_sampling": LOCAL_SAMPLING,
                "local_range_scale": LOCAL_RANGE_SCALE,
                "sampling_center_q": center_q.copy() if LOCAL_SAMPLING else None,
                "fixed_direction": direction,
                "cluster_threshold": CLUSTER_THRESHOLD,
            },
                "all_results": all_results,
                "successful_sorted_indices": [item["global_index"] for item in successful_results],
                "top_analysis": top_analysis,
            },
            f,
        )
    print(f"[INFO] Saved full results to: {RESULT_PATH}")

    if successful_results:
        visualize_top_candidates(base, successful_results, robot)

    base.run()
