import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath

import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
from wrs import wd

warnings.filterwarnings("ignore")

NUM_RANDOM_STARTS = 250
STEP_SIZE = 0.005
MAX_STEPS = 240
TOP_K = 15
LOCAL_SAMPLING = True
LOCAL_RANGE_RADIUS = 0.05
LOCAL_RANGE_SCALE = 2 * LOCAL_RANGE_RADIUS
RESULT_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine/xarm_trail1_xyz_results.pkl")
COMBINED_FIG_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine/xarm_trail1_xyz_joint_trends.png")

DIRECTION_CONFIGS = {
    "x": {"vec": np.array([1.0, 0.0, 0.0], dtype=float), "color": "red"},
    "y": {"vec": np.array([0.0, 1.0, 0.0], dtype=float), "color": "green"},
    "z": {"vec": np.array([0.0, 0.0, 1.0], dtype=float), "color": "blue"},
}


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
    center_q = np.asarray(robot.rand_conf() if center_q is None else center_q, dtype=float)
    span = (jnt_ranges[:, 1] - jnt_ranges[:, 0]) * range_scale
    low = np.maximum(jnt_ranges[:, 0], center_q - 0.5 * span)
    high = np.minimum(jnt_ranges[:, 1], center_q + 0.5 * span)
    return np.random.uniform(low=low, high=high, size=(num_samples, robot.n_dof)), low, high


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


def evaluate_direction(robot, contour, start_q_batch, direction_name, direction_vec):
    print(f"\n{'=' * 72}")
    print(f"Direction {direction_name.upper()} | vec={direction_vec}")
    print(f"{'=' * 72}")

    all_results = []
    for idx, start_q in enumerate(start_q_batch):
        result = trace_line_by_ik(
            robot=robot,
            contour=contour,
            start_q=start_q,
            direction=direction_vec,
            step_size=STEP_SIZE,
            max_steps=MAX_STEPS,
        )
        result["global_index"] = idx
        result["direction"] = direction_name
        all_results.append(result)
        # if (idx + 1) % 50 == 0:
        #     print(f"[INFO][{direction_name}] Processed {idx + 1}/{len(start_q_batch)}")

    successful_results = [item for item in all_results if item["success"]]
    successful_results.sort(key=lambda x: x["line_length"], reverse=True)
    return all_results, successful_results


def summarize_direction(direction_name, successful_results):
    print(f"\nDirection {direction_name.upper()} summary:")
    print(f"Successful starts: {len(successful_results)}")
    if not successful_results:
        print("No start configuration can advance in this direction.")
        return
    print(f"Top {min(TOP_K, len(successful_results))} ranked starts:")
    for rank, item in enumerate(successful_results[:TOP_K], start=1):
        print(
            f"#{rank:02d} | "
            f"L={item['line_length']:.4f} m | "
            f"steps={item['num_success_steps']:3d} | "
            f"reason={item['termination_reason']} | "
            f"q0={np.array2string(item['start_q'], precision=3, separator=', ')}"
        )


def plot_joint_trends_xyz(direction_results, jnt_ranges, fig_path, sample_low=None, sample_high=None, top_k=TOP_K):
    fig, ax = plt.subplots(figsize=(10.5, 6))
    joint_ids = np.arange(1, jnt_ranges.shape[0] + 1)
    y_min = float(np.min(jnt_ranges[:, 0]))
    y_max = float(np.max(jnt_ranges[:, 1]))

    for joint_idx, joint_id in enumerate(joint_ids):
        ax.vlines(
            joint_id,
            jnt_ranges[joint_idx, 0],
            jnt_ranges[joint_idx, 1],
            color="lightgray",
            linewidth=10,
            alpha=0.65,
            zorder=0,
        )
        if sample_low is not None and sample_high is not None:
            half_width = 0.14
            ax.hlines(sample_low[joint_idx], joint_id - half_width, joint_id + half_width, color="black", linewidth=2.0, zorder=3)
            ax.hlines(sample_high[joint_idx], joint_id - half_width, joint_id + half_width, color="black", linewidth=2.0, zorder=3)

    for direction_name, cfg in DIRECTION_CONFIGS.items():
        top_results = direction_results[direction_name][:top_k]
        if not top_results:
            continue
        q_top = np.stack([item["start_q"] for item in top_results], axis=0)
        q_mean = np.mean(q_top, axis=0)
        q_std = np.std(q_top, axis=0)
        ax.fill_between(
            joint_ids,
            q_mean - q_std,
            q_mean + q_std,
            color=cfg["color"],
            alpha=0.18,
            zorder=1,
        )
        ax.plot(
            joint_ids,
            q_mean,
            color=cfg["color"],
            linewidth=3.0,
            marker="o",
            markersize=5.0,
            label=f"{direction_name.upper()} mean ± std",
            zorder=3,
        )

    ax.set_xlim(0.7, joint_ids[-1] + 0.3)
    ax.set_ylim(y_min - 0.08 * (y_max - y_min), y_max + 0.08 * (y_max - y_min))
    ax.set_xticks(joint_ids)
    ax.set_xticklabels([f"q{i}" for i in joint_ids])
    ax.set_xlabel("Joint dimension")
    ax.set_ylabel("Joint angle (rad)")
    ax.set_title("Top-15 start joint trends for X / Y / Z directions")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def visualize_direction_results(base, robot, direction_success_results, top_k=TOP_K):
    for direction_name, cfg in DIRECTION_CONFIGS.items():
        color_name = cfg["color"]
        if color_name == "red":
            rgb = [0.9, 0.2, 0.2]
        elif color_name == "green":
            rgb = [0.2, 0.75, 0.2]
        else:
            rgb = [0.2, 0.35, 0.9]

        top_results = direction_success_results[direction_name][:top_k]
        for rank, item in enumerate(top_results, start=1):
            start_q = item["start_q"]
            traj_pos = item["traj_pos"]
            robot.goto_given_conf(start_q)
            robot_alpha = 0.12 if rank > 5 else 0.22
            robot.gen_meshmodel(rgb=rgb, alpha=robot_alpha).attach_to(base)

            start_p = traj_pos[0]
            end_p = traj_pos[-1] if len(traj_pos) >= 2 else traj_pos[0]
            mgm.gen_stick(start_p, end_p, radius=0.0025, rgb=rgb).attach_to(base)
            mgm.gen_sphere(start_p, radius=0.004, rgb=rgb).attach_to(base)


if __name__ == "__main__":
    base = wd.World(cam_pos=[1.2, 0.5, 0.5], lookat_pos=[0.3, 0.0, 0.0])
    mgm.gen_frame().attach_to(base)

    contour = WorkspaceContour(
        contour_path="0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl",
        z_value=0.0,
    )
    robot = xarm6_sim.XArmLite6Miller(enable_cc=True)

    center_q = None
    sample_low = None
    sample_high = None
    if LOCAL_SAMPLING:
        # center_q = np.asarray(robot.rand_conf(), dtype=float)
        # center_q = np.array([-0.804,  1.791,  5.193,  0.565, -0.265,  2.875], dtype=float)
        center_q = np.array([-1.228,  1.207,  5.228,  1.323, -0.67 ,  2.91 ], dtype=float)
        all_start_q, sample_low, sample_high = sample_local_start_q(
            robot=robot,
            num_samples=NUM_RANDOM_STARTS,
            center_q=center_q,
            range_scale=LOCAL_RANGE_SCALE,
        )
        print(f"[INFO] Generated {NUM_RANDOM_STARTS} local random starts around fixed center configuration.")
        print(f"[INFO] Sampling center q: {np.array2string(center_q, precision=3, separator=', ')}")
        print(f"[INFO] Local sampling low : {np.array2string(sample_low, precision=3, separator=', ')}")
        print(f"[INFO] Local sampling high: {np.array2string(sample_high, precision=3, separator=', ')}")
    else:
        all_start_q = np.asarray([robot.rand_conf() for _ in range(NUM_RANDOM_STARTS)])
        print(f"[INFO] Generated {NUM_RANDOM_STARTS} global random starts.")

    direction_all_results = {}
    direction_success_results = {}
    for direction_name, cfg in DIRECTION_CONFIGS.items():
        all_results, successful_results = evaluate_direction(
            robot=robot,
            contour=contour,
            start_q_batch=all_start_q,
            direction_name=direction_name,
            direction_vec=cfg["vec"],
        )
        direction_all_results[direction_name] = all_results
        direction_success_results[direction_name] = successful_results
        summarize_direction(direction_name, successful_results)

    plot_joint_trends_xyz(
        direction_results=direction_success_results,
        jnt_ranges=np.asarray(robot.jnt_ranges, dtype=float),
        fig_path=COMBINED_FIG_PATH,
        sample_low=sample_low if LOCAL_SAMPLING else None,
        sample_high=sample_high if LOCAL_SAMPLING else None,
        top_k=TOP_K,
    )
    print(f"\n[INFO] Saved combined XYZ joint trend figure to: {COMBINED_FIG_PATH}")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "wb") as f:
        pickle.dump(
            {
                "config": {
                    "num_random_starts": NUM_RANDOM_STARTS,
                    "step_size": STEP_SIZE,
                    "max_steps": MAX_STEPS,
                    "top_k": TOP_K,
                    "local_sampling": LOCAL_SAMPLING,
                    "local_range_scale": LOCAL_RANGE_SCALE,
                    "sampling_center_q": center_q,
                    "sample_low": sample_low,
                    "sample_high": sample_high,
                    "directions": {name: cfg["vec"] for name, cfg in DIRECTION_CONFIGS.items()},
                },
                "all_start_q": all_start_q,
                "direction_all_results": direction_all_results,
                "direction_success_results": direction_success_results,
                "combined_figure": str(COMBINED_FIG_PATH),
            },
            f,
        )
    print(f"[INFO] Saved XYZ experiment results to: {RESULT_PATH}")

    visualize_direction_results(base, robot, direction_success_results, top_k=TOP_K)
    base.run()
