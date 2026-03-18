import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath

import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim

N_VALUES = [100, 500, 800, 1000, 2000, 5000, 8000, 10000]
NUM_RANDOM_CENTERS = 1
STEP_SIZE = 0.005
MAX_STEPS = 240
LOCAL_SAMPLING = True
LOCAL_RANGE_RADIUS = 0.1
LOCAL_RANGE_SCALE = 2 * LOCAL_RANGE_RADIUS
SATURATION_EPS = 0.005

DIRECTION_CONFIGS = {
    "x": {"vec": np.array([1.0, 0.0, 0.0], dtype=float), "color": "red"},
    "y": {"vec": np.array([0.0, 1.0, 0.0], dtype=float), "color": "green"},
    "z": {"vec": np.array([0.0, 0.0, 1.0], dtype=float), "color": "blue"},
}

RESULT_JSON_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine/num_random_starts_sweep.json")
RESULT_FIG_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine/num_random_starts_sweep.png")


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
        return 0.0, "start_outside_workspace"

    robot.goto_given_conf(start_q)
    if robot.is_collided():
        return 0.0, "start_in_collision"

    current_q = start_q.copy()
    success_steps = 0

    for step_idx in range(1, max_steps + 1):
        tgt_pos = start_pos + direction * (step_idx * step_size)
        if not is_pose_inside_workspace(contour, tgt_pos):
            return success_steps * step_size, "out_of_workspace"

        ik_solutions = robot.ik(
            tgt_pos=tgt_pos,
            tgt_rotmat=start_rot,
            seed_jnt_values=current_q,
            option="multiple",
        )
        if ik_solutions is None or len(ik_solutions) == 0:
            return success_steps * step_size, "ik_failed"

        next_q = select_closest_solution(ik_solutions, current_q)
        if next_q is None or not robot.are_jnts_in_ranges(next_q):
            return success_steps * step_size, "joint_limit"

        robot.goto_given_conf(next_q)
        if robot.is_collided():
            return success_steps * step_size, "self_collision"

        current_q = next_q.copy()
        success_steps += 1

    return success_steps * step_size, "max_steps_reached"


def determine_saturation(records, eps=SATURATION_EPS):
    if not records:
        return None
    if len(records) == 1:
        return records[0]["N"]
    for idx in range(1, len(records)):
        improvement = records[idx]["L_max"] - records[idx - 1]["L_max"]
        records[idx]["delta_from_prev"] = improvement
        if improvement <= eps:
            return records[idx]["N"]
    return None


def evaluate_prefix_records(all_lengths, all_reasons):
    records = []
    for n in N_VALUES:
        prefix_lengths = all_lengths[:n]
        top_idx = int(np.argmax(prefix_lengths))
        l_max = float(prefix_lengths[top_idx])
        success_count = int(np.sum(prefix_lengths > 0.0))
        records.append(
            {
                "N": int(n),
                "L_max": l_max,
                "best_index_within_prefix": top_idx,
                "best_reason": all_reasons[top_idx],
                "success_count": success_count,
            }
        )
    saturation_n = determine_saturation(records, eps=SATURATION_EPS)
    return records, saturation_n


def aggregate_direction_stats(center_runs):
    aggregated = {}
    for direction_name in DIRECTION_CONFIGS:
        per_n_lmax = {n: [] for n in N_VALUES}
        saturation_values = []
        for center_run in center_runs:
            direction_run = center_run["directions"][direction_name]
            for record in direction_run["records"]:
                per_n_lmax[record["N"]].append(record["L_max"])
            if direction_run["saturation_n"] is not None:
                saturation_values.append(direction_run["saturation_n"])

        aggregated_records = []
        prev_mean = None
        saturation_mean_based = None
        for n in N_VALUES:
            values = np.asarray(per_n_lmax[n], dtype=float)
            mean_lmax = float(values.mean())
            std_lmax = float(values.std())
            record = {
                "N": int(n),
                "L_max_mean": mean_lmax,
                "L_max_std": std_lmax,
            }
            if prev_mean is not None:
                delta = mean_lmax - prev_mean
                record["delta_from_prev_mean"] = delta
                if saturation_mean_based is None and delta <= SATURATION_EPS:
                    saturation_mean_based = n
            aggregated_records.append(record)
            prev_mean = mean_lmax

        aggregated[direction_name] = {
            "records": aggregated_records,
            "saturation_n_mean_based": saturation_mean_based,
            "saturation_n_across_centers_mean": (None if not saturation_values else float(np.mean(saturation_values))),
            "saturation_n_across_centers_all": saturation_values,
        }
    return aggregated


def plot_sweep(center_runs, aggregated, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    for direction_name, cfg in DIRECTION_CONFIGS.items():
        color = cfg["color"]
        for center_run in center_runs:
            records = center_run["directions"][direction_name]["records"]
            ax.plot(
                [item["N"] for item in records],
                [item["L_max"] for item in records],
                color=color,
                alpha=0.16,
                linewidth=1.2,
            )
        agg_records = aggregated[direction_name]["records"]
        x = [item["N"] for item in agg_records]
        y = np.asarray([item["L_max_mean"] for item in agg_records], dtype=float)
        y_std = np.asarray([item["L_max_std"] for item in agg_records], dtype=float)
        ax.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.15)
        ax.plot(x, y, color=color, linewidth=3.0, marker="o", markersize=5.5, label=f"{direction_name.upper()} mean")

    ax.set_xscale("log")
    ax.set_xlabel("NUM_RANDOM_STARTS (N)")
    ax.set_ylabel("Top-1 line length L_max (m)")
    ax.set_title("Per-center curves and cross-center mean")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=True)

    ax = axes[1]
    directions = list(DIRECTION_CONFIGS.keys())
    sat_mean = [aggregated[d]["saturation_n_across_centers_mean"] or 0.0 for d in directions]
    sat_based = [aggregated[d]["saturation_n_mean_based"] or 0.0 for d in directions]
    x = np.arange(len(directions))
    width = 0.34
    ax.bar(x - width / 2, sat_mean, width=width, color=[DIRECTION_CONFIGS[d]["color"] for d in directions], alpha=0.45, label="center saturation mean")
    ax.bar(x + width / 2, sat_based, width=width, color=[DIRECTION_CONFIGS[d]["color"] for d in directions], alpha=0.85, label="mean-curve saturation")
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in directions])
    ax.set_ylabel("Suggested saturation N")
    ax.set_title("Saturation summary")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    contour = WorkspaceContour(
        contour_path="0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl",
        z_value=0.0,
    )
    robot = xarm6_sim.XArmLite6Miller(enable_cc=True)
    max_n = max(N_VALUES)

    center_runs = []
    for center_idx in range(NUM_RANDOM_CENTERS):
        print("\n" + "=" * 72)
        print(f"Center {center_idx + 1}/{NUM_RANDOM_CENTERS}")
        print("=" * 72)

        if LOCAL_SAMPLING:
            # center_q = np.asarray(robot.rand_conf(), dtype=float)
            center_q = np.array([-1.228,  1.207,  5.228,  1.323, -0.67 ,  2.91 ], dtype=float)
            all_start_q, sample_low, sample_high = sample_local_start_q(
                robot=robot,
                num_samples=max_n,
                center_q=center_q,
                range_scale=LOCAL_RANGE_SCALE,
            )
            print(f"[INFO] Sampling center q: {np.array2string(center_q, precision=3, separator=', ')}")
        else:
            center_q = None
            sample_low = None
            sample_high = None
            all_start_q = np.asarray([robot.rand_conf() for _ in range(max_n)])

        center_run = {
            "center_index": center_idx,
            "sampling_center_q": None if center_q is None else center_q.tolist(),
            "sample_low": None if sample_low is None else sample_low.tolist(),
            "sample_high": None if sample_high is None else sample_high.tolist(),
            "directions": {},
        }

        for direction_name, cfg in DIRECTION_CONFIGS.items():
            print(f"[INFO] Evaluating direction {direction_name.upper()} for center {center_idx + 1}")
            all_lengths = np.zeros(max_n, dtype=float)
            all_reasons = []
            for idx, start_q in enumerate(all_start_q):
                line_length, reason = trace_line_by_ik(
                    robot=robot,
                    contour=contour,
                    start_q=start_q,
                    direction=cfg["vec"],
                    step_size=STEP_SIZE,
                    max_steps=MAX_STEPS,
                )
                all_lengths[idx] = line_length
                all_reasons.append(reason)
                if (idx + 1) % 200 == 0:
                    print(f"[INFO][C{center_idx + 1}][{direction_name.upper()}] Evaluated {idx + 1}/{max_n}")

            records, saturation_n = evaluate_prefix_records(all_lengths, all_reasons)
            center_run["directions"][direction_name] = {
                "records": records,
                "saturation_n": saturation_n,
            }

            print(f"Direction {direction_name.upper()} summary:")
            for item in records:
                delta_txt = ""
                if "delta_from_prev" in item:
                    delta_txt = f" | Δ={item['delta_from_prev']:.4f}"
                print(
                    f"N={item['N']:5d} | "
                    f"L_max={item['L_max']:.4f} | "
                    f"success={item['success_count']:5d} | "
                    f"best_reason={item['best_reason']}{delta_txt}"
                )
            print(f"Saturation N (eps={SATURATION_EPS:.4f}): {saturation_n}")

        center_runs.append(center_run)

    aggregated = aggregate_direction_stats(center_runs)
    print("\n" + "=" * 72)
    print("Aggregated across centers")
    print("=" * 72)
    for direction_name in DIRECTION_CONFIGS:
        print(f"Direction {direction_name.upper()}:")
        for item in aggregated[direction_name]["records"]:
            delta_txt = ""
            if "delta_from_prev_mean" in item:
                delta_txt = f" | Δmean={item['delta_from_prev_mean']:.4f}"
            print(
                f"N={item['N']:5d} | "
                f"L_max_mean={item['L_max_mean']:.4f} | "
                f"L_max_std={item['L_max_std']:.4f}{delta_txt}"
            )
        print(f"Mean-curve saturation N: {aggregated[direction_name]['saturation_n_mean_based']}")
        print(f"Center saturation N mean: {aggregated[direction_name]['saturation_n_across_centers_mean']}")

    plot_sweep(center_runs, aggregated, RESULT_FIG_PATH)
    print(f"[INFO] Saved sweep figure to: {RESULT_FIG_PATH}")

    summary = {
        "n_values": N_VALUES,
        "num_random_centers": NUM_RANDOM_CENTERS,
        "step_size": STEP_SIZE,
        "max_steps": MAX_STEPS,
        "local_sampling": LOCAL_SAMPLING,
        "local_range_scale": LOCAL_RANGE_SCALE,
        "saturation_eps": SATURATION_EPS,
        "directions": {name: cfg["vec"].tolist() for name, cfg in DIRECTION_CONFIGS.items()},
        "center_runs": center_runs,
        "aggregated": aggregated,
        "figure_path": str(RESULT_FIG_PATH),
    }
    RESULT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved sweep summary to: {RESULT_JSON_PATH}")
