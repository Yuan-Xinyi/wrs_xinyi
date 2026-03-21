import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib.pyplot as plt
import numpy as np

import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
from wrs import wd
from xarm_trail1 import MAX_STEPS, STEP_SIZE, WorkspaceContour, trace_line_by_ik


BASE_DIR = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine")
CONTOUR_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl")
DEFAULT_H5_PATH = BASE_DIR / "datasets" / "xarm_trail1_large_scale_top10.h5"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def create_timestamped_result_dir(root_dir: Path, prefix: str):
    root_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = root_dir / f"{prefix}_{timestamp}"
    suffix = 1
    while result_dir.exists():
        result_dir = root_dir / f"{prefix}_{timestamp}_{suffix:02d}"
        suffix += 1
    result_dir.mkdir(parents=True, exist_ok=False)
    return result_dir


def load_flat_entries_from_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        done_mask = np.asarray(f["done_mask"][:], dtype=bool)
        valid_mask = np.asarray(f["top_valid_mask"][:], dtype=bool)
        start_q = np.asarray(f["top_start_q"][:], dtype=np.float32)
        start_pos = np.asarray(f["top_start_pos"][:], dtype=np.float32)
        direction_vec = np.asarray(f["top_direction_vec"][:], dtype=np.float32)
        line_length = np.asarray(f["top_line_length"][:], dtype=np.float32)
        kernel_q = np.asarray(f["kernel_qs"][:], dtype=np.float32)

    entry_kernel_idx, entry_slot_idx = np.where(valid_mask & done_mask[:, None])
    return {
        "entry_indices": np.arange(len(entry_kernel_idx), dtype=np.int32),
        "kernel_idx": entry_kernel_idx.astype(np.int32),
        "slot_idx": entry_slot_idx.astype(np.int32),
        "start_q": start_q[entry_kernel_idx, entry_slot_idx].astype(np.float32),
        "start_pos": start_pos[entry_kernel_idx, entry_slot_idx].astype(np.float32),
        "direction_vec": direction_vec[entry_kernel_idx, entry_slot_idx].astype(np.float32),
        "line_length": line_length[entry_kernel_idx, entry_slot_idx].astype(np.float32),
        "kernel_q": kernel_q[entry_kernel_idx].astype(np.float32),
    }


def load_validation_entry_indices(bundle_path: Path):
    payload = torch_load(bundle_path)
    metadata = payload.get("metadata", {})
    indices = metadata.get("val_entry_indices")
    if indices is None:
        raise RuntimeError(f"No val_entry_indices found in bundle: {bundle_path}")
    return np.asarray(indices, dtype=np.int32)


def torch_load(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def choose_validation_subset(val_entry_indices: np.ndarray, num_samples: int, seed: int | None):
    rng = np.random.default_rng(seed)
    num_samples = min(num_samples, len(val_entry_indices))
    choice = rng.choice(val_entry_indices, size=num_samples, replace=False)
    return np.asarray(choice, dtype=np.int32)


def build_robot_and_contour():
    contour = WorkspaceContour(contour_path=str(CONTOUR_PATH), z_value=0.0)
    robot = xarm6_sim.XArmLite6Miller(enable_cc=True)
    return robot, contour


def direction_name_from_vec(direction_vec: np.ndarray):
    direction_vec = np.asarray(direction_vec, dtype=float)
    axis = int(np.argmax(np.abs(direction_vec)))
    return ["x", "y", "z"][axis]


def evaluate_q_with_ground_truth_method(robot, contour, start_q: np.ndarray, direction_vec: np.ndarray):
    return trace_line_by_ik(
        robot=robot,
        contour=contour,
        start_q=np.asarray(start_q, dtype=float),
        direction=np.asarray(direction_vec, dtype=float),
        step_size=STEP_SIZE,
        max_steps=MAX_STEPS,
    )


def evaluate_prediction_set(robot, contour, predicted_qs: np.ndarray, direction_vec: np.ndarray):
    evaluated = []
    for q in predicted_qs:
        q = np.asarray(q, dtype=np.float32)
        result = evaluate_q_with_ground_truth_method(robot, contour, q, direction_vec)
        evaluated.append((q, result))

    top1_q, top1_result = evaluated[0]
    best_q, best_result = max(evaluated, key=lambda item: item[1]["line_length"])
    return {
        "top1_q": np.asarray(top1_q, dtype=np.float32),
        "top1_result": top1_result,
        "best_q": np.asarray(best_q, dtype=np.float32),
        "best_result": best_result,
    }


def summarize_results(records):
    gt_lengths = np.asarray([item["gt_length"] for item in records], dtype=np.float32)
    top1_lengths = np.asarray([item["top1_length"] for item in records], dtype=np.float32)
    best8_lengths = np.asarray([item["best8_length"] for item in records], dtype=np.float32)
    return {
        "num_samples": int(len(records)),
        "gt_mean_length": float(gt_lengths.mean()),
        "top1_mean_length": float(top1_lengths.mean()),
        "best8_mean_length": float(best8_lengths.mean()),
        "top1_better_count": int(np.sum(top1_lengths > gt_lengths)),
        "top1_equal_or_better_count": int(np.sum(top1_lengths >= gt_lengths)),
        "best8_better_count": int(np.sum(best8_lengths > gt_lengths)),
        "best8_equal_or_better_count": int(np.sum(best8_lengths >= gt_lengths)),
    }


def save_plot(records, fig_path: Path, title: str):
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    gt_lengths = np.asarray([item["gt_length"] for item in records], dtype=np.float32)
    top1_lengths = np.asarray([item["top1_length"] for item in records], dtype=np.float32)
    best8_lengths = np.asarray([item["best8_length"] for item in records], dtype=np.float32)
    sorted_order = np.argsort(-gt_lengths)
    sample_rank = np.arange(len(records))

    colors = {
        "gt": "#d95f02",
        "top1": "#1f77b4",
        "best8": "#1b9e77",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(sample_rank, gt_lengths[sorted_order], label="ground truth", linewidth=3.0, color=colors["gt"])
    ax.plot(sample_rank, top1_lengths[sorted_order], label="top-1 prediction", linewidth=2.5, color=colors["top1"])
    ax.plot(sample_rank, best8_lengths[sorted_order], label="best-of-8 prediction", linewidth=2.5, color=colors["best8"])
    ax.set_title("Sorted Length Curves")
    ax.set_xlabel("Validation sample rank")
    ax.set_ylabel("Line length (m)")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True)

    ax = axes[0, 1]
    bins = np.linspace(
        min(gt_lengths.min(), top1_lengths.min(), best8_lengths.min()),
        max(gt_lengths.max(), top1_lengths.max(), best8_lengths.max()),
        18,
    )
    ax.hist(gt_lengths, bins=bins, alpha=0.45, color=colors["gt"], label="ground truth", density=True)
    ax.hist(top1_lengths, bins=bins, alpha=0.45, color=colors["top1"], label="top-1", density=True)
    ax.hist(best8_lengths, bins=bins, alpha=0.45, color=colors["best8"], label="best-of-8", density=True)
    ax.set_title("Length Distribution")
    ax.set_xlabel("Line length (m)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.18)
    ax.legend(frameon=True)

    ax = axes[1, 0]
    data = [gt_lengths, top1_lengths, best8_lengths]
    box = ax.boxplot(
        data,
        patch_artist=True,
        labels=["GT", "Top-1", "Best-of-8"],
        widths=0.55,
        medianprops={"color": "black", "linewidth": 2.0},
    )
    for patch, color in zip(box["boxes"], [colors["gt"], colors["top1"], colors["best8"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_title("Length Summary")
    ax.set_ylabel("Line length (m)")
    ax.grid(True, axis="y", alpha=0.2)

    ax = axes[1, 1]
    mean_vals = [float(gt_lengths.mean()), float(top1_lengths.mean()), float(best8_lengths.mean())]
    ax.bar(["GT", "Top-1", "Best-of-8"], mean_vals, color=[colors["gt"], colors["top1"], colors["best8"]], alpha=0.8)
    for idx, val in enumerate(mean_vals):
        ax.text(idx, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_title("Mean Length")
    ax.set_ylabel("Line length (m)")
    ax.grid(True, axis="y", alpha=0.2)

    fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metrics_json(metrics: dict, records, json_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": to_jsonable(metrics),
        "records": to_jsonable(records),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def visualize_record_comparison(record: dict, title: str):
    base = wd.World(cam_pos=[1.2, 0.5, 0.5], lookat_pos=[0.3, 0.0, 0.0])
    mgm.gen_frame().attach_to(base)
    robot = xarm6_sim.XArmLite6Miller(enable_cc=False)

    robot.goto_given_conf(record["gt_q"])
    robot.gen_meshmodel(rgb=[0.95, 0.45, 0.15], alpha=0.45).attach_to(base)

    robot.goto_given_conf(record["top1_q"])
    robot.gen_meshmodel(rgb=[0.15, 0.35, 0.95], alpha=0.32).attach_to(base)

    robot.goto_given_conf(record["best8_q"])
    robot.gen_meshmodel(rgb=[0.1, 0.7, 0.35], alpha=0.42).attach_to(base)

    gt_traj = record["gt_result"]["traj_pos"]
    top1_traj = record["top1_result"]["traj_pos"]
    best8_traj = record["best8_result"]["traj_pos"]
    if len(gt_traj) >= 2:
        mgm.gen_stick(gt_traj[0], gt_traj[-1], radius=0.0025, rgb=[0.95, 0.45, 0.15]).attach_to(base)
    if len(top1_traj) >= 2:
        mgm.gen_stick(top1_traj[0], top1_traj[-1], radius=0.0023, rgb=[0.15, 0.35, 0.95]).attach_to(base)
    if len(best8_traj) >= 2:
        mgm.gen_stick(best8_traj[0], best8_traj[-1], radius=0.0027, rgb=[0.1, 0.7, 0.35]).attach_to(base)
    mgm.gen_sphere(record["start_pos"], radius=0.004, rgb=[0.9, 0.1, 0.1]).attach_to(base)

    print(title)
    print(f"  kernel_idx={record['kernel_idx']} slot_idx={record['slot_idx']} direction={record['direction_name']}")
    print(
        f"  gt_length={record['gt_length']:.4f} "
        f"top1_length={record['top1_length']:.4f} "
        f"best8_length={record['best8_length']:.4f}"
    )
    print("  orange robot/stick: ground-truth start_q and traced line")
    print("  blue robot/stick: top-1 predicted start_q and traced line")
    print("  green robot/stick: best-of-8 predicted start_q and traced line")
    base.run()
