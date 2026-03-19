import os
import argparse
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np

import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
from wrs import wd
from xarm_trail1 import MAX_STEPS, STEP_SIZE, WorkspaceContour, trace_line_by_ik


BASE_DIR = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine")
DEFAULT_H5_PATH = BASE_DIR / "datasets" / "checkpoints" / "checkpoint_00100.h5"
CONTOUR_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl")
DEFAULT_KERNEL_IDX = 0
DEFAULT_SLOT_IDX = None
DEFAULT_DIRECTION = None
DEFAULT_LIST_ONLY = False


def _decode_direction(value):
    if isinstance(value, bytes):
        return value.decode("ascii")
    if hasattr(value, "decode"):
        return value.decode("ascii")
    return str(value)


def _pick_slot(valid_mask, direction_names, slot_idx=None, direction=None):
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        raise ValueError("This kernel has no valid records.")

    if slot_idx is not None:
        if slot_idx < 0 or slot_idx >= len(valid_mask):
            raise IndexError(f"slot_idx={slot_idx} out of range [0, {len(valid_mask) - 1}]")
        if not valid_mask[slot_idx]:
            raise ValueError(f"slot_idx={slot_idx} is invalid according to top_valid_mask.")
        return int(slot_idx)

    if direction is not None:
        direction = direction.lower()
        for idx in valid_indices:
            if _decode_direction(direction_names[idx]).lower() == direction:
                return int(idx)
        raise ValueError(f"No valid record found for direction={direction!r}.")

    return int(valid_indices[0])


def load_record(h5_path, kernel_idx, slot_idx=None, direction=None):
    with h5py.File(h5_path, "r") as f:
        num_kernels = int(f["kernel_qs"].shape[0])
        if kernel_idx < 0 or kernel_idx >= num_kernels:
            raise IndexError(f"kernel_idx={kernel_idx} out of range [0, {num_kernels - 1}]")

        kernel_q = np.asarray(f["kernel_qs"][kernel_idx], dtype=float)
        valid_mask = np.asarray(f["top_valid_mask"][kernel_idx], dtype=bool)
        direction_names = np.asarray(f["top_direction_name"][kernel_idx])
        line_lengths = np.asarray(f["top_line_length"][kernel_idx], dtype=float)
        start_qs = np.asarray(f["top_start_q"][kernel_idx], dtype=float)
        start_poss = np.asarray(f["top_start_pos"][kernel_idx], dtype=float)
        direction_vecs = np.asarray(f["top_direction_vec"][kernel_idx], dtype=float)
        done_mask = bool(f["done_mask"][kernel_idx])
        valid_count = int(f["top_valid_count"][kernel_idx])

        chosen_slot = _pick_slot(
            valid_mask=valid_mask,
            direction_names=direction_names,
            slot_idx=slot_idx,
            direction=direction,
        )

        slot_summaries = []
        for idx in np.flatnonzero(valid_mask):
            slot_summaries.append(
                {
                    "slot_idx": int(idx),
                    "direction": _decode_direction(direction_names[idx]),
                    "line_length": float(line_lengths[idx]),
                    "start_q": start_qs[idx].copy(),
                    "start_pos": start_poss[idx].copy(),
                    "direction_vec": direction_vecs[idx].copy(),
                }
            )

        chosen_summary = next(item for item in slot_summaries if item["slot_idx"] == chosen_slot)

        return {
            "kernel_idx": int(kernel_idx),
            "chosen_slot": int(chosen_slot),
            "done_mask": done_mask,
            "valid_count": valid_count,
            "kernel_q": kernel_q,
            "slot_summaries": slot_summaries,
            "start_q": chosen_summary["start_q"].copy(),
            "start_pos": chosen_summary["start_pos"].copy(),
            "direction_vec": chosen_summary["direction_vec"].copy(),
            "direction_name": chosen_summary["direction"],
            "line_length": float(chosen_summary["line_length"]),
        }


def print_record_summary(record):
    print(f"h5 kernel_idx: {record['kernel_idx']}")
    print(f"done_mask: {record['done_mask']}")
    print(f"valid_count: {record['valid_count']}")
    print(f"chosen_slot: {record['chosen_slot']}")
    print(f"kernel_q: {np.array2string(record['kernel_q'], precision=4, separator=', ')}")
    print("")
    print("valid slots:")
    for item in record["slot_summaries"]:
        print(
            f"  slot={item['slot_idx']:02d} | dir={item['direction']} | "
            f"L={item['line_length']:.4f} m | "
            f"start_pos={np.array2string(item['start_pos'], precision=4, separator=', ')}"
        )
    print("")
    print("chosen record:")
    print(f"  direction: {record['direction_name']}")
    print(f"  line_length: {record['line_length']:.4f} m")
    print(f"  start_q: {np.array2string(record['start_q'], precision=4, separator=', ')}")
    print(f"  direction_vec: {np.array2string(record['direction_vec'], precision=4, separator=', ')}")
    print(f"  start_pos: {np.array2string(record['start_pos'], precision=4, separator=', ')}")


def visualize_record(record):
    base = wd.World(cam_pos=[1.2, 0.5, 0.5], lookat_pos=[0.3, 0.0, 0.0])
    mgm.gen_frame().attach_to(base)

    robot = xarm6_sim.XArmLite6Miller(enable_cc=False)
    robot.goto_given_conf(record["kernel_q"])
    robot.gen_meshmodel(rgb=[0.75, 0.75, 0.75], alpha=0.18).attach_to(base)

    direction_colors = {
        "x": [0.9, 0.2, 0.2],
        "y": [0.2, 0.75, 0.2],
        "z": [0.2, 0.35, 0.9],
    }

    for item in record["slot_summaries"]:
        direction_name = item["direction"].lower()
        rgb = direction_colors.get(direction_name, [0.7, 0.7, 0.7])
        start_pos = np.asarray(item["start_pos"], dtype=float)
        direction_vec = np.asarray(item["direction_vec"], dtype=float)
        line_length = float(item["line_length"])
        end_pos = start_pos + direction_vec * line_length

        robot.goto_given_conf(item["start_q"])
        alpha = 0.28 if item["slot_idx"] == record["chosen_slot"] else 0.12
        robot.gen_meshmodel(rgb=rgb, alpha=alpha).attach_to(base)

        mgm.gen_sphere(start_pos, radius=0.0045, rgb=rgb).attach_to(base)
        mgm.gen_stick(start_pos, end_pos, radius=0.0025, rgb=rgb).attach_to(base)
        mgm.gen_sphere(end_pos, radius=0.0035, rgb=rgb).attach_to(base)
        mgm.gen_arrow(
            spos=start_pos,
            epos=start_pos + direction_vec * 0.06,
            stick_radius=0.0016,
            rgb=rgb,
        ).attach_to(base)

    print("")
    print("visualization:")
    print("  gray robot : kernel center q")
    print("  red / green / blue robots : all valid X / Y / Z records in this kernel")
    print("  same-color spheres and sticks : start_pos and valid straight segment")
    print("  chosen_slot is rendered with slightly higher alpha")
    base.run()


def verify_record_batch(record, atol=1e-6):
    contour = WorkspaceContour(contour_path=str(CONTOUR_PATH), z_value=0.0)
    robot = xarm6_sim.XArmLite6Miller(enable_cc=True)

    print("")
    print("verification:")
    all_ok = True
    for item in record["slot_summaries"]:
        recomputed = trace_line_by_ik(
            robot=robot,
            contour=contour,
            start_q=np.asarray(item["start_q"], dtype=float),
            direction=np.asarray(item["direction_vec"], dtype=float),
            step_size=STEP_SIZE,
            max_steps=MAX_STEPS,
        )
        stored_length = float(item["line_length"])
        recomputed_length = float(recomputed["line_length"])
        length_diff = abs(stored_length - recomputed_length)
        start_pos_diff = float(
            np.linalg.norm(np.asarray(item["start_pos"], dtype=float) - np.asarray(recomputed["start_pos"], dtype=float))
        )
        ok = (length_diff <= atol) and (start_pos_diff <= atol)
        all_ok = all_ok and ok
        print(
            f"  slot={item['slot_idx']:02d} | dir={item['direction']} | "
            f"stored_L={stored_length:.4f} | recomputed_L={recomputed_length:.4f} | "
            f"diff={length_diff:.6f} | start_pos_diff={start_pos_diff:.6f} | "
            f"reason={recomputed['termination_reason']} | ok={ok}"
        )

    print("")
    print(f"verification_result: all_ok={all_ok}")
    return all_ok


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect and visualize one kernel batch from xarm_trail1 HDF5 dataset.")
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--kernel-idx", type=int, default=DEFAULT_KERNEL_IDX)
    parser.add_argument("--slot-idx", type=int, default=DEFAULT_SLOT_IDX, help="Optional chosen slot in [0, 14].")
    parser.add_argument("--direction", type=str, default=DEFAULT_DIRECTION, help="Optional chosen direction: x/y/z.")
    parser.add_argument("--list-only", action="store_true", default=DEFAULT_LIST_ONLY, help="Only print the record summary without opening GUI.")
    parser.add_argument("--verify", action="store_true", help="Recompute all valid records in this kernel and compare with HDF5.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {args.h5_path}")

    record = load_record(
        h5_path=args.h5_path,
        kernel_idx=args.kernel_idx,
        slot_idx=args.slot_idx,
        direction=args.direction,
    )
    print_record_summary(record)

    if args.verify:
        verify_record_batch(record)

    if not args.list_only:
        visualize_record(record)


if __name__ == "__main__":
    main()
