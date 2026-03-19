import argparse
import os
import pickle
import time
from pathlib import Path

for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(env_name, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import math
import numpy as np
from matplotlib.path import Path as MplPath

import h5py
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
from collision_checker import XarmCollisionChecker

BASE_DIR = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine")
DATASET_DIR = BASE_DIR / "datasets"
CHECKPOINT_DIR = DATASET_DIR / "checkpoints"
INPUT_KERNEL_PATH = DATASET_DIR / "cvt_kernels_collision_free.npy"
OUTPUT_H5_PATH = DATASET_DIR / "xarm_trail1_large_scale_top10.h5"
CONTOUR_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl")

N_RAW = 500
LOCAL_RANGE_RADIUS = 0.1
LOCAL_RANGE_SCALE = 2 * LOCAL_RANGE_RADIUS
STEP_SIZE = 0.005
MAX_STEPS = 240
MIN_LINE_LENGTH = 0.1
TOP_K = 10
CHECKPOINT_INTERVAL = 1000

DIRECTION_CONFIGS = {
    "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}


def wrap_to_range(value, jnt_range):
    low, high = jnt_range
    while value < low:
        value += 2 * math.pi
    while value > high:
        value -= 2 * math.pi
    return value


class XArmLite6MillerLite(mi.ManipulatorInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="xarm_lite6_miller_lite"):
        home_conf = np.array([0.0, 0.173311, 0.555015, 0.0, 0.381703, 0.0], dtype=float)
        super().__init__(pos=pos, rotmat=rotmat, home_conf=home_conf, name=name, enable_cc=False)

        self.jlc.jnts[0].loc_pos = np.array([0.0, 0.0, 0.2433])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-math.pi, math.pi])

        self.jlc.jnts[1].loc_pos = np.array([0.0, 0.0, 0.0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(1.5708, -1.5708, 3.1416)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[1].motion_range = np.array([-2.61799, 2.61799])

        self.jlc.jnts[2].loc_pos = np.array([0.2, 0.0, 0.0])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(-3.1416, 0.0, 1.5708)
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].motion_range = np.array([-0.061087, 5.235988])

        self.jlc.jnts[3].loc_pos = np.array([0.087, -0.2276, 0.0])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(1.5708, 0.0, 0.0)
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[3].motion_range = np.array([-math.pi, math.pi])

        self.jlc.jnts[4].loc_pos = np.array([0.0, 0.0, 0.0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(1.5708, 0.0, 0.0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-2.1642, 2.1642])

        self.jlc.jnts[5].loc_pos = np.array([0.0, 0.0615, 0.0])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(-1.5708, 0.0, 0.0)
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].motion_range = np.array([-math.pi, math.pi])

        self.jlc.set_flange(loc_flange_pos=np.array([0.0, 0.0, 0.0]), loc_flange_rotmat=np.eye(3))
        self.jlc.finalize(ik_solver="a", identifier_str=name)
        self.loc_tcp_pos = np.array([0.0, 0.0, 0.22])
        self.loc_tcp_rotmat = np.eye(3)

    def ik(self, tgt_pos, tgt_rotmat, seed_jnt_values=None, option="single", toggle_dbg=False):
        solutions = []
        tcp_loc_pos = self.loc_tcp_pos
        tcp_loc_rotmat = self.loc_tcp_rotmat
        tgt_flange_rotmat = tgt_rotmat @ tcp_loc_rotmat.T
        tgt_flange_pos = tgt_pos - tgt_flange_rotmat @ tcp_loc_pos
        rrr_pos = tgt_flange_pos - tgt_flange_rotmat[:, 2] * np.linalg.norm(self.jlc.jnts[5].loc_pos)
        rrr_x, rrr_y, rrr_z = ((rrr_pos - self.pos) @ self.rotmat).tolist()

        j0_value_candidates = [math.atan2(rrr_y, rrr_x)]
        for j0_value in j0_value_candidates:
            if not self._is_jnt_in_range(jnt_id=0, jnt_value=j0_value):
                continue
            c = math.sqrt(rrr_x ** 2 + rrr_y ** 2 + (rrr_z - self.jlc.jnts[0].loc_pos[2]) ** 2)
            a = self.jlc.jnts[2].loc_pos[0]
            b = np.linalg.norm(self.jlc.jnts[3].loc_pos)
            tmp_acos_target = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
            if tmp_acos_target > 1 or tmp_acos_target < -1:
                continue

            j2_value_candidates = []
            j2_value = math.acos(tmp_acos_target)
            j2_initial_offset = math.atan(abs(self.jlc.jnts[3].loc_pos[0] / self.jlc.jnts[3].loc_pos[1]))
            j2_value_candidates.append(j2_value - j2_initial_offset)
            j2_value_candidates.append(wrap_to_range(-(j2_value + j2_initial_offset), self.jnt_ranges[2]))

            for idx, j2_value in enumerate(j2_value_candidates):
                if not self._is_jnt_in_range(jnt_id=2, jnt_value=j2_value):
                    continue
                tmp_acos_target = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
                if tmp_acos_target > 1 or tmp_acos_target < -1:
                    continue
                j1_value_upper = math.acos(tmp_acos_target)
                d = self.jlc.jnts[0].loc_pos[2]
                e = math.sqrt(rrr_x ** 2 + rrr_y ** 2 + rrr_z ** 2)
                tmp_acos_target = (d ** 2 + c ** 2 - e ** 2) / (2 * d * c)
                if tmp_acos_target > 1 or tmp_acos_target < -1:
                    continue
                j1_value_lower = math.acos(tmp_acos_target)
                if idx == 0:
                    j1_value = math.pi - (j1_value_lower + j1_value_upper)
                else:
                    j1_value = j1_value_upper + math.pi - j1_value_lower
                if not self._is_jnt_in_range(jnt_id=1, jnt_value=j1_value):
                    continue

                anchor_gl_rotmatq = self.rotmat
                j0_gl_rotmat0 = anchor_gl_rotmatq @ self.jlc.jnts[0].loc_rotmat
                j0_gl_rotmatq = j0_gl_rotmat0 @ rm.rotmat_from_axangle(self.jlc.jnts[0].loc_motion_ax, j0_value)
                j1_gl_rotmat0 = j0_gl_rotmatq @ self.jlc.jnts[1].loc_rotmat
                j1_gl_rotmatq = j1_gl_rotmat0 @ rm.rotmat_from_axangle(self.jlc.jnts[1].loc_motion_ax, j1_value)
                j2_gl_rotmat0 = j1_gl_rotmatq @ self.jlc.jnts[2].loc_rotmat
                j2_gl_rotmatq = j2_gl_rotmat0 @ rm.rotmat_from_axangle(self.jlc.jnts[2].loc_motion_ax, j2_value)
                rrr_g_rotmat = (
                    j2_gl_rotmatq
                    @ self.jlc.jnts[3].loc_rotmat
                    @ self.jlc.jnts[4].loc_rotmat
                    @ self.jlc.jnts[5].loc_rotmat
                )
                j3_value, j4_value, j5_value = rm.rotmat_to_euler(
                    rrr_g_rotmat.T @ tgt_flange_rotmat, order="rzyz"
                ).tolist()
                j4_value = -j4_value
                solutions.append(np.array([j0_value, j1_value, j2_value, j3_value, j4_value, j5_value]))

                if self._is_jnt_in_range(jnt_id=4, jnt_value=j4_value) and self._is_jnt_in_range(
                    jnt_id=4, jnt_value=-j4_value
                ):
                    j4_value = -j4_value
                    j3_value -= np.pi
                    j5_value -= np.pi
                    solutions.append(np.array([j0_value, j1_value, j2_value, j3_value, j4_value, j5_value]))

        if option == "single":
            return None if len(solutions) == 0 else solutions[0]
        return solutions


class WorkspaceContour:
    def __init__(self, contour_path, z_value=0.0):
        with open(contour_path, "rb") as f:
            self.contour = pickle.load(f)
        self.path = MplPath(self.contour)
        self.z_value = z_value

    def contains_xy(self, xy_points):
        return self.path.contains_points(xy_points)


def is_pose_inside_workspace(contour, pos):
    return bool(contour.contains_xy(np.asarray(pos[:2], dtype=float).reshape(1, 2))[0])


def wrap_angle_difference(delta):
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def select_closest_solution(solutions, seed_q):
    if not solutions:
        return None
    best_q = None
    best_dist = None
    for q in solutions:
        q = np.asarray(q, dtype=np.float32)
        dist = np.linalg.norm(wrap_angle_difference(q - seed_q))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_q = q
    return best_q


def sample_local_start_q(robot, num_samples, center_q=None, range_scale=LOCAL_RANGE_SCALE):
    jnt_ranges = np.asarray(robot.jnt_ranges, dtype=np.float32)
    center_q = np.asarray(robot.rand_conf() if center_q is None else center_q, dtype=np.float32)
    span = (jnt_ranges[:, 1] - jnt_ranges[:, 0]) * range_scale
    low = np.maximum(jnt_ranges[:, 0], center_q - 0.5 * span)
    high = np.minimum(jnt_ranges[:, 1], center_q + 0.5 * span)
    samples = np.random.uniform(low=low, high=high, size=(num_samples, robot.n_dof)).astype(np.float32)
    return samples, low, high


def check_robot_collision(robot, collision_checker, q):
    if collision_checker.check_collision(q):
        return True, "self_collision"
    return False, None


def evaluate_start_state(robot, contour, collision_checker, start_q):
    start_q = np.asarray(start_q, dtype=np.float32)
    start_pos, start_rot = robot.fk(start_q)
    start_pos = np.asarray(start_pos, dtype=np.float32)
    start_rot = np.asarray(start_rot, dtype=np.float32)

    if not is_pose_inside_workspace(contour, start_pos):
        return False, start_q, start_pos, start_rot, "start_outside_workspace"

    is_collided, collision_reason = check_robot_collision(robot, collision_checker, start_q)
    if is_collided:
        return False, start_q, start_pos, start_rot, collision_reason if collision_reason is not None else "start_in_collision"

    return True, start_q, start_pos, start_rot, None


def trace_line_by_ik_from_state(
    robot,
    contour,
    collision_checker,
    start_q,
    start_pos,
    start_rot,
    direction,
    step_size,
    max_steps,
):
    start_q = np.asarray(start_q, dtype=np.float32)
    start_pos = np.asarray(start_pos, dtype=np.float32)
    start_rot = np.asarray(start_rot, dtype=np.float32)

    current_q = start_q.copy()
    success_steps = 0

    for step_idx in range(1, max_steps + 1):
        tgt_pos = start_pos + direction * (step_idx * step_size)
        if not is_pose_inside_workspace(contour, tgt_pos):
            return {
                "start_q": start_q,
                "direction_vec": np.asarray(direction, dtype=np.float32),
                "start_pos": start_pos,
                "line_length": success_steps * step_size,
                "success": success_steps > 0,
                "termination_reason": "out_of_workspace",
            }

        ik_solutions = robot.ik(
            tgt_pos=tgt_pos,
            tgt_rotmat=start_rot,
            seed_jnt_values=current_q,
            option="multiple",
        )
        if ik_solutions is None or len(ik_solutions) == 0:
            return {
                "start_q": start_q,
                "direction_vec": np.asarray(direction, dtype=np.float32),
                "start_pos": start_pos,
                "line_length": success_steps * step_size,
                "success": success_steps > 0,
                "termination_reason": "ik_failed",
            }

        next_q = select_closest_solution(ik_solutions, current_q)
        if next_q is None or not robot.are_jnts_in_ranges(next_q):
            return {
                "start_q": start_q,
                "direction_vec": np.asarray(direction, dtype=np.float32),
                "start_pos": start_pos,
                "line_length": success_steps * step_size,
                "success": success_steps > 0,
                "termination_reason": "joint_limit",
            }

        is_collided, collision_reason = check_robot_collision(robot, collision_checker, next_q)
        if is_collided:
            return {
                "start_q": start_q,
                "direction_vec": np.asarray(direction, dtype=np.float32),
                "start_pos": start_pos,
                "line_length": success_steps * step_size,
                "success": success_steps > 0,
                "termination_reason": collision_reason,
            }

        current_q = np.asarray(next_q, dtype=np.float32)
        success_steps += 1

    return {
        "start_q": start_q,
        "direction_vec": np.asarray(direction, dtype=np.float32),
        "start_pos": start_pos,
        "line_length": success_steps * step_size,
        "success": success_steps > 0,
        "termination_reason": "max_steps_reached",
    }


def collect_kernel_records(
    kernel_idx,
    kernel_q,
    robot,
    contour,
    collision_checker,
    n_raw,
    range_scale,
    step_size,
    max_steps,
    min_line_length,
):
    np.random.seed(20260319 + int(kernel_idx))
    sampled_qs, _, _ = sample_local_start_q(robot=robot, num_samples=n_raw, center_q=kernel_q, range_scale=range_scale)

    records = []
    for start_q in sampled_qs:
        is_valid_start, start_q, start_pos, start_rot, invalid_reason = evaluate_start_state(
            robot=robot,
            contour=contour,
            collision_checker=collision_checker,
            start_q=start_q,
        )
        if not is_valid_start:
            continue

        for direction_name, direction_vec in DIRECTION_CONFIGS.items():
            result = trace_line_by_ik_from_state(
                robot=robot,
                contour=contour,
                collision_checker=collision_checker,
                start_q=start_q,
                start_pos=start_pos,
                start_rot=start_rot,
                direction=direction_vec,
                step_size=step_size,
                max_steps=max_steps,
            )
            if result["line_length"] < min_line_length:
                continue
            records.append(
                (
                    float(result["line_length"]),
                    np.asarray(result["start_q"], dtype=np.float32),
                    np.asarray(result["direction_vec"], dtype=np.float32),
                    np.asarray(result["start_pos"], dtype=np.float32),
                    direction_name,
                )
            )

    records.sort(key=lambda item: item[0], reverse=True)
    top_records = records[:TOP_K]

    start_qs = np.full((TOP_K, 6), np.nan, dtype=np.float32)
    direction_vecs = np.full((TOP_K, 3), np.nan, dtype=np.float32)
    start_poss = np.full((TOP_K, 3), np.nan, dtype=np.float32)
    line_lengths = np.full((TOP_K,), np.nan, dtype=np.float32)
    direction_names = np.full((TOP_K,), "", dtype="S1")
    valid_mask = np.zeros((TOP_K,), dtype=bool)

    for rank, record in enumerate(top_records):
        line_length, start_q, direction_vec, start_pos, direction_name = record
        start_qs[rank] = start_q
        direction_vecs[rank] = direction_vec
        start_poss[rank] = start_pos
        line_lengths[rank] = line_length
        direction_names[rank] = direction_name.encode("ascii")
        valid_mask[rank] = True

    return {
        "kernel_idx": int(kernel_idx),
        "valid_count": int(len(top_records)),
        "start_qs": start_qs,
        "direction_vecs": direction_vecs,
        "start_poss": start_poss,
        "line_lengths": line_lengths,
        "direction_names": direction_names,
        "valid_mask": valid_mask,
    }


def create_or_open_h5(output_path, kernel_qs, resume, input_kernel_path, checkpoint_interval):
    if h5py is None:
        raise ImportError("h5py is required to write the dataset in HDF5 format.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and output_path.exists() else "w"
    h5_file = h5py.File(output_path, mode)

    num_kernels = kernel_qs.shape[0]
    if "kernel_qs" not in h5_file:
        h5_file.create_dataset("kernel_qs", data=kernel_qs.astype(np.float32), compression="gzip", shuffle=True)
        h5_file.create_dataset(
            "top_start_q",
            shape=(num_kernels, TOP_K, 6),
            dtype=np.float32,
            compression="gzip",
            shuffle=True,
            fillvalue=np.nan,
        )
        h5_file.create_dataset(
            "top_direction_vec",
            shape=(num_kernels, TOP_K, 3),
            dtype=np.float32,
            compression="gzip",
            shuffle=True,
            fillvalue=np.nan,
        )
        h5_file.create_dataset(
            "top_start_pos",
            shape=(num_kernels, TOP_K, 3),
            dtype=np.float32,
            compression="gzip",
            shuffle=True,
            fillvalue=np.nan,
        )
        h5_file.create_dataset(
            "top_line_length",
            shape=(num_kernels, TOP_K),
            dtype=np.float32,
            compression="gzip",
            shuffle=True,
            fillvalue=np.nan,
        )
        h5_file.create_dataset(
            "top_direction_name",
            shape=(num_kernels, TOP_K),
            dtype="S1",
            compression="gzip",
            shuffle=True,
        )
        h5_file.create_dataset(
            "top_valid_mask",
            shape=(num_kernels, TOP_K),
            dtype=np.bool_,
            compression="gzip",
            shuffle=True,
        )
        h5_file.create_dataset("top_valid_count", shape=(num_kernels,), dtype=np.int32, compression="gzip", shuffle=True)
        h5_file.create_dataset("done_mask", shape=(num_kernels,), dtype=np.bool_, compression="gzip", shuffle=True)
        h5_file["done_mask"][:] = False
        h5_file.attrs["created_unix_time"] = time.time()

    stored_kernel_shape = tuple(h5_file["kernel_qs"].shape)
    if stored_kernel_shape != tuple(kernel_qs.shape):
        raise ValueError(
            f"Existing HDF5 kernel shape {stored_kernel_shape} does not match current kernel shape {kernel_qs.shape}."
        )

    h5_file.attrs["n_raw"] = N_RAW
    h5_file.attrs["local_range_radius"] = LOCAL_RANGE_RADIUS
    h5_file.attrs["local_range_scale"] = LOCAL_RANGE_SCALE
    h5_file.attrs["step_size"] = STEP_SIZE
    h5_file.attrs["max_steps"] = MAX_STEPS
    h5_file.attrs["min_line_length"] = MIN_LINE_LENGTH
    h5_file.attrs["top_k"] = TOP_K
    h5_file.attrs["checkpoint_interval"] = checkpoint_interval
    h5_file.attrs["contour_path"] = str(CONTOUR_PATH)
    h5_file.attrs["input_kernel_path"] = str(input_kernel_path)
    h5_file.attrs["environment_obstacles"] = "none"
    h5_file.attrs["collision_backend"] = "XarmCollisionChecker"
    h5_file.attrs["collision_device"] = "cpu"
    return h5_file


def write_kernel_result(h5_file, result):
    idx = result["kernel_idx"]
    h5_file["top_start_q"][idx] = result["start_qs"]
    h5_file["top_direction_vec"][idx] = result["direction_vecs"]
    h5_file["top_start_pos"][idx] = result["start_poss"]
    h5_file["top_line_length"][idx] = result["line_lengths"]
    h5_file["top_direction_name"][idx] = result["direction_names"]
    h5_file["top_valid_mask"][idx] = result["valid_mask"]
    h5_file["top_valid_count"][idx] = result["valid_count"]
    h5_file["done_mask"][idx] = True


def flush_checkpoint(h5_file, output_path, started_at):
    done_count = int(np.count_nonzero(h5_file["done_mask"][:]))
    h5_file.attrs["completed_kernels"] = done_count
    h5_file.attrs["last_flush_unix_time"] = time.time()
    h5_file.flush()
    elapsed = time.time() - started_at
    print(
        f"[Checkpoint] completed={done_count}/{h5_file['done_mask'].shape[0]} | "
        f"elapsed={elapsed / 60.0:.1f} min | file={output_path}"
    , flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Large-scale XArm straight-line dataset collection with checkpointing.")
    parser.add_argument("--input-npy", type=Path, default=INPUT_KERNEL_PATH)
    parser.add_argument("--output-h5", type=Path, default=OUTPUT_H5_PATH)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    parser.add_argument("--resume", action="store_true", help="Resume from existing HDF5 if present.")
    return parser.parse_args()


def main():
    args = parse_args()

    if h5py is None:
        raise ImportError("h5py is not available in the current environment. Install it in the 'wrs' environment first.")

    if not args.input_npy.exists():
        raise FileNotFoundError(f"Input kernel file not found: {args.input_npy}")

    kernel_qs = np.load(args.input_npy).astype(np.float32)
    if kernel_qs.ndim != 2 or kernel_qs.shape[1] != 6:
        raise ValueError(f"Expected input kernel array shape [N, 6], got {kernel_qs.shape}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    h5_file = create_or_open_h5(
        args.output_h5,
        kernel_qs,
        resume=args.resume,
        input_kernel_path=args.input_npy,
        checkpoint_interval=args.checkpoint_interval,
    )
    done_mask = h5_file["done_mask"][:]
    pending_indices = np.flatnonzero(~done_mask)

    print(f"[Config] input={args.input_npy}", flush=True)
    print(f"[Config] output={args.output_h5}", flush=True)
    print(f"[Config] kernels={len(kernel_qs)} | pending={len(pending_indices)}", flush=True)
    print(f"[Config] execution=single_process | n_raw={N_RAW} | top_k={TOP_K}", flush=True)
    print(
        f"[Config] local_range_radius={LOCAL_RANGE_RADIUS} | "
        f"local_range_scale={LOCAL_RANGE_SCALE} | "
        f"step={STEP_SIZE} | max_steps={MAX_STEPS} | min_length={MIN_LINE_LENGTH}"
    , flush=True)

    if len(pending_indices) == 0:
        print("[INFO] No pending kernels. The HDF5 dataset is already complete.", flush=True)
        h5_file.close()
        return

    started_at = time.time()
    processed_since_flush = 0
    total_done_before = int(np.count_nonzero(done_mask))
    print("[Init] Building lightweight robot model...", flush=True)
    robot = XArmLite6MillerLite()
    print("[Init] Loading workspace contour...", flush=True)
    contour = WorkspaceContour(contour_path=str(CONTOUR_PATH), z_value=0.0)
    print("[Init] Building collision checker on CPU...", flush=True)
    collision_checker = XarmCollisionChecker(device="cpu")
    print("[Init] Initialization complete. Starting dataset collection...", flush=True)

    try:
        for completed_idx, kernel_idx in enumerate(pending_indices, start=1):
            kernel_start_time = time.time()
            result = collect_kernel_records(
                kernel_idx=int(kernel_idx),
                kernel_q=kernel_qs[int(kernel_idx)],
                robot=robot,
                contour=contour,
                collision_checker=collision_checker,
                n_raw=N_RAW,
                range_scale=LOCAL_RANGE_SCALE,
                step_size=STEP_SIZE,
                max_steps=MAX_STEPS,
                min_line_length=MIN_LINE_LENGTH,
            )
            write_kernel_result(h5_file, result)
            processed_since_flush += 1

            if completed_idx % 1 == 0:
                total_done_now = total_done_before + completed_idx
                elapsed = time.time() - started_at
                kernel_elapsed = time.time() - kernel_start_time
                print(
                    f"[Progress] processed={completed_idx}/{len(pending_indices)} | "
                    f"done_total={total_done_now}/{len(kernel_qs)} | "
                    f"kernel_dt={kernel_elapsed:.1f}s | "
                    f"elapsed={elapsed / 60.0:.1f} min"
                , flush=True)

            if processed_since_flush >= args.checkpoint_interval:
                flush_checkpoint(h5_file, args.output_h5, started_at)
                processed_since_flush = 0
    finally:
        flush_checkpoint(h5_file, args.output_h5, started_at)
        h5_file.close()


if __name__ == "__main__":
    main()
