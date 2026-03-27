from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial.transform import Rotation

from .franka_single_point_config import CONTOUR_PATH, TASK_SAMPLE_RETRIES, WORKSPACE_Z, TrackerConfig
from .math_utils import damped_pseudoinverse, linear_map_range_to_tanh, linear_map_tanh_to_range, normalize, random_unit_vector, wrap_angle_difference
from .pybullet_franka import PyBulletFranka


class WorkspaceContour:
    def __init__(self, contour_path: str | Path = CONTOUR_PATH, z_value: float = WORKSPACE_Z):
        with open(contour_path, "rb") as f:
            contour = pickle.load(f)
        self.contour = np.asarray(contour, dtype=np.float64)
        self.path = MplPath(self.contour)
        self.z_value = float(z_value)

    def contains_xy(self, xy_points: np.ndarray) -> np.ndarray:
        return self.path.contains_points(np.asarray(xy_points, dtype=np.float64))


def is_pose_inside_workspace(contour: WorkspaceContour, pos: np.ndarray) -> bool:
    return bool(contour.contains_xy(np.asarray(pos[:2], dtype=np.float64).reshape(1, 2))[0])


def canonicalize_quaternion_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)
    if quat_xyzw[3] < 0.0:
        quat_xyzw = -quat_xyzw
    return quat_xyzw


def random_quaternion_xyzw(rng: np.random.Generator) -> np.ndarray:
    return canonicalize_quaternion_xyzw(Rotation.random(random_state=rng).as_quat())


def random_task(robot: PyBulletFranka, contour: WorkspaceContour, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for _ in range(TASK_SAMPLE_RETRIES):
        q = robot.sample_random_configuration(rng)
        fk = robot.fk(q)
        if not is_pose_inside_workspace(contour, fk.position):
            continue
        if robot.self_collision():
            continue
        return q, fk.position.copy(), random_unit_vector(rng)
    raise RuntimeError("Failed to sample a valid task from random Franka configurations.")


def action_to_joint_positions(action: np.ndarray, robot: PyBulletFranka) -> np.ndarray:
    low, high = robot.get_joint_limits()
    return linear_map_tanh_to_range(action, low, high)


def joint_positions_to_action(q: np.ndarray, robot: PyBulletFranka) -> np.ndarray:
    low, high = robot.get_joint_limits()
    return linear_map_range_to_tanh(q, low, high)


def jacobian_position_correction(
    robot: PyBulletFranka,
    q_init: np.ndarray,
    target_position: np.ndarray,
    config: TrackerConfig,
) -> tuple[np.ndarray, float, bool]:
    """Iteratively project q to the exact target position using DLS Jacobian steps."""
    q = np.asarray(q_init, dtype=np.float64).copy()
    for _ in range(config.correction_max_iters):
        fk = robot.fk(q)
        err = np.asarray(target_position, dtype=np.float64) - fk.position
        if float(np.linalg.norm(err)) < config.correction_tol:
            q = robot.clip_to_limits(q, margin=config.joint_limit_margin)
            return q, float(np.linalg.norm(wrap_angle_difference(q - q_init))), True
        j_pos = robot.position_jacobian(q)
        dq = damped_pseudoinverse(j_pos, config.correction_damping) @ err
        q = robot.clip_to_limits(q + dq, margin=config.joint_limit_margin)
        if robot.self_collision():
            break
    return q, float(np.linalg.norm(wrap_angle_difference(q - q_init))), False
