from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .franka_single_point_config import TrackerConfig
from .math_utils import directional_manipulability, directional_manipulability_gradient, damped_pseudoinverse, normalize, nullspace_projector
from .pybullet_franka import PyBulletFranka


@dataclass
class TrackerResult:
    line_length: float
    average_mu: float
    cumulative_mu: float
    final_q: np.ndarray
    q_traj: np.ndarray
    pos_traj: np.ndarray
    mu_traj: np.ndarray
    termination_reason: str


class NullSpaceTracker:
    """Low-level executor

    Integrates the differential command

        q_dot = J^# v + (I - J^# J) q_dot_null

    where the primary task moves the TCP along a fixed Cartesian direction d, and
    the null-space term ascends the directional manipulability gradient.
    """

    def __init__(self, robot: PyBulletFranka, config: TrackerConfig | None = None) -> None:
        self.robot = robot
        self.config = config or TrackerConfig()

    def _mu_at(self, q: np.ndarray, direction: np.ndarray) -> float:
        return directional_manipulability(self.robot.position_jacobian(q), direction, self.config.damping)

    def run(self, q0: np.ndarray, direction: np.ndarray) -> TrackerResult:
        cfg = self.config
        direction = normalize(direction)
        q = np.asarray(q0, dtype=np.float64).copy()
        fk = self.robot.fk(q)
        q_traj = [q.copy()]
        pos_traj = [fk.position.copy()]
        mu_traj = []
        total_length = 0.0
        cumulative_mu = 0.0
        termination_reason = "max_steps_reached"

        for _ in range(cfg.max_steps):
            j_pos = self.robot.position_jacobian(q)
            mu = directional_manipulability(j_pos, direction, cfg.damping)
            mu_traj.append(mu)
            if mu < cfg.mu_threshold:
                termination_reason = "mu_below_threshold"
                break

            mu_grad = directional_manipulability_gradient(
                q,
                mu_fn=lambda q_eval: self._mu_at(q_eval, direction),
                step=cfg.gradient_step,
            )
            qdot_task = damped_pseudoinverse(j_pos, cfg.damping) @ (direction * cfg.task_speed)
            qdot_null = cfg.nullspace_gain * mu_grad
            qdot = qdot_task + nullspace_projector(j_pos, cfg.damping) @ qdot_null
            q_next = q + qdot * cfg.integration_dt

            if not self.robot.in_joint_limits(q_next, margin=cfg.joint_limit_margin):
                termination_reason = "joint_limit"
                break
            q = self.robot.clip_to_limits(q_next, margin=cfg.joint_limit_margin)
            fk_next = self.robot.fk(q)
            if self.robot.self_collision():
                termination_reason = "self_collision"
                break

            delta_pos = fk_next.position - pos_traj[-1]
            step_length = float(np.dot(delta_pos, direction))
            if step_length <= 0.0:
                termination_reason = "stalled"
                break
            total_length += step_length
            cumulative_mu += mu * step_length
            q_traj.append(q.copy())
            pos_traj.append(fk_next.position.copy())

        average_mu = cumulative_mu / total_length if total_length > 1e-12 else 0.0
        return TrackerResult(
            line_length=float(total_length),
            average_mu=float(average_mu),
            cumulative_mu=float(cumulative_mu),
            final_q=q.copy(),
            q_traj=np.asarray(q_traj, dtype=np.float64),
            pos_traj=np.asarray(pos_traj, dtype=np.float64),
            mu_traj=np.asarray(mu_traj, dtype=np.float64),
            termination_reason=termination_reason,
        )
