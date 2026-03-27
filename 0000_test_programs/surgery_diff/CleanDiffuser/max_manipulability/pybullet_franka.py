from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:  # pragma: no cover
    raise ImportError("pybullet is required for this framework.") from exc


@dataclass
class FKResult:
    position: np.ndarray
    orientation_xyzw: np.ndarray


class PyBulletFranka:
    """Thin PyBullet wrapper around the Panda 7-DoF arm."""

    def __init__(self, use_gui: bool = False) -> None:
        self.client_id = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.body_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.client_id,
        )
        self.arm_joint_indices = []
        self.finger_joint_indices = []
        lower = []
        upper = []
        for joint_idx in range(p.getNumJoints(self.body_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.body_id, joint_idx, physicsClientId=self.client_id)
            name = info[1].decode("utf-8")
            joint_type = info[2]
            if name.startswith("panda_joint") and joint_type == p.JOINT_REVOLUTE:
                self.arm_joint_indices.append(joint_idx)
                lower.append(info[8])
                upper.append(info[9])
            elif name.startswith("panda_finger_joint") and joint_type == p.JOINT_PRISMATIC:
                self.finger_joint_indices.append(joint_idx)
        if len(self.arm_joint_indices) != 7:
            raise RuntimeError("Expected 7 revolute Franka arm joints.")
        if len(self.finger_joint_indices) != 2:
            raise RuntimeError("Expected 2 prismatic Franka finger joints.")
        self.arm_joint_indices = tuple(self.arm_joint_indices)
        self.finger_joint_indices = tuple(self.finger_joint_indices)
        self.all_movable_joint_indices = self.arm_joint_indices + self.finger_joint_indices
        self.joint_lower = np.asarray(lower, dtype=np.float64)
        self.joint_upper = np.asarray(upper, dtype=np.float64)
        self.joint_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
        self.finger_home = np.array([0.04, 0.04], dtype=np.float64)
        self.ee_link_index = self._find_ee_link()
        self.set_joint_positions(self.joint_home)

    def _find_ee_link(self) -> int:
        preferred = ["panda_grasptarget", "panda_hand", "panda_link8"]
        for target in preferred:
            for joint_idx in range(p.getNumJoints(self.body_id, physicsClientId=self.client_id)):
                name = p.getJointInfo(self.body_id, joint_idx, physicsClientId=self.client_id)[12].decode("utf-8")
                if name == target:
                    return joint_idx
        return self.arm_joint_indices[-1]

    @property
    def dof(self) -> int:
        return 7

    def disconnect(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self.joint_lower.copy(), self.joint_upper.copy()

    def get_joint_positions(self) -> np.ndarray:
        states = p.getJointStates(self.body_id, self.arm_joint_indices, physicsClientId=self.client_id)
        return np.asarray([state[0] for state in states], dtype=np.float64)

    def _set_finger_positions(self, finger_q: Sequence[float] | None = None) -> None:
        finger_q = self.finger_home if finger_q is None else np.asarray(finger_q, dtype=np.float64)
        for joint_idx, value in zip(self.finger_joint_indices, finger_q):
            p.resetJointState(self.body_id, joint_idx, float(value), physicsClientId=self.client_id)

    def _get_all_movable_positions(self) -> np.ndarray:
        states = p.getJointStates(self.body_id, self.all_movable_joint_indices, physicsClientId=self.client_id)
        return np.asarray([state[0] for state in states], dtype=np.float64)

    def set_joint_positions(self, q: Sequence[float]) -> None:
        q = np.asarray(q, dtype=np.float64)
        for joint_idx, value in zip(self.arm_joint_indices, q):
            p.resetJointState(self.body_id, joint_idx, float(value), physicsClientId=self.client_id)
        self._set_finger_positions()

    def fk(self, q: Sequence[float]) -> FKResult:
        self.set_joint_positions(q)
        state = p.getLinkState(
            self.body_id,
            self.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        return FKResult(
            position=np.asarray(state[4], dtype=np.float64),
            orientation_xyzw=np.asarray(state[5], dtype=np.float64),
        )

    def position_jacobian(self, q: Sequence[float]) -> np.ndarray:
        self.set_joint_positions(q)
        q_curr = self._get_all_movable_positions()
        zeros = np.zeros_like(q_curr)
        jac_t, _ = p.calculateJacobian(
            self.body_id,
            self.ee_link_index,
            [0.0, 0.0, 0.0],
            q_curr.tolist(),
            zeros.tolist(),
            zeros.tolist(),
            physicsClientId=self.client_id,
        )
        return np.asarray(jac_t, dtype=np.float64)[:, : self.dof]

    def inverse_kinematics(self, position: Sequence[float], orientation_xyzw: Sequence[float] | None = None, seed_q: Sequence[float] | None = None) -> np.ndarray:
        if seed_q is not None:
            self.set_joint_positions(seed_q)
        kwargs = dict(
            bodyUniqueId=self.body_id,
            endEffectorLinkIndex=self.ee_link_index,
            targetPosition=np.asarray(position, dtype=np.float64),
            lowerLimits=self.joint_lower.tolist(),
            upperLimits=self.joint_upper.tolist(),
            jointRanges=(self.joint_upper - self.joint_lower).tolist(),
            restPoses=(np.asarray(seed_q, dtype=np.float64) if seed_q is not None else self.joint_home).tolist(),
            maxNumIterations=200,
            residualThreshold=1e-6,
            physicsClientId=self.client_id,
        )
        if orientation_xyzw is not None:
            kwargs["targetOrientation"] = np.asarray(orientation_xyzw, dtype=np.float64)
        ik = p.calculateInverseKinematics(**kwargs)
        return np.asarray(ik[: self.dof], dtype=np.float64)

    def sample_random_configuration(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.joint_lower, self.joint_upper)

    def clip_to_limits(self, q: Sequence[float], margin: float = 0.0) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        return np.clip(q, self.joint_lower + margin, self.joint_upper - margin)

    def in_joint_limits(self, q: Sequence[float], margin: float = 0.0) -> bool:
        q = np.asarray(q, dtype=np.float64)
        return bool(np.all(q >= self.joint_lower + margin) and np.all(q <= self.joint_upper - margin))

    def self_collision(self) -> bool:
        contacts = p.getContactPoints(bodyA=self.body_id, bodyB=self.body_id, physicsClientId=self.client_id)
        return len(contacts) > 0
