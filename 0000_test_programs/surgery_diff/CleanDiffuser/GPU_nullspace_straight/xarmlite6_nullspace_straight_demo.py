import argparse
from dataclasses import dataclass

import numpy as np

import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller


def current_position_jacobian(robot: XArmLite6Miller, q: np.ndarray) -> np.ndarray:
    robot.goto_given_conf(q)
    return robot.jacobian()[:3, :]


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("Zero-length vector is not allowed.")
    return vec / norm


def damped_pseudoinverse(j_mat: np.ndarray, damping: float) -> np.ndarray:
    jj_t = j_mat @ j_mat.T
    return j_mat.T @ np.linalg.inv(jj_t + (damping ** 2) * np.eye(j_mat.shape[0]))


def nullspace_projector(j_mat: np.ndarray, damping: float) -> np.ndarray:
    j_pinv = damped_pseudoinverse(j_mat, damping)
    return np.eye(j_mat.shape[1]) - j_pinv @ j_mat


def directional_manipulability(j_pos: np.ndarray, direction: np.ndarray, damping: float) -> float:
    metric = j_pos @ j_pos.T + (damping ** 2) * np.eye(3)
    value = float(direction.T @ np.linalg.inv(metric) @ direction)
    return value ** -0.5


def directional_manipulability_gradient(
    robot: XArmLite6Miller,
    q: np.ndarray,
    direction: np.ndarray,
    damping: float,
    fd_eps: float,
) -> np.ndarray:
    grad = np.zeros_like(q)
    for idx in range(q.size):
        q_pos = q.copy()
        q_neg = q.copy()
        q_pos[idx] += fd_eps
        q_neg[idx] -= fd_eps
        mu_pos = directional_manipulability(current_position_jacobian(robot, q_pos), direction, damping)
        mu_neg = directional_manipulability(current_position_jacobian(robot, q_neg), direction, damping)
        grad[idx] = (mu_pos - mu_neg) / (2.0 * fd_eps)
    return grad


def sample_valid_start(robot: XArmLite6Miller, rng: np.random.Generator, mu_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(2000):
        q = robot.rand_conf()
        direction = normalize(rng.normal(size=3))
        robot.goto_given_conf(q)
        if robot.is_collided():
            continue
        j_pos = robot.jacobian()[:3, :]
        mu_val = directional_manipulability(j_pos, direction, damping=1e-3)
        if mu_val > mu_threshold:
            return q, direction
    raise RuntimeError("Failed to sample a valid XArmLite6Miller start configuration.")


@dataclass
class TrackerConfig:
    dt: float = 0.01
    task_speed: float = 0.1
    damping: float = 1e-3
    null_gain: float = 0.6
    grad_fd_eps: float = 1e-4
    max_steps: int = 2000
    mu_threshold: float = 0.03


@dataclass
class TrackerResult:
    q_path: list[np.ndarray]
    tcp_path: list[np.ndarray]
    mu_path: list[float]
    projected_length: float
    euclidean_length: float
    mean_mu: float
    termination_reason: str


class NullspaceStraightTracker:
    def __init__(self, robot: XArmLite6Miller, config: TrackerConfig):
        self.robot = robot
        self.config = config

    def run(self, q0: np.ndarray, direction: np.ndarray) -> TrackerResult:
        direction = normalize(direction)
        q = q0.astype(float).copy()
        self.robot.goto_given_conf(q)
        start_pos, _ = self.robot.fk(q, update=False)
        tcp_path = [start_pos.copy()]
        q_path = [q.copy()]
        mu_path: list[float] = []
        termination_reason = "max_steps"

        for _ in range(self.config.max_steps):
            j_pos = current_position_jacobian(self.robot, q)
            mu_val = directional_manipulability(j_pos, direction, self.config.damping)
            mu_path.append(mu_val)
            if mu_val < self.config.mu_threshold:
                termination_reason = "low_mu"
                break

            grad_mu = directional_manipulability_gradient(
                robot=self.robot,
                q=q,
                direction=direction,
                damping=self.config.damping,
                fd_eps=self.config.grad_fd_eps,
            )
            j_pinv = damped_pseudoinverse(j_pos, self.config.damping)
            projector = nullspace_projector(j_pos, self.config.damping)
            v_task = self.config.task_speed * direction
            q_dot_task = j_pinv @ v_task
            q_dot_null = self.config.null_gain * grad_mu
            q_dot = q_dot_task + projector @ q_dot_null
            q_next = q + self.config.dt * q_dot

            if not self.robot.are_jnts_in_ranges(q_next):
                termination_reason = "joint_limit"
                break

            self.robot.goto_given_conf(q_next)
            if self.robot.is_collided():
                termination_reason = "self_collision"
                break

            q = q_next
            tcp_pos, _ = self.robot.fk(q, update=False)
            q_path.append(q.copy())
            tcp_path.append(tcp_pos.copy())

        end_pos = tcp_path[-1]
        delta = end_pos - start_pos
        projected_length = float(delta @ direction)
        euclidean_length = float(np.linalg.norm(delta))
        mean_mu = float(np.mean(mu_path)) if mu_path else 0.0
        return TrackerResult(
            q_path=q_path,
            tcp_path=tcp_path,
            mu_path=mu_path,
            projected_length=projected_length,
            euclidean_length=euclidean_length,
            mean_mu=mean_mu,
            termination_reason=termination_reason,
        )


def visualize_result(
    robot: XArmLite6Miller,
    direction: np.ndarray,
    result: TrackerResult,
    show_waypoints: bool,
) -> None:
    start_pos = result.tcp_path[0]
    end_pos = result.tcp_path[-1]
    ideal_end = start_pos + direction * result.projected_length
    world = wd.World(cam_pos=[1.6, -1.4, 1.0], lookat_pos=[0.25, 0.0, 0.25])
    mgm.gen_frame().attach_to(world)
    mgm.gen_arrow(
        spos=start_pos,
        epos=start_pos + 0.25 * direction,
        stick_radius=0.006,
        rgb=np.array([1.0, 0.2, 0.2]),
    ).attach_to(world)
    line_segs = [[result.tcp_path[i], result.tcp_path[i + 1]] for i in range(len(result.tcp_path) - 1)]
    if line_segs:
        mgm.gen_linesegs(line_segs, thickness=0.004, rgb=np.array([0.1, 0.8, 0.2]), alpha=1.0).attach_to(world)
    mgm.gen_stick(
        spos=start_pos,
        epos=ideal_end,
        radius=0.003,
        rgb=np.array([0.1, 0.4, 1.0]),
        alpha=0.5,
    ).attach_to(world)
    mgm.gen_sphere(start_pos, radius=0.01, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)
    mgm.gen_sphere(end_pos, radius=0.012, rgb=np.array([1.0, 0.5, 0.0]), alpha=1.0).attach_to(world)

    if show_waypoints:
        for pos in result.tcp_path[1:-1]:
            mgm.gen_sphere(pos, radius=0.004, rgb=np.array([0.0, 0.8, 0.2]), alpha=0.4).attach_to(world)

    robot.goto_given_conf(result.q_path[0])
    robot.gen_meshmodel(rgb=np.array([0.2, 0.5, 1.0]), alpha=0.25, toggle_tcp_frame=True).attach_to(world)
    robot.goto_given_conf(result.q_path[-1])
    robot.gen_meshmodel(rgb=np.array([1.0, 0.5, 0.0]), alpha=0.9, toggle_tcp_frame=True).attach_to(world)
    world.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XArmLite6Miller null-space straight-line differential tracker demo.")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--speed", type=float, default=0.10)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--null-gain", type=float, default=0.6)
    parser.add_argument("--fd-eps", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--mu-threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--show-waypoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    robot = XArmLite6Miller(enable_cc=True)
    q0, direction = sample_valid_start(robot, rng=rng, mu_threshold=args.mu_threshold)
    tracker = NullspaceStraightTracker(
        robot=robot,
        config=TrackerConfig(
            dt=args.dt,
            task_speed=args.speed,
            damping=args.damping,
            null_gain=args.null_gain,
            grad_fd_eps=args.fd_eps,
            max_steps=args.max_steps,
            mu_threshold=args.mu_threshold,
        ),
    )
    result = tracker.run(q0=q0, direction=direction)
    print("start_q =", np.array2string(q0, precision=4, separator=", "))
    print("direction =", np.array2string(direction, precision=4, separator=", "))
    print(f"projected_length = {result.projected_length:.4f} m")
    print(f"euclidean_length = {result.euclidean_length:.4f} m")
    print(f"mean_mu = {result.mean_mu:.6f}")
    print(f"steps = {len(result.tcp_path) - 1}")
    print("termination_reason =", result.termination_reason)
    visualize_result(robot=robot, direction=direction, result=result, show_waypoints=args.show_waypoints)


if __name__ == "__main__":
    main()
