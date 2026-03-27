from __future__ import annotations

import argparse
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from max_manipulability.franka_single_point_config import (
    BCConfig,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TB_DIR,
    DEFAULT_WANDB_PROJECT,
    DEFAULT_WANDB_RUN_NAME,
    RewardConfig,
    TrackerConfig,
    TrainConfig,
)
from max_manipulability.franka_single_point_utils import action_to_joint_positions, jacobian_position_correction, random_task, WorkspaceContour
from max_manipulability.nullspace_tracker import NullSpaceTracker
from max_manipulability.offline_bc import collect_offline_bc_dataset, copy_bc_to_ppo, pretrain_bc_actor
from max_manipulability.pybullet_franka import PyBulletFranka

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except ImportError:  # pragma: no cover
    wandb = None
    WandbCallback = None


class GoalConditionedFrankaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        robot: PyBulletFranka,
        contour: WorkspaceContour,
        tracker_config: TrackerConfig | None = None,
        reward_config: RewardConfig | None = None,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.robot = robot
        self.contour = contour
        self.tracker_config = tracker_config or TrackerConfig()
        self.reward_config = reward_config or RewardConfig()
        self.tracker = NullSpaceTracker(robot, self.tracker_config)
        self.rng = np.random.default_rng(seed)
        self.current_start_pos = None
        self.current_direction = None
        self.reward_history: list[float] = []
        self.length_history: list[float] = []
        self.mu_history: list[float] = []
        self.correction_history: list[float] = []
        self.success_history: list[float] = []
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def _sample_task(self) -> np.ndarray:
        _q, start_pos, direction = random_task(self.robot, self.contour, self.rng)
        self.current_start_pos = start_pos
        self.current_direction = direction
        return np.concatenate([start_pos, direction], axis=0).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        return self._sample_task(), {}

    def step(self, action: np.ndarray):
        proposed_q = action_to_joint_positions(action, self.robot)
        corrected_q, correction_distance, success = jacobian_position_correction(
            self.robot,
            proposed_q,
            self.current_start_pos,
            self.tracker_config,
        )
        if success:
            result = self.tracker.run(corrected_q, self.current_direction)
            reward = (
                self.reward_config.w_length * result.line_length
                + self.reward_config.w_avg_mu * result.average_mu
                - self.reward_config.w_correction * correction_distance
            )
            info = {
                "line_length": result.line_length,
                "average_mu": result.average_mu,
                "cumulative_mu": result.cumulative_mu,
                "correction_distance": correction_distance,
                "correction_success": 1.0,
                "tracker_term": result.termination_reason,
            }
        else:
            reward = -self.reward_config.w_correction * correction_distance
            info = {
                "line_length": 0.0,
                "average_mu": 0.0,
                "cumulative_mu": 0.0,
                "correction_distance": correction_distance,
                "correction_success": 0.0,
                "tracker_term": "correction_failed",
            }
        self.reward_history.append(float(reward))
        self.length_history.append(float(info["line_length"]))
        self.mu_history.append(float(info["average_mu"]))
        self.correction_history.append(float(info["correction_distance"]))
        self.success_history.append(float(info["correction_success"]))
        obs = np.concatenate([self.current_start_pos, self.current_direction], axis=0).astype(np.float32)
        return obs, float(reward), True, False, info


class MetricsCallback(BaseCallback):
    def __init__(self, env_ref: GoalConditionedFrankaEnv):
        super().__init__()
        self.env_ref = env_ref
        self.last_idx = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        new_slice = slice(self.last_idx, len(self.env_ref.reward_history))
        rewards = np.asarray(self.env_ref.reward_history[new_slice], dtype=np.float64)
        lengths = np.asarray(self.env_ref.length_history[new_slice], dtype=np.float64)
        mus = np.asarray(self.env_ref.mu_history[new_slice], dtype=np.float64)
        corrections = np.asarray(self.env_ref.correction_history[new_slice], dtype=np.float64)
        successes = np.asarray(self.env_ref.success_history[new_slice], dtype=np.float64)
        self.last_idx = len(self.env_ref.reward_history)
        if rewards.size == 0:
            return
        metrics = {
            "task/avg_reward": float(np.mean(rewards)),
            "task/avg_line_length": float(np.mean(lengths)),
            "task/avg_mu": float(np.mean(mus)),
            "task/avg_correction_distance": float(np.mean(corrections)),
            "task/correction_success_rate": float(np.mean(successes)),
            "task/num_episodes": int(len(self.env_ref.reward_history)),
        }
        for key, value in metrics.items():
            self.logger.record(key, value)
        if wandb is not None and wandb.run is not None:
            wandb.log(metrics, step=self.num_timesteps)


def build_ppo_env(seed: int = 7):
    robot = PyBulletFranka(use_gui=False)
    contour = WorkspaceContour()
    return GoalConditionedFrankaEnv(robot=robot, contour=contour, seed=seed)


def parse_args():
    parser = argparse.ArgumentParser(description="BC + PPO framework for Franka null-space tracking with continuous q0 actions.")
    parser.add_argument("--timesteps", type=int, default=TrainConfig().total_timesteps)
    parser.add_argument("--dataset-size", type=int, default=BCConfig().dataset_size)
    parser.add_argument("--seed", type=int, default=TrainConfig().seed)
    parser.add_argument("--skip-bc", action="store_true")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name", type=str, default=DEFAULT_WANDB_RUN_NAME)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    train_cfg = TrainConfig(total_timesteps=args.timesteps, seed=args.seed)
    bc_cfg = BCConfig(dataset_size=args.dataset_size)
    reward_cfg = RewardConfig()

    env = build_ppo_env(seed=args.seed)
    vec_env = DummyVecEnv([lambda: env])
    policy_kwargs = dict(net_arch=dict(pi=[bc_cfg.hidden_dim, bc_cfg.hidden_dim], vf=[128, 128]))

    run = None
    if not args.disable_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed, but W&B logging was requested.")
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "timesteps": train_cfg.total_timesteps,
                "dataset_size": bc_cfg.dataset_size,
                "seed": train_cfg.seed,
                "device": args.device,
                "tracker": vars(TrackerConfig()),
                "reward": vars(reward_cfg),
                "bc": vars(bc_cfg),
                "train": vars(train_cfg),
            },
            sync_tensorboard=True,
            save_code=True,
        )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=train_cfg.learning_rate,
        n_steps=train_cfg.n_steps,
        batch_size=train_cfg.batch_size,
        gamma=train_cfg.gamma,
        gae_lambda=train_cfg.gae_lambda,
        ent_coef=train_cfg.ent_coef,
        verbose=1,
        seed=train_cfg.seed,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(DEFAULT_TB_DIR),
        device=args.device,
    )
    print(f"ppo_device={model.device}")

    if not args.skip_bc:
        dataset = collect_offline_bc_dataset(
            env.robot,
            env.contour,
            bc_cfg.dataset_size,
            np.random.default_rng(args.seed),
            env.tracker_config,
            reward_cfg,
        )
        dataset.save(args.dataset_path)
        if run is not None:
            wandb.log(
                {
                    "bc_dataset/avg_reward": float(np.mean(dataset.rewards)),
                    "bc_dataset/avg_line_length": float(np.mean(dataset.lengths)),
                    "bc_dataset/avg_mu": float(np.mean(dataset.average_mu)),
                },
                step=0,
            )
        bc_device = "cuda" if args.device == "cuda" else "cpu"
        bc_actor = pretrain_bc_actor(dataset, config=bc_cfg, device=bc_device)
        copy_bc_to_ppo(model, bc_actor)

    callbacks = [MetricsCallback(env)]
    if run is not None and WandbCallback is not None:
        callbacks.append(WandbCallback(model_save_path=None, gradient_save_freq=0, verbose=0))
    model.learn(total_timesteps=train_cfg.total_timesteps, callback=CallbackList(callbacks))
    model.save(args.model_path)

    rewards = np.asarray(env.reward_history, dtype=np.float64)
    lengths = np.asarray(env.length_history, dtype=np.float64)
    mus = np.asarray(env.mu_history, dtype=np.float64)
    corrections = np.asarray(env.correction_history, dtype=np.float64)
    successes = np.asarray(env.success_history, dtype=np.float64)
    final_metrics = {
        "final/avg_reward": float(np.mean(rewards)) if rewards.size else 0.0,
        "final/avg_line_length": float(np.mean(lengths)) if lengths.size else 0.0,
        "final/avg_mu": float(np.mean(mus)) if mus.size else 0.0,
        "final/avg_correction_distance": float(np.mean(corrections)) if corrections.size else 0.0,
        "final/correction_success_rate": float(np.mean(successes)) if successes.size else 0.0,
    }
    if run is not None:
        wandb.log(final_metrics, step=train_cfg.total_timesteps)
    for key, value in final_metrics.items():
        print(f"{key}={value:.6f}")
    print(f"saved_model={Path(args.model_path).resolve()}")
    env.robot.disconnect()
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
