from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackerConfig:
    damping: float = 1e-4
    correction_damping: float = 1e-3
    integration_dt: float = 0.05
    task_speed: float = 0.02
    max_steps: int = 240
    mu_threshold: float = 1e-3
    nullspace_gain: float = 0.10
    gradient_step: float = 1e-3
    joint_limit_margin: float = 1e-3
    correction_max_iters: int = 120
    correction_tol: float = 1e-5


@dataclass(frozen=True)
class RewardConfig:
    w_length: float = 1.0
    w_avg_mu: float = 0.15
    w_correction: float = 0.10


@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int = 200000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    seed: int = 7


@dataclass(frozen=True)
class BCConfig:
    dataset_size: int = 512
    top_quantile: float = 0.25
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_dim: int = 128


BASE_DIR = Path("0000_test_programs/surgery_diff/CleanDiffuser/max_manipulability")
CONTOUR_PATH = Path("0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl")
DEFAULT_DATASET_PATH = BASE_DIR / "offline_bc_dataset.npz"
DEFAULT_MODEL_PATH = BASE_DIR / "ppo_nullspace_actor.zip"
DEFAULT_VIS_PATH = BASE_DIR / "rl_vs_random_length.png"
DEFAULT_TB_DIR = BASE_DIR / "tb"
WORKSPACE_Z = 0.0
TASK_SAMPLE_RETRIES = 200
DEFAULT_NUM_VIS_SAMPLES = 32
DEFAULT_WANDB_PROJECT = "max_manipulability"
DEFAULT_WANDB_RUN_NAME = "franka_nullspace_rl"
