from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from stable_baselines3 import PPO

from max_manipulability.franka_single_point_config import DEFAULT_MODEL_PATH, DEFAULT_NUM_VIS_SAMPLES, DEFAULT_VIS_PATH
from max_manipulability.franka_single_point_rl import build_ppo_env
from max_manipulability.franka_single_point_utils import action_to_joint_positions, jacobian_position_correction


def parse_args():
    parser = argparse.ArgumentParser(description="Compare RL-selected q0 against random q0 on path length.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_VIS_SAMPLES)
    parser.add_argument("--save-path", type=str, default=str(DEFAULT_VIS_PATH))
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    env = build_ppo_env(seed=args.seed)
    model = PPO.load(args.model_path)
    rl_lengths = []
    random_lengths = []
    for _ in range(args.num_samples):
        obs, _ = env.reset()
        rl_action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, rl_info = env.step(rl_action)
        rl_lengths.append(float(rl_info["line_length"]))

        obs, _ = env.reset()
        random_action = env.action_space.sample()
        _, _, _, _, rand_info = env.step(random_action)
        random_lengths.append(float(rand_info["line_length"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([random_lengths, rl_lengths], labels=["Random q0", "RL q0"])
    ax.set_ylabel("Straight-line length")
    ax.set_title("RL-selected q0 vs Random q0")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"random_avg_length={np.mean(random_lengths):.6f}")
    print(f"rl_avg_length={np.mean(rl_lengths):.6f}")
    print(f"saved_plot={save_path.resolve()}")
    env.robot.disconnect()


if __name__ == "__main__":
    main()
