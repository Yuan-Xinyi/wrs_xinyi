from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .franka_single_point_config import BCConfig, DEFAULT_DATASET_PATH, RewardConfig, TrackerConfig
from .franka_single_point_utils import joint_positions_to_action, random_task
from .nullspace_tracker import NullSpaceTracker


@dataclass
class OfflineDataset:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    lengths: np.ndarray
    average_mu: np.ndarray

    def save(self, path: str | Path = DEFAULT_DATASET_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            lengths=self.lengths,
            average_mu=self.average_mu,
        )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_DATASET_PATH) -> "OfflineDataset":
        payload = np.load(path)
        return cls(
            observations=payload["observations"],
            actions=payload["actions"],
            rewards=payload["rewards"],
            lengths=payload["lengths"],
            average_mu=payload["average_mu"],
        )


class BCActor(nn.Module):
    def __init__(self, obs_dim: int = 6, action_dim: int = 7, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def collect_offline_bc_dataset(
    robot,
    contour,
    num_samples: int,
    rng: np.random.Generator,
    tracker_config: TrackerConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> OfflineDataset:
    tracker = NullSpaceTracker(robot, tracker_config)
    reward_config = reward_config or RewardConfig()
    obs_list = []
    action_list = []
    reward_list = []
    length_list = []
    mu_list = []
    for idx in range(num_samples):
        q0, start_pos, direction = random_task(robot, contour, rng)
        result = tracker.run(q0, direction)
        reward = reward_config.w_length * result.line_length + reward_config.w_avg_mu * result.average_mu
        obs_list.append(np.concatenate([start_pos, direction], axis=0))
        action_list.append(joint_positions_to_action(q0, robot))
        reward_list.append(reward)
        length_list.append(result.line_length)
        mu_list.append(result.average_mu)
        if (idx + 1) % 50 == 0 or (idx + 1) == num_samples:
            print(f"[bc-data] {idx + 1}/{num_samples} collected", flush=True)
    return OfflineDataset(
        observations=np.asarray(obs_list, dtype=np.float32),
        actions=np.asarray(action_list, dtype=np.float32),
        rewards=np.asarray(reward_list, dtype=np.float32),
        lengths=np.asarray(length_list, dtype=np.float32),
        average_mu=np.asarray(mu_list, dtype=np.float32),
    )


def pretrain_bc_actor(dataset: OfflineDataset, config: BCConfig | None = None, device: str = "cpu") -> BCActor:
    config = config or BCConfig()
    reward_threshold = float(np.quantile(dataset.rewards, 1.0 - config.top_quantile))
    mask = dataset.rewards >= reward_threshold
    obs = torch.as_tensor(dataset.observations[mask], dtype=torch.float32)
    actions = torch.as_tensor(dataset.actions[mask], dtype=torch.float32)
    loader = DataLoader(TensorDataset(obs, actions), batch_size=config.batch_size, shuffle=True)
    model = BCActor(hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(config.epochs):
        losses = []
        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            pred = model(batch_obs)
            loss = loss_fn(pred, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"[bc] epoch={epoch + 1}/{config.epochs} loss={np.mean(losses):.6f}", flush=True)
    return model


def copy_bc_to_ppo(ppo_model, bc_actor: BCActor) -> None:
    bc_linears = [module for module in bc_actor.net if isinstance(module, nn.Linear)]
    pi_linears = [module for module in ppo_model.policy.mlp_extractor.policy_net if isinstance(module, nn.Linear)]
    for src, dst in zip(bc_linears[:-1], pi_linears):
        dst.weight.data.copy_(src.weight.data)
        dst.bias.data.copy_(src.bias.data)
    ppo_model.policy.action_net.weight.data.copy_(bc_linears[-1].weight.data)
    ppo_model.policy.action_net.bias.data.copy_(bc_linears[-1].bias.data)
