from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = spectral_norm(nn.Linear(in_dim, width))
        self.fc2 = spectral_norm(nn.Linear(width, width))
        self.act = nn.SiLU()
        self.shortcut = spectral_norm(nn.Linear(in_dim, width)) if in_dim != width else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        return self.shortcut(x) + h


class LNet(nn.Module):
    def __init__(
        self,
        q_min: torch.Tensor,
        q_max: torch.Tensor,
        in_min: torch.Tensor,
        in_max: torch.Tensor,
        width: int = 512,
    ) -> None:
        super().__init__()
        self.register_buffer('q_min', q_min.float().view(1, -1))
        self.register_buffer('q_max', q_max.float().view(1, -1))
        self.register_buffer('in_min', in_min.float().view(1, -1))
        self.register_buffer('in_max', in_max.float().view(1, -1))

        input_dim = int(q_min.numel()) + 9 + int(q_min.numel())
        self.blocks = nn.ModuleList([
            ResidualBlock(input_dim, width),
            ResidualBlock(width, width),
            ResidualBlock(width, width),
        ])
        self.out_norm = nn.LayerNorm(width)
        self.out = spectral_norm(nn.Linear(width, 1))

    def _scale_to_minus1_1(self, x: torch.Tensor) -> torch.Tensor:
        denom = (self.in_max - self.in_min).clamp_min(1e-6)
        return 2.0 * (x - self.in_min) / denom - 1.0

    def _augment(self, q: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        dist_to_limit = torch.minimum(q - self.q_min, self.q_max - q)
        base = torch.cat([q, cond], dim=-1)
        base = self._scale_to_minus1_1(base)
        dist_scale = (self.q_max - self.q_min).clamp_min(1e-6)
        dist_norm = 2.0 * dist_to_limit / dist_scale - 1.0
        return torch.cat([base, dist_norm], dim=-1)

    def forward(self, q: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self._augment(q, cond)
        for block in self.blocks:
            x = block(x)
        x = self.out(self.out_norm(x))
        return x.squeeze(-1)

    def get_gradient(self, q: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        with torch.enable_grad():
            q_in = q.detach().clone().requires_grad_(True)
            pred = self.forward(q_in, cond)
            grad = torch.autograd.grad(pred.sum(), q_in, create_graph=False, retain_graph=False)[0]
        if was_training:
            self.train()
        return grad


def pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.01) -> torch.Tensor:
    diff_target = target[:, None] - target[None, :]
    diff_pred = pred[:, None] - pred[None, :]
    valid = diff_target > 0.0
    if not valid.any():
        return pred.new_tensor(0.0)
    return F.relu(margin - diff_pred[valid]).mean()


def lnet_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 1.0,
    rank_weight: float = 0.2,
    rank_margin: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    mse = F.mse_loss(pred, target)
    rank = pairwise_ranking_loss(pred, target, margin=rank_margin)
    total = mse_weight * mse + rank_weight * rank
    return total, {'mse': float(mse.detach()), 'rank': float(rank.detach())}
