from __future__ import annotations

"""Contrastive L-Net for length scoring and gradient guidance."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class ResidualBlock(nn.Module):
    """Two-layer residual MLP block with LayerNorm and spectral normalization."""

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


class LNetContrastive(nn.Module):
    def __init__(
        self,
        q_min: torch.Tensor,
        q_max: torch.Tensor,
        in_min: torch.Tensor,
        in_max: torch.Tensor,
        width: int = 512,
        num_blocks: int = 6,
        pair_threshold: float = 0.05,
        pair_margin: float = 0.05,
        mse_weight: float = 0.2,
        rank_weight: float = 1.0,
        max_pairs: int = 4096,
    ) -> None:
        super().__init__()
        self.register_buffer('q_min', q_min.float().view(1, -1))
        self.register_buffer('q_max', q_max.float().view(1, -1))
        self.register_buffer('in_min', in_min.float().view(1, -1))
        self.register_buffer('in_max', in_max.float().view(1, -1))
        self.pair_threshold = float(pair_threshold)
        self.pair_margin = float(pair_margin)
        self.mse_weight = float(mse_weight)
        self.rank_weight = float(rank_weight)
        self.max_pairs = int(max_pairs)

        input_dim = int(q_min.numel()) + 9 + int(q_min.numel())
        blocks = []
        cur_dim = input_dim
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(cur_dim, width))
            cur_dim = width
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(width)
        self.score_head = spectral_norm(nn.Linear(width, 1))
        self.length_head = spectral_norm(nn.Linear(width, 1))

    def _scale_to_minus1_1(self, x: torch.Tensor) -> torch.Tensor:
        denom = (self.in_max - self.in_min).clamp_min(1e-6)
        return 2.0 * (x - self.in_min) / denom - 1.0

    def _augment(self, q: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        dist_to_limit = torch.minimum(q - self.q_min, self.q_max - q)
        base = torch.cat([q, cond], dim=-1)
        base = self._scale_to_minus1_1(base)
        joint_span = (self.q_max - self.q_min).clamp_min(1e-6)
        dist_norm = 2.0 * dist_to_limit / joint_span - 1.0
        return torch.cat([base, dist_norm], dim=-1)

    def encode(self, q: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self._augment(q, cond)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, q: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encode(q, cond)
        score = self.score_head(feat).squeeze(-1)
        length = self.length_head(feat).squeeze(-1)
        return score, length

    def compute_pair_loss(
        self,
        q_i: torch.Tensor,
        cond_i: torch.Tensor,
        l_i: torch.Tensor,
        q_j: torch.Tensor,
        cond_j: torch.Tensor,
        l_j: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        score_i, length_i = self.forward(q_i, cond_i)
        score_j, length_j = self.forward(q_j, cond_j)
        length_gap = l_i - l_j
        valid = length_gap.abs() > self.pair_threshold
        if valid.any():
            sign = torch.sign(length_gap[valid])
            score_diff = score_i[valid] - score_j[valid]
            rank_loss = F.relu(self.pair_margin - sign * score_diff).mean()
            pair_count = int(valid.sum().item())
        else:
            rank_loss = score_i.new_tensor(0.0)
            pair_count = 0
        mse_loss = 0.5 * (F.mse_loss(length_i, l_i) + F.mse_loss(length_j, l_j))
        total = self.rank_weight * rank_loss + self.mse_weight * mse_loss
        aux = {
            'rank_loss': float(rank_loss.detach()),
            'mse_loss': float(mse_loss.detach()),
            'pair_count': pair_count,
            'score_i_mean': float(score_i.detach().mean()),
            'score_j_mean': float(score_j.detach().mean()),
            'length_i_mean': float(length_i.detach().mean()),
            'length_j_mean': float(length_j.detach().mean()),
        }
        return total, aux

    def compute_loss(self, q: torch.Tensor, cond: torch.Tensor, l_gt: torch.Tensor) -> tuple[torch.Tensor, dict]:
        score, length = self.forward(q, cond)
        n = l_gt.shape[0]
        if n < 2:
            rank_loss = score.new_tensor(0.0)
            pair_count = 0
        else:
            diff = l_gt[:, None] - l_gt[None, :]
            valid = diff.abs() > self.pair_threshold
            valid.fill_diagonal_(False)
            pair_idx = valid.nonzero(as_tuple=False)
            if pair_idx.numel() == 0:
                rank_loss = score.new_tensor(0.0)
                pair_count = 0
            else:
                if pair_idx.shape[0] > self.max_pairs:
                    perm = torch.randperm(pair_idx.shape[0], device=l_gt.device)[: self.max_pairs]
                    pair_idx = pair_idx[perm]
                i = pair_idx[:, 0]
                j = pair_idx[:, 1]
                sign = torch.sign(l_gt[i] - l_gt[j])
                rank_loss = F.relu(self.pair_margin - sign * (score[i] - score[j])).mean()
                pair_count = int(i.numel())
        mse_loss = F.mse_loss(length, l_gt)
        total = self.rank_weight * rank_loss + self.mse_weight * mse_loss
        aux = {
            'rank_loss': float(rank_loss.detach()),
            'mse_loss': float(mse_loss.detach()),
            'pair_count': pair_count,
            'score_mean': float(score.detach().mean()),
            'length_mean': float(length.detach().mean()),
        }
        return total, aux

    def get_guidance_gradient(self, q: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        with torch.enable_grad():
            q_in = q.detach().clone().requires_grad_(True)
            score, _ = self.forward(q_in, cond)
            grad = torch.autograd.grad(score.sum(), q_in, create_graph=False, retain_graph=False)[0]
        if was_training:
            self.train()
        return grad
