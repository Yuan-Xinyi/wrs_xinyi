import math
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn

from cleandiffuser.utils import SinusoidalEmbedding
from cleandiffuser.nn_diffusion import BaseNNDiffusion


class EmphasisProjection(nn.Module):
    def __init__(self, state_dim: int, global_idx: List[int], c: float = 2.0):
        super().__init__()
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.randn(state_dim, state_dim), requires_grad=False)
        b = torch.ones(state_dim)
        if len(global_idx) > 0:
            b[torch.as_tensor(global_idx, dtype=torch.long)] = c
        self.register_buffer("b", b)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: (B, T, state_dim)
        s_scaled = s * self.b
        proj = torch.matmul(s_scaled, self.A.t())
        return torch.cat([proj, s], dim=-1)


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, d_model), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class CloC1d(BaseNNDiffusion):
    """
    按论文结构的模型定义：
      - 输入 x: (B, T, S+A) 或 (states, actions)
      - noise: (B,) 或 (B,T)；自动广播为状态噪声和动作噪声
      - condition: 可选（默认不使用）
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        depth: int = 6,
        noise_emb_dim: int = 64,
        state_mlp_hidden: int = 512,
        max_tokens: int = 256,
        emphasis_c: float = 2.0,
        global_state_idx: Optional[list] = None,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ):
        super().__init__(emb_dim=d_model, timestep_emb_type=timestep_emb_type, timestep_emb_params=timestep_emb_params)
        self.S, self.A = state_dim, action_dim
        self.d = d_model

        self.emphasis = EmphasisProjection(state_dim, global_state_idx or [], emphasis_c)

        self.state_enc = nn.Sequential(
            nn.Linear(state_dim * 2, state_mlp_hidden), nn.GELU(),
            nn.Linear(state_mlp_hidden, d_model // 2)
        )
        self.action_enc = nn.Linear(action_dim, d_model // 2)

        self.emb_noise_state = SinusoidalEmbedding(noise_emb_dim)
        self.emb_noise_action = SinusoidalEmbedding(noise_emb_dim)
        self.state_tok_proj = nn.Linear(d_model // 2 + noise_emb_dim, d_model)
        self.action_tok_proj = nn.Linear(d_model // 2 + noise_emb_dim, d_model)

        self.pos = nn.Parameter(torch.zeros(max_tokens, d_model))
        nn.init.normal_(self.pos, std=0.02)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads) for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        self.state_head = nn.Linear(d_model, state_dim)
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self,
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                noise: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                condition: Optional[torch.Tensor] = None):
        if isinstance(x, (tuple, list)):
            states, actions = x
        else:
            states, actions = x[..., :self.S], x[..., self.S:]

        if isinstance(noise, (tuple, list)):
            k_states, k_actions = noise
        else:
            k_states = noise
            k_actions = noise

        # Emphasis projection on states
        states_emph = self.emphasis(states)
        s_feat = self.state_enc(states_emph)
        a_feat = self.action_enc(actions)

        s_noise = self.emb_noise_state(k_states)
        a_noise = self.emb_noise_action(k_actions)

        s_tok = self.state_tok_proj(torch.cat([s_feat, s_noise], dim=-1))
        a_tok = self.action_tok_proj(torch.cat([a_feat, a_noise], dim=-1))

        tokens = []
        types, times = [], []
        T = states.size(1)
        for t in range(T):
            tokens.append(s_tok[:, t])
            types.append("s"); times.append(t)
            if t < actions.size(1):
                tokens.append(a_tok[:, t])
                types.append("a"); times.append(t)
        x = torch.stack(tokens, dim=1)

        pos = self.pos[:x.size(1)].unsqueeze(0)
        x = x + pos

        attn_mask = self._build_attn_mask(types, times).to(x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.ln_f(x)

        s_idx = [i for i, t in enumerate(types) if t == "s"]
        a_idx = [i for i, t in enumerate(types) if t == "a"]
        s_pred = self.state_head(x[:, s_idx])
        a_pred = self.action_head(x[:, a_idx])
        return torch.cat([s_pred, a_pred], dim=-1)

    @staticmethod
    def _build_attn_mask(types: List[str], times: List[int]) -> torch.Tensor:
        L = len(types)
        mask = torch.zeros(L, L)
        neg_inf = torch.finfo(mask.dtype).min
        for i in range(L):
            qi_type, qi_time = types[i], times[i]
            for j in range(L):
                kj_type, kj_time = types[j], times[j]
                allow = False
                if qi_type == "s":
                    allow = (kj_type == "s")
                else:
                    if kj_time <= qi_time and (kj_type in ("s", "a")):
                        allow = True
                if not allow:
                    mask[i, j] = neg_inf
        return mask