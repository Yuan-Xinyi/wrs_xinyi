import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

BASE_DIR = Path(__file__).resolve().parent
CLEANDIFFUSER_ROOT = BASE_DIR.parent
if str(CLEANDIFFUSER_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEANDIFFUSER_ROOT))

from cleandiffuser.nn_diffusion.dit import DiT1d


DEFAULT_H5_PATH = BASE_DIR / "xarmlite6_gpu_trajectories_100000_sub10.hdf5"
DEFAULT_CACHE_DIR = BASE_DIR / "kinematic_token_cache_sub10"
DEFAULT_WORKDIR = BASE_DIR / "dit_kinematic_inpainting_runs"
DEFAULT_RUN_NAME = "dit_inpaint_qrot_from_posdirL_sub10"


@dataclass
class FeatureLayout:
    q_dim: int

    @property
    def q_slice(self) -> slice:
        return slice(0, self.q_dim)

    @property
    def pos_slice(self) -> slice:
        return slice(self.q_dim, self.q_dim + 3)

    @property
    def rot_slice(self) -> slice:
        return slice(self.q_dim + 3, self.q_dim + 7)

    @property
    def dir_slice(self) -> slice:
        return slice(self.q_dim + 7, self.q_dim + 10)

    @property
    def length_slice(self) -> slice:
        return slice(self.q_dim + 10, self.q_dim + 11)

    @property
    def token_dim(self) -> int:
        return self.q_dim + 11

    @property
    def primary_unknown_mask(self) -> np.ndarray:
        mask = np.ones(self.token_dim, dtype=np.float32)
        mask[self.q_slice] = 0.0
        mask[self.rot_slice] = 0.0
        return mask


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray
    data_min: np.ndarray
    data_max: np.ndarray

    def transform_np(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_transform_np(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)

    def transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return (x - mean) / std

    def inverse_transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return x * std + mean


class TokenMemmapDataset(Dataset):
    def __init__(self, tokens_path: Path, indices: np.ndarray):
        self.tokens_path = Path(tokens_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self._tokens = None

    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens = np.load(self.tokens_path, mmap_mode="r")
        return self._tokens

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict:
        token = np.asarray(self.tokens[self.indices[idx]], dtype=np.float32)
        return {"token": torch.from_numpy(token.copy())}


class JointDistributionMasker:
    def __init__(self, layout: FeatureLayout, primary_prob: float = 0.5):
        self.layout = layout
        self.primary_prob = float(primary_prob)

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        mask = torch.ones(batch_size, self.layout.token_dim, device=device, dtype=torch.float32)
        primary_flags = torch.rand(batch_size, device=device) < self.primary_prob
        if primary_flags.any():
            primary_mask = torch.as_tensor(self.layout.primary_unknown_mask, device=device, dtype=torch.float32)
            mask[primary_flags] = primary_mask.unsqueeze(0)
        random_flags = ~primary_flags
        if random_flags.any():
            rand_mask = (torch.rand(int(random_flags.sum().item()), self.layout.token_dim, device=device) > 0.5).float()
            all_known = rand_mask.sum(dim=1) >= self.layout.token_dim
            all_unknown = rand_mask.sum(dim=1) <= 0
            if all_known.any():
                rand_mask[all_known, self.layout.length_slice] = 0.0
            if all_unknown.any():
                rand_mask[all_unknown, self.layout.pos_slice] = 1.0
                rand_mask[all_unknown, self.layout.dir_slice] = 1.0
                rand_mask[all_unknown, self.layout.length_slice] = 1.0
            mask[random_flags] = rand_mask
        return mask


class MaskConditionedDiT(nn.Module):
    def __init__(
        self,
        token_dim: int,
        emb_dim: int = 256,
        d_model: int = 384,
        n_heads: int = 6,
        depth: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mask_encoder = nn.Sequential(
            nn.Linear(token_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.diffusion = DiT1d(
            in_dim=token_dim,
            emb_dim=emb_dim,
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            dropout=dropout,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, known_mask: torch.Tensor) -> torch.Tensor:
        cond = self.mask_encoder(known_mask)
        return self.diffusion(x_t.unsqueeze(1), t, cond).squeeze(1)


class ResidualMLPDenoiser(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int = 512, depth: int = 4, time_dim: int = 128):
        super().__init__()
        self.token_dim = token_dim
        self.time_dim = time_dim
        in_dim = token_dim * 2 + time_dim
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, token_dim)]
        self.net = nn.Sequential(*layers)

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.time_dim:
            emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=t.device)], dim=1)
        return emb

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, known_mask: torch.Tensor) -> torch.Tensor:
        t_emb = self.timestep_embedding(t)
        inp = torch.cat([x_t, known_mask, t_emb], dim=1)
        return self.net(inp)


class MaskedDiffusionModel(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        token_dim: int,
        diffusion_steps: int = 64,
        predict_x0: bool = True,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.denoiser = denoiser
        self.token_dim = int(token_dim)
        self.diffusion_steps = int(diffusion_steps)
        self.predict_x0 = bool(predict_x0)
        self.device = torch.device(device)

        betas = cosine_beta_schedule(self.diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]], dim=0)
        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_var[0] = 1e-8

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("posterior_var", posterior_var)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alpha_bar[t].unsqueeze(1)
        xt = a.sqrt() * x0 + (1.0 - a).sqrt() * noise
        return xt, noise

    def training_loss(
        self,
        x0: torch.Tensor,
        known_mask: torch.Tensor,
        unknown_weight: torch.Tensor,
        known_weight: float = 0.05,
    ) -> tuple[torch.Tensor, dict]:
        batch_size = x0.shape[0]
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=x0.device)
        x_noisy, noise = self.q_sample(x0, t)
        x_inpaint = known_mask * x0 + (1.0 - known_mask) * x_noisy
        pred = self.denoiser(x_inpaint, t, known_mask)
        target = x0 if self.predict_x0 else noise
        weights = known_mask * known_weight + (1.0 - known_mask) * unknown_weight.unsqueeze(0)
        mse = (pred - target).pow(2)
        loss = (mse * weights).mean()
        masked_mse = ((mse * (1.0 - known_mask)).sum(dim=1) / (1.0 - known_mask).sum(dim=1).clamp_min(1.0)).mean()
        return loss, {"masked_mse": float(masked_mse.item())}

    @torch.no_grad()
    def sample_inpaint(
        self,
        known_values: torch.Tensor,
        known_mask: torch.Tensor,
        n_samples: int,
        temperature: float = 1.0,
        clip_min: Optional[torch.Tensor] = None,
        clip_max: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.randn(n_samples, self.token_dim, device=known_values.device) * temperature
        x = known_mask * known_values + (1.0 - known_mask) * x
        for step in range(self.diffusion_steps - 1, -1, -1):
            t = torch.full((n_samples,), step, device=known_values.device, dtype=torch.long)
            x0_pred = self.denoiser(x, t, known_mask)
            x0_pred = known_mask * known_values + (1.0 - known_mask) * x0_pred
            if clip_min is not None or clip_max is not None:
                lo = clip_min if clip_min is not None else -torch.inf
                hi = clip_max if clip_max is not None else torch.inf
                x0_pred = torch.maximum(torch.minimum(x0_pred, hi), lo)
                x0_pred = known_mask * known_values + (1.0 - known_mask) * x0_pred
            if step == 0:
                x = x0_pred
                break
            a_t = self.alphas[step]
            ab_t = self.alpha_bar[step]
            var_t = self.posterior_var[step]
            eps_pred = (x - ab_t.sqrt() * x0_pred) / max(float((1.0 - ab_t).sqrt().item()), 1e-8)
            x = (1.0 / a_t.sqrt()) * (x - (self.betas[step] / (1.0 - ab_t).sqrt()) * eps_pred)
            x = x + var_t.sqrt() * torch.randn_like(x)
            x = known_mask * known_values + (1.0 - known_mask) * x
        return x


def cosine_beta_schedule(diffusion_steps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, diffusion_steps, diffusion_steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((x / diffusion_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-5, 0.999).float()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonicalize_quaternion_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    quat_xyzw = quat_xyzw / max(np.linalg.norm(quat_xyzw), 1e-12)
    if quat_xyzw[3] < 0.0:
        quat_xyzw = -quat_xyzw
    return quat_xyzw.astype(np.float32)


def rotmats_to_quats_xyzw(rotmats: np.ndarray) -> np.ndarray:
    quats = Rotation.from_matrix(rotmats).as_quat().astype(np.float32)
    flip = quats[:, 3] < 0.0
    quats[flip] *= -1.0
    return quats


def infer_layout_from_h5(h5_path: Path) -> FeatureLayout:
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f["trajectories"].keys())
        if not keys:
            raise RuntimeError(f"No trajectories found in {h5_path}")
        q_shape = f["trajectories"][keys[0]]["q"].shape
    q_dim = int(q_shape[1])
    return FeatureLayout(q_dim=q_dim)


def prepare_token_cache(
    h5_path: Path,
    cache_dir: Path,
    max_trajectories: Optional[int] = None,
    progress_every: int = 500,
) -> tuple[Path, StandardScaler, FeatureLayout, dict]:
    h5_path = Path(h5_path).resolve()
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    layout = infer_layout_from_h5(h5_path)
    suffix = f"_maxtraj{max_trajectories}" if max_trajectories is not None else ""
    tokens_path = cache_dir / f"{h5_path.stem}_tokens_q{layout.q_dim}{suffix}.npy"
    stats_path = cache_dir / f"{h5_path.stem}_tokens_q{layout.q_dim}{suffix}_stats.npz"
    meta_path = cache_dir / f"{h5_path.stem}_tokens_q{layout.q_dim}{suffix}_meta.json"

    if tokens_path.exists() and stats_path.exists() and meta_path.exists():
        stats_npz = np.load(stats_path)
        scaler = StandardScaler(
            mean=stats_npz["mean"].astype(np.float32),
            std=stats_npz["std"].astype(np.float32),
            data_min=stats_npz["data_min"].astype(np.float32),
            data_max=stats_npz["data_max"].astype(np.float32),
        )
        metadata = json.loads(meta_path.read_text())
        return tokens_path, scaler, layout, metadata

    with h5py.File(h5_path, "r") as f:
        group_names = sorted(f["trajectories"].keys())
        if max_trajectories is not None:
            group_names = group_names[: max_trajectories]
        total_samples = int(sum(int(f["trajectories"][name].attrs["num_points"]) for name in group_names))

        mmap = np.lib.format.open_memmap(tokens_path, mode="w+", dtype=np.float32, shape=(total_samples, layout.token_dim))
        sum_x = np.zeros(layout.token_dim, dtype=np.float64)
        sum_sq = np.zeros(layout.token_dim, dtype=np.float64)
        data_min = np.full(layout.token_dim, np.inf, dtype=np.float64)
        data_max = np.full(layout.token_dim, -np.inf, dtype=np.float64)

        cursor = 0
        for traj_idx, name in enumerate(group_names):
            grp = f["trajectories"][name]
            q = np.asarray(grp["q"][:], dtype=np.float32)
            pos = np.asarray(grp["tcp_pos"][:], dtype=np.float32)
            rot_q = rotmats_to_quats_xyzw(np.asarray(grp["tcp_rotmat"][:], dtype=np.float32))
            direction = np.asarray(grp.attrs["direction"], dtype=np.float32)
            remaining_length = np.asarray(grp["remaining_length"][:], dtype=np.float32).reshape(-1, 1)
            d = np.repeat(direction.reshape(1, 3), q.shape[0], axis=0)
            tokens = np.concatenate([q, pos, rot_q, d, remaining_length], axis=1).astype(np.float32)
            mmap[cursor: cursor + q.shape[0]] = tokens
            sum_x += tokens.sum(axis=0, dtype=np.float64)
            sum_sq += np.square(tokens, dtype=np.float64).sum(axis=0)
            data_min = np.minimum(data_min, tokens.min(axis=0))
            data_max = np.maximum(data_max, tokens.max(axis=0))
            cursor += q.shape[0]
            if progress_every > 0 and ((traj_idx + 1) % progress_every == 0 or traj_idx + 1 == len(group_names)):
                print(f"[cache] trajectories={traj_idx + 1}/{len(group_names)} samples={cursor}/{total_samples}")

    mean = (sum_x / max(total_samples, 1)).astype(np.float32)
    var = np.maximum(sum_sq / max(total_samples, 1) - np.square(mean, dtype=np.float64), 1e-8)
    std = np.sqrt(var).astype(np.float32)
    scaler = StandardScaler(
        mean=mean,
        std=std,
        data_min=data_min.astype(np.float32),
        data_max=data_max.astype(np.float32),
    )
    metadata = {
        "h5_path": str(h5_path),
        "tokens_path": str(tokens_path),
        "num_samples": total_samples,
        "q_dim": layout.q_dim,
        "token_dim": layout.token_dim,
    }
    np.savez(stats_path, mean=scaler.mean, std=scaler.std, data_min=scaler.data_min, data_max=scaler.data_max)
    meta_path.write_text(json.dumps(metadata, indent=2))
    return tokens_path, scaler, layout, metadata
