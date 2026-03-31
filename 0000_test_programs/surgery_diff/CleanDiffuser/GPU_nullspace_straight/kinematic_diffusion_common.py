import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

BASE_DIR = Path(__file__).resolve().parent
CLEANDIFFUSER_ROOT = BASE_DIR.parent
if str(CLEANDIFFUSER_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEANDIFFUSER_ROOT))

from cleandiffuser.diffusion.ddpm import DDPM
from cleandiffuser.nn_diffusion.dit import DiT1d

DEFAULT_H5_PATH = BASE_DIR / "xarmlite6_gpu_trajectories_100000_sub10.hdf5"
DEFAULT_CACHE_DIR = BASE_DIR / "kinematic_token_cache_qL_normal_sub10"
DEFAULT_WORKDIR = BASE_DIR / "dit_kinematic_inpainting_runs"
DEFAULT_RUN_NAME = "ddpm32_dit_inpaint_qL_from_posdirnormal_sub10"


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
    def dir_slice(self) -> slice:
        return slice(self.q_dim + 3, self.q_dim + 6)

    @property
    def normal_slice(self) -> slice:
        return slice(self.q_dim + 6, self.q_dim + 9)

    @property
    def length_slice(self) -> slice:
        return slice(self.q_dim + 9, self.q_dim + 10)

    @property
    def token_dim(self) -> int:
        return self.q_dim + 10


def set_seed(seed: int):
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


def to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def infer_layout_from_h5(h5_path: Path) -> FeatureLayout:
    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f['trajectories'].keys())
        if not keys:
            raise RuntimeError(f'No trajectories found in {h5_path}')
        q_shape = f['trajectories'][keys[0]]['q'].shape
    return FeatureLayout(q_dim=int(q_shape[1]))


def prepare_raw_token_cache(
    h5_path: Path,
    cache_dir: Path,
    max_trajectories: Optional[int] = None,
    progress_every: int = 500,
) -> tuple[Path, FeatureLayout, dict]:
    h5_path = Path(h5_path).resolve()
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    layout = infer_layout_from_h5(h5_path)
    suffix = f'_maxtraj{max_trajectories}' if max_trajectories is not None else ''
    tokens_path = cache_dir / f'{h5_path.stem}_qLnormal_tokens_q{layout.q_dim}{suffix}.npy'
    meta_path = cache_dir / f'{h5_path.stem}_qLnormal_tokens_q{layout.q_dim}{suffix}_meta.json'

    if tokens_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        return tokens_path, layout, metadata

    with h5py.File(h5_path, 'r') as f:
        group_names = sorted(f['trajectories'].keys())
        if max_trajectories is not None:
            group_names = group_names[: max_trajectories]
        total_samples = int(sum(int(f['trajectories'][name].attrs['num_points']) for name in group_names))

        mmap = np.lib.format.open_memmap(tokens_path, mode='w+', dtype=np.float32, shape=(total_samples, layout.token_dim))
        cursor = 0
        for traj_idx, name in enumerate(group_names):
            grp = f['trajectories'][name]
            q = np.asarray(grp['q'][:], dtype=np.float32)
            pos = np.asarray(grp['tcp_pos'][:], dtype=np.float32)
            direction = np.asarray(grp.attrs['direction'], dtype=np.float32)
            target_normal = np.asarray(grp.attrs['target_normal'], dtype=np.float32)
            d = np.repeat(direction.reshape(1, 3), q.shape[0], axis=0)
            n = np.repeat(target_normal.reshape(1, 3), q.shape[0], axis=0)
            remaining_length = np.asarray(grp['remaining_length'][:], dtype=np.float32).reshape(-1, 1)
            tokens = np.concatenate([q, pos, d, n, remaining_length], axis=1).astype(np.float32)
            mmap[cursor: cursor + q.shape[0]] = tokens
            cursor += q.shape[0]
            if progress_every > 0 and ((traj_idx + 1) % progress_every == 0 or traj_idx + 1 == len(group_names)):
                print(f'[cache] trajectories={traj_idx + 1}/{len(group_names)} samples={cursor}/{total_samples}')

    metadata = {
        'h5_path': str(h5_path),
        'tokens_path': str(tokens_path),
        'num_samples': total_samples,
        'q_dim': layout.q_dim,
        'token_dim': layout.token_dim,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    return tokens_path, layout, metadata


def compute_stats(train_tokens: np.ndarray, layout: FeatureLayout):
    q = train_tokens[:, layout.q_slice]
    pos = train_tokens[:, layout.pos_slice]
    return {
        'q_mean': q.mean(axis=0).astype(np.float32),
        'q_std': np.maximum(q.std(axis=0).astype(np.float32), 1e-6),
        'pos_mean': pos.mean(axis=0).astype(np.float32),
        'pos_std': np.maximum(pos.std(axis=0).astype(np.float32), 1e-6),
        'q_dim': int(layout.q_dim),
        'token_dim': int(layout.token_dim),
    }


def normalize_q(q: np.ndarray, stats: dict):
    return ((q - stats['q_mean']) / stats['q_std']).astype(np.float32)


def denormalize_q(q: np.ndarray, stats: dict):
    return (q * stats['q_std'] + stats['q_mean']).astype(np.float32)


def normalize_condition(condition: np.ndarray, stats: dict):
    pos = (condition[:, :3] - stats['pos_mean']) / stats['pos_std']
    direction = condition[:, 3:6]
    normal = condition[:, 6:9]
    return np.concatenate([pos.astype(np.float32), direction.astype(np.float32), normal.astype(np.float32)], axis=1)


def build_inpainting_x(raw_tokens: np.ndarray, stats: dict, layout: FeatureLayout):
    q_norm = normalize_q(raw_tokens[:, layout.q_slice], stats)
    cond = np.concatenate([raw_tokens[:, layout.pos_slice], raw_tokens[:, layout.dir_slice], raw_tokens[:, layout.normal_slice]], axis=1).astype(np.float32)
    cond_norm = normalize_condition(cond, stats)
    remaining_length = raw_tokens[:, layout.length_slice].astype(np.float32)
    return np.concatenate([q_norm, cond_norm, remaining_length], axis=1).astype(np.float32)


class InpaintingDataset(Dataset):
    def __init__(self, x0: np.ndarray, line_length: np.ndarray):
        self.x0 = torch.from_numpy(x0).float().unsqueeze(1)
        self.line_length = torch.from_numpy(line_length).float()

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, idx):
        return {'x0': self.x0[idx], 'line_length': self.line_length[idx]}


def create_loader(dataset: InpaintingDataset, batch_size: int, weighted: bool, shuffle: bool = False):
    if weighted:
        weights = dataset.line_length.numpy().copy()
        weights = weights / np.maximum(weights.mean(), 1e-6)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(weights),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def create_model(device: torch.device, x_min: np.ndarray, x_max: np.ndarray, diffusion_steps: int, q_dim: int):
    x_dim = q_dim + 10
    nn_diffusion = DiT1d(
        in_dim=x_dim,
        emb_dim=256,
        d_model=384,
        n_heads=6,
        depth=8,
        dropout=0.0,
    )
    fix_mask = np.ones((1, x_dim), dtype=np.float32)
    fix_mask[:, :q_dim] = 0.0
    fix_mask[:, -1] = 0.0
    model = DDPM(
        nn_diffusion=nn_diffusion,
        nn_condition=None,
        fix_mask=fix_mask,
        loss_weight=np.ones((1, x_dim), dtype=np.float32),
        grad_clip_norm=1.0,
        diffusion_steps=diffusion_steps,
        ema_rate=0.999,
        optim_params={'lr': 2e-4, 'weight_decay': 1e-4},
        x_min=torch.tensor(x_min, dtype=torch.float32, device=device).view(1, 1, x_dim),
        x_max=torch.tensor(x_max, dtype=torch.float32, device=device).view(1, 1, x_dim),
        predict_noise=True,
        device=device,
    )
    return model


@torch.no_grad()
def sample_q_length_from_condition(model, stats: dict, condition: np.ndarray, device: torch.device, q_dim: int, n_samples: int, sample_steps: int, temperature: float = 1.0):
    cond_norm = normalize_condition(condition[None, :].astype(np.float32), stats)[0]
    x_dim = q_dim + 10
    prior = np.zeros((n_samples, 1, x_dim), dtype=np.float32)
    prior[:, 0, q_dim:q_dim + 9] = cond_norm[None, :]
    prior_t = torch.from_numpy(prior).float().to(device)
    samples, _ = model.sample(
        prior=prior_t,
        n_samples=n_samples,
        sample_steps=sample_steps,
        use_ema=True,
        temperature=temperature,
    )
    samples_np = samples[:, 0, :].detach().cpu().numpy()
    q_norm = samples_np[:, :q_dim]
    q = denormalize_q(q_norm, stats)
    pred_length = samples_np[:, -1].astype(np.float32)
    return q, pred_length, samples_np


def save_bundle(run_dir: Path, model, stats: dict, x_min: np.ndarray, x_max: np.ndarray, args, metadata: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(run_dir / 'model_latest.pt'))
    payload = {
        'stats': {k: np.asarray(v, dtype=np.float32) if isinstance(v, np.ndarray) else v for k, v in stats.items()},
        'x_min': np.asarray(x_min, dtype=np.float32),
        'x_max': np.asarray(x_max, dtype=np.float32),
        'args': dict(vars(args)),
        'metadata': metadata,
    }
    torch.save(payload, run_dir / 'bundle_latest.pt')
    with open(run_dir / 'metadata_latest.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'args': to_jsonable(dict(vars(args))),
                'metadata': to_jsonable(metadata),
                'stats': to_jsonable({k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()}),
                'x_min': to_jsonable(np.asarray(x_min).tolist()),
                'x_max': to_jsonable(np.asarray(x_max).tolist()),
            },
            f,
            indent=2,
        )
