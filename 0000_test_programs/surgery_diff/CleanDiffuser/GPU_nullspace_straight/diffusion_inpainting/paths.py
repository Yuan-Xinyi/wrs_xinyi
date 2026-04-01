from __future__ import annotations

from pathlib import Path


DIFFUSION_DIR = Path(__file__).resolve().parent
GPU_NULLSPACE_DIR = DIFFUSION_DIR.parent
DATASETS_DIR = GPU_NULLSPACE_DIR / 'datasets'
RUNS_DIR = GPU_NULLSPACE_DIR / 'runs'
UTILS_DIR = GPU_NULLSPACE_DIR / 'utils'

DEFAULT_H5_PATH = DATASETS_DIR / 'xarmlite6_gpu_trajectories_100000_sub10.hdf5'
DEFAULT_CACHE_DIR = RUNS_DIR / 'kinematic_token_cache_qL_normal_sub10'
DEFAULT_WORKDIR = RUNS_DIR / 'dit_kinematic_inpainting_runs'
DEFAULT_RUN_NAME = 'ddpm32_dit_inpaint_qL_from_posdirnormal_sub10'
