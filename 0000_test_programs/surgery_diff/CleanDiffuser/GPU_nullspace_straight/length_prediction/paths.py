from __future__ import annotations

from pathlib import Path


LENGTH_PREDICTION_DIR = Path(__file__).resolve().parent
GPU_NULLSPACE_DIR = LENGTH_PREDICTION_DIR.parent
DATASETS_DIR = GPU_NULLSPACE_DIR / 'datasets'
RUNS_DIR = GPU_NULLSPACE_DIR / 'runs'

DEFAULT_H5 = DATASETS_DIR / 'xarmlite6_gpu_trajectories_100000_sub10.hdf5'
LNET_RUNS_DIR = RUNS_DIR / 'lnet_runs'
LNET_CONTRASTIVE_RUNS_DIR = RUNS_DIR / 'lnet_contrastive_runs'
