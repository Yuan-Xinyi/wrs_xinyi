from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from paths import DATASETS_DIR

DEFAULT_H5 = DATASETS_DIR / 'fr3_cvt_test_tasks.hdf5'
DEFAULT_PLOT_DIR = DATASETS_DIR / 'fr3_cvt_test_tasks_plots'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize FR3 CVT test task distributions.')
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5)
    parser.add_argument('--plot-dir', type=Path, default=DEFAULT_PLOT_DIR)
    return parser.parse_args()


def load_group(group: h5py.Group) -> dict[str, np.ndarray]:
    return {key: np.asarray(group[key][:]) for key in group.keys()}


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_joint_histograms(all_data: dict[str, np.ndarray], plot_dir: Path) -> str:
    q = all_data['q']
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    axes = axes.reshape(-1)
    for j in range(q.shape[1]):
        ax = axes[j]
        ax.hist(q[:, j], bins=40, color='#4682b4', edgecolor='black', alpha=0.85)
        ax.set_title(f'Joint q{j + 1}')
        ax.grid(alpha=0.2)
    axes[-1].axis('off')
    path = plot_dir / 'joint_histograms.png'
    save_fig(fig, path)
    return str(path)


def plot_workspace_scatter(all_data: dict[str, np.ndarray], hard_union: dict[str, np.ndarray], plot_dir: Path) -> str:
    pos_all = all_data['pos']
    pos_hard = hard_union['pos']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [(0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ')]
    for ax, (a, b, label) in zip(axes, pairs):
        ax.scatter(pos_all[:, a], pos_all[:, b], s=6, alpha=0.18, label='all_tasks')
        ax.scatter(pos_hard[:, a], pos_hard[:, b], s=10, alpha=0.75, label='hard_union')
        ax.set_xlabel(['x', 'x', 'y'][pairs.index((a, b, label))])
        ax.set_ylabel(['y', 'z', 'z'][pairs.index((a, b, label))])
        ax.set_title(f'Workspace {label}')
        ax.grid(alpha=0.2)
    axes[0].legend()
    path = plot_dir / 'workspace_scatter.png'
    save_fig(fig, path)
    return str(path)


def plot_direction_normal(all_data: dict[str, np.ndarray], hard_union: dict[str, np.ndarray], plot_dir: Path) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    names = ['x', 'y', 'z']
    for i in range(3):
        axes[0, i].hist(all_data['direction'][:, i], bins=40, alpha=0.75, label='all_tasks', color='#2e8b57')
        axes[0, i].hist(hard_union['direction'][:, i], bins=40, alpha=0.55, label='hard_union', color='#b22222')
        axes[0, i].set_title(f'Direction {names[i]}')
        axes[0, i].grid(alpha=0.2)
        axes[1, i].hist(all_data['normal'][:, i], bins=40, alpha=0.75, label='all_tasks', color='#2e8b57')
        axes[1, i].hist(hard_union['normal'][:, i], bins=40, alpha=0.55, label='hard_union', color='#b22222')
        axes[1, i].set_title(f'Normal {names[i]}')
        axes[1, i].grid(alpha=0.2)
    axes[0, 0].legend()
    path = plot_dir / 'direction_normal_histograms.png'
    save_fig(fig, path)
    return str(path)


def plot_metric_distributions(
    all_data: dict[str, np.ndarray],
    hard_limit: dict[str, np.ndarray],
    hard_singularity: dict[str, np.ndarray],
    plot_dir: Path,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(all_data['joint_limit_margin'], bins=50, alpha=0.6, label='all_tasks', color='#4682b4')
    axes[0].hist(hard_limit['joint_limit_margin'], bins=50, alpha=0.8, label='hard_joint_limit', color='#d2691e')
    axes[0].set_title('Joint-limit Margin')
    axes[0].set_xlabel('min normalized distance to joint limits')
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].hist(all_data['min_sigma'], bins=50, alpha=0.6, label='all_tasks', color='#4682b4')
    axes[1].hist(hard_singularity['min_sigma'], bins=50, alpha=0.8, label='hard_singularity', color='#8a2be2')
    axes[1].set_title('Minimum Jacobian Singular Value')
    axes[1].set_xlabel('min singular value')
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    path = plot_dir / 'metric_distributions.png'
    save_fig(fig, path)
    return str(path)


def plot_metric_scatter(
    all_data: dict[str, np.ndarray],
    hard_limit: dict[str, np.ndarray],
    hard_singularity: dict[str, np.ndarray],
    plot_dir: Path,
) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_data['joint_limit_margin'], all_data['min_sigma'], s=8, alpha=0.15, label='all_tasks')
    ax.scatter(hard_limit['joint_limit_margin'], hard_limit['min_sigma'], s=16, alpha=0.7, label='hard_joint_limit')
    ax.scatter(hard_singularity['joint_limit_margin'], hard_singularity['min_sigma'], s=16, alpha=0.7, label='hard_singularity')
    ax.set_xlabel('joint_limit_margin')
    ax.set_ylabel('min_sigma')
    ax.set_title('Hard-set Selection in Metric Space')
    ax.grid(alpha=0.2)
    ax.legend()
    path = plot_dir / 'metric_scatter.png'
    save_fig(fig, path)
    return str(path)


def write_summary(all_data: dict[str, np.ndarray], hard_limit: dict[str, np.ndarray], hard_singularity: dict[str, np.ndarray], hard_union: dict[str, np.ndarray], plot_dir: Path) -> str:
    payload = {
        'all_tasks': {
            'count': int(all_data['q'].shape[0]),
            'joint_limit_margin_mean': float(np.mean(all_data['joint_limit_margin'])),
            'joint_limit_margin_min': float(np.min(all_data['joint_limit_margin'])),
            'min_sigma_mean': float(np.mean(all_data['min_sigma'])),
            'min_sigma_min': float(np.min(all_data['min_sigma'])),
            'pos_mean': np.mean(all_data['pos'], axis=0).tolist(),
        },
        'hard_joint_limit': {
            'count': int(hard_limit['q'].shape[0]),
            'joint_limit_margin_mean': float(np.mean(hard_limit['joint_limit_margin'])),
            'joint_limit_margin_max': float(np.max(hard_limit['joint_limit_margin'])),
            'min_sigma_mean': float(np.mean(hard_limit['min_sigma'])),
        },
        'hard_singularity': {
            'count': int(hard_singularity['q'].shape[0]),
            'joint_limit_margin_mean': float(np.mean(hard_singularity['joint_limit_margin'])),
            'min_sigma_mean': float(np.mean(hard_singularity['min_sigma'])),
            'min_sigma_max': float(np.max(hard_singularity['min_sigma'])),
        },
        'hard_union': {
            'count': int(hard_union['q'].shape[0]),
            'joint_limit_margin_mean': float(np.mean(hard_union['joint_limit_margin'])),
            'min_sigma_mean': float(np.mean(hard_union['min_sigma'])),
            'pos_mean': np.mean(hard_union['pos'], axis=0).tolist(),
        },
    }
    path = plot_dir / 'summary.json'
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return str(path)


def main() -> None:
    args = parse_args()
    if plt is None:
        raise RuntimeError('matplotlib is required for visualization.')
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.h5_path, 'r') as f:
        all_data = load_group(f['all_tasks'])
        hard_limit = load_group(f['hard_joint_limit'])
        hard_singularity = load_group(f['hard_singularity'])
        hard_union = load_group(f['hard_union'])

    outputs = {
        'joint_histograms': plot_joint_histograms(all_data, args.plot_dir),
        'workspace_scatter': plot_workspace_scatter(all_data, hard_union, args.plot_dir),
        'direction_normal_histograms': plot_direction_normal(all_data, hard_union, args.plot_dir),
        'metric_distributions': plot_metric_distributions(all_data, hard_limit, hard_singularity, args.plot_dir),
        'metric_scatter': plot_metric_scatter(all_data, hard_limit, hard_singularity, args.plot_dir),
        'summary': write_summary(all_data, hard_limit, hard_singularity, hard_union, args.plot_dir),
    }
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
