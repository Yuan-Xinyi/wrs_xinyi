from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from paths import DATASETS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a contrastive preference index with near-cond neighbors for LNetContrastive.')
    parser.add_argument('--input', type=Path, default=DATASETS_DIR / 'xarmlite6_gpu_trajectories_100000_sub10.hdf5')
    parser.add_argument('--output', type=Path, default=DATASETS_DIR / 'xarmlite6_gpu_trajectories_100000_sub10_pref.hdf5')
    parser.add_argument('--pos-radius', type=float, default=0.02, help='Position radius in meters.')
    parser.add_argument('--angle-radius-deg', type=float, default=30.0, help='Angular radius for direction/normal matching, in degrees.')
    parser.add_argument('--k', type=int, default=200, help='Max nearest neighbors queried per sample before filtering.')
    parser.add_argument('--pair-threshold', type=float, default=0.03, help='Minimum absolute length gap to keep a neighbor pair.')
    parser.add_argument('--hard-mu-quantile', type=float, default=0.9, help='High-mu quantile used for hard-negative prioritization.')
    parser.add_argument('--hard-len-quantile', type=float, default=0.1, help='Low-length quantile used for hard-negative prioritization.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--print-every', type=int, default=50000)
    return parser.parse_args()


def copy_attrs(src_attrs: h5py.AttributeManager, dst_attrs: h5py.AttributeManager) -> None:
    for key, value in src_attrs.items():
        dst_attrs[key] = value


def copy_group_contents(src_grp: h5py.Group, dst_grp: h5py.Group) -> None:
    copy_attrs(src_grp.attrs, dst_grp.attrs)
    for key in src_grp.keys():
        src_ds = src_grp[key]
        dst_grp.create_dataset(key, data=src_ds[...], compression='gzip')


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-12, None)
    return x / norm


def build_scaled_cond(pos: np.ndarray, direction: np.ndarray, normal: np.ndarray, pos_radius: float, angle_radius_deg: float) -> np.ndarray:
    chord = 2.0 * np.sin(np.deg2rad(angle_radius_deg) / 2.0)
    pos_scaled = pos / max(pos_radius, 1e-8)
    dir_scaled = normalize_rows(direction) / max(chord, 1e-8)
    normal_scaled = normalize_rows(normal) / max(chord, 1e-8)
    return np.concatenate([pos_scaled, dir_scaled, normal_scaled], axis=1).astype(np.float32)


def pack_source(src: h5py.File) -> dict[str, np.ndarray]:
    traj_keys = sorted(src['trajectories'].keys())
    q_list = []
    pos_list = []
    dir_list = []
    normal_list = []
    len_list = []
    mu_list = []
    traj_idx_list = []
    point_idx_list = []
    traj_names = []
    for traj_idx, key in enumerate(traj_keys):
        grp = src['trajectories'][key]
        q = np.asarray(grp['q'][:], dtype=np.float32)
        pos = np.asarray(grp['tcp_pos'][:], dtype=np.float32)
        n = q.shape[0]
        direction = np.repeat(np.asarray(grp.attrs['direction'], dtype=np.float32).reshape(1, 3), n, axis=0)
        normal = np.repeat(np.asarray(grp.attrs['target_normal'], dtype=np.float32).reshape(1, 3), n, axis=0)
        length = np.asarray(grp['remaining_length'][:], dtype=np.float32)
        mu = np.asarray(grp['mu'][:], dtype=np.float32) if 'mu' in grp else np.zeros(n, dtype=np.float32)
        q_list.append(q)
        pos_list.append(pos)
        dir_list.append(direction)
        normal_list.append(normal)
        len_list.append(length)
        mu_list.append(mu)
        traj_idx_list.append(np.full(n, traj_idx, dtype=np.int32))
        point_idx_list.append(np.arange(n, dtype=np.int32))
        traj_names.append(key)
    return {
        'traj_keys': np.asarray(traj_names, dtype=object),
        'q': np.concatenate(q_list, axis=0),
        'pos': np.concatenate(pos_list, axis=0),
        'direction': np.concatenate(dir_list, axis=0),
        'normal': np.concatenate(normal_list, axis=0),
        'length': np.concatenate(len_list, axis=0),
        'mu': np.concatenate(mu_list, axis=0),
        'traj_idx': np.concatenate(traj_idx_list, axis=0),
        'point_idx': np.concatenate(point_idx_list, axis=0),
    }


def build_pair_index(packed: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, np.ndarray]:
    cond_scaled = build_scaled_cond(packed['pos'], packed['direction'], packed['normal'], args.pos_radius, args.angle_radius_deg)
    tree = cKDTree(cond_scaled)
    n = cond_scaled.shape[0]
    k = min(max(int(args.k), 2), n)
    dists, neigh = tree.query(cond_scaled, k=k)
    if k == 1:
        dists = dists[:, None]
        neigh = neigh[:, None]

    mu = packed['mu']
    length = packed['length']
    mu_hi = float(np.quantile(mu, args.hard_mu_quantile))
    len_lo = float(np.quantile(length, args.hard_len_quantile))
    len_hi = float(np.quantile(length, 1.0 - args.hard_len_quantile))

    offsets = np.zeros(n + 1, dtype=np.int64)
    all_neighbors: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_dists: list[np.ndarray] = []
    cursor = 0
    radius = 1.0
    for i in range(n):
        nn_idx = neigh[i]
        nn_dist = dists[i]
        valid = (nn_idx != i) & np.isfinite(nn_dist) & (nn_dist <= radius)
        if valid.any():
            nn_idx = nn_idx[valid].astype(np.int32)
            nn_dist = nn_dist[valid].astype(np.float32)
            gap = np.abs(length[i] - length[nn_idx])
            keep = gap > args.pair_threshold
            nn_idx = nn_idx[keep]
            nn_dist = nn_dist[keep]
            gap = gap[keep]
        else:
            nn_idx = np.empty(0, dtype=np.int32)
            nn_dist = np.empty(0, dtype=np.float32)
            gap = np.empty(0, dtype=np.float32)

        if nn_idx.size > 0:
            pseudo_i = (mu[i] >= mu_hi) and (length[i] <= len_lo)
            good_i = length[i] >= len_hi
            pseudo_j = (mu[nn_idx] >= mu_hi) & (length[nn_idx] <= len_lo)
            good_j = length[nn_idx] >= len_hi
            hard = (pseudo_i & good_j) | (good_i & pseudo_j)
            score = hard.astype(np.int32) * 10_000_000 + np.round(gap * 1000).astype(np.int32)
            order = np.lexsort((nn_dist, -score))
            nn_idx = nn_idx[order]
            nn_dist = nn_dist[order]
            score = score[order].astype(np.int32)
        else:
            score = np.empty(0, dtype=np.int32)

        all_neighbors.append(nn_idx)
        all_dists.append(nn_dist)
        all_scores.append(score)
        cursor += nn_idx.size
        offsets[i + 1] = cursor
        if args.print_every > 0 and ((i + 1) % args.print_every == 0 or i + 1 == n):
            print(f'[pref] indexed={i + 1}/{n} pairs={cursor}')

    neighbors = np.concatenate(all_neighbors, axis=0) if cursor > 0 else np.empty(0, dtype=np.int32)
    neigh_dists = np.concatenate(all_dists, axis=0) if cursor > 0 else np.empty(0, dtype=np.float32)
    neigh_scores = np.concatenate(all_scores, axis=0) if cursor > 0 else np.empty(0, dtype=np.int32)
    neighbor_count = offsets[1:] - offsets[:-1]
    return {
        'offsets': offsets,
        'neighbors': neighbors,
        'neighbor_dists': neigh_dists,
        'neighbor_scores': neigh_scores,
        'neighbor_count': neighbor_count.astype(np.int32),
        'cond_scaled': cond_scaled,
        'mu_hi': np.float32(mu_hi),
        'len_lo': np.float32(len_lo),
        'len_hi': np.float32(len_hi),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f'Output already exists: {output_path}. Use --overwrite to replace it.')

    t0 = time.perf_counter()
    mode = 'w' if args.overwrite else 'x'
    with h5py.File(input_path, 'r') as src:
        packed = pack_source(src)
        pref = build_pair_index(packed, args)

        with h5py.File(output_path, mode) as dst:
            copy_attrs(src.attrs, dst.attrs)
            dst_root = dst.create_group('trajectories')
            for key in sorted(src['trajectories'].keys()):
                copy_group_contents(src['trajectories'][key], dst_root.create_group(key))

            g = dst.create_group('contrastive_pref')
            str_dtype = h5py.string_dtype(encoding='utf-8')
            g.create_dataset('traj_key', data=packed['traj_keys'].astype(str_dtype), dtype=str_dtype)
            g.create_dataset('traj_idx', data=packed['traj_idx'], compression='gzip')
            g.create_dataset('point_idx', data=packed['point_idx'], compression='gzip')
            g.create_dataset('q', data=packed['q'], compression='gzip')
            g.create_dataset('pos', data=packed['pos'], compression='gzip')
            g.create_dataset('direction', data=packed['direction'], compression='gzip')
            g.create_dataset('normal', data=packed['normal'], compression='gzip')
            g.create_dataset('length', data=packed['length'], compression='gzip')
            g.create_dataset('mu', data=packed['mu'], compression='gzip')
            g.create_dataset('cond_scaled', data=pref['cond_scaled'], compression='gzip')
            g.create_dataset('neighbor_offsets', data=pref['offsets'], compression='gzip')
            g.create_dataset('neighbor_index', data=pref['neighbors'], compression='gzip')
            g.create_dataset('neighbor_distance', data=pref['neighbor_dists'], compression='gzip')
            g.create_dataset('neighbor_score', data=pref['neighbor_scores'], compression='gzip')
            g.create_dataset('neighbor_count', data=pref['neighbor_count'], compression='gzip')
            valid_anchor_index = np.flatnonzero(pref['neighbor_count'] > 0).astype(np.int32)
            g.create_dataset('valid_anchor_index', data=valid_anchor_index, compression='gzip')

            g.attrs['source_dataset'] = str(input_path)
            g.attrs['num_samples'] = int(packed['length'].shape[0])
            g.attrs['num_valid_anchors'] = int(valid_anchor_index.shape[0])
            g.attrs['num_pairs'] = int(pref['neighbors'].shape[0])
            g.attrs['pos_radius'] = float(args.pos_radius)
            g.attrs['angle_radius_deg'] = float(args.angle_radius_deg)
            g.attrs['k'] = int(args.k)
            g.attrs['pair_threshold'] = float(args.pair_threshold)
            g.attrs['hard_mu_quantile'] = float(args.hard_mu_quantile)
            g.attrs['hard_len_quantile'] = float(args.hard_len_quantile)
            g.attrs['hard_mu_value'] = float(pref['mu_hi'])
            g.attrs['hard_len_low_value'] = float(pref['len_lo'])
            g.attrs['hard_len_high_value'] = float(pref['len_hi'])

    print(f'[done] output={output_path} samples={packed["length"].shape[0]} valid_anchors={int((pref["neighbor_count"] > 0).sum())} pairs={pref["neighbors"].shape[0]} elapsed={time.perf_counter() - t0:.2f}s')


if __name__ == '__main__':
    main()
