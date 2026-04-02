from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion import DEFAULT_H5_PATH, DEFAULT_RUN_NAME, DEFAULT_WORKDIR, normalize_condition
from diffusion_sample import load_model
from diffusion_eval_batch_candidates_lnet import (
    batch_position_error_and_correction,
    build_tracker,
    load_lnet_contrastive_model,
    rollout_lengths_batch,
)
from length_prediction.paths import LNET_CONTRASTIVE_RUNS_DIR


class ContrastiveScoreAdapter:
    def __init__(self, lnet_contrastive, q_mean: np.ndarray, q_std: np.ndarray, q_dim: int, device: torch.device, fixed_guidance_step: float = 1.0, grad_eps: float = 1e-6):
        self.lnet_contrastive = lnet_contrastive
        self.q_mean = torch.from_numpy(q_mean.astype(np.float32)).to(device).view(1, -1)
        self.q_std = torch.from_numpy(q_std.astype(np.float32)).to(device).view(1, -1)
        self.q_dim = int(q_dim)
        self.device = device
        self.fixed_guidance_step = float(fixed_guidance_step)
        self.grad_eps = float(grad_eps)

    def gradients(self, x: torch.Tensor, t: torch.Tensor, condition_vec_cg: torch.Tensor):
        q_norm = x[:, 0, :self.q_dim]
        q_raw = q_norm * self.q_std + self.q_mean
        cond_raw = condition_vec_cg.to(self.device)
        grad_q_raw = self.lnet_contrastive.get_guidance_gradient(q_raw, cond_raw)
        grad_q_norm = grad_q_raw * self.q_std
        grad_norm = torch.linalg.norm(grad_q_norm, dim=1, keepdim=True)
        grad_q_norm = self.fixed_guidance_step * grad_q_norm / (grad_norm + self.grad_eps)
        grad = torch.zeros_like(x)
        grad[:, 0, :self.q_dim] = grad_q_norm
        with torch.no_grad():
            score, _ = self.lnet_contrastive(q_raw, cond_raw)
        return score.detach(), grad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Try contrastive classifier guidance during diffusion denoising and visualize unguided vs guided trajectories.')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--lnet-contrastive-ckpt', type=Path, default=LNET_CONTRASTIVE_RUNS_DIR / 'lnet_contrastive_q_cond_to_length_sub10_pref' / 'lnet_contrastive_best.pt')
    parser.add_argument('--h5-path', type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument('--traj-id', type=str, default=None)
    parser.add_argument('--point-idx', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[0.0, 0.1, 1.0, 5.0, 10.0])
    parser.add_argument('--vis-lambda', type=float, default=5.0)
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--fixed-guidance-step', type=float, default=1.0)
    parser.add_argument('--guidance-grad-eps', type=float, default=1e-6)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'guidance_vis')
    parser.add_argument('--no-vis', action='store_true')
    return parser.parse_args()


def normalize_direction(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    return vec / max(float(np.linalg.norm(vec)), 1e-12)


def load_anchor(h5_path: Path, traj_id: str | None, point_idx: int, rng: np.random.Generator) -> dict:
    with h5py.File(h5_path, 'r') as f:
        traj_root = f['trajectories']
        if traj_id is None:
            keys = sorted(traj_root.keys())
            traj_id = str(rng.choice(keys))
        grp = traj_root[traj_id]
        n = int(grp.attrs['num_points'])
        idx = int(np.clip(point_idx, 0, n - 1))
        q = np.asarray(grp['q'][idx], dtype=np.float32)
        pos = np.asarray(grp['tcp_pos'][idx], dtype=np.float32)
        direction = normalize_direction(np.asarray(grp.attrs['direction'], dtype=np.float32))
        target_normal = normalize_direction(np.asarray(grp.attrs['target_normal'], dtype=np.float32))
        gt_length = float(np.asarray(grp['remaining_length'][idx], dtype=np.float32))
        return {
            'traj_id': traj_id,
            'point_idx': idx,
            'q': q,
            'pos': pos,
            'direction': direction,
            'target_normal': target_normal,
            'gt_length': gt_length,
        }


def sample_with_guidance(
    model,
    lnet_contrastive,
    stats: dict,
    q_dim: int,
    condition_raw_np: np.ndarray,
    prior: torch.Tensor,
    init_noise: torch.Tensor,
    sample_steps: int,
    temperature: float,
    lambda_guidance: float,
    device: torch.device,
    fixed_guidance_step: float = 1.0,
    guidance_grad_eps: float = 1e-6,
):
    if sample_steps != int(model.diffusion_steps):
        raise ValueError(f'sample_steps must equal diffusion_steps={model.diffusion_steps} for this script.')

    adapter = ContrastiveScoreAdapter(
        lnet_contrastive,
        np.asarray(stats['q_mean'], dtype=np.float32),
        np.asarray(stats['q_std'], dtype=np.float32),
        q_dim,
        device,
        fixed_guidance_step=fixed_guidance_step,
        grad_eps=guidance_grad_eps,
    )
    cond_raw = torch.from_numpy(condition_raw_np[None, :].astype(np.float32)).to(device)
    xt = init_noise.clone().to(device) * float(temperature)
    xt = xt * (1.0 - model.fix_mask) + prior * model.fix_mask

    history_q = []
    history_score = []
    history_pred_length = []
    history_step = []

    def record(step_tag: int, x: torch.Tensor) -> None:
        q_norm = x[:, 0, :q_dim]
        q_raw = q_norm * adapter.q_std + adapter.q_mean
        with torch.no_grad():
            score, _ = lnet_contrastive(q_raw, cond_raw)
        history_q.append(q_raw[0].detach().cpu().numpy().astype(np.float32))
        history_score.append(float(score[0].detach().cpu()))
        history_pred_length.append(float(x[0, 0, -1].detach().cpu()))
        history_step.append(int(step_tag))

    old_classifier = getattr(model, 'classifier', None)
    model.classifier = adapter if float(lambda_guidance) != 0.0 else None
    try:
        record(sample_steps, xt)
        for t in range(model.diffusion_steps - 1, -1, -1):
            t_batch = torch.tensor(t, device=device, dtype=torch.long).repeat(prior.shape[0])
            bar_alpha = model.bar_alpha[t]
            bar_alpha_prev = model.bar_alpha[t - 1] if t > 0 else torch.tensor(1.0, device=device)
            alpha = model.alpha[t]
            beta = model.beta[t]
            pred_theta, _ = model.predict_function(
                xt,
                t_batch,
                bar_alpha,
                use_ema=True,
                requires_grad=float(lambda_guidance) != 0.0,
                condition_vec_cfg=None,
                condition_vec_cg=cond_raw if float(lambda_guidance) != 0.0 else None,
                w_cfg=0.0,
                w_cg=float(lambda_guidance),
            )
            if model.predict_noise:
                xt = 1.0 / alpha.sqrt() * (xt - beta / (1.0 - bar_alpha).sqrt() * pred_theta)
            else:
                xt = 1.0 / (1.0 - bar_alpha) * (
                    alpha.sqrt() * (1.0 - bar_alpha_prev) * xt + beta * bar_alpha_prev.sqrt() * pred_theta
                )
            if t != 0:
                xt = xt + (beta * (1.0 - bar_alpha_prev) / (1.0 - bar_alpha)).sqrt() * torch.randn_like(xt)
            xt = xt * (1.0 - model.fix_mask) + prior * model.fix_mask
            record(t, xt)
    finally:
        model.classifier = old_classifier

    return {
        'lambda': float(lambda_guidance),
        'final_q': history_q[-1],
        'final_score': float(history_score[-1]),
        'final_pred_length': float(history_pred_length[-1]),
        'history_step': np.asarray(history_step, dtype=np.int32),
        'history_q': np.asarray(history_q, dtype=np.float32),
        'history_score': np.asarray(history_score, dtype=np.float32),
        'history_pred_length': np.asarray(history_pred_length, dtype=np.float32),
    }


def save_figure(results: list[dict], anchor: dict, vis_lambda: float, output_path: Path) -> None:
    lambda_map = {float(r['lambda']): r for r in results}
    if 0.0 not in lambda_map:
        raise ValueError('lambdas must include 0.0 for unguided baseline.')
    if float(vis_lambda) not in lambda_map:
        raise ValueError(f'vis_lambda={vis_lambda} is not in lambdas.')

    base = lambda_map[0.0]
    guided = lambda_map[float(vis_lambda)]

    fig, axes = plt.subplots(7, 1, figsize=(10, 16), sharex=True)
    for j in range(6):
        ax = axes[j]
        ax.plot(base['history_step'], base['history_q'][:, j], label='unguided', color='tab:gray', linewidth=2)
        ax.plot(guided['history_step'], guided['history_q'][:, j], label=f'guided λ={vis_lambda:g}', color='tab:blue', linewidth=2)
        ax.axhline(float(anchor['q'][j]), color='tab:green', linestyle='--', linewidth=1)
        ax.set_ylabel(f'q{j + 1}')
        if j == 0:
            ax.legend(loc='best')
    score_ax = axes[6]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(results)))
    for color, result in zip(colors, results):
        score_ax.plot(result['history_step'], result['history_score'], color=color, linewidth=2, label=f"λ={result['lambda']:g}")
    score_ax.set_ylabel('contrastive score')
    score_ax.set_xlabel('reverse diffusion step')
    score_ax.legend(loc='best', ncol=3)
    fig.suptitle(f"traj={anchor['traj_id']} point={anchor['point_idx']} gt_len={anchor['gt_length']:.3f}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed if args.seed is not None else int(np.random.SeedSequence().entropy))
    device = torch.device(args.device)

    _, stats, model, q_dim, diffusion_steps = load_model(args.bundle, device)
    lnet_contrastive = load_lnet_contrastive_model(args.lnet_contrastive_ckpt, device)
    tracker, tracker_device = build_tracker(device)

    anchor = load_anchor(args.h5_path, args.traj_id, args.point_idx, rng)
    condition_raw = np.concatenate([anchor['pos'], anchor['direction'], anchor['target_normal']], axis=0).astype(np.float32)
    condition_norm = normalize_condition(condition_raw[None, :], stats)[0]
    x_dim = q_dim + 10
    prior_np = np.zeros((1, 1, x_dim), dtype=np.float32)
    prior_np[:, 0, q_dim:q_dim + 9] = condition_norm[None, :]
    prior = torch.from_numpy(prior_np).float().to(device)
    init_noise = torch.randn_like(prior)
    steps = int(args.sample_steps) if args.sample_steps is not None else int(diffusion_steps)

    results = []
    for lam in args.lambdas:
        results.append(
            sample_with_guidance(
                model=model,
                lnet_contrastive=lnet_contrastive,
                stats=stats,
                q_dim=q_dim,
                condition_raw_np=condition_raw,
                prior=prior,
                init_noise=init_noise,
                sample_steps=steps,
                temperature=float(args.temperature),
                lambda_guidance=float(lam),
                device=device,
                fixed_guidance_step=float(args.fixed_guidance_step),
                guidance_grad_eps=float(args.guidance_grad_eps),
            )
        )

    q_pred_batch = np.stack([r['final_q'] for r in results], axis=0).astype(np.float32)
    q_corr_batch, raw_pos_err = batch_position_error_and_correction(
        tracker.robot,
        q_pred_batch,
        anchor['pos'],
        args.correction_damping,
        args.correction_iters,
        args.correction_tol,
        tracker_device,
    )
    real_len_batch = rollout_lengths_batch(tracker, tracker_device, q_corr_batch, anchor['direction'], anchor['target_normal'])
    gt_real = float(rollout_lengths_batch(tracker, tracker_device, anchor['q'][None, :], anchor['direction'], anchor['target_normal'])[0])

    summary_rows = []
    for idx, result in enumerate(results):
        summary_rows.append({
            'lambda': float(result['lambda']),
            'final_score': float(result['final_score']),
            'diff_pred_length': float(result['final_pred_length']),
            'raw_pos_err_mm': float(raw_pos_err[idx] * 1e3),
            'guided_real_len': float(real_len_batch[idx]),
            'gain_vs_gt': float(real_len_batch[idx] - gt_real),
        })
    summary_rows = sorted(summary_rows, key=lambda x: x['lambda'])
    best = max(summary_rows, key=lambda x: x['guided_real_len'])

    payload = {
        'traj_id': anchor['traj_id'],
        'point_idx': int(anchor['point_idx']),
        'gt_dataset_length': float(anchor['gt_length']),
        'gt_real_length': gt_real,
        'selection_condition': {
            'pos': anchor['pos'].tolist(),
            'direction': anchor['direction'].tolist(),
            'target_normal': anchor['target_normal'].tolist(),
        },
        'lambdas': summary_rows,
        'best_guided_by_real_len': best,
    }
    print(json.dumps(payload, indent=2))

    if not args.no_vis:
        safe_traj = anchor['traj_id'].replace('/', '_')
        fig_path = args.output_dir / f'contrastive_guidance_traj_{safe_traj}_pt_{anchor["point_idx"]}.png'
        save_figure(results, anchor, args.vis_lambda, fig_path)
        print(f'[saved] {fig_path}')


if __name__ == '__main__':
    main()
