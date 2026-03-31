import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kinematic_diffusion_common import (
    DEFAULT_RUN_NAME,
    DEFAULT_WORKDIR,
    denormalize_q,
    normalize_condition,
    sample_q_length_from_condition,
)
from kinematic_diffusion_common import create_model
from xarmlite6_nullspace_straight_demo import TrackerConfig, NullspaceStraightTracker, damped_pseudoinverse
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inpaint q and remaining length from (pos, direction) using a trained DDPM model.')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / 'bundle_latest.pt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pos', type=float, nargs=3, required=True)
    parser.add_argument('--direction', type=float, nargs=3, required=True)
    parser.add_argument('--n-samples', type=int, default=10)
    parser.add_argument('--sample-steps', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--correction-iters', type=int, default=50)
    parser.add_argument('--correction-tol', type=float, default=1e-4)
    parser.add_argument('--correction-damping', type=float, default=1e-3)
    parser.add_argument('--eval-real-length', action='store_true')
    parser.add_argument('--tracker-max-steps', type=int, default=400)
    return parser.parse_args()


class JacobianCorrection:
    def __init__(self, robot: XArmLite6Miller, damping: float = 1e-3):
        self.robot = robot
        self.damping = float(damping)

    def run(self, q_init: np.ndarray, target_pos: np.ndarray, max_iters: int, tol: float) -> tuple[np.ndarray, float]:
        q = np.asarray(q_init, dtype=np.float64).copy()
        joint_ranges = getattr(self.robot, 'jnt_ranges', None)
        for _ in range(max_iters):
            self.robot.goto_given_conf(q)
            cur_pos, _ = self.robot.fk(q, update=False)
            err = target_pos - cur_pos
            err_norm = float(np.linalg.norm(err))
            if err_norm < tol:
                return q.astype(np.float32), err_norm
            j_pos = self.robot.jacobian()[:3, :]
            dq = damped_pseudoinverse(j_pos, self.damping) @ err
            q = q + dq
            if joint_ranges is not None:
                q = np.clip(q, joint_ranges[:, 0], joint_ranges[:, 1])
        self.robot.goto_given_conf(q)
        cur_pos, _ = self.robot.fk(q, update=False)
        err_norm = float(np.linalg.norm(target_pos - cur_pos))
        return q.astype(np.float32), err_norm


def normalize_direction(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float32)
    direction /= max(np.linalg.norm(direction), 1e-12)
    return direction


def load_model(bundle_path: Path, device: torch.device):
    bundle = torch.load(bundle_path, map_location=device, weights_only=False)
    stats = {k: np.asarray(v, dtype=np.float32) if isinstance(v, (list, tuple, np.ndarray)) else v for k, v in bundle['stats'].items()}
    q_dim = int(stats['q_dim'])
    x_min = np.asarray(bundle['x_min'], dtype=np.float32)
    x_max = np.asarray(bundle['x_max'], dtype=np.float32)
    args = bundle.get('args', {})
    diffusion_steps = int(args.get('diffusion_steps', 32))
    model = create_model(device=device, x_min=x_min, x_max=x_max, diffusion_steps=diffusion_steps, q_dim=q_dim)
    model_path = bundle_path.with_name('model_best.pt')
    if not model_path.exists():
        model_path = bundle_path.with_name('model_latest.pt')
    model.load(str(model_path))
    model.eval()
    return bundle, stats, model, q_dim, diffusion_steps


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    bundle, stats, model, q_dim, diffusion_steps = load_model(args.bundle, device)

    pos = np.asarray(args.pos, dtype=np.float32)
    direction = normalize_direction(np.asarray(args.direction, dtype=np.float32))
    condition = np.concatenate([pos, direction], axis=0).astype(np.float32)
    steps = args.sample_steps if args.sample_steps is not None else diffusion_steps
    q_samples, pred_lengths, raw_samples = sample_q_length_from_condition(
        model=model,
        stats=stats,
        condition=condition,
        device=device,
        q_dim=q_dim,
        n_samples=args.n_samples,
        sample_steps=int(steps),
        temperature=float(args.temperature),
    )

    robot = XArmLite6Miller(enable_cc=True)
    correction = JacobianCorrection(robot=robot, damping=args.correction_damping)
    tracker = NullspaceStraightTracker(robot=robot, config=TrackerConfig(max_steps=args.tracker_max_steps))

    corrected_q_list = []
    raw_q_list = []
    length_list = []
    pos_error_list = []
    pred_length_list = []

    for idx, (q_pred, pred_length) in enumerate(zip(q_samples, pred_lengths)):
        q_corr, pos_err = correction.run(q_pred, pos, max_iters=args.correction_iters, tol=args.correction_tol)
        corrected_q_list.append(q_corr)
        raw_q_list.append(q_pred.astype(np.float32))
        pos_error_list.append(pos_err)
        pred_length_list.append(float(pred_length))
        if args.eval_real_length:
            result = tracker.run(q_corr, direction)
            length_list.append(float(result.projected_length))
        print(
            f'[sample {idx}] pred_length={float(pred_length):.6f} pos_err={pos_err:.6f} '
            f'q_pred={np.array2string(q_pred, precision=4, separator=", ")} '
            f'q_corr={np.array2string(q_corr, precision=4, separator=", ")}'
        )
        if args.eval_real_length:
            print(f'           real_projected_length={length_list[-1]:.6f} m')

    corrected_q_arr = np.stack(corrected_q_list, axis=0)
    raw_q_arr = np.stack(raw_q_list, axis=0)
    q_diversity = float(np.mean(np.std(corrected_q_arr, axis=0)))
    print(f'[summary] q_diversity_mean_std={q_diversity:.6f}')
    print(f'[summary] mean_predicted_length={float(np.mean(pred_length_list)):.6f} m')
    print(f'[summary] mean_pos_error={float(np.mean(pos_error_list)):.6f} m')
    if args.eval_real_length and length_list:
        lengths = np.asarray(length_list, dtype=np.float32)
        print(f'[summary] mean_real_projected_length={float(lengths.mean()):.6f} m')
        print(f'[summary] max_real_projected_length={float(lengths.max()):.6f} m')

    payload = {
        'input_pos': pos.tolist(),
        'input_direction': direction.tolist(),
        'predicted_q': raw_q_arr.tolist(),
        'corrected_q': corrected_q_arr.tolist(),
        'predicted_length': [float(v) for v in pred_length_list],
        'pos_error': [float(v) for v in pos_error_list],
        'real_projected_length': [float(v) for v in length_list],
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
