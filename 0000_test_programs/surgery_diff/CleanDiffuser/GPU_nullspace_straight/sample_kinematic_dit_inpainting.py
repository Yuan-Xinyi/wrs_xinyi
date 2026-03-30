import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from kinematic_diffusion_common import (
    DEFAULT_RUN_NAME,
    DEFAULT_WORKDIR,
    FeatureLayout,
    MaskConditionedDiT,
    MaskedDiffusionModel,
    ResidualMLPDenoiser,
    StandardScaler,
    canonicalize_quaternion_xyzw,
)

from xarmlite6_nullspace_straight_demo import TrackerConfig, NullspaceStraightTracker, damped_pseudoinverse
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller
import wrs.basis.robot_math as rm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inpaint q and rot_q from (pos, direction, target_length) using a trained DiT model.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_WORKDIR / DEFAULT_RUN_NAME / "bundle_latest.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pos", type=float, nargs=3, required=True)
    parser.add_argument("--direction", type=float, nargs=3, required=True)
    parser.add_argument("--target-length", type=float, default=0.6)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--correction-iters", type=int, default=50)
    parser.add_argument("--correction-tol", type=float, default=1e-4)
    parser.add_argument("--correction-damping", type=float, default=1e-3)
    parser.add_argument("--eval-real-length", action="store_true")
    parser.add_argument("--tracker-max-steps", type=int, default=400)
    return parser.parse_args()


class JacobianCorrection:
    def __init__(self, robot: XArmLite6Miller, damping: float = 1e-3):
        self.robot = robot
        self.damping = float(damping)

    def run(self, q_init: np.ndarray, target_pos: np.ndarray, max_iters: int, tol: float) -> tuple[np.ndarray, float]:
        q = np.asarray(q_init, dtype=np.float64).copy()
        joint_ranges = getattr(self.robot, "jnt_ranges", None)
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



def build_model(bundle: dict, device: torch.device):
    args = bundle["args"]
    token_dim = int(bundle["layout_token_dim"])
    backbone = args.get("backbone", "dit")
    if backbone == "dit":
        denoiser = MaskConditionedDiT(token_dim=token_dim)
    else:
        denoiser = ResidualMLPDenoiser(token_dim=token_dim)
    model = MaskedDiffusionModel(
        denoiser=denoiser,
        token_dim=token_dim,
        diffusion_steps=int(args.get("diffusion_steps", 64)),
        predict_x0=True,
        device=device,
    ).to(device)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return model


@torch.no_grad()
def sample_tokens(model, scaler: StandardScaler, layout: FeatureLayout, pos: np.ndarray, direction: np.ndarray, target_length: float, n_samples: int, temperature: float, device: torch.device):
    known_values = np.zeros((n_samples, layout.token_dim), dtype=np.float32)
    known_mask = np.zeros((n_samples, layout.token_dim), dtype=np.float32)
    known_values[:, layout.pos_slice] = pos.reshape(1, 3)
    known_values[:, layout.dir_slice] = direction.reshape(1, 3)
    known_values[:, layout.length_slice] = float(target_length)
    known_mask[:, layout.pos_slice] = 1.0
    known_mask[:, layout.dir_slice] = 1.0
    known_mask[:, layout.length_slice] = 1.0

    known_values_norm = scaler.transform_np(known_values)
    samples_norm = model.sample_inpaint(
        known_values=torch.from_numpy(known_values_norm).to(device),
        known_mask=torch.from_numpy(known_mask).to(device),
        n_samples=n_samples,
        temperature=temperature,
        clip_min=torch.as_tensor((scaler.data_min - scaler.mean) / scaler.std, dtype=torch.float32, device=device),
        clip_max=torch.as_tensor((scaler.data_max - scaler.mean) / scaler.std, dtype=torch.float32, device=device),
    )
    return scaler.inverse_transform_np(samples_norm.detach().cpu().numpy())



def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    bundle = torch.load(args.bundle, map_location=device, weights_only=False)
    layout = FeatureLayout(q_dim=int(bundle["layout_q_dim"]))
    scaler = StandardScaler(
        mean=np.asarray(bundle["scaler_mean"], dtype=np.float32),
        std=np.asarray(bundle["scaler_std"], dtype=np.float32),
        data_min=np.asarray(bundle["scaler_min"], dtype=np.float32),
        data_max=np.asarray(bundle["scaler_max"], dtype=np.float32),
    )
    model = build_model(bundle, device)

    pos = np.asarray(args.pos, dtype=np.float32)
    direction = normalize_direction(np.asarray(args.direction, dtype=np.float32))
    samples = sample_tokens(
        model=model,
        scaler=scaler,
        layout=layout,
        pos=pos,
        direction=direction,
        target_length=float(args.target_length),
        n_samples=args.n_samples,
        temperature=float(args.temperature),
        device=device,
    )

    robot = XArmLite6Miller(enable_cc=True)
    correction = JacobianCorrection(robot=robot, damping=args.correction_damping)
    tracker = NullspaceStraightTracker(robot=robot, config=TrackerConfig(max_steps=args.tracker_max_steps))

    corrected_q_list = []
    pred_rot_list = []
    fk_rot_list = []
    length_list = []
    pos_error_list = []

    for idx, token in enumerate(samples):
        q_pred = token[layout.q_slice].astype(np.float32)
        rot_pred = canonicalize_quaternion_xyzw(token[layout.rot_slice])
        q_corr, pos_err = correction.run(q_pred, pos, max_iters=args.correction_iters, tol=args.correction_tol)
        robot.goto_given_conf(q_corr)
        fk_pos, fk_rot = robot.fk(q_corr, update=False)
        fk_rot_q = canonicalize_quaternion_xyzw(rm.rotmat_to_quaternion(fk_rot))
        corrected_q_list.append(q_corr)
        pred_rot_list.append(rot_pred)
        fk_rot_list.append(fk_rot_q)
        pos_error_list.append(pos_err)
        if args.eval_real_length:
            result = tracker.run(q_corr, direction)
            length_list.append(float(result.projected_length))
        print(
            f"[sample {idx}] pos_err={pos_err:.6f} q={np.array2string(q_corr, precision=4, separator=', ')} "
            f"pred_rot_q={np.array2string(rot_pred, precision=4, separator=', ')} "
            f"fk_rot_q={np.array2string(fk_rot_q, precision=4, separator=', ')}"
        )
        if args.eval_real_length:
            print(f"           real_projected_length={length_list[-1]:.6f} m")

    corrected_q_arr = np.stack(corrected_q_list, axis=0)
    pred_rot_arr = np.stack(pred_rot_list, axis=0)
    fk_rot_arr = np.stack(fk_rot_list, axis=0)
    q_diversity = float(np.mean(np.std(corrected_q_arr, axis=0)))
    rot_diversity = float(np.mean(np.std(pred_rot_arr, axis=0)))
    print(f"[summary] q_diversity_mean_std={q_diversity:.6f}")
    print(f"[summary] pred_rot_diversity_mean_std={rot_diversity:.6f}")
    print(f"[summary] mean_pos_error={float(np.mean(pos_error_list)):.6f} m")
    if args.eval_real_length and length_list:
        lengths = np.asarray(length_list, dtype=np.float32)
        print(f"[summary] mean_real_projected_length={float(lengths.mean()):.6f} m")
        print(f"[summary] max_real_projected_length={float(lengths.max()):.6f} m")

    payload = {
        "input_pos": pos.tolist(),
        "input_direction": direction.tolist(),
        "input_target_length": float(args.target_length),
        "corrected_q": corrected_q_arr.tolist(),
        "pred_rot_q": pred_rot_arr.tolist(),
        "fk_rot_q": fk_rot_arr.tolist(),
        "pos_error": [float(v) for v in pos_error_list],
        "real_projected_length": [float(v) for v in length_list],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
