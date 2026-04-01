import json

import numpy as np
import torch

import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd

from diffusion import DEFAULT_RUN_NAME, DEFAULT_WORKDIR, sample_q_length_from_condition
from diffusion_sample import load_model, normalize_direction
from wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill import XArmLite6Miller

BUNDLE_PATH = DEFAULT_WORKDIR / DEFAULT_RUN_NAME / "bundle_latest.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 64
SAMPLE_STEPS = None
TEMPERATURE = 1.0
SEED = None
DIRECTION_AXIS = 0
NORMAL_AXIS = 2
VIS_DIRECTION_SCALE = 0.18


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normal / max(np.linalg.norm(normal), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-12)
    return np.column_stack((x_axis, y_axis, z_axis))


def sample_target_pose(robot: XArmLite6Miller, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for _ in range(2000):
        q = robot.rand_conf().astype(np.float32)
        robot.goto_given_conf(q)
        if robot.is_collided():
            continue
        pos, rotmat = robot.fk(q, update=False)
        return q, np.asarray(pos, dtype=np.float32), np.asarray(rotmat, dtype=np.float32)
    raise RuntimeError("Failed to sample a valid target pose.")


def main() -> None:
    rng = np.random.default_rng(SEED if SEED is not None else int(np.random.SeedSequence().entropy))
    device = torch.device(DEVICE)
    _, stats, model, q_dim, diffusion_steps = load_model(BUNDLE_PATH, device)

    gt_robot = XArmLite6Miller(enable_cc=True)
    q_gt, pos, rotmat = sample_target_pose(gt_robot, rng)
    direction = normalize_direction(rotmat[:, DIRECTION_AXIS])
    normal = normalize_direction(rotmat[:, NORMAL_AXIS])
    condition = np.concatenate([pos, direction, normal], axis=0).astype(np.float32)

    q_preds, pred_lengths, _ = sample_q_length_from_condition(
        model=model,
        stats=stats,
        condition=condition,
        device=device,
        q_dim=q_dim,
        n_samples=int(N_SAMPLES),
        sample_steps=int(SAMPLE_STEPS) if SAMPLE_STEPS is not None else int(diffusion_steps),
        temperature=float(TEMPERATURE),
    )

    eval_robot = XArmLite6Miller(enable_cc=True)
    raw_pos_errs = []
    for q_pred in q_preds.astype(np.float32):
        q_pred64 = q_pred.astype(np.float64)
        eval_robot.goto_given_conf(q_pred64)
        cur_pos, _ = eval_robot.fk(q_pred64, update=False)
        raw_pos_errs.append(float(np.linalg.norm(pos - cur_pos)))
    raw_pos_errs = np.asarray(raw_pos_errs, dtype=np.float32)
    best_idx = int(np.argmin(raw_pos_errs))

    result = {
        "q_gt": q_gt.tolist(),
        "pos": pos.tolist(),
        "rotmat": rotmat.tolist(),
        "direction": direction.tolist(),
        "normal": normal.tolist(),
        "best_idx": best_idx,
        "q_pred": q_preds[best_idx].tolist(),
        "pred_length": float(pred_lengths[best_idx]),
        "raw_pos_err_mm": float(raw_pos_errs[best_idx] * 1e3),
    }
    print(json.dumps(result, indent=2))

    world = wd.World(cam_pos=[1.7, -1.5, 1.05], lookat_pos=[0.25, 0.0, 0.25])
    mgm.gen_frame().attach_to(world)
    mgm.gen_frame(pos=pos, rotmat=rotmat, ax_length=0.12).attach_to(world)
    mgm.gen_arrow(spos=pos, epos=pos + direction * VIS_DIRECTION_SCALE, rgb=np.array([1.0, 0.1, 0.1]), alpha=0.9).attach_to(world)
    mgm.gen_arrow(spos=pos, epos=pos + normal * VIS_DIRECTION_SCALE, rgb=np.array([0.1, 0.5, 1.0]), alpha=0.9).attach_to(world)
    mgm.gen_sphere(pos, radius=0.010, rgb=np.array([0.0, 0.7, 1.0]), alpha=1.0).attach_to(world)

    plane_size = 1.2
    plane_rotmat = rotation_matrix_from_normal(normal)
    plane_center = pos + 0.5 * plane_size * direction
    mcm.gen_box(
        xyz_lengths=[plane_size, plane_size, 0.001],
        pos=plane_center,
        rotmat=plane_rotmat,
        rgb=[180 / 255, 211 / 255, 217 / 255],
        alpha=0.5,
    ).attach_to(world)

    gt_color = np.array([0.1, 0.75, 0.2], dtype=np.float32)
    pred_color = np.array([0.15, 0.45, 0.95], dtype=np.float32)

    gt_robot.goto_given_conf(q_gt)
    gt_robot.gen_meshmodel(rgb=gt_color, alpha=0.55, toggle_tcp_frame=True).attach_to(world)

    eval_robot.goto_given_conf(q_preds[best_idx].astype(np.float32))
    eval_robot.gen_meshmodel(rgb=pred_color, alpha=0.60, toggle_tcp_frame=True).attach_to(world)

    gt_end = pos + direction * VIS_DIRECTION_SCALE
    pred_end = pos + direction * float(pred_lengths[best_idx])
    mgm.gen_stick(spos=pos, epos=gt_end, radius=0.0045, rgb=gt_color, alpha=0.90).attach_to(world)
    mgm.gen_stick(spos=pos, epos=pred_end, radius=0.0045, rgb=pred_color, alpha=0.95).attach_to(world)
    mgm.gen_sphere(pred_end, radius=0.009, rgb=pred_color, alpha=0.95).attach_to(world)

    world.run()


if __name__ == "__main__":
    main()
