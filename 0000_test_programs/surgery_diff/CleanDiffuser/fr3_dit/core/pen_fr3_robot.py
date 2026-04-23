from __future__ import annotations

import numpy as np
import torch

import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.neuro._kinematics.jlchain as jlc
from wrs.robot_sim.robots.franka_research_3.franka_research_3 import FrankaResearch3


PEN_LENGTH = 0.10
# In the WRS articulated robot, the manipulator flange already includes the arm's
# 0.107 m flange offset. The hand acting center is 0.1034 m from that flange.
ORIGINAL_TCP_OFFSET = np.array([0.0, 0.0, 0.1034], dtype=np.float32)


class PenFrankaResearch3GPU:
    def __init__(self, device: torch.device):
        self.device = device
        self.robot = jlc.JLChain(n_dof=7, pos=torch.zeros(3, device=device), rotmat=torch.eye(3, device=device))
        self._build_robot()
        self.robot.finalize()

    def _build_robot(self) -> None:
        r = self.robot
        d = self.device

        r.jnts[0].loc_pos = torch.tensor([0.0, 0.0, 0.333], dtype=torch.float32, device=d)
        r.jnts[0].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[0].motion_range = torch.tensor([-2.8973, 2.8973], dtype=torch.float32, device=d)

        r.jnts[1].loc_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=d)
        r.jnts[1].loc_rotmat = torch.tensor(rm.rotmat_from_euler(-np.pi / 2, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[1].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[1].motion_range = torch.tensor([-1.8326, 1.8326], dtype=torch.float32, device=d)

        r.jnts[2].loc_pos = torch.tensor([0.0, -0.316, 0.0], dtype=torch.float32, device=d)
        r.jnts[2].loc_rotmat = torch.tensor(rm.rotmat_from_euler(np.pi / 2, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[2].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[2].motion_range = torch.tensor([-2.8972, 2.8972], dtype=torch.float32, device=d)

        r.jnts[3].loc_pos = torch.tensor([0.0825, 0.0, 0.0], dtype=torch.float32, device=d)
        r.jnts[3].loc_rotmat = torch.tensor(rm.rotmat_from_euler(np.pi / 2, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[3].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[3].motion_range = torch.tensor([-3.0718, -0.1222], dtype=torch.float32, device=d)

        r.jnts[4].loc_pos = torch.tensor([-0.0825, 0.384, 0.0], dtype=torch.float32, device=d)
        r.jnts[4].loc_rotmat = torch.tensor(rm.rotmat_from_euler(-np.pi / 2, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[4].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[4].motion_range = torch.tensor([-2.8798, 2.8798], dtype=torch.float32, device=d)

        r.jnts[5].loc_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=d)
        r.jnts[5].loc_rotmat = torch.tensor(rm.rotmat_from_euler(np.pi / 2, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[5].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[5].motion_range = torch.tensor([0.4364, 4.6251], dtype=torch.float32, device=d)

        r.jnts[6].loc_pos = torch.tensor([0.088, 0.0, 0.0], dtype=torch.float32, device=d)
        r.jnts[6].loc_rotmat = torch.tensor(rm.rotmat_from_euler(np.pi / 2, 0.0, 0.0), dtype=torch.float32, device=d)
        r.jnts[6].loc_motion_ax = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=d)
        r.jnts[6].motion_range = torch.tensor([-3.0543, 3.0543], dtype=torch.float32, device=d)

        r._loc_flange_pos = torch.tensor([0.0, 0.0, 0.2104 + PEN_LENGTH], dtype=torch.float32, device=d)


class PenFrankaResearch3(FrankaResearch3):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="pen_franka_research_3", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos + np.array([0.0, 0.0, PEN_LENGTH], dtype=np.float32)
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat

    def _pen_segment(self) -> tuple[np.ndarray, np.ndarray]:
        flange_pos = self.manipulator.gl_flange_pos
        flange_rotmat = self.manipulator.gl_flange_rotmat
        pen_start = flange_pos + flange_rotmat @ ORIGINAL_TCP_OFFSET
        pen_tip = flange_pos + flange_rotmat @ (ORIGINAL_TCP_OFFSET + np.array([0.0, 0.0, PEN_LENGTH], dtype=np.float32))
        return pen_start, pen_tip

    def gen_meshmodel(self, rgb=None, alpha=None, toggle_tcp_frame=False, toggle_jnt_frames=False, toggle_cdprim=False, toggle_cdmesh=False):
        model = super().gen_meshmodel(
            rgb=rgb,
            alpha=alpha,
            toggle_tcp_frame=toggle_tcp_frame,
            toggle_jnt_frames=toggle_jnt_frames,
            toggle_cdprim=toggle_cdprim,
            toggle_cdmesh=toggle_cdmesh,
        )
        pen_start, pen_tip = self._pen_segment()
        pen_rgb = np.array([0.15, 0.15, 0.15]) if rgb is None else np.asarray(rgb[:3], dtype=np.float32)
        pen_alpha = 0.95 if alpha is None else float(alpha)
        mgm.gen_stick(spos=pen_start, epos=pen_tip, radius=0.006, rgb=pen_rgb, alpha=pen_alpha).attach_to(model)
        mgm.gen_sphere(pos=pen_tip, radius=0.0065, rgb=pen_rgb, alpha=pen_alpha).attach_to(model)
        return model

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False):
        model = super().gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame, toggle_jnt_frames=toggle_jnt_frames)
        pen_start, pen_tip = self._pen_segment()
        mgm.gen_stick(spos=pen_start, epos=pen_tip, radius=0.0045, rgb=np.array([0.15, 0.15, 0.15]), alpha=0.95).attach_to(model)
        mgm.gen_sphere(pos=pen_tip, radius=0.0055, rgb=np.array([0.15, 0.15, 0.15]), alpha=0.95).attach_to(model)
        return model


if __name__ == "__main__":
    import wrs.visualization.panda.world as wd

    world = wd.World(cam_pos=[2.0, -1.8, 1.2], lookat_pos=[0.2, 0.0, 0.4])
    mgm.gen_frame().attach_to(world)

    robot = PenFrankaResearch3(name="pen", enable_cc=True)
    robot.gen_meshmodel(alpha=0.6, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(world)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(world)
    mgm.gen_frame(pos=robot.manipulator.gl_tcp_pos, rotmat=robot.manipulator.gl_tcp_rotmat, ax_length=0.08).attach_to(world)

    print(f"[pen-fr3] pen_length = {PEN_LENGTH:.3f} m")
    print(f"[pen-fr3] tcp_pos    = {np.array2string(robot.manipulator.gl_tcp_pos, precision=4, suppress_small=True)}")
    world.run()
