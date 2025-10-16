import os
import math
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface_neuro as mi
import wrs.neuro._kinematics.jlchain as jlc
import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import torch
import wrs.visualization.panda.world as wd
import wrs.basis.constant as bc

base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

XArmLite6 = jlc.JLChain(n_dof=6)
# anchor
XArmLite6.jnts[0].loc_pos = torch.tensor([.0, .0, .2433], dtype=torch.float32, device=device)
XArmLite6.jnts[0].loc_motion_ax = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
XArmLite6.jnts[0].motion_range = torch.tensor([-math.pi, math.pi], dtype=torch.float32, device=device)
# first joint and link
XArmLite6.jnts[1].loc_pos = torch.tensor([.0, .0, .0], dtype=torch.float32, device=device)
XArmLite6.jnts[1].loc_rotmat = torch.tensor(rm.rotmat_from_euler(1.5708, -1.5708, 3.1416),
                                            dtype=torch.float32, device=device)
XArmLite6.jnts[1].loc_motion_ax = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
XArmLite6.jnts[1].motion_range = torch.tensor([-2.61799, 2.61799], dtype=torch.float32, device=device)
# second joint and link
XArmLite6.jnts[2].loc_pos = torch.tensor([.2, .0, .0], dtype=torch.float32, device=device)
XArmLite6.jnts[2].loc_rotmat = torch.tensor(rm.rotmat_from_euler(-3.1416, 0., 1.5708),
                                            dtype=torch.float32, device=device)
XArmLite6.jnts[2].loc_motion_ax = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
XArmLite6.jnts[2].motion_range = torch.tensor([-0.061087, 5.235988], dtype=torch.float32, device=device)
# fourth joint and link
XArmLite6.jnts[3].loc_pos = torch.tensor([.087, -.2276, .0], dtype=torch.float32, device=device)
XArmLite6.jnts[3].loc_rotmat = torch.tensor(rm.rotmat_from_euler(1.5708, 0., 0.),
                                            dtype=torch.float32, device=device)
XArmLite6.jnts[3].loc_motion_ax = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
XArmLite6.jnts[3].motion_range = torch.tensor([-math.pi, math.pi], dtype=torch.float32, device=device)
# fifth joint and link
XArmLite6.jnts[4].loc_pos = torch.tensor([.0, .0, .0], dtype=torch.float32, device=device)
XArmLite6.jnts[4].loc_rotmat = torch.tensor(rm.rotmat_from_euler(1.5708, 0., 0.),
                                            dtype=torch.float32, device=device)
XArmLite6.jnts[4].loc_motion_ax = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
XArmLite6.jnts[4].motion_range = torch.tensor([-2.1642, 2.1642], dtype=torch.float32, device=device)
# sixth joint and link
XArmLite6.jnts[5].loc_pos = torch.tensor([.0, .0615, .0], dtype=torch.float32, device=device)
XArmLite6.jnts[5].loc_rotmat = torch.tensor(rm.rotmat_from_euler(-1.5708, 0., 0.),
                                            dtype=torch.float32, device=device)
XArmLite6.jnts[5].loc_motion_ax = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
XArmLite6.jnts[5].motion_range = torch.tensor([-math.pi, math.pi], dtype=torch.float32, device=device)
# finalizing
XArmLite6._loc_flange_pos = torch.tensor([0.0, 0.0, 0.01], device=device)
XArmLite6._loc_flange_pos = torch.tensor([0.1, 0.1, 0.1], device=device)
result = XArmLite6.finalize()
XArmLite6.gen_stickmodel(stick_rgba=bc.navy_blue, toggle_jnt_frames=True, toggle_flange_frame=True).attach_to(base)
base.run()


def ik(self,
        tgt_pos: torch.tensor,
        tgt_rotmat: torch.tensor,
        seed_jnt_values=None,
        option="single",
        toggle_dbg=False):
    """
    :param tgt_pos:
    :param tgt_rotmat:
    :param seed_jnt_values:
    :return:
    """
    tcp_loc_pos = self.loc_tcp_pos
    tcp_loc_rotmat = self.loc_tcp_rotmat
    tgt_flange_rotmat = tgt_rotmat @ tcp_loc_rotmat.T
    tgt_flange_pos = tgt_pos - tgt_flange_rotmat @ tcp_loc_pos
    rrr_pos = tgt_flange_pos - tgt_flange_rotmat[:, 2] * torch.linalg.norm(self.jlc.jnts[5].loc_pos)
    rrr_x, rrr_y, rrr_z = ((rrr_pos - self.pos) @ self.rotmat).tolist()  # in base coordinate system
    j0_value = torch.pi / 2 - math.atan2(rrr_x, rrr_y)
    if not self._is_jnt_in_range(jnt_id=0, jnt_value=j0_value):
        return None
    # assume a, b, c are the axis_length of shoulders and bottom of the big triangle formed by the robot arm
    c = math.sqrt(rrr_x ** 2 + rrr_y ** 2 + (rrr_z - self.jlc.jnts[0].loc_pos[2]) ** 2)
    a = self.jlc.jnts[2].loc_pos[0]
    b = torch.linalg.norm(self.jlc.jnts[3].loc_pos)
    tmp_acos_target = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    if tmp_acos_target > 1 or tmp_acos_target < -1:
        print("Analytical IK Failure: The triangle formed by the robot arm is violated!")
        return None
    j2_value = math.acos(tmp_acos_target)
    j2_initial_offset = math.atan(abs(self.jlc.jnts[3].loc_pos[0] / self.jlc.jnts[3].loc_pos[1]))
    j2_value = j2_value - j2_initial_offset
    if not self._is_jnt_in_range(jnt_id=2, jnt_value=j2_value):
        # ignore reversed elbow
        # j2_value = math.acos(tmp_acos_target) - math.pi
        # if not self._is_jnt_in_range(jnt_id=2, jnt_value=j2_value):
        return None
    tmp_acos_target = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    if tmp_acos_target > 1 or tmp_acos_target < -1:
        print("Analytical IK Failure: The triangle formed by the robot arm is violated!")
        return None
    j1_value_upper = math.acos(tmp_acos_target)
    # assume d, c, e are the edges of the lower triangle formed with the ground
    d = self.jlc.jnts[0].loc_pos[2]
    e = math.sqrt(rrr_x ** 2 + rrr_y ** 2 + rrr_z ** 2)
    tmp_acos_target = (d ** 2 + c ** 2 - e ** 2) / (2 * d * c)
    if tmp_acos_target > 1 or tmp_acos_target < -1:
        print("Analytical IK Failure: The triangle formed with the ground is violated!")
        return None
    j1_value_lower = math.acos(tmp_acos_target)
    j1_value = math.pi - (j1_value_lower + j1_value_upper)
    if not self._is_jnt_in_range(jnt_id=1, jnt_value=j1_value):
        return None
    # RRR
    anchor_gl_rotmatq = self.rotmat
    j0_gl_rotmat0 = anchor_gl_rotmatq @ self.jlc.jnts[0].loc_rotmat
    j0_gl_rotmatq = j0_gl_rotmat0 @ rm.rotmat_from_axangle(self.jlc.jnts[0].loc_motion_ax, j0_value)
    j1_gl_rotmat0 = j0_gl_rotmatq @ self.jlc.jnts[1].loc_rotmat
    j1_gl_rotmatq = j1_gl_rotmat0 @ rm.rotmat_from_axangle(self.jlc.jnts[1].loc_motion_ax, j1_value)
    j2_gl_rotmat0 = j1_gl_rotmatq @ self.jlc.jnts[2].loc_rotmat
    j2_gl_rotmatq = j2_gl_rotmat0 @ rm.rotmat_from_axangle(self.jlc.jnts[2].loc_motion_ax, j2_value)
    rrr_g_rotmat = (j2_gl_rotmatq @ self.jlc.jnts[3].loc_rotmat @
                    self.jlc.jnts[4].loc_rotmat @ self.jlc.jnts[5].loc_rotmat)
    j3_value, j4_value, j5_value = rm.rotmat_to_euler(rrr_g_rotmat.T @ tgt_flange_rotmat, order='rzyz').tolist()
    j4_value = -j4_value
    # print(j3_value, j4_value, j5_value)
    # if not (self._is_jnt_in_range(jnt_id=3, jnt_value=j3_value) and
    #         self._is_jnt_in_range(jnt_id=4, jnt_value=j4_value) and
    #         self._is_jnt_in_range(jnt_id=5, jnt_value=j5_value)):
    #     return None
    return torch.tensor([j0_value, j1_value, j2_value, j3_value, j4_value, j5_value])
