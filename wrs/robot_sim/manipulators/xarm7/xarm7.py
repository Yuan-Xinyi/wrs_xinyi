import os
import math
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim.manipulators.manipulator_interface as mi


class XArm7(mi.ManipulatorInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 home_conf=np.zeros(7),
                 name='xarm7',
                 enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=home_conf, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link_base.stl"), name="xarm7_base")
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.const.tab20_list[15]
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([.0, .0, .267])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-np.pi, np.pi])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link1.stl"), name="xarm7_link1")
        self.jlc.jnts[0].lnk.cmodel.rgba = rm.const.tab20_list[15]
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(-1.5708, 0, 0)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[1].motion_range = np.array([-2.18, 2.18])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link2.stl"), name="xarm7_link2")
        self.jlc.jnts[1].lnk.cmodel.rgba = rm.const.tab20_list[15]
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([.0, -.293, .0])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(1.5708, 0, 0)
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].motion_range = np.array([-np.pi, np.pi])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link3.stl"), name="xarm7_link3")
        self.jlc.jnts[2].lnk.cmodel.rgba = rm.const.tab20_list[15]
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([.0525, .0, .0])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(1.5708, 0, 0)
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[3].motion_range = np.array([-0.11, np.pi])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link4.stl"), name="xarm7_link4")
        self.jlc.jnts[3].lnk.cmodel.rgba = rm.const.tab20_list[15]
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([0.0775, -0.3425, 0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(1.5708, 0, 0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-math.pi, np.pi])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link5.stl"), name="xarm7_link5")
        self.jlc.jnts[4].lnk.cmodel.rgba = rm.const.tab20_list[15]
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(1.5708, 0, 0)
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].motion_range = np.array([-1.75, np.pi])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link6.stl"), name="xarm7_link6")
        self.jlc.jnts[5].lnk.cmodel.rgba = rm.const.tab20_list[15]
        # seventh joint and link
        self.jlc.jnts[6].loc_pos = np.array([0.076, 0.097, 0])
        self.jlc.jnts[6].loc_rotmat = rm.rotmat_from_euler(-1.5708, 0, 0)
        self.jlc.jnts[6].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[6].motion_range = np.array([-math.pi, np.pi])
        self.jlc.jnts[6].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link7.stl"), name="xarm7_link7")
        self.jlc.jnts[6].lnk.cmodel.rgba = rm.const.tab20_list[14]
        self.jlc.finalize(ik_solver='s', identifier_str=name)
        # tcp
        self.loc_tcp_pos = np.array([0, 0, 0])
        self.loc_tcp_rotmat = np.eye(3)
        # set up cc
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        l6 = self.cc.add_cce(self.jlc.jnts[6].lnk)
        from_list = [lb, l0, l1]
        into_list = [l4, l5, l6]
        self.cc.set_cdpair_by_ids(from_list, into_list)


if __name__ == '__main__':
    from wrs import wd, mgm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0.5])
    mgm.gen_frame().attach_to(base)
    arm = XArm7(enable_cc=True)
    # arm.gen_meshmodel(alpha=1).attach_to(base)
    # arm.gen_stickmodel(toggle_jnt_frames=True).attach_to(base)
    # arm.show_cdprim()

    '''random a joint configuration and calculate the target by FK'''
    jnt_values = arm.rand_conf()
    arm.goto_given_conf(jnt_values=jnt_values)
    arm.gen_meshmodel(alpha=0.3, rgb=[0,1,0]).attach_to(base)
    tgt_pos, tgt_rotmat = arm.fk(jnt_values=jnt_values)
    mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

    '''calculate the predicted joint configuration by IK'''
    result = arm.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    if result is not None:
        print('ik result:', result)
        pred_pos, pred_rotmat = arm.fk(jnt_values=result)
        arm.goto_given_conf(jnt_values=result)
        arm.gen_meshmodel(alpha=0.3, rgb=[0,0,1]).attach_to(base)
        mcm.mgm.gen_dashed_frame(pos=pred_pos, rotmat=pred_rotmat).attach_to(base)

    base.run()

    