import os
import math
import copy
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.robots.robot_interface as ri
import wrs.modeling.model_collection as mmc
import wrs.robot_sim.robots.yumi.yumi_single_arm as ysa
import wrs.robot_sim._kinematics.collision_checker as cc


class Yumi(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='yumi', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # the body anchor
        self.body = rkjlc.rkjl.Anchor(name="yumi_body", pos=self.pos, rotmat=self.rotmat, n_flange=2, n_lnk=9)
        self.body.loc_flange_pose_list[0] = [np.array([0.05355, 0.07250, 0.41492]),
                                             (rm.rotmat_from_euler(0.9781, -0.5716, 2.3180) @
                                              rm.rotmat_from_euler(0.0, 0.0, -np.pi))]
        self.body.loc_flange_pose_list[1] = [np.array([0.05355, -0.07250, 0.41492]),
                                             (rm.rotmat_from_euler(-0.9781, -0.5682, -2.3155) @
                                              rm.rotmat_from_euler(0.0, 0.0, -np.pi))]
        self.body.lnk_list[0].name = "yumi_body_main"
        self.body.lnk_list[0].cmodel = mcm.CollisionModel(initor=os.path.join(current_file_dir, "meshes", "body.stl"),
                                                          cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
                                                          userdef_cdprim_fn=self._base_cdprim)
        self.body.lnk_list[0].cmodel.rgba = rm.const.hug_gray
        # table
        self.body.lnk_list[1].name = "yumi_body_table_top"
        self.body.lnk_list[1].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "yumi_tablenotop.stl"), name="yumi_body_table_top")
        self.body.lnk_list[1].cmodel.rgba = rm.const.steel_gray
        # lft column
        self.body.lnk_list[2].name = "yumi_body_lft_column"
        self.body.lnk_list[2].loc_pos = np.array([-.327, -.24, -1.015])
        self.body.lnk_list[2].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "yumi_column60602100.stl"), name="yumi_body_lft_column")
        self.body.lnk_list[2].cmodel.rgba = rm.const.steel_gray
        # rgt column
        self.body.lnk_list[3].name = "yumi_body_rgt_column"
        self.body.lnk_list[3].loc_pos = np.array([-.327, .24, -1.015])
        self.body.lnk_list[3].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "yumi_column60602100.stl"), name="yumi_body_rgt_column")
        self.body.lnk_list[3].cmodel.rgba = rm.const.steel_gray
        # top back column
        self.body.lnk_list[4].name = "yumi_body_top_back_column"
        self.body.lnk_list[4].loc_pos = np.array([-.327, 0, 1.085])
        self.body.lnk_list[4].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "yumi_column6060540.stl"), name="yumi_body_top_back_column")
        self.body.lnk_list[4].cmodel.rgba = rm.const.steel_gray
        # top lft column
        self.body.lnk_list[5].name = "yumi_body_top_lft_column"
        self.body.lnk_list[5].loc_pos = np.array([-.027, -.24, 1.085])
        self.body.lnk_list[5].loc_rotmat = rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
        self.body.lnk_list[5].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "yumi_column6060540.stl"), name="yumi_body_top_lft_column")
        self.body.lnk_list[5].cmodel.rgba = rm.const.steel_gray
        # top rgt column
        self.body.lnk_list[6].name = "yumi_body_top_rgt_column"
        self.body.lnk_list[6].loc_pos = np.array([-.027, .24, 1.085])
        self.body.lnk_list[6].loc_rotmat = rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
        self.body.lnk_list[6].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "yumi_column6060540.stl"), name="yumi_body_top_rgt_column")
        self.body.lnk_list[6].cmodel.rgba = rm.const.steel_gray
        # top front column
        self.body.lnk_list[7].name = "yumi_body_top_front_column"
        self.body.lnk_list[7].loc_pos = np.array([.273, 0, 1.085])
        self.body.lnk_list[7].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "yumi_column6060540.stl"),
            name="yumi_body_top_front_column")
        self.body.lnk_list[7].cmodel.rgba = rm.const.steel_gray
        # phoxi
        self.body.lnk_list[8].name = "phoxi"
        self.body.lnk_list[8].loc_pos = np.array([.273, 0, 1.085])
        self.body.lnk_list[8].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "phoxi_m.stl"), name="yumi_phoxi")
        self.body.lnk_list[8].cmodel.rgba = rm.const.black
        # left arm
        self.lft_arm = ysa.YumiSglArm(pos=self.body.gl_flange_pose_list[0][0],
                                      rotmat=self.body.gl_flange_pose_list[0][1],
                                      name='yumi_lft_arm', enable_cc=False)
        self.lft_arm.home_conf = np.radians(np.array([20, -90, 120, 30, 0, 40, 0]))
        # right arm
        self.rgt_arm = ysa.YumiSglArm(pos=self.body.gl_flange_pose_list[1][0],
                                      rotmat=self.body.gl_flange_pose_list[1][1],
                                      name='yumi_rgt_arm', enable_cc=False)
        self.rgt_arm.home_conf = np.radians(np.array([-20, -90, -120, 30, .0, 40, 0]))
        if self.cc is not None:
            self.setup_cc()
        # go home
        self.goto_home_conf()

    @staticmethod
    def _base_cdprim(name="auto", ex_radius=None):
        pdcnd = CollisionNode(name + "_cnode")
        collision_primitive_c0 = CollisionBox(Point3(-.2, 0, 0.04),
                                              x=.16 + ex_radius, y=.2 + ex_radius, z=.04 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(-.24, 0, 0.24),
                                              x=.12 + ex_radius, y=.125 + ex_radius, z=.24 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(-.07, 0, 0.4),
                                              x=.075 + ex_radius, y=.125 + ex_radius, z=.06 + ex_radius)
        pdcnd.addSolid(collision_primitive_c2)
        collision_primitive_l0 = CollisionBox(Point3(0, 0.145, 0.03),
                                              x=.135 + ex_radius, y=.055 + ex_radius, z=.03 + ex_radius)
        pdcnd.addSolid(collision_primitive_l0)
        collision_primitive_r0 = CollisionBox(Point3(0, -0.145, 0.03),
                                              x=.135 + ex_radius, y=.055 + ex_radius, z=.03 + ex_radius)
        pdcnd.addSolid(collision_primitive_r0)
        cdprim = NodePath(name + "_cdprim")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    @property
    def n_dof(self):
        if self.delegator is None:
            return self.lft_arm.n_dof + self.rgt_arm.n_dof
        else:
            return self.delegator.n_dof

    def _enable_lft_cc(self):
        self.lft_arm.cc = cc.CollisionChecker("lft_arm_collision_checker")
        # body
        bd = self.lft_arm.cc.add_cce(self.body.lnk_list[0])
        tbl = self.lft_arm.cc.add_cce(self.body.lnk_list[1])
        lc = self.lft_arm.cc.add_cce(self.body.lnk_list[2])
        rc = self.lft_arm.cc.add_cce(self.body.lnk_list[3])
        tbc = self.lft_arm.cc.add_cce(self.body.lnk_list[4])
        tlc = self.lft_arm.cc.add_cce(self.body.lnk_list[5])
        trc = self.lft_arm.cc.add_cce(self.body.lnk_list[6])
        tfc = self.lft_arm.cc.add_cce(self.body.lnk_list[7])
        phx = self.lft_arm.cc.add_cce(self.body.lnk_list[8])
        # left ee
        lft_ee_cces = []
        for id, cdlnk in enumerate(self.lft_arm.end_effector.cdelements):
            lft_ee_cces.append(self.lft_arm.cc.add_cce(cdlnk))
        # left manipulator
        lft_ml0 = self.lft_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[0].lnk)
        lft_ml1 = self.lft_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[1].lnk)
        lft_ml2 = self.lft_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[2].lnk)
        lft_ml3 = self.lft_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[3].lnk)
        lft_ml4 = self.lft_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[4].lnk)
        lft_ml5 = self.lft_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[5].lnk)
        # right ee
        rgt_ee_cces = []
        for id, cdlnk in enumerate(self.rgt_arm.end_effector.cdelements):
            rgt_ee_cces.append(self.lft_arm.cc.add_cce(cdlnk))
        # right manipulator
        rgt_ml0 = self.lft_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[0].lnk)
        rgt_ml1 = self.lft_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[1].lnk)
        rgt_ml2 = self.lft_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[2].lnk)
        rgt_ml3 = self.lft_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[3].lnk)
        rgt_ml4 = self.lft_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[4].lnk)
        rgt_ml5 = self.lft_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[5].lnk)
        # first pairs
        from_list = [lft_ml4, lft_ml5] + lft_ee_cces + [rgt_ml4, rgt_ml5] + rgt_ee_cces
        into_list = [bd, tbl, lc, rc, tbc, tlc, trc, tfc, phx, lft_ml0, rgt_ml0]
        self.lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs
        from_list = lft_ee_cces + rgt_ee_cces
        into_list = [lft_ml0, lft_ml1, rgt_ml0, rgt_ml1]
        self.lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs
        from_list = [lft_ml1, lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces
        into_list = [rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces
        self.lft_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self.lft_arm.cc.enable_extcd_by_id_list(
            id_list=[lft_ml1, lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces, type="from")
        self.lft_arm.cc.enable_innercd_by_id_list(
            id_list=[bd, tbl, lc, rc, tbc, tlc, trc, tfc, phx, lft_ml1, lft_ml2, lft_ml3, rgt_ml1, rgt_ml2, rgt_ml3,
                     rgt_ml4] + rgt_ee_cces, type="into")
        self.lft_arm.cc.dynamic_into_list = [tbl]
        self.lft_arm.cc.dynamic_ext_list = lft_ee_cces[1:]

    def _enable_rgt_cc(self):
        self.rgt_arm.cc = cc.CollisionChecker("rgt_arm_collision_checker")
        # body
        bd = self.rgt_arm.cc.add_cce(self.body.lnk_list[0])
        tbl = self.rgt_arm.cc.add_cce(self.body.lnk_list[1])
        lc = self.rgt_arm.cc.add_cce(self.body.lnk_list[2])
        rc = self.rgt_arm.cc.add_cce(self.body.lnk_list[3])
        tbc = self.rgt_arm.cc.add_cce(self.body.lnk_list[4])
        tlc = self.rgt_arm.cc.add_cce(self.body.lnk_list[5])
        trc = self.rgt_arm.cc.add_cce(self.body.lnk_list[6])
        tfc = self.rgt_arm.cc.add_cce(self.body.lnk_list[7])
        phx = self.rgt_arm.cc.add_cce(self.body.lnk_list[8])
        # left ee
        lft_ee_cces = []
        for id, cdlnk in enumerate(self.lft_arm.end_effector.cdelements):
            lft_ee_cces.append(self.rgt_arm.cc.add_cce(cdlnk))
        # left manipulator
        lft_ml0 = self.rgt_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[0].lnk)
        lft_ml1 = self.rgt_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[1].lnk)
        lft_ml2 = self.rgt_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[2].lnk)
        lft_ml3 = self.rgt_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[3].lnk)
        lft_ml4 = self.rgt_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[4].lnk)
        lft_ml5 = self.rgt_arm.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[5].lnk)
        # right ee
        rgt_ee_cces = []
        for id, cdlnk in enumerate(self.rgt_arm.end_effector.cdelements):
            rgt_ee_cces.append(self.rgt_arm.cc.add_cce(cdlnk))
        # right manipulator
        rgt_ml0 = self.rgt_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[0].lnk)
        rgt_ml1 = self.rgt_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[1].lnk)
        rgt_ml2 = self.rgt_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[2].lnk)
        rgt_ml3 = self.rgt_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[3].lnk)
        rgt_ml4 = self.rgt_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[4].lnk)
        rgt_ml5 = self.rgt_arm.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[5].lnk)
        # first pairs
        from_list = [lft_ml4, lft_ml5] + lft_ee_cces + [rgt_ml4, rgt_ml5] + rgt_ee_cces
        into_list = [bd, tbl, lc, rc, tbc, tlc, trc, tfc, phx, lft_ml0, rgt_ml0]
        self.rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # second pairs
        from_list = lft_ee_cces + rgt_ee_cces
        into_list = [lft_ml0, lft_ml1, rgt_ml0, rgt_ml1]
        self.rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # third pairs
        from_list = [lft_ml1, lft_ml2, lft_ml3, lft_ml4, lft_ml5] + lft_ee_cces
        into_list = [rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces
        self.rgt_arm.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self.rgt_arm.cc.enable_extcd_by_id_list(
            id_list=[rgt_ml1, rgt_ml2, rgt_ml3, rgt_ml4, rgt_ml5] + rgt_ee_cces, type="from")
        self.rgt_arm.cc.enable_innercd_by_id_list(
            id_list=[bd, tbl, lc, rc, tbc, tlc, trc, tfc, phx, rgt_ml1, rgt_ml2, rgt_ml3, lft_ml1, lft_ml2, lft_ml3,
                     lft_ml4] + lft_ee_cces, type="into")
        self.rgt_arm.cc.dynamic_into_list = [tbl]
        self.rgt_arm.cc.dynamic_ext_list = rgt_ee_cces[1:]

    def setup_cc(self):
        """
        author: weiwei
        date: 20241030
        """
        # dual arm
        self._enable_lft_cc()
        self._enable_rgt_cc()
        if self.delegator is not None:
            self.cc = self.delegator.cc
        else:
            self.cc = self.lft_arm.cc  # set left to default

    def use_both(self):
        self.delegator = None
        self.cc = self.lft_arm.cc

    def use_lft(self):
        self.delegator = self.lft_arm
        self.cc = self.delegator.cc

    def use_rgt(self):
        self.delegator = self.rgt_arm
        self.cc = self.delegator.cc

    def backup_state(self):
        if self.delegator is None:
            self.rgt_arm.backup_state()
            self.lft_arm.backup_state()
        else:
            self.delegator.backup_state()

    def restore_state(self):
        if self.delegator is None:
            self.rgt_arm.restore_state()
            self.lft_arm.restore_state()
        else:
            self.delegator.restore_state()

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.body.pos = self.pos
        self.body.rotmat = self.rotmat
        self.lft_arm.fix_to(pos=self.body.gl_flange_pose_list[0][0],
                            rotmat=self.body.gl_flange_pose_list[0][1])
        self.rgt_arm.fix_to(pos=self.body.gl_flange_pose_list[1][0],
                            rotmat=self.body.gl_flange_pose_list[1][1])

    def fk(self, jnt_values, toggle_jacobian=False):
        if self.delegator is None:
            raise AttributeError("FK is not available in multi-arm mode.")
        else:
            return self.delegator.fk(jnt_values=jnt_values, toggle_jacobian=toggle_jacobian)

    def goto_given_conf(self, jnt_values, ee_values=None):
        """
        :param jnt_values: nparray 1x14, 0:7lft, 7:14rgt
        :return:
        author: weiwei
        date: 20240307
        """
        if self.delegator is None:
            if len(jnt_values) != self.lft_arm.manipulator.n_dof + self.rgt_arm.manipulator.n_dof:
                raise ValueError("The given joint values do not match total n_dof")
            self.lft_arm.goto_given_conf(jnt_values=jnt_values[:self.lft_arm.manipulator.n_dof])
            self.rgt_arm.goto_given_conf(jnt_values=jnt_values[self.rgt_arm.manipulator.n_dof:])  # TODO
        else:
            self.delegator.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)

    def goto_home_conf(self):
        if self.delegator is None:
            self.lft_arm.goto_home_conf()
            self.rgt_arm.goto_home_conf()
        else:
            self.delegator.goto_home_conf()

    def get_jnt_values(self):
        if self.delegator is None:
            return np.concatenate((self.lft_arm.get_jnt_values(), self.rgt_arm.get_jnt_values()))
        else:
            return self.delegator.get_jnt_values()

    def rand_conf(self):
        """
        :return:
        author: weiwei
        date: 20210406
        """
        if self.delegator is None:
            return np.concatenate((self.lft_arm.rand_conf(), self.rgt_arm.rand_conf()))
        else:
            return self.delegator.rand_conf()

    def are_jnts_in_ranges(self, jnt_values):
        if self.delegator is None:
            return self.lft_arm.are_jnts_in_ranges(
                jnt_values=jnt_values[:self.lft_arm.manipulator.n_dof]) and self.rgt_arm.are_jnts_in_ranges(
                jnt_values=jnt_values[self.rgt_arm.manipulator.n_dof:])
        else:
            return self.delegator.are_jnts_in_ranges(jnt_values=jnt_values)

    def get_jaw_width(self):
        return self.get_ee_values()

    def change_jaw_width(self, jaw_width):
        self.change_ee_values(ee_values=jaw_width)

    def is_collided(self, obstacle_list=None, other_robot_list=None, toggle_contacts=False, toggle_dbg=False):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param other_robot_list:
        :param toggle_contacts: debug
        :param toggle_dbg: debug
        :return: see CollisionChecker is_collided for details
        author: weiwei
        date: 20240307
        """
        collision_info = self.cc.is_collided(obstacle_list=obstacle_list,
                                             other_robot_list=other_robot_list,
                                             toggle_contacts=toggle_contacts,
                                             toggle_dbg=toggle_dbg)
        return collision_info

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False):
        m_col = mmc.ModelCollection(name=self.name + "_stickmodel")
        self.body.gen_stickmodel(name=self.name + "_body_stickmodel",
                                 toggle_root_frame=toggle_jnt_frames,
                                 toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        self.lft_arm.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                    toggle_jnt_frames=toggle_jnt_frames,
                                    toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        self.rgt_arm.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                    toggle_jnt_frames=toggle_jnt_frames,
                                    toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False):
        m_col = mmc.ModelCollection(name=self.name + "_meshmodel")
        self.body.gen_meshmodel(rgb=rgb, alpha=alpha, toggle_flange_frame=toggle_flange_frame,
                                toggle_root_frame=toggle_jnt_frames, toggle_cdprim=toggle_cdprim,
                                toggle_cdmesh=toggle_cdmesh, name=self.name + "_body_meshmodel").attach_to(m_col)
        self.lft_arm.gen_meshmodel(rgb=rgb,
                                   alpha=alpha,
                                   toggle_tcp_frame=toggle_tcp_frame,
                                   toggle_jnt_frames=toggle_jnt_frames,
                                   toggle_flange_frame=toggle_flange_frame,
                                   toggle_cdprim=toggle_cdprim,
                                   toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        self.rgt_arm.gen_meshmodel(rgb=rgb,
                                   alpha=alpha,
                                   toggle_tcp_frame=toggle_tcp_frame,
                                   toggle_jnt_frames=toggle_jnt_frames,
                                   toggle_flange_frame=toggle_flange_frame,
                                   toggle_cdprim=toggle_cdprim,
                                   toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        return m_col


if __name__ == '__main__':
    import time
    from wrs import wd, mgm

    base = wd.World(cam_pos=[3, 1, 1], lookat_pos=[0, 0, 0.5])
    mgm.gen_frame().attach_to(base)
    robot = Yumi(enable_cc=True)
    robot.gen_meshmodel(toggle_cdprim=True).attach_to(base)
    # robot.gen_stickmodel().attach_to(base)
    # robot.show_cdprim()
    base.run()

    # ik test
    tgt_pos = np.array([.6, .0, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    robot.use_rgt()
    jnt_values = robot.ik(tgt_pos, tgt_rotmat)
    robot.goto_given_conf(jnt_values=jnt_values)
    robot.gen_meshmodel().attach_to(base)
    base.run()

    tic = time.time()
    jnt_values = robot.rgt_arm.ik(tgt_pos, tgt_rotmat)
    toc = time.time()
    print(toc - tic)
    if jnt_values is not None:
        robot.rgt_arm.goto_given_conf(jnt_values=jnt_values)
        robot.gen_meshmodel().attach_to(base)
    tic = time.time()
    result = robot.is_collided()
    toc = time.time()
    print(result, toc - tic)
    robot.show_cdprim()
    # robot.lft_arm.show_cdprim()
    base.run()
