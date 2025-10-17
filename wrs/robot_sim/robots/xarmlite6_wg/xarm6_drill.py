import numpy as np
import wrs.robot_sim.manipulators.xarm_lite6 as manipulator
# import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v2 as end_effector
import wrs.robot_sim.end_effectors.single_contact.milling.spine_miller as end_effector
import wrs.robot_sim.robots.single_arm_robot_interface as sari


class XArmLite6Miller(sari.SglArmRobotInterface):
    """
    Simulation for the XArm Lite 6 With the WRS grippers
    Author: Chen Hao (chen960216@gmail.com), Updated by Weiwei
    Date: 20220925osaka, 20240318
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='xarm_lite6_miller', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self.manipulator = manipulator.XArmLite6(pos=pos, rotmat=rotmat, name="xarmlite6g2_arm",
                                                 enable_cc=False)
        self.end_effector = end_effector.SpineMiller(pos=self.manipulator.gl_flange_pos,
                                                     rotmat=self.manipulator.gl_flange_rotmat, name="miller")
        # tool center point
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        # end effector
        ee_cces = []
        for id, cdlnk in enumerate(self.end_effector.cdelements):
            ee_cces.append(self.cc.add_cce(cdlnk))
        # manipulator
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        from_list = [mlb, ml0, ml1]
        into_list = ee_cces + [ml4, ml5]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self.cc.enable_extcd_by_id_list(id_list=[ml0, ml1, ml2, ml3, ml4, ml5], type="from")
        self.cc.enable_innercd_by_id_list(id_list=[mlb, ml0, ml1, ml2, ml3], type="into")
        self.cc.dynamic_ext_list = ee_cces

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def change_jaw_width(self, jaw_width):
        return self.change_ee_values(ee_values=jaw_width)

    def get_jaw_width(self):
        return self.get_ee_values()


if __name__ == '__main__':
    from wrs import wd, mgm, rm

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    mgm.gen_frame().attach_to(base)
    robot = XArmLite6Miller(pos=np.array([0, 0, 0]), enable_cc=True)
    jnt = np.array([ 2.5870,  1.9920,  0.2887, -1.2303,  0.1405, -0.2582])
    robot.goto_given_conf(jnt_values=jnt)
    tgt_pos, tgt_rotmat = robot.fk(jnt_values=jnt)
    print(tgt_pos)
    print(tgt_rotmat)
    base.run()
