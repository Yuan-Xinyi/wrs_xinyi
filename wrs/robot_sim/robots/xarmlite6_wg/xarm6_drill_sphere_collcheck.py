import string
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
    
    def are_jnts_in_ranges(self, jnt_values):
        return super().are_jnts_in_ranges(jnt_values)


if __name__ == '__main__':
    from wrs import wd, mgm, rm, mcm

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    mgm.gen_frame().attach_to(base)

    table_size = np.array([1.5, 1.5, 0.03])
    table_pos  = np.array([0.2, 0, -0.025])
    table = mcm.gen_box(xyz_lengths=table_size, pos=table_pos, rgb=np.array([0.6, 0.4, 0.2]), alpha=1)
    # table.attach_to(base)

    paper_size = np.array([1.2, 1.2, 0.002])
    paper_pos = table_pos.copy()
    paper_pos[2] = table_pos[2] + table_size[2]/2 + paper_size[2]/2
    print("paper pos:", paper_pos)
    paper = mcm.gen_box(xyz_lengths=paper_size, pos=paper_pos, rgb=np.array([1, 1, 1]), alpha=1)
    # paper.attach_to(base)

    robot = XArmLite6Miller(enable_cc=True)
    # jnt = robot.rand_conf()
    # robot.goto_given_conf(jnt_values=jnt)
    # robot.gen_meshmodel(rgb=[0,1,0]).attach_to(base)
    # tgt_pos, tgt_rotmat = robot.fk(jnt_values=jnt)
    tgt_pos = np.array([0.3, 0.0, 0.0])
    tgt_rotmat = np.eye(3)
    tgt_rotmat[:3,2] = np.array([0,0,-1])
    jnt_ik = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    jnt_ik = robot.rand_conf()  # --- IGNORE ---
    # jnt_ik = np.array([-0.51425886, 1.61529928, 4.88470885, 2.42885323, 1.0333872, 0.98597998])
    robot.goto_given_conf(jnt_values=jnt_ik)
    robot.gen_meshmodel(alpha=0.2).attach_to(base)
    # mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    print("jnt from ik:", jnt_ik)
    print(tgt_pos)
    print(tgt_rotmat)
    

    from sphere_collision_checker import SphereCollisionChecker
    model = SphereCollisionChecker('wrs/robot_sim/robots/xarmlite6_wg/xarm6_sphere_visuals.urdf')
    import time
    import jax.numpy as jnp
    # warm up the jax to avoid initial compilation time
    _ = model.update(jnp.array(np.zeros(robot.n_dof)))
    t1 = time.time()
    q_gpu = jnp.array(jnt_ik)
    positions = model.update(q_gpu)
    t2 = time.time()
    print("[INFO] Time for sphere collision check update:", t2 - t1)

    # collision visualization
    spheres_pos, collision_flags = model.check_collisions(q_gpu)
    radii = model.sphere_radii

    for id in range(positions.shape[0]):
        if collision_flags[id]:
            sphere = mcm.gen_sphere(radius=model.sphere_radii[id], pos=positions[id], rgb=[1,0,0], alpha=0.2)
        else:
            sphere = mcm.gen_sphere(radius=model.sphere_radii[id], pos=positions[id], rgb=[0,0,1], alpha=0.2)
        sphere.attach_to(base)
    
    base.run()
