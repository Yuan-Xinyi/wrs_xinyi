import numpy as np
import wrs.robot_sim.manipulators.franka_research_3_arm.franka_research_3_arm as manipulator
import wrs.robot_sim.end_effectors.grippers.franka_hand.franka_hand as end_effector
import wrs.robot_sim.robots.single_arm_robot_interface as sari


class FrankaResearch3SphereCollCheck(sari.SglArmRobotInterface):
    """Simulation for Franka Research 3 with FrakaHand.

    Structured to mirror ``xarm6_drill_sphere_collcheck.py``.
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='franka_research_3_sphere_collcheck', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        home_conf = np.zeros(7)
        home_conf[1] = -0.785398163
        home_conf[3] = -2.35619449
        home_conf[5] = 1.57079632679
        home_conf[6] = 0.785398163397
        self.manipulator = manipulator.FrankaResearch3Arm(
            pos=pos,
            rotmat=rotmat,
            home_conf=home_conf,
            name='franka_research_3_arm',
            enable_cc=False,
        )
        self.end_effector = end_effector.FrankaHand(
            pos=self.manipulator.gl_flange_pos,
            rotmat=self.manipulator.gl_flange_rotmat,
            name='franka_hand',
        )
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        ee_cces = []
        for _, cdlnk in enumerate(self.end_effector.cdelements):
            ee_cces.append(self.cc.add_cce(cdlnk))
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        ml6 = self.cc.add_cce(self.manipulator.jlc.jnts[6].lnk)
        from_list = ee_cces + [ml5, ml6]
        into_list = [mlb, ml0, ml1, ml2, ml3]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        self.cc.enable_extcd_by_id_list(id_list=[ml0, ml1, ml2, ml3, ml4, ml5] + ee_cces, type='from')
        self.cc.enable_innercd_by_id_list(id_list=[mlb, ml0, ml1, ml2, ml3], type='into')
        self.cc.dynamic_ext_list = ee_cces[1:]

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
    from wrs import wd, mgm, mcm
    from sphere_collision_checker import SphereCollisionChecker
    from wrs.robot_sim.robots.franka_research_3.franka_research_3 import FrankaResearch3
    import time
    import jax.numpy as jnp

    # Calibration demo:
    # - robot mesh uses the official WRS kinematic chain defined by FrankaResearch3
    # - collision spheres use the independent sphere-URDF kinematic chain
    # The overlap quality between the two is the calibration target.
    base = wd.World(cam_pos=[2, 2, 0.8], lookat_pos=[0, 0, 0.5])
    mgm.gen_frame().attach_to(base)

    robot = FrankaResearch3(enable_cc=True)
    q = robot.rand_conf()
    # q = np.zeros(7, dtype=np.float64)
    robot.goto_given_conf(jnt_values=q)
    robot.gen_meshmodel(alpha=0.6, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)

    model = SphereCollisionChecker('wrs/robot_sim/robots/franka_research_3/franka_research_3_ccsphere.urdf')
    _ = model.update(jnp.array(np.zeros(robot.n_dof)))
    t1 = time.time()
    q_gpu = jnp.array(q)
    positions = model.update(q_gpu)
    positions = positions.block_until_ready()
    t2 = time.time()
    print('[INFO] sphere-URDF update time:', t2 - t1)
    print('[INFO] mesh source: wrs.robot_sim.robots.franka_research_3.FrankaResearch3')
    print('[INFO] q =', q)

    spheres_pos, collision_flags = model._jit_check_collisions(q_gpu)
    spheres_pos = spheres_pos.block_until_ready()
    collision_flags = collision_flags.block_until_ready()

    for idx in range(positions.shape[0]):
        if collision_flags[idx]:
            sphere = mcm.gen_sphere(radius=float(model.sphere_radii[idx]), pos=positions[idx], rgb=[1, 0, 0], alpha=0.2)
        else:
            sphere = mcm.gen_sphere(radius=float(model.sphere_radii[idx]), pos=positions[idx], rgb=[0, 0, 1], alpha=0.2)
        sphere.attach_to(base)

    print(f'[INFO] self collision cost = {model.self_collision_cost(q_gpu, scale=1)}')
    base.run()
