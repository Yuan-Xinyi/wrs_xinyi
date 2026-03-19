import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
from wrs import wd
import numpy as np
from collision_checker import XarmCollisionChecker

if __name__ == "__main__":
    base = wd.World(cam_pos=[1.2, 0.5, 0.5], lookat_pos=[0.3, 0.0, 0.0])
    mgm.gen_frame().attach_to(base)

    robot = xarm6_sim.XArmLite6Miller(enable_cc=True)
    cc_checker = XarmCollisionChecker()
    
    '''single configuration collision check'''
    # jnt = robot.rand_conf()
    # robot.goto_given_conf(jnt_values=jnt)
    # in_collision = cc_checker.check_collision(jnt)
    # print(f"Random configuration in collision: {bool(in_collision)}")
    # robot.gen_meshmodel().attach_to(base)
    # base.run()

    '''visulize collision datasets'''
    collision_free_kernels_path = "0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine/datasets/cvt_kernels_in_collision.npy"
    collision_jnts = np.load(collision_free_kernels_path)
    print(f"Loaded {len(collision_jnts)} collision configurations.")

    idx = 10000
    jnt = collision_jnts[idx]
    robot.goto_given_conf(jnt_values=jnt)
    robot.gen_meshmodel(alpha=0.3).attach_to(base)
    in_collision = cc_checker.check_collision(jnt)
    print(f"Collision kernel {idx} in collision: {bool(in_collision)}")

    base.run()