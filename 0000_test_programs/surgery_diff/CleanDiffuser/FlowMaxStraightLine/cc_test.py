import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
from wrs import wd
from collision_checker import XarmCollisionChecker

if __name__ == "__main__":
    base = wd.World(cam_pos=[1.2, 0.5, 0.5], lookat_pos=[0.3, 0.0, 0.0])
    mgm.gen_frame().attach_to(base)

    robot = xarm6_sim.XArmLite6Miller(enable_cc=True)
    cc_checker = XarmCollisionChecker()
    jnt = robot.rand_conf()
    
    robot.goto_given_conf(jnt_values=jnt)
    in_collision = cc_checker.check_collision(jnt)
    print(f"Random configuration in collision: {bool(in_collision)}")
    robot.gen_meshmodel().attach_to(base)
    base.run()
