import numpy as np
import samply

import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
from wrs import wd

if __name__ == "__main__":
    base = wd.World(cam_pos=[1.2, 0.5, 0.5], lookat_pos=[0.3, 0.0, 0.0])
    mgm.gen_frame().attach_to(base)
    robot = xarm6_sim.XArmLite6Miller(enable_cc=True)

    sample_number = 1000

    normalized_qs = samply.hypercube.cvt(sample_number, robot.n_dof)
    print(f"Building Data for SELIK using CVT for {sample_number} uniform samples...")
    sampled_qs = robot.jnt_ranges[:, 0] + normalized_qs * (robot.jnt_ranges[:, 1] - robot.jnt_ranges[:, 0])