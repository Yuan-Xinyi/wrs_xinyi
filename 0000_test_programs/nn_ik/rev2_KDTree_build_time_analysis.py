import scipy.spatial
import numpy as np
from shapely import points
from tqdm import tqdm
import wrs.basis.robot_math as rm
import time
import copy

import samply
from wrs.basis.robot_math import vec
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.robots.cobotta_pro1300.cobotta_pro1300 as cbtpro1300
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi

# N6 = [4096, 5120, 12000, 25200, 40320, 60480, 151200, 166320, 259200, 2592000]
N6 = [166320, 259200, 2592000]
N7 = [16384, 20480, 48000, 100800, 201600, 362880, 604800, 1663200, 1814400, 3628800]

for n in N6:
    robot_constructors = {
        'cobotta': lambda: cbt.Cobotta(pos=vec(0.1, .3, .5), enable_cc=False),
        'cobotta_pro1300': lambda: cbtpro1300.CobottaPro1300WithRobotiq140(pos=vec(0.1, .3, .5), enable_cc=False),
        'ur3': lambda: ur3.UR3(pos=vec(0.1, .3, .5), enable_cc=False),
    }
    sampled_qs = samply.hypercube.cvt(n, 6)
    for name, ctor in robot_constructors.items():
        robot = ctor()
        low = np.array([rng[0] for rng in robot.jnt_ranges])
        high = np.array([rng[1] for rng in robot.jnt_ranges])
        scaled_qs = low + (high - low) * copy.deepcopy(sampled_qs)

        query_data = []
        jnt_data = []
        jinv_data = []
        for id in range(len(scaled_qs)):
            jnt_values = scaled_qs[id]
            # pinv of jacobian
            flange_pos, flange_rotmat = robot.fk(jnt_values=jnt_values, toggle_jacobian=False)
            rel_pos, rel_rotmat = rm.rel_pose(robot.pos, robot.rotmat, flange_pos, flange_rotmat)
            rel_rotvec = rm.rotmat_to_wvec(rel_rotmat)
            query_data.append(rel_pos.tolist() + rel_rotvec.tolist())
        start = time.perf_counter()
        query_tree = scipy.spatial.cKDTree(query_data)
        end = time.perf_counter()
        elapsed_us = (end - start) * 1e3
        print(f"[{name}] 6 dof {n} samples: {elapsed_us:.2f} ms")

for n in N7:
    robot_constructors = {
        'yumi': lambda: yumi.YumiSglArm(pos=vec(0.1, .3, .5), enable_cc=False),
    }
    sampled_qs = samply.hypercube.cvt(n, 7)
    for name, ctor in robot_constructors.items():
        robot = ctor()
        low = np.array([rng[0] for rng in robot.jnt_ranges])
        high = np.array([rng[1] for rng in robot.jnt_ranges])
        scaled_qs = low + (high - low) * copy.deepcopy(sampled_qs)

        query_data = []
        jnt_data = []
        jinv_data = []
        for id in range(len(scaled_qs)):
            jnt_values = scaled_qs[id]
            # pinv of jacobian
            flange_pos, flange_rotmat = robot.fk(jnt_values=jnt_values, toggle_jacobian=False)
            rel_pos, rel_rotmat = rm.rel_pose(robot.pos, robot.rotmat, flange_pos, flange_rotmat)
            rel_rotvec = rm.rotmat_to_wvec(rel_rotmat)
            query_data.append(rel_pos.tolist() + rel_rotvec.tolist())
        start = time.perf_counter()
        query_tree = scipy.spatial.cKDTree(query_data)
        end = time.perf_counter()
        elapsed_us = (end - start) * 1e3
        print(f"[{name}] 7 dof {n} samples: {elapsed_us:.2f} ms")