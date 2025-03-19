from wrs import wd, rm, mcm, mgm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.rs007l.rs007l as rs007l
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.manipulators.ur3e.ur3e as ur3e
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi
import wrs.robot_sim.robots.cobotta_pro1300.cobotta_pro1300 as cbtpro1300
import wrs.robot_sim.robots.cobotta_pro900.cobotta_pro900_spine as cbtpro900


import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)


c0 = cbt.Cobotta(pos=rm.vec(0.3, -.3, .0), enable_cc=True)
c1 = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0., .0, .0), enable_cc=True)

c0_gth = [ 1.80237981,  0.17540694,  1.52333545, -2.12703468,  0.74813403,
       -1.2165423 ]
c0_seed = [ 0.29088822, -0.69813213,  1.22671717, -2.11932857, -0.98902017,
       -0.593412  ]
c1_gth = [ 0.21871924, -1.86396316,  0.81752382,  0.4906148 ,  0.15849782,
        3.76826096]
c1_seed = [ 0.52359878, -1.96349541,  1.12199738, -0.67319843,  0.        ,
       -1.25663706]


c0.goto_given_conf(c0_gth)
arm_mesh = c0.gen_meshmodel(alpha=.5, rgb=[0,1,0])
arm_mesh.attach_to(base)
c0.goto_given_conf(c0_seed)
arm_mesh = c0.gen_meshmodel(alpha=.5, rgb=[1,0,0])
arm_mesh.attach_to(base)
c1.goto_given_conf(c1_gth)
arm_mesh = c1.gen_meshmodel(alpha=.5, rgb=[0,1,0])
arm_mesh.attach_to(base)
c1.goto_given_conf(c1_seed)
arm_mesh = c1.gen_meshmodel(alpha=.5, rgb=[0,0,1])
arm_mesh.attach_to(base)

base.run()