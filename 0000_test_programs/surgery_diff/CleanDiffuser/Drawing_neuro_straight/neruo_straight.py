'''wrs reliance'''
from wrs import wd, rm, mcm
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
import wrs.modeling.geometric_model as mgm

'''global variables'''
import time
import pickle
import numpy as np
import torch
torch.autograd.set_detect_anomaly(False)

import matplotlib.pyplot as plt

# initialize robot models and simulation
xarm_gpu = xarm6_gpu.XArmLite6GPU()
xarm_sim = xarm6_sim.XArmLite6Miller(enable_cc=True)
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)
xarm_sim.goto_given_conf([0, 0, 0, 0, 0, 0])
xarm_sim.gen_meshmodel().attach_to(base)

table_size = np.array([1.5, 1.5, 0.03])
table_pos  = np.array([0.2, 0, -0.025])
table = mcm.gen_box(xyz_lengths=table_size, pos=table_pos, rgb=np.array([0.6, 0.4, 0.2]), alpha=1)
table.attach_to(base)

paper_size = np.array([1.2, 1.2, 0.002])
paper_pos = table_pos.copy()
paper_pos[2] = table_pos[2] + table_size[2]/2 + paper_size[2]/2
print("paper pos:", paper_pos)
paper = mcm.gen_box(xyz_lengths=paper_size, pos=paper_pos, rgb=np.array([1, 1, 1]), alpha=1)
paper.attach_to(base)

device = xarm_gpu.device


if __name__ == "__main__":

    base.run()

