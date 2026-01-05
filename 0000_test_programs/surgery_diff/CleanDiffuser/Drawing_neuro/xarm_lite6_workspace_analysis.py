'''wrs reliance'''
from tqdm import tqdm
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

xarm = xarm6_sim.XArmLite6Miller(enable_cc=True)
pos_list = []
rotmat_list = []

for i in tqdm(range(int(1e5))):
    jnt = xarm.rand_conf()
    pos, rotmat = xarm.fk(jnt)
    pos_list.append(pos.numpy())
    rotmat_list.append(rotmat.numpy())