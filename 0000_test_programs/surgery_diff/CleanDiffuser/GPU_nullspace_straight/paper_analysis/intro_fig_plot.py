#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np
import torch

import wrs.modeling.collision_model as mcm
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd
from wrs.robot_sim.robots.franka_research_3.franka_research_3 import FrankaResearch3

# ==========================================
# 1. 基础数据准备 (所有块共享)
# ==========================================
INPUT_PATH = Path(__file__).resolve().parent.parent / "paper_analysis" / "outputs" / "intro_pair" / "selected_pair_with_paths.json"
ROOT_DIR = Path(__file__).resolve().parent.parent
LENGTH_PREDICTION_DIR = ROOT_DIR / "length_prediction"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(LENGTH_PREDICTION_DIR) not in sys.path:
    sys.path.insert(0, str(LENGTH_PREDICTION_DIR))

from length_prediction.lnet_contrastive_fr3_rotation_cone_eval import build_tracker, collect_candidate_qs

data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

pos = np.array(data["task"]["pos"])
normal = np.array(data["task"]["normal"]); normal /= np.linalg.norm(normal)
direction = np.array(data["task"]["direction"]); direction /= np.linalg.norm(direction)

# 计算旋转矩阵 (用于平面和锥体)
z_axis = normal
helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) <= 0.9 else np.array([0.0, 1.0, 0.0])
x_axis = np.cross(helper, z_axis); x_axis /= np.linalg.norm(x_axis)
y_axis = np.cross(z_axis, x_axis)
rotmat = np.column_stack((x_axis, y_axis, z_axis))

robot = FrankaResearch3(enable_cc=True)
short_q = np.array(data["short_solution"]["q"])
long_q = np.array(data["long_solution"]["q"])
short_path = np.array(data["short_trajectory"]["q"])
long_path = np.array(data["long_trajectory"]["q"])[:-150]
short_tcp = np.array(data["short_trajectory"]["tcp_pos"])
long_tcp = np.array(data["long_trajectory"]["tcp_pos"])[:-150]

# 通用环境初始化
def get_base_world():
    world = wd.World(cam_pos=[1.8, -1.6, 1.1], lookat_pos=[0.2, 0.0, 0.35])
    mgm.gen_frame().attach_to(world)
    mcm.gen_box(xyz_lengths=[0.7, 0.7, 0.001], pos=pos + 0.3 * direction, 
                rotmat=rotmat, rgb=[0.82, 0.87, 0.94], alpha=0.3).attach_to(world)
    return world

# ==========================================
# [BLOCK 1] 任务平面 + 30度锥体 + 代表性姿态
# ==========================================
world = get_base_world()
mgm.gen_arrow(spos=pos, epos=pos + 0.2 * normal, stick_radius=0.005, rgb=[1.0, 0.2, 0.2]).attach_to(world)

tracker, tracker_device = build_tracker(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
candidate_qs, _ = collect_candidate_qs(
    tracker,
    tracker_device,
    pos.astype(np.float32),
    direction.astype(np.float32),
    normal.astype(np.float32),
    64,
    512,
    2.0,
    50,
    1e-4,
    1e-3,
)
for q in candidate_qs:
    robot.goto_given_conf(q)
    robot.gen_meshmodel(alpha=0.1).attach_to(world)
    # t_p, t_r = robot.fk(jnt_values=q)
    # mgm.gen_frame(pos=t_p, rotmat=t_r).attach_to(world)
world.run()

# ==========================================
# [BLOCK 2] 两个候选姿态叠加 (红 vs 蓝)
# ==========================================
# world = get_base_world()
# for q, color in [(long_q, [0, 0, 1]), (short_q, [1, 0, 0])]:
#     robot.goto_given_conf(q)
#     robot.gen_meshmodel(rgb=color, alpha=0.2).attach_to(world)
#     t_p, t_r = robot.fk(jnt_values=q)
#     mgm.gen_frame(pos=t_p, rotmat=t_r).attach_to(world)
# world.run()

# ==========================================
# [BLOCK 3] 失败情况轨迹 (红色短路径)
# ==========================================
world = get_base_world()
robot.goto_given_conf(short_path[0])
robot.gen_meshmodel(rgb=[1, 0, 0], alpha=0.2).attach_to(world)
# 绘制起点到终点的直线连线
mgm.gen_stick(spos=short_tcp[0], epos=short_tcp[-1], radius=0.004, rgb=[0, 0, 0]).attach_to(world)
# 轨迹阴影 (步长100)
for i in list(range(0, len(short_path), 100)) + [len(short_path)-1]:
    robot.goto_given_conf(short_path[i])
    robot.gen_meshmodel(rgb=[1, 0, 0], alpha=0.1).attach_to(world)
    t_p, t_r = robot.fk(jnt_values=short_path[i])
    mgm.gen_frame(pos=t_p, rotmat=t_r).attach_to(world)
# 终点姿态
robot.goto_given_conf(short_path[-1])
robot.gen_meshmodel(rgb=[1, 0, 0], alpha=0.2).attach_to(world)
world.run()

# ==========================================
# [BLOCK 4] 成功情况轨迹 (蓝色长路径)
# ==========================================
world = get_base_world()
robot.goto_given_conf(long_path[0])
robot.gen_meshmodel(rgb = [0,0,1],alpha=0.2).attach_to(world)
mgm.gen_stick(spos=long_tcp[0], epos=long_tcp[-1], radius=0.004, rgb=[0, 0, 0]).attach_to(world)
for i in list(range(0, len(long_path), 100)) + [len(long_path)-1]:
    robot.goto_given_conf(long_path[i])
    robot.gen_meshmodel(rgb=[0,0,1], alpha=0.1).attach_to(world)
    t_p, t_r = robot.fk(jnt_values=long_path[i])
    mgm.gen_frame(pos=t_p, rotmat=t_r).attach_to(world)
robot.goto_given_conf(long_path[-1])
robot.gen_meshmodel(rgb = [0,0,1],alpha=0.2).attach_to(world)
t_p, t_r = robot.fk(jnt_values=long_path[-1])
mgm.gen_frame(pos=t_p, rotmat=t_r).attach_to(world)
world.run()
