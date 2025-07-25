import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi
import wrs.robot_sim.robots.cobotta_pro1300.cobotta_pro1300 as cbtpro1300
import wrs.basis.robot_math as rm
import wrs.motion.probabilistic.rrt_rtree as rrt

base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)

def extended_rand_conf(robot, expand_ratio=0.2):
    jnt_min, jnt_max = robot.jnt_ranges[:, 0], robot.jnt_ranges[:, 1]
    jnt_range = jnt_max - jnt_min
    return np.random.uniform(jnt_min - expand_ratio * jnt_range,
                             jnt_max + expand_ratio * jnt_range)


def generate_uniform_points_by_rrt_extend(robot, n_points=500, ext_dist=0.3):
    joint_lower, joint_upper = robot.jnt_ranges[:, 0], robot.jnt_ranges[:, 1]
    planner = rrt.RRT(robot)
    planner.roadmap.clear()
    start_conf = (joint_lower + joint_upper) / 2
    planner.start_conf = start_conf
    planner.roadmap.add_node("start", conf=start_conf)

    pbar = tqdm(total=n_points, desc=f"[{type(robot).__name__}] Sampling")
    while len(planner.roadmap.nodes) < n_points + 1:
        prev_n = len(planner.roadmap.nodes)
        rand_conf = extended_rand_conf(robot, expand_ratio=1.0)
        # planner._extend_roadmap(planner.roadmap, rand_conf, ext_dist, rand_conf, [], [], False)
        planner._extend_roadmap(rand_conf, ext_dist, rand_conf, [], [])
        pbar.update(len(planner.roadmap.nodes) - prev_n)
    pbar.close()

    print(f"[{type(robot).__name__}] Generated {len(planner.roadmap.nodes) - 1} configurations.")
    conf_array = np.array([data["conf"] for nid, data in planner.roadmap.nodes.items() if nid != "start"])
    np.save(f"{type(robot).__name__}_configs.npy", conf_array)

    sampled_min, sampled_max = np.min(conf_array, axis=0), np.max(conf_array, axis=0)
    print('for robot:', type(robot).__name__)
    print("\n=== Joint Ranges (robot vs. sampled data) ===")
    print(f"Sampled {len(conf_array)} configurations from RRT.")
    for i, (jmin, jmax, smin, smax) in enumerate(zip(joint_lower, joint_upper, sampled_min, sampled_max)):
        print(f"Joint {i}: range = [{jmin:.3f}, {jmax:.3f}], sampled = [{smin:.3f}, {smax:.3f}]")

    import math

    n_joints = conf_array.shape[1]
    n_cols = 3  # 每行 3 个子图
    n_rows = math.ceil(n_joints / n_cols)  # 自动计算行数

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i in range(n_joints):
        ax = axes[i]
        ax.hist(conf_array[:, i], bins=50, alpha=0.7, color='steelblue')
        ax.set_title(f"Joint {i}")
        ax.set_xlabel("Joint Value")
        ax.set_ylabel("Count")
        ax.grid(True)

    for j in range(n_joints, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    fig_path = '0000_test_programs/nn_ik/res_figs/0723_save'
    os.makedirs(fig_path, exist_ok=True)
    fig.savefig(os.path.join(fig_path, f"{type(robot).__name__}_joint_histograms.png"), dpi=300)

    return conf_array


if __name__ == '__main__':
    # robot_name_list = ['cbt','cbtpro1300', 'ur3', 'yumi']  # 支持多个机器人
    robot_name_list = ['ur3']
    for name in robot_name_list:
        if name == 'yumi':
            robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5), enable_cc=True)
            n_points = 1814400 # 201600
        elif name == 'cbt':
            robot = cbt.Cobotta(pos=rm.vec(0.1, .3, .5), enable_cc=True)
            n_points = 259200 # 40320
        elif name == 'ur3':
            robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), enable_cc=True)
            n_points = 40320 # 259200 # 40320
        elif name == 'cbtpro1300':
            robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)
            n_points = 259200 # 40320
        else:
            raise ValueError(f"Invalid robot name: {name}")
        conf_array = generate_uniform_points_by_rrt_extend(robot, n_points=n_points, ext_dist=0.4)
        print(f"Generated {len(conf_array)} configurations for {name}.")
        save_path = f"wrs/robot_sim/{name}_configs_rrt_rtree_0724.npy"
        np.save(save_path, conf_array)
        print(f"Configurations saved to {save_path}.")
        print(f'config_array shape: {conf_array.shape}')
