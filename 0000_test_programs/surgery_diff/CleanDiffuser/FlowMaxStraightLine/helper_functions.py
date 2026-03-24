import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline, BSpline

from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka

def visualize_anime_path(base, robot, path):
    class Data(object):
        def __init__(self):
            self.counter = 0
            self.path = path
            self.current_model = None
            self.current_frame = None

    anime_data = Data()

    def update(robot, anime_data, task):
        if anime_data.counter >= len(anime_data.path):
            if anime_data.current_model:
                anime_data.current_model.detach()
            if anime_data.current_frame:
                anime_data.current_frame.detach()
            anime_data.counter = 0
            return task.again

        if anime_data.current_model:
            anime_data.current_model.detach()
        if anime_data.current_frame:
            anime_data.current_frame.detach()

        conf = anime_data.path[anime_data.counter]
        robot.goto_given_conf(conf)
        anime_data.current_model = robot.gen_meshmodel(alpha=1.0)
        anime_data.current_model.attach_to(base)

        ee_pos, ee_rotmat = robot.fk(conf)
        anime_data.current_frame = mcm.mgm.gen_frame(pos=ee_pos, rotmat=ee_rotmat)
        anime_data.current_frame.attach_to(base)

        anime_data.counter += 1
        return task.again

    def start_animation(task):
        taskMgr.doMethodLater(0.02, update, "update",
                              extraArgs=[robot, anime_data],
                              appendTask=True)
        return task.done

    taskMgr.doMethodLater(1.0, start_animation, "start_animation_delay")
    base.run()

def visualize_static_path(base, robot, path):
    for jnt in path:
        robot.goto_given_conf(jnt)
        pos, rot = robot.fk(jnt)
        mcm.mgm.gen_frame(pos=pos, rotmat=rot).attach_to(base)
        robot.gen_meshmodel(alpha=.1).attach_to(base)
    base.run()

def workspace_plot(robot, jnt_path):
    pos_list = []
    for jnt_conf in jnt_path:
        pos, _ = robot.fk(jnt_conf)
        pos_list.append(pos)

    pos_list = np.array(pos_list)
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axs[0].plot(pos_list[:, 0], label='X', color='r')
    axs[0].set_title('X Axis')
    axs[0].set_ylabel('Position (X)')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(pos_list[:, 1], label='Y', color='g')
    axs[1].set_title('Y Axis')
    axs[1].set_ylabel('Position (Y)')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(pos_list[:, 2], label='Z', color='b')
    axs[2].set_title('Z Axis')
    axs[2].set_ylabel('Position (Z)')
    axs[2].set_xlabel('Time Steps')
    axs[2].grid(True)
    axs[2].legend()

    fig.suptitle("Workspace Trajectory - X, Y, Z Axes", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
