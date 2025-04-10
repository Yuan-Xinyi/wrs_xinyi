import sys
sys.path.append('/home/lqin/wrs_xinyi/wrs')
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from wrs import wd, rm, mcm
robot_s = franka.FrankaResearch3(enable_cc=True)


# init
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)


def visualize_anime_diffusion(robot, path, start_conf, goal_conf):
    class Data(object):
        def __init__(self):
            self.counter = 0
            self.path = None
            self.current_model = None  # 当前帧的模型

    robot.goto_given_conf(jnt_values=start_conf)
    robot.gen_meshmodel(rgb=[0, 0, 1], alpha=.3).attach_to(base)
    robot.goto_given_conf(jnt_values=goal_conf)
    robot.gen_meshmodel(rgb=[0, 1, 0], alpha=.3).attach_to(base)

    anime_data = Data()
    anime_data.path = path

    def update(robot, anime_data, task):
        if anime_data.counter >= len(anime_data.path):
            if anime_data.current_model:
                anime_data.current_model.detach()  # 移除最后一帧的模型
            anime_data.counter = 0  # 重置计数器，循环播放

        # 移除上一帧的模型
        if anime_data.current_model:
            anime_data.current_model.detach()

        # 更新机器人位置并生成当前帧的模型
        conf = anime_data.path[anime_data.counter]
        robot.goto_given_conf(conf)
        anime_data.current_model = robot.gen_meshmodel(alpha=0.8)
        anime_data.current_model.attach_to(base)

        anime_data.counter += 1
        return task.again  # 继续任务，循环播放

    taskMgr.doMethodLater(0.001, update, "update",
                          extraArgs=[robot, anime_data],
                          appendTask=True)
    base.run()

if __name__ == '__main__':
    '''anime the npz'''
    # import numpy as np
    # with open('jnt_info.npz', 'rb') as f:
    #     data = np.load(f)
    #     jnt_pos = data['jnt_pos']

    '''anime zarr'''
    import zarr
    import numpy as np
    root = zarr.open('/home/lqin/zarr_datasets/test.zarr', mode='r')
    traj_id = 2
    traj_end = int(root['meta']['episode_ends'][traj_id])
    jnt_pos = root['data']['jnt_pos'][:traj_end+1]

    visualize_anime_diffusion(robot_s, jnt_pos, start_conf=jnt_pos[0], goal_conf=jnt_pos[-1])