from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm


def visualize_anime_path(robot, path):
    class Data(object):
        def __init__(self):
            self.counter = 0
            self.path = path
            self.current_model = None
            self.current_frame = None  # 用于记录并移除上一帧的末端坐标系

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
        taskMgr.doMethodLater(0.08, update, "update",
                              extraArgs=[robot, anime_data],
                              appendTask=True)
        return task.done

    taskMgr.doMethodLater(1.0, start_animation, "start_animation_delay")
    base.run()