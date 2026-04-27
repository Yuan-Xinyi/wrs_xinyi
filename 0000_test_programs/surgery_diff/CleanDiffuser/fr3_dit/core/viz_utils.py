import numpy as np

import wrs.modeling.geometric_model as mgm


def visualize_anime_path(base, robot, path, frame_delay: float = 0.2):
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
        anime_data.current_frame = mgm.gen_frame(pos=ee_pos, rotmat=ee_rotmat)
        anime_data.current_frame.attach_to(base)

        anime_data.counter += 1
        return task.again

    def start_animation(task):
        base.taskMgr.doMethodLater(
            float(frame_delay),
            update,
            "update",
            extraArgs=[robot, anime_data],
            appendTask=True,
        )
        return task.done

    base.taskMgr.doMethodLater(1.0, start_animation, "start_animation_delay")
    base.run()


def visualize_anime_dual(base, entries, frame_delay: float = 0.05):
    """Animate multiple robots in lock-step, each with its own trajectory.

    ``entries`` is a list of dicts:
        {"robot": <PenFrankaResearch3>, "path": np.ndarray (T, 7),
         "rgb": (3,) or None, "alpha": float, "name": str (optional)}

    Trajectories are right-padded with their last frame so the animation runs
    for max(len(path)) steps. After all frames the loop restarts.
    """
    if not entries:
        raise ValueError("entries must be non-empty")

    paths = []
    for e in entries:
        p = np.asarray(e["path"], dtype=np.float32)
        if p.ndim != 2 or p.shape[1] != 7:
            raise ValueError(f"path shape must be (T, 7), got {p.shape}")
        paths.append(p)
    n_frames = max(p.shape[0] for p in paths)
    rgbs = [e.get("rgb") for e in entries]
    alphas = [float(e.get("alpha", 1.0)) for e in entries]

    class Data:
        def __init__(self):
            self.counter = 0
            self.current_models = [None] * len(entries)

    state = Data()

    def update(task):
        if state.counter >= n_frames:
            for m in state.current_models:
                if m is not None:
                    m.detach()
            state.counter = 0
            return task.again

        for i, e in enumerate(entries):
            if state.current_models[i] is not None:
                state.current_models[i].detach()
            idx = min(state.counter, paths[i].shape[0] - 1)
            conf = paths[i][idx].astype(np.float32)
            e["robot"].goto_given_conf(conf)
            kwargs = {"alpha": alphas[i]}
            if rgbs[i] is not None:
                kwargs["rgb"] = np.asarray(rgbs[i], dtype=np.float32)
            mdl = e["robot"].gen_meshmodel(**kwargs)
            mdl.attach_to(base)
            state.current_models[i] = mdl

        state.counter += 1
        return task.again

    def start_animation(task):
        base.taskMgr.doMethodLater(
            float(frame_delay), update, "dual_update", appendTask=True,
        )
        return task.done

    base.taskMgr.doMethodLater(1.0, start_animation, "dual_start_delay")
    base.run()
