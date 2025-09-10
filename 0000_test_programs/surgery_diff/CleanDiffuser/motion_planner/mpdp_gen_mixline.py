# -*- coding: utf-8 -*-
import os
import numpy as np
import zarr
from tqdm import tqdm
from scipy import interpolate

# wrs imports
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import wrs.basis.robot_math as rm


def fibonacci_sphere(samples=20):
    """Uniformly sample `samples` unit vectors on the sphere"""
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def axis_directions():
    """Six Cartesian axis directions"""
    return np.array([
        [1, 0, 0],   # +x
        [-1, 0, 0],  # -x
        [0, 1, 0],   # +y
        [0, -1, 0],  # -y
        [0, 0, 1],   # +z
        [0, 0, -1],  # -z
    ], dtype=float)


def sample_spline_traj(start_pos, n_ctrl=5, n_points=40, radius=0.15):
    """Generate a local B-spline trajectory around start_pos.
    
    Args:
        start_pos: np.array(3,), start position of trajectory
        n_ctrl: number of control points
        n_points: number of sampled points on spline
        radius: max offset for control points from start_pos
    """
    # Random control points around start_pos
    ctrl_points = start_pos + np.random.uniform(
        low=-radius, high=radius, size=(n_ctrl, 3)
    )
    ctrl_points[0] = start_pos  # ensure starting point fixed
    
    # B-spline fitting
    tck, _ = interpolate.splprep(
        [ctrl_points[:, 0], ctrl_points[:, 1], ctrl_points[:, 2]], k=3, s=0
    )
    u_new = np.linspace(0, 1, n_points)
    x_new, y_new, z_new = interpolate.splev(u_new, tck)
    traj = np.vstack([x_new, y_new, z_new]).T
    
    return traj


def visualize_anime_path_colored(robot, path, colors):
    """Visualization: animate trajectory with colors"""
    class Data(object):
        def __init__(self, path, colors):
            self.counter = 0
            self.path = path
            self.colors = colors
            self.current_model = None
            self.current_frame = None

    anime_data = Data(path, colors)

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
        color = anime_data.colors[anime_data.counter]

        robot.goto_given_conf(conf)
        anime_data.current_model = robot.gen_meshmodel(rgb=color)
        anime_data.current_model.attach_to(base)

        ee_pos, ee_rotmat = robot.fk(conf)
        anime_data.current_frame = mgm.gen_frame(pos=ee_pos, rotmat=ee_rotmat)
        anime_data.current_frame.attach_to(base)

        anime_data.counter += 1
        return task.again

    def start_animation(task):
        print("[INFO] Animation started")
        taskMgr.doMethodLater(0.08, update, "update",
                              extraArgs=[robot, anime_data],
                              appendTask=True)
        return task.done

    taskMgr.doMethodLater(1.0, start_animation, "start_animation_delay")
    base.run()


if __name__ == '__main__':
    MAX_WAYPOINT = 80
    waypoint_interval = 0.02
    n_ctrl = 5
    n_points = 40

    '''Initialize the world and robot'''
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = franka.FrankaResearch3(enable_cc=True)

    jnt_samples = np.load('cvt_joint_samples_fr3_scale5.npy')
    jnt_samples = jnt_samples[3956:]
    print(f"[INFO] Total {len(jnt_samples)} joint configurations sampled.")
    print('--' * 100)

    #------------------------simple visualization test------------------------
    '''simple visualization test: comment this part if you only want dataset generation'''
    # # Use only the first initial configuration
    # pos_init, rotmat_init = robot.fk(jnt_values=jnt_samples[0])
    # seed_jnt = jnt_samples[0]
    # print(f"[DEBUG] Initial end-effector position: {pos_init}")

    # path = []
    # colors = []

    # # ===== Line trajectories (red) =====
    # directions = axis_directions()
    # for dir_id, direction in enumerate(directions):
    #     jnt_list, pos = [seed_jnt], pos_init.copy()
    #     print(f"[INFO] Generating line trajectory {dir_id}, direction {direction}")
    #     for step in range(MAX_WAYPOINT):
    #         pos_try = pos + direction * waypoint_interval
    #         jnt = robot.ik(tgt_pos=pos_try, tgt_rotmat=rotmat_init, seed_jnt_values=jnt_list[-1])
    #         if jnt is None:
    #             print(f"[WARN] IK failed at step {step}, pos {pos_try}")
    #             break
    #         pos = pos_try
    #         jnt_list.append(jnt)
    #     print(f"[INFO] Line trajectory {dir_id} generated {len(jnt_list)} points")
    #     path.extend(jnt_list)
    #     colors.extend([[1, 0, 0, 1]] * len(jnt_list))  # red

    # # ===== Random spline trajectories (blue) =====
    # for curve_id in range(3):  # generate 3 random curves
    #     print(f"[INFO] Generating spline trajectory {curve_id}")
    #     spline_traj = sample_spline_traj(pos_init, n_ctrl=4, n_points=40)
    #     jnt_list = [seed_jnt]
    #     for i, pos_target in enumerate(spline_traj[1:]):
    #         jnt = robot.ik(tgt_pos=pos_target, tgt_rotmat=rotmat_init, seed_jnt_values=jnt_list[-1])
    #         if jnt is None:
    #             print(f"[WARN] IK failed at curve {curve_id}, point {i}, pos {pos_target}")
    #             break
    #         jnt_list.append(jnt)
    #     print(f"[INFO] Curve {curve_id} generated {len(jnt_list)} points")
    #     path.extend(jnt_list)
    #     colors.extend([[0, 0, 1, 1]] * len(jnt_list))  # blue

    # # ===== Visualization =====
    # print(f"[INFO] Visualizing trajectory with total {len(path)} configs")
    # visualize_anime_path_colored(robot, path, colors)
    # base.run()
    #------------------------visulization finished------------------------

    '''dataset generation'''
    dataset_name = os.path.join('/home/lqin/zarr_datasets', f'fr3_mixed_trajs_clean.zarr')
    dof = robot.n_dof
    if os.path.exists(dataset_name):
        root = zarr.open(dataset_name, mode='a')  # Open the dataset in append mode
        jnt_p = root['data']["jnt_pos"]
        workspc_pos = root['data']["position"]
        workspc_rotq = root['data']["rotation"]
        traj_type = root['data']["traj_type"]
        episode_ends_ds = root['meta']["episode_ends"]
        episode_ends_counter = root['meta']['episode_ends'][-1]
        print(f"[INFO] Dataset opened: {dataset_name}")
    else:
        store = zarr.DirectoryStore(dataset_name)
        root = zarr.group(store=store)
        # create meta and data groups
        meta_group = root.create_group("meta")
        data_group = root.create_group("data")
        print(f"[INFO] dataset created in: {dataset_name}")
        jnt_p = data_group.create_dataset("jnt_pos", shape=(0, dof), chunks=(1, dof), dtype=np.float32, append=True)
        workspc_pos = data_group.create_dataset("position", shape=(0, 3), chunks=(1, 3), dtype=np.float32, append=True)
        workspc_rotq = data_group.create_dataset("rotation", shape=(0, 4), chunks=(1, 4), dtype=np.float32, append=True)
        traj_type = data_group.create_dataset("traj_type", shape=(0,), chunks=(1,), dtype=np.int32, append=True)  # 0=line, 1=spline
        episode_ends_ds = meta_group.create_dataset("episode_ends", shape=(0,), chunks=(1,), dtype=np.int64, append=True)
        episode_ends_counter = 0
        print(f"[INFO] New dataset created: {dataset_name}")

    line_success, spline_success = 0, 0  # statistics

    # direction set
    directions = axis_directions()

    for idx in tqdm(range(len(jnt_samples))):
        pos_init, rotmat_init = robot.fk(jnt_values=jnt_samples[idx])

        # ====== Line trajectories ======
        for dir_id, direction in enumerate(directions):
            jnt_list = []
            pos = pos_init.copy()
            seed_jnt = jnt_samples[idx]

            for step in range(MAX_WAYPOINT):
                pos_try = pos + direction * waypoint_interval
                jnt = robot.ik(tgt_pos=pos_try, tgt_rotmat=rotmat_init, seed_jnt_values=seed_jnt)
                if jnt is None:
                    break
                jnt_p.append(np.array(jnt).reshape(1, dof))
                workspc_pos.append(np.array(pos_try).reshape(1, 3))
                workspc_rotq.append(np.array(rm.rotmat_to_quaternion(rotmat_init)).reshape(1, 4))
                traj_type.append(np.array([0]))
                jnt_list.append(jnt)
                seed_jnt = jnt
                pos = pos_try

            if len(jnt_list) > 0:
                episode_ends_counter += len(jnt_list)
                episode_ends_ds.append(np.array([episode_ends_counter]))
                line_success += 1
                print(f"[INFO] Sample {idx}, Line {dir_id}: length={len(jnt_list)}")


        # ====== Spline trajectories ======
        for curve_id in range(5):
            # real workspace range: workspace=[(-0.9, 0.9), (-0.9, 0.9), (-0.3, 1.26)]
            spline_traj = sample_spline_traj(pos_init, n_ctrl=n_ctrl, n_points=n_points)
            jnt_list = []
            seed_jnt = jnt_samples[idx]

            for pos_target in spline_traj:
                jnt = robot.ik(tgt_pos=pos_target, tgt_rotmat=rotmat_init, seed_jnt_values=seed_jnt)
                if jnt is None:
                    break
                jnt_p.append(np.array(jnt).reshape(1, dof))
                workspc_pos.append(np.array(pos_target).reshape(1, 3))
                workspc_rotq.append(np.array(rm.rotmat_to_quaternion(rotmat_init)).reshape(1, 4))
                traj_type.append(np.array([1]))
                jnt_list.append(jnt)
                seed_jnt = jnt

            if len(jnt_list) > 0:
                episode_ends_counter += len(jnt_list)
                episode_ends_ds.append(np.array([episode_ends_counter]))
                spline_success += 1
                print(f"[INFO] Sample {idx}, Spline {curve_id}: length={len(jnt_list)}")



    # ===== Final statistics =====
    print("========== Summary ==========")
    print(f"Line trajectories succeeded: {line_success}")
    print(f"Spline trajectories succeeded: {spline_success}")
    print(f"Total episodes: {len(episode_ends_ds[:])}")
    print("=============================")
