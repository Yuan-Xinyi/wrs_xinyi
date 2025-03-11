import copy
import cv2
import time
import numpy as np
import math
import os
import cobotta2 as cbtx
# import wrs.robot_sim.robots.cobotta.cobotta as cbt
from drivers.devices.realsense.realsense_d400s import RealSenseD405
import wrs.basis.robot_math as rm
import matplotlib.pyplot as plt

obs_height = 0.201
obs_pose_list = [np.array([ 2.8031e-01,  3.0601e-02,  2.3300e-01,  1.5708e+00, 0,  1.5708e+00,  5]),
                 # -5
                 np.array([2.8057e-01, 3.0621e-02, 2.3298e-01, 1.5706e+00, 0, 1.4837e+00, 5]),
                 np.array([2.8086e-01, 3.0227e-02, 2.3298e-01, 1.5705e+00, 0, 1.3962e+00, 5]),
                 np.array([2.8115e-01, 3.0099e-02, 2.3298e-01, 1.5705e+00, 0, 1.3091e+00, 5]),
                 np.array([2.8132e-01, 2.9767e-02, 2.3299e-01, 1.5705e+00, 0, 1.2220e+00, 5]),
                 np.array([2.8161e-01, 2.9390e-02, 2.3299e-01, 1.5705e+00, 0, 1.1347e+00, 5]),
                 np.array([2.8159e-01, 2.8688e-02, 2.3297e-01, 1.5707e+00, 0, 1.0474e+00, 5]),
                 np.array([2.8168e-01, 2.8434e-02, 2.3296e-01, 1.5708e+00, 0, 9.6013e-01, 5]),
                 np.array([2.8186e-01, 2.8127e-02, 2.3296e-01, 1.5708e+00, 0, 8.7280e-01, 5]),
                 np.array([2.8214e-01, 2.7833e-02, 2.3293e-01, 1.5708e+00,0, 7.8553e-01, 5]),

                 # +5
                 np.array([2.8012e-01, 3.0733e-02, 2.3303e-01, 1.5706e+00, 0, 1.6579e+00, 5]),
                 np.array([2.7984e-01, 3.0974e-02, 2.3306e-01, 1.5704e+00, 0, 1.7451e+00, 5]),
                 np.array([2.7945e-01, 3.1136e-02, 2.3307e-01, 1.5703e+00, 0, 1.8324e+00, 5]),
                 np.array([2.7907e-01, 3.1277e-02, 2.3309e-01, 1.5702e+00, 0, 1.9196e+00, 5]),
                 np.array([2.7855e-01, 3.1463e-02, 2.3303e-01, 1.5704e+00, 0, 2.0070e+00, 5]),
                 np.array([2.7816e-01, 3.1352e-02, 2.3302e-01, 1.5705e+00, 0, 2.0944e+00, 5]),
                 np.array([2.7777e-01, 3.1371e-02, 2.3301e-01, 1.5705e+00, 0, 2.1815e+00, 5]),
                 np.array([2.7738e-01, 3.1299e-02, 2.3301e-01, 1.5705e+00, 0, 2.2689e+00, 5]),
                 np.array([2.7709e-01, 3.1351e-02, 2.3303e-01, 1.5706e+00, 0, 2.3564e+00, 5]),
                 ]


def capture_save(rs_pipe, save_path, pic_id):
    img = rs_pipe.get_color_img()



def get_spiral_pose_values_list(start_pose, layer, edge_length, plot_spiral=False):
    start_pos = np.asarray(start_pose[:3])
    # start_rotmat = rm.rotmat_from_euler(start_pose[3],start_pose[4],start_pose[5])
    spiral_points = rm.gen_3d_equilateral_verts(start_pos, rm.rotmat_from_axangle(np.array([0, 0, 1]), -math.pi / 2),
                                                layer, edge_length)
    total_num = len(spiral_points)
    if plot_spiral:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(spiral_points[:,0], spiral_points[:,1], spiral_points[:,2])
        plt.show()
    output_pose_list = np.asarray([start_pose] * total_num)
    output_pose_list[:, :3] = spiral_points
    return output_pose_list

def get_3D_spiral_pose_values_list(start_pose, layer, edge_length, z_layer, z_height = 0.003, plot_spiral=False):
    '''2d spiral'''
    start_pos = np.asarray(start_pose[:3])
    # start_rotmat = rm.rotmat_from_euler(start_pose[3],start_pose[4],start_pose[5])
    spiral_points = rm.gen_3d_equilateral_verts(start_pos, rm.rotmat_from_axangle(np.array([0, 0, 1]), -math.pi / 2),
                                                layer, edge_length)
    total_num = len(spiral_points) * z_layer
    
    '''3d augumentation'''
    z_list = np.arange(start_pos[2], start_pos[2] + z_height * (z_layer-1), z_height)
    augumented_spiral = np.tile(spiral_points, (z_layer, 1, 1))
    for index, z in enumerate(z_list):
        augumented_spiral[index, :, :][:,-1] = z
    spiral_points = augumented_spiral.reshape(-1, 3)
    
    if plot_spiral:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(spiral_points[:,0], spiral_points[:,1], spiral_points[:,2], s=10)
        plt.show()
    output_pose_list = np.asarray([start_pose] * total_num)
    output_pose_list[:, :3] = spiral_points
    return output_pose_list


def create_dataset_spiral(robot_x, camera, obs_pose_list, save_path_parent):
    dataset_id = 0
    camera_num = 2
    init_pose = copy.deepcopy(obs_pose_list[0])
    init_pose[2] = obs_height
    # robot_x.move_pose(init_pose)
    
    for i in range(1000):
        # real_id = 50
        print("*" * 100)
        input(f"Press enter to capture Dataset {dataset_id}.")
        for obs_pose in obs_pose_list:
            start_pose = copy.deepcopy(obs_pose)
            start_pose[2] = 0.236
            # robot_x.move_pose(start_pose)
            input(f"Press enter if the robot is in the right position.")
            time.sleep(3)
            
            save_address = save_path_parent + f"dataset{dataset_id}/"
            if not os.path.exists(save_address):
                os.mkdir(save_address)
            for i in range(camera_num):
                tmp_address = save_address + f"camera{i}/"
                if not os.path.exists(tmp_address):
                    os.mkdir(tmp_address)
                print(f"Dataset{dataset_id}, camera{i} saved in: {tmp_address}")

            # input(f"frame{pic_id}")
            
            '''spiral pose img capture'''
            sample_pos_list = get_spiral_pose_values_list(start_pose, layer=4, edge_length=0.0009, plot_spiral=False)
            # sample_pos_list = get_3D_spiral_pose_values_list(start_pose, layer=3, edge_length=0.0009, z_layer=3, z_height=0.003, plot_spiral=False)
            
            for id, pose in enumerate(sample_pos_list):
                robot_x.move_pose(pose)
                capture_save(camera, save_address, id)
                # # print(pose)
                # robot_x.move_pose(pose)
                # img_tip = fcam.get_frame_cut_combine_row()
                # time.sleep(0.5)
                # # fcam.save_combine_real(save_address, id)
                # fcam.save_combine_row_L(save_address, id)
                # fcam.save_combine_original(save_address_L, id)
            dataset_id += 1

        robot_x.move_pose(init_pose)
        print(f"{dataset_id - 1} capture finish.")
        print("*" * 100)



if __name__ == "__main__":
    
    '''init the robot and camera'''
    robot_x = cbtx.Cobotta(host = '192.168.0.11')
    print('current pose:', repr(robot_x.get_pose_values()))
    rs_pipe = RealSenseD405()

    '''warm up the camera'''
    img = rs_pipe.get_color_img()
    img = rs_pipe.get_color_img()
    img = rs_pipe.get_color_img()
    img = rs_pipe.get_color_img()
    img = rs_pipe.get_color_img()
    time.sleep(1)
    
    '''image display'''
    cv2.imshow("img", img)
    np.set_printoptions(precision=4,linewidth=np.inf)  # format the print

    '''parameter setting'''
    parent_path = "/home/lqin/wrs_xinyi/0000_test_programs/surgery_diff/datasets/"
    dataset_id = 0

    '''get and test the manual rotation pose'''
    # for id,obs_pose in enumerate(obs_pose_list):
    #     print(f"np.{repr(obs_pose)}",end=",\n")
    #     start_pose = copy.deepcopy(obs_pose)
    #     start_pose[2] = obs_height
    #     robot_x.move_pose(start_pose)
    #     input(f"pose{id}")
    
    '''get and test the spiral pose'''
    # sample_pos_list = get_spiral_pose_values_list(start_pose, layer=4, edge_length=0.0009, plot_spiral=False)
    # for id,obs_pose in enumerate(sample_pos_list):
    #     print(f"np.{repr(obs_pose)}",end=",\n")
    #     start_pose = copy.deepcopy(obs_pose)
    #     start_pose[2] = obs_height
    #     robot_x.move_pose(start_pose)
    #     input(f"pose{id}")

    print(rm.rotmat_to_euler(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T))
    create_dataset_spiral(robot_x, rs_pipe, obs_pose_list, save_path_parent=parent_path)
    
