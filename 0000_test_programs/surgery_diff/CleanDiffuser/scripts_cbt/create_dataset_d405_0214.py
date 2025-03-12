import copy
import cv2
import time
import numpy as np
import math
import os
# import cobotta2 as cbtx
import wrs.robot_con.cobotta.cobotta_x as cbtx
# import wrs.robot_sim.robots.cobotta.cobotta as cbt
# from drivers.devices.realsense.realsense_d400s import RealSenseD405
import wrs.basis.robot_math as rm
import matplotlib.pyplot as plt

# from xinyi_dosing_volume.create_dataset_d405 import obs_pose_list

# from dosing_volume.create_dataset_d405 import obs_pose_list

np.set_printoptions(precision=4,linewidth=np.inf)  # format the print
# obs_pose_list = [np.array([ 5.2183e-01,  3.4382e-03,  3.8953e-01,  3.1416e+00, -5.0819e-05, -2.8703e+00,  2.6100e+02]),
#                  np. array([ 5.2183e-01,  3.4381e-03,  3.8953e-01,  3.1416e+00, -4.8082e-05, -2.3467e+00,  2.6100e+02]),
#                  ]
# obs_pose_list = [
#                 np. array([ 5.2182e-01,  3.4469e-03,  3.8953e-01,  3.1414e+00, -1.0586e-04,  1.3197e+00,  2.6100e+02]),
#                 np. array([ 5.2183e-01,  3.4451e-03,  3.8954e-01,  3.1414e+00, -9.4282e-05,  1.8435e+00,  2.6100e+02]),
#                 np. array([ 5.2183e-01,  3.4376e-03,  3.8952e-01,  3.1414e+00, -2.6442e-05,  2.3673e+00,  2.6100e+02]),
#                 np. array([ 5.2181e-01,  3.4505e-03,  3.8952e-01,  3.1413e+00, -2.1970e-04,  2.8911e+00,  2.6100e+02]),
#                 np. array([ 5.2186e-01,  3.4412e-03,  3.8953e-01,  3.1412e+00, -4.6732e-05, -2.8684e+00,  2.6100e+02]),
#                 np. array([ 5.2182e-01,  3.4559e-03,  3.8953e-01,  3.1411e+00, -2.5873e-04, -2.3445e+00,  2.6100e+02]),
#                 np. array([ 5.2184e-01,  3.4618e-03,  3.8953e-01,  3.1412e+00, -2.7733e-04, -1.8206e+00,  2.6100e+02]),
#                 np. array([ 5.2182e-01,  3.4586e-03,  3.8951e-01,  3.1410e+00, -2.0709e-04, -1.2965e+00,  2.6100e+02]),
#                 np. array([ 5.2183e-01,  3.4559e-03,  3.8955e-01,  3.1411e+00, -3.3963e-04, -7.7294e-01,  2.6100e+02]),
#                 np. array([ 5.2181e-01,  3.4617e-03,  3.8952e-01,  3.1410e+00, -3.1592e-04, -2.4908e-01,  2.6100e+02])
# ]
# obs_pose_list = [
# np. array([ 5.2104e-01,  2.1877e-03,  3.8939e-01, -3.1413e+00, -3.9413e-04,  1.0111e-02,  2.6100e+02]),
# np. array([ 5.2104e-01,  2.1751e-03,  3.8938e-01, -3.1413e+00, -3.9399e-04,  1.8478e-01,  2.6100e+02]),
# np. array([ 5.2104e-01,  2.1830e-03,  3.8939e-01, -3.1413e+00, -4.1660e-04,  3.5933e-01,  2.6100e+02]),
# np. array([ 5.2104e-01,  2.1815e-03,  3.8938e-01, -3.1414e+00, -4.3889e-04,  5.3410e-01,  2.6100e+02]),
# np. array([ 5.2104e-01,  2.1802e-03,  3.8937e-01, -3.1414e+00, -4.4203e-04,  7.0881e-01,  2.6100e+02]),
# np. array([ 5.2103e-01,  2.1826e-03,  3.8936e-01, -3.1414e+00, -3.9845e-04,  8.8351e-01,  2.6100e+02]),
# np. array([ 5.2103e-01,  2.1829e-03,  3.8936e-01, -3.1413e+00, -4.0020e-04,  1.0582e+00,  2.6100e+02]),
# np. array([ 5.2104e-01,  2.1828e-03,  3.8937e-01, -3.1413e+00, -4.2412e-04,  1.2329e+00,  2.6100e+02]),
# np. array([ 5.2103e-01,  2.1860e-03,  3.8938e-01, -3.1412e+00, -4.6633e-04,  1.4075e+00,  2.6100e+02]),
# np. array([ 5.2104e-01,  2.1993e-03,  3.8940e-01, -3.1411e+00, -5.2131e-04,  1.5822e+00,  2.6100e+02]),
# np. array([ 5.2103e-01,  2.1853e-03,  3.8938e-01, -3.1411e+00, -4.3344e-04,  1.7569e+00,  2.6100e+02]),
# np. array([ 5.2104e-01,  2.2016e-03,  3.8941e-01, -3.1411e+00, -5.0704e-04,  1.9316e+00,  2.6100e+02]),
# np. array([ 5.2102e-01,  2.1955e-03,  3.8939e-01, -3.1410e+00, -4.8971e-04,  2.1063e+00,  2.6100e+02]),
# np. array([ 5.2103e-01,  2.2019e-03,  3.8940e-01, -3.1410e+00, -5.1971e-04,  2.2810e+00,  2.6100e+02]),
# np. array([ 5.2103e-01,  2.1996e-03,  3.8939e-01, -3.1409e+00, -4.4809e-04,  2.4558e+00,  2.6100e+02]),
# np. array([ 5.2103e-01,  2.2008e-03,  3.8939e-01, -3.1409e+00, -5.0579e-04,  2.6305e+00,  2.6100e+02]),
# np. array([ 5.2102e-01,  2.2062e-03,  3.8938e-01, -3.1409e+00, -5.4751e-04,  2.8052e+00,  2.6100e+02]),
# np. array([ 5.2100e-01,  2.2022e-03,  3.8938e-01, -3.1409e+00, -5.9832e-04,  2.9800e+00,  2.6100e+02]),
# np. array([ 5.2101e-01,  2.2010e-03,  3.8939e-01, -3.1408e+00, -4.9843e-04, -3.1285e+00,  2.6100e+02]),
# np. array([ 5.2103e-01,  2.1924e-03,  3.8941e-01, -3.1406e+00, -3.3823e-04, -2.9535e+00,  2.6100e+02]),
# np. array([ 5.2102e-01,  2.1928e-03,  3.8940e-01, -3.1406e+00, -3.8485e-04, -2.7790e+00,  2.6100e+02]),
# np. array([ 5.2101e-01,  2.1954e-03,  3.8939e-01, -3.1406e+00, -4.4438e-04, -2.6043e+00,  2.6100e+02]),
# np. array([ 5.2099e-01,  2.1945e-03,  3.8938e-01, -3.1406e+00, -4.6885e-04, -2.4293e+00,  2.6100e+02]),
# np. array([ 5.2099e-01,  2.1828e-03,  3.8936e-01, -3.1405e+00, -4.4768e-04, -2.2548e+00,  2.6100e+02]),
# np. array([ 5.2099e-01,  2.1792e-03,  3.8937e-01, -3.1405e+00, -4.2276e-04, -2.0802e+00,  2.6100e+02]),
# np. array([ 5.2100e-01,  2.2030e-03,  3.8939e-01, -3.1405e+00, -2.6857e-04, -1.9055e+00,  2.6100e+02]),
# np. array([ 5.2099e-01,  2.1873e-03,  3.8938e-01, -3.1405e+00, -2.8714e-04, -1.7308e+00,  2.6100e+02]),
# np. array([ 5.2100e-01,  2.1870e-03,  3.8938e-01, -3.1404e+00, -2.1051e-04, -1.5562e+00,  2.6100e+02]),
# np. array([ 5.2102e-01,  2.1923e-03,  3.8940e-01, -3.1403e+00, -2.0782e-04, -1.3815e+00,  2.6100e+02]),
# np. array([ 5.2100e-01,  2.1626e-03,  3.8938e-01, -3.1403e+00, -3.3544e-04, -1.2066e+00,  2.6100e+02]),
# np. array([ 5.2101e-01,  2.1831e-03,  3.8939e-01, -3.1402e+00, -2.1948e-04, -1.0322e+00,  2.6100e+02]),
# np. array([ 5.2100e-01,  2.1597e-03,  3.8937e-01, -3.1403e+00, -3.5006e-04, -8.5725e-01,  2.6100e+02]),
# np. array([ 5.2101e-01,  2.1573e-03,  3.8936e-01, -3.1403e+00, -3.7132e-04, -6.8260e-01,  2.6100e+02]),
# np. array([ 5.2099e-01,  2.1702e-03,  3.8935e-01, -3.1403e+00, -1.7136e-04, -5.0819e-01,  2.6100e+02]),
# np. array([ 5.2097e-01,  2.1783e-03,  3.8934e-01, -3.1402e+00,  8.1044e-06, -3.3357e-01,  2.6100e+02])
# ]

obs_pose_list = [
np. array([ 5.2099e-01,  2.1998e-03,  3.8259e-01, -3.1401e+00, -2.4332e-04,  1.5898e-02,  2.6100e+02]),
np. array([ 5.2102e-01,  2.1939e-03,  3.8263e-01, -3.1401e+00, -1.7515e-04,  1.9080e-01,  2.6100e+02]),
np. array([ 5.2102e-01,  2.1938e-03,  3.8264e-01, -3.1400e+00, -2.0459e-04,  3.6519e-01,  2.6100e+02]),
np. array([ 5.2101e-01,  2.1915e-03,  3.8264e-01, -3.1400e+00, -1.8020e-04,  5.3988e-01,  2.6100e+02]),
np. array([ 5.2102e-01,  2.1853e-03,  3.8265e-01, -3.1400e+00, -1.5935e-04,  7.1456e-01,  2.6100e+02]),
np. array([ 5.2103e-01,  2.1744e-03,  3.8263e-01, -3.1400e+00, -1.0813e-04,  8.8928e-01,  2.6100e+02]),
np. array([ 5.2102e-01,  2.1925e-03,  3.8264e-01, -3.1399e+00, -6.3165e-05,  1.0640e+00,  2.6100e+02]),
np. array([ 5.2105e-01,  2.1906e-03,  3.8264e-01, -3.1401e+00, -4.8438e-05,  1.2390e+00,  2.6100e+02]),
np. array([ 5.2105e-01,  2.1964e-03,  3.8263e-01, -3.1400e+00, -1.2073e-05,  1.4135e+00,  2.6100e+02]),
np. array([ 5.2104e-01,  2.2018e-03,  3.8263e-01, -3.1399e+00,  3.3447e-06,  1.5882e+00,  2.6100e+02]),
np. array([ 5.2103e-01,  2.2162e-03,  3.8263e-01, -3.1398e+00, -2.3290e-05,  1.7629e+00,  2.6100e+02]),
np. array([ 5.2103e-01,  2.2023e-03,  3.8262e-01, -3.1398e+00, -2.5905e-05,  1.9376e+00,  2.6100e+02]),
np. array([ 5.2104e-01,  2.2211e-03,  3.8263e-01, -3.1397e+00, -9.2311e-05,  2.1123e+00,  2.6100e+02]),
np. array([ 5.2102e-01,  2.2345e-03,  3.8266e-01, -3.1397e+00, -7.9240e-05,  2.2873e+00,  2.6100e+02]),
np. array([ 5.2105e-01,  2.2320e-03,  3.8267e-01, -3.1398e+00, -1.0985e-04,  2.4617e+00,  2.6100e+02]),
np. array([ 5.2105e-01,  2.2243e-03,  3.8268e-01, -3.1397e+00,  6.1692e-05,  2.6365e+00,  2.6100e+02]),
np. array([ 5.2109e-01,  2.2241e-03,  3.8272e-01, -3.1398e+00,  2.0788e-04,  2.8112e+00,  2.6100e+02]),
np. array([ 5.2105e-01,  2.2284e-03,  3.8268e-01, -3.1397e+00,  1.8398e-04,  2.9859e+00,  2.6100e+02]),
np. array([ 5.2104e-01,  2.2310e-03,  3.8268e-01, -3.1396e+00,  2.0042e-04, -3.1225e+00,  2.6100e+02]),
np. array([ 5.2103e-01,  2.2291e-03,  3.8265e-01, -3.1394e+00,  1.9103e-04, -2.9477e+00,  2.6100e+02]),
np. array([ 5.2102e-01,  2.2285e-03,  3.8265e-01, -3.1395e+00,  2.0419e-04, -2.7731e+00,  2.6100e+02]),
np. array([ 5.2101e-01,  2.2176e-03,  3.8264e-01, -3.1395e+00,  2.1643e-04, -2.5984e+00,  2.6100e+02]),
np. array([ 5.2102e-01,  2.2174e-03,  3.8265e-01, -3.1395e+00,  2.7647e-04, -2.4237e+00,  2.6100e+02]),
np. array([ 5.2099e-01,  2.2144e-03,  3.8264e-01, -3.1396e+00,  2.1758e-04, -2.2490e+00,  2.6100e+02]),
np. array([ 5.2100e-01,  2.2041e-03,  3.8264e-01, -3.1395e+00,  2.9308e-04, -2.0744e+00,  2.6100e+02]),
np. array([ 5.2098e-01,  2.2027e-03,  3.8264e-01, -3.1396e+00,  2.8930e-04, -1.8997e+00,  2.6100e+02]),
np. array([ 5.2099e-01,  2.1993e-03,  3.8265e-01, -3.1395e+00,  3.3764e-04, -1.7250e+00,  2.6100e+02]),
np. array([ 5.2099e-01,  2.1958e-03,  3.8266e-01, -3.1394e+00,  3.5039e-04, -1.5504e+00,  2.6100e+02]),
np. array([ 5.2099e-01,  2.1885e-03,  3.8266e-01, -3.1393e+00,  3.4454e-04, -1.3757e+00,  2.6100e+02]),
np. array([ 5.2101e-01,  2.2003e-03,  3.8267e-01, -3.1392e+00,  2.7756e-04, -1.2010e+00,  2.6100e+02]),
np. array([ 5.2102e-01,  2.1707e-03,  3.8263e-01, -3.1392e+00,  2.6310e-04, -1.0261e+00,  2.6100e+02]),
np. array([ 5.2098e-01,  2.1977e-03,  3.8263e-01, -3.1393e+00,  4.0522e-04, -8.5164e-01,  2.6100e+02]),
np. array([ 5.2095e-01,  2.1828e-03,  3.8257e-01, -3.1395e+00,  4.7659e-04, -6.7690e-01,  2.6100e+02]),
np. array([ 5.2095e-01,  2.1817e-03,  3.8258e-01, -3.1394e+00,  6.1634e-04, -5.0227e-01,  2.6100e+02]),
np. array([ 5.2094e-01,  2.1823e-03,  3.8258e-01, -3.1393e+00,  7.1165e-04, -3.2770e-01,  2.6100e+02])
]

def capture_save(camera, save_path, pic_id):
    for id, single_camera in enumerate(camera):
        ret, frame = single_camera.read()
        if ret:
            '''if crop'''
            x1, y1 = 240,195
            x2, y2 = 320, 275
            frame = frame[y1:y2, x1:x2, :]

            # cv2.imshow('captured frame', frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            camera_path = os.path.join(save_path, f"camera{id}")
            img_path = os.path.join(camera_path,  f'image{pic_id}.jpg')
            cv2.imwrite(img_path, frame)
            print(f'image {pic_id} camera {id} saved in {img_path}')
        else:
            print('no image captured')


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

def junbo_get_spiral_pose_values_list(start_pose, layer, edge_length):
    start_pos = np.asarray(start_pose[:3])
    # start_rotmat = rm.rotmat_from_euler(start_pose[3],start_pose[4],start_pose[5])
    spiral_points = rm.gen_3d_equilateral_verts(start_pos, rm.rotmat_from_axangle(np.array([0, 0, 1]), -math.pi / 2),
                                                layer, edge_length)
    total_num = len(spiral_points)
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
    camera_num = len(camera)
    obs_height = obs_pose_list[0][2]
    init_pose = copy.deepcopy(obs_pose_list[0])
    init_pose[2] = obs_height
    robot_x.move_pose(init_pose)

    for i in range(1):
        # real_id = 50
        print("*" * 100)
        input(f"Press enter to capture Dataset {i}.")
        for obs_pose_id, obs_pose in enumerate(obs_pose_list):
            start_pose = copy.deepcopy(obs_pose)
            start_pose[2] = obs_height
            robot_x.move_pose(start_pose)
            # input(f"Press enter if the robot is in the right position.")
            time.sleep(1)

            save_address = os.path.join(save_path_parent, f"dataset{obs_pose_id}")
            if not os.path.exists(save_address):
                os.mkdir(save_address)
            for j in range(camera_num):
                tmp_address = os.path.join(save_address,  f"camera{j}")
                if not os.path.exists(tmp_address):
                    os.mkdir(tmp_address)
                print(f"Dataset{obs_pose_id}, camera{j} saved in: {tmp_address}")

            # input(f"frame{pic_id}")

            '''spiral pose img capture'''
            sample_pos_list = get_spiral_pose_values_list(start_pose, layer=5, edge_length=0.001, plot_spiral=False)
            # sample_pos_list = get_3D_spiral_pose_values_list(start_pose, layer=3, edge_length=0.001, z_layer=3, z_height=0.001, plot_spiral=False)
            
            for id, pose in enumerate(sample_pos_list):
                print(pose)
                robot_x.move_pose(pose)
                # time.sleep(0.5)
                capture_save(camera, save_address, id)
                # # print(pose)
                # robot_x.move_pose(pose)
                # img_tip = fcam.get_frame_cut_combine_row()
                # time.sleep(0.5)
                # # fcam.save_combine_real(save_address, id)
                # fcam.save_combine_row_L(save_address, id)
                # fcam.save_combine_original(save_address_L, id)

        robot_x.move_pose(init_pose)
        print(f"{i - 1} capture finish.")
        print("*" * 100)



if __name__ == "__main__":
    
    '''init the robot and camera'''
    robot_x = cbtx.CobottaX(host='169.254.1.3')
    print(repr(robot_x.get_pose_values()))
    current_file_dir = os.path.dirname(__file__)
    parent_path = os.path.join(os.path.dirname(current_file_dir), "datasets")
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
    print('datasets parent path is: ', parent_path)
    pose = obs_pose_list[0]
    robot_x.move_pose(pose)

    '''camera testing'''
    rgb_camera = []
    rgb_camera.append(cv2.VideoCapture(0))
    rgb_camera.append(cv2.VideoCapture(1))

    # for camera in rgb_camera:
    #     ret, frame = camera.read()
    #     if ret:
    #         cv2.imshow('captured frame', frame)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #     else:
    #         print('no image captured')
    #
    #     time.sleep(1)


    '''get and test the manual rotation pose'''
    # for id,obs_pose in enumerate(obs_pose_list):
    #     print(f"np.{repr(obs_pose)}",end=",\n")
    #     start_pose = copy.deepcopy(obs_pose)
    #     start_pose[2] = obs_height
    #     robot_x.move_pose(start_pose)
    #     input(f"pose{id}")

    '''get and test the spiral pose'''
    # sample_pos_list = get_spiral_pose_values_list(start_pose=obs_pose_list[0],
    #                                               layer=5, edge_length=0.001, plot_spiral=True)
    # sample_pos_list = get_3D_spiral_pose_values_list(start_pose=obs_pose_list[0],
    #                                               layer=5, z_layer=5, edge_length=0.003, plot_spiral=True)
    # sample_pos_list = get_3D_spiral_pose_values_list(start_pose, layer=3, edge_length=0.001, z_layer=2, z_height=0.001,
    #                                                  plot_spiral=True)
    # obs_height = obs_pose_list[0][2]
    # for id,obs_pose in enumerate(sample_pos_list):
    #     print(f"np.{repr(obs_pose)}",end=",\n")
    #     start_pose = copy.deepcopy(obs_pose)
    #     start_pose[2] = obs_height
    #     robot_x.move_pose(start_pose)
    #     time.sleep(0.2)
    #     print(f"pose{id}")

    # print(rm.rotmat_to_euler(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T))
    create_dataset_spiral(robot_x, rgb_camera, obs_pose_list, save_path_parent=parent_path)
    
