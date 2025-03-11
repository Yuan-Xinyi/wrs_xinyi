import wrs.robot_con.xarm_lite6.xarm_lite6_x as xarm_x
import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm_s
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm

import cv2
import time
import yaml
import numpy as np
import zarr
import os

current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))

'''load the config file'''
with open("0000_test_programs/surgery_diff/scripts/gendata_config.yaml", "r") as file:
    config = yaml.safe_load(file)

'''init the camera'''
rgb_camera = []
cam_idx = config['camera_idx']
rgb_camera.append(cv2.VideoCapture(cam_idx[0]))
rgb_camera.append(cv2.VideoCapture(cam_idx[1]))

'''img capture test'''
for camera in rgb_camera:
    ret, frame = camera.read()
    if not ret:
        print('no image captured')

    h, w, _ = frame.shape
    crop_size = config['crop_size']
    cv2.namedWindow("Cropped Image")

    def on_change(val):
        pass

    cv2.createTrackbar("Center X", "Cropped Image", w // 2, w, on_change)
    cv2.createTrackbar("Center Y", "Cropped Image", h // 2, h, on_change)


    while True:
        ret, frame = camera.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        center_x = cv2.getTrackbarPos("Center X", "Cropped Image")
        center_y = cv2.getTrackbarPos("Center Y", "Cropped Image")

        x1 = max(center_x - crop_size // 2, 0)
        x2 = min(center_x + crop_size // 2, w)
        y1 = max(center_y - crop_size // 2, 0)
        y2 = min(center_y + crop_size // 2, h)

        cropped = frame[y1:y2, x1:x2]
        print(camera, center_x, center_y, x1, x2, y1, y2)

        cv2.imshow("Cropped Image", cropped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
