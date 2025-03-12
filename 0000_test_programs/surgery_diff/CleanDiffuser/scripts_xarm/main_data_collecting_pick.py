
from hand_detector_wilor import HandDetector_wilor
from data_class import animation,handdata,Data
from keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

import pickle
from pathlib import Path
import pyrealsense2 as rs
import multiprocessing
import torch
import time
import cv2
import numpy as np
from queue import Empty
from loguru import logger
from wrs import wd, rm, ur3d, rrtc, mgm, mcm
import wrs.robot_con.ur.ur3_rtq85_x as ur3

from ultralytics import YOLO
from precise_sleep import precise_wait
from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
import struct

WIDTH = 1280
HEIGHT = 720



def wilor_to_wrs(queue1: multiprocessing.Queue):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, model_cfg = load_wilor(checkpoint_path='./pretrained_models/wilor_final.ckpt',
                                  cfg_path='./pretrained_models/model_config.yaml')
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)  ##3Dグラフィックを表示するためのオブジェクト
    model = model.to(device)
    model.eval()

    detector = YOLO(f'./pretrained_models/detector.pt').to(device)
    hand_detector = HandDetector_wilor(device, model, model_cfg, renderer, detector, hand_type="Right",
                                       detect_hand_num=1, WIDTH=1280, HEIGHT=720)

    # RealSenseカメラの設定
    pipeline_hand = rs.pipeline()
    config_hand = rs.config()
    config_hand.enable_device('243122072240')

    # カメラのストリームを設定（RGBと深度）
    config_hand.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)  # 30は毎秒フレーム数
    config_hand.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)


    # spatial_filterのパラメータ
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 1)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    # hole_filling_filterのパラメータ
    hole_filling = rs.hole_filling_filter()
    # disparity
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    ##depthとcolorの画角がずれないようにalignを生成
    align = rs.align(rs.stream.color)
    # パイプラインを開始
    pipeline_hand.start(config_hand)
    try:
        while True:
            # 1つ目のフレームを取得
            frames = pipeline_hand.wait_for_frames()
            aligned_frames1 = align.process(frames)
            color_frame = aligned_frames1.get_color_frame()
            if not color_frame:
                continue

            # カラーカメラの内部パラメータを取得
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

            ##深度フレームの取得
            depth_frame = aligned_frames1.get_depth_frame()
            if not depth_frame:
                continue


            # フィルタ処理
            filter_frame = spatial.process(depth_frame)
            filter_frame = disparity_to_depth.process(filter_frame)
            filter_frame = hole_filling.process(filter_frame)
            result_depth_frame = filter_frame.as_depth_frame()

            if not result_depth_frame:
                continue

            # BGR画像をNumPy配列に変換
            frame = np.asanyarray(color_frame.get_data())



            img_vis, detected_hand_count, reconstructions = hand_detector.run_wilow_model(frame, IoU_threshold=0.3)
            detected_time = time.time()
            # output_img, text = hand_detector.render_reconstruction(img_vis, detected_hand_count, reconstructions)

            # 結果を表示
            cv2.imshow('Hand Tracking', img_vis)

            if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
                break

            if detected_hand_count == 1:
                kpts_2d = hand_detector.keypoints_2d_on_image(reconstructions['joints'][0], reconstructions['cam_t'][0],
                                                              reconstructions['focal'])
                pos_3d_wrist = hand_detector.calib(kpts_2d[0].tolist(), result_depth_frame, intrinsics)
                index_finger_mcp=np.array(reconstructions['joints'][0][5].tolist())
                pos_3d = hand_detector.coor_trans_from_rs_to_wrs(pos_3d_wrist+index_finger_mcp)
                jaw_width = hand_detector.distance_between_fingers_normalization(
                    reconstructions['joints'][0][4].tolist(), reconstructions['joints'][0][8].tolist())
                # rotmat = reconstructions['rotmat'][0]
                # rotmat = hand_detector.rotmat_to_wrs(rotmat)
                rotmat = hand_detector.fixed_rotmat_to_wrs(reconstructions['joints'][0])
                q = [pos_3d, rotmat, jaw_width, detected_time]
            else:
                q = None

            queue1.put(q)




    finally:
        # パイプラインを停止
        pipeline_hand.stop()
        cv2.destroyAllWindows()

def get_frame_robot(queue2: multiprocessing.Queue):
    # RealSenseカメラの設定
    pipeline_robot = rs.pipeline()
    config_robot = rs.config()
    config_robot.enable_device('408322072412')

    # カメラのストリームを設定（RGBと深度）
    config_robot.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 60)  # 30は毎秒フレーム数
    pipeline_robot.start(config_robot)

    try:
        while True:
            frame_robot = pipeline_robot.wait_for_frames(timeout_ms=10000)
            color_frame_robot = frame_robot.get_color_frame()
            if not color_frame_robot:
                continue
            frame_robot_numpy = np.asanyarray(color_frame_robot.get_data())

            queue2.put(frame_robot_numpy)
    finally:
        pipeline_robot.stop()



def get_frame_robot_wrist(queue3: multiprocessing.Queue):
    # RealSenseカメラの設定
    pipeline_robot_wrist= rs.pipeline()
    config_robot_wrist = rs.config()
    config_robot_wrist.enable_device('243122071603')

    # カメラのストリームを設定（RGBと深度）
    config_robot_wrist.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 60)  # 30は毎秒フレーム数
    pipeline_robot_wrist.start(config_robot_wrist)

    try:
        while True:
            frame_robot_wrist = pipeline_robot_wrist.wait_for_frames(timeout_ms=10000)
            color_frame_robot_wrist = frame_robot_wrist.get_color_frame()
            if not color_frame_robot_wrist:
                continue
            frame_robot_wrist_numpy = np.asanyarray(color_frame_robot_wrist.get_data())

            queue3.put(frame_robot_wrist_numpy)
            t2 = time.time()
    finally:
        pipeline_robot_wrist.stop()





def wrs(queue1: multiprocessing.Queue,queue2: multiprocessing.Queue,queue3: multiprocessing.Queue):
    file_path="grouping_and_picking_two_images_1_22.pkl"
    file_path = Path(file_path)  # Pathオブジェクトに変換


    # ファイルが存在する場合、読み込む
    if file_path.exists():
        with file_path.open("rb") as f:
            try:
                train_data = pickle.load(f)
                print("既存のデータを読み込みました。")
                count = 0
                count_true=False
                for value in train_data["dones"]:
                    if value is True and count_true is True:
                        print("error!!!!!")
                    count_true=value
                    if value is True:
                        count += 1
                print(f"episode_num:{count}")
                # print(f"actions:{train_data['actions']}")

            except EOFError:
                # ファイルが空の場合の初期化
                train_data = {"observations":{"rgb_robot":list(),
                                              "rgb_robot_wrist":list(),
                                              "state":list(),},
                              "actions":list(),
                              "rewards":list(),
                              "dones":list(),
                              }
    else:
        print("ファイルが存在しないため、新規作成します。")
        train_data = {"observations":{"rgb_robot":list(),
                                      "rgb_robot_wrist":list(),
                                      "state":list(),},
                              "actions":list(),
                              "rewards":list(),
                              "dones":list(),
                              }

    this_episode_dict={"observations":{"rgb_robot":list(),
                                       "rgb_robot_wrist":list(),
                                       "state":list(),},
                              "actions":list(),
                              "rewards":list(),
                              "dones":list(),
                              }


    robot = ur3d.UR3Dual()
    robot.use_rgt()

    # executor
    robotx = ur3.UR3Rtq85X(robot_ip="10.2.0.51", pc_ip="10.2.0.101")


    start_conf = np.asarray(robotx.arm.getj())

    start_pos, start_rotmat = robot.fk(start_conf, toggle_jacobian=False)

    print(f"start_pos:{start_pos}")

    robot.goto_given_conf(jnt_values=start_conf)
    animation_data = animation(start_pos, start_rotmat, start_conf)
    hand_data = handdata()




    recording=False
    is_recorded=False
    last_record=False
    gripper_action=0
    robotx._arm.send_program(robotx._modern_driver_urscript)
    rbt_socket, rbt_socket_addr = robotx._pc_server_socket.accept()
    animation_data.current_jnt_values = np.asarray(robotx.get_jnt_values())
    jnt_list=[animation_data.current_jnt_values,animation_data.current_jnt_values]
    value = 0
    # t_start = time.perf_counter()
    iter_idx = 0
    command_latency = 0.1
    dt = 0.1
    with KeystrokeCounter() as key_counter:
        try:
            while True:
                # calculate timing
                t_cycle_end = time.monotonic() + dt  ##indexが終わるまでの時間
                t_sample = t_cycle_end - command_latency
                # t_command_target = t_cycle_end + dt
                # t1 = time.time()


                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='s'):
                        recording = True
                        print("episode started")
                        key_counter.clear()
                    elif key_stroke == KeyCode(char='q'):
                        last_record = True
                        is_recorded=True
                        print("episode end")

                q = queue1.get(timeout=10)


                precise_wait(t_sample)
                if q is not None:
                    if q[0] is not None and q[1] is not None and q[2] is not None and q[3] is not None:
                        hand_data.hand_pos = q[0]
                        hand_data.hand_rotmat = q[1]
                        hand_data.jaw_width = q[2]
                        hand_data.detected_time = q[3]
                        # print(f"hand_data.detected_time:{hand_data.detected_time}")
                        if animation_data.pos_error(hand_data.hand_pos, animation_data.tgt_rotmat) > 0.000:
                            animation_data.tgt_pos = hand_data.hand_pos
                        if animation_data.rotmat_error(hand_data.hand_rotmat, animation_data.tgt_rotmat) > 0.000:
                            animation_data.tgt_rotmat = hand_data.hand_rotmat
                        animation_data.jaw_width = hand_data.jaw_width
                        if animation_data.tgt_pos[2] < 1.158:
                            animation_data.tgt_pos[2] = 1.158
                            if animation_data.tgt_pos[2] > 1.458:
                                animation_data.tgt_pos[2] = 1.458

                    frame_robot_numpy = queue2.get()
                    frame_robot_wrist_numpy = queue3.get()

                    output_image = np.hstack((frame_robot_numpy, frame_robot_wrist_numpy))

                    # 画像を表示
                    cv2.imshow('Robot', output_image)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
                        break

                    animation_data.current_jnt_values = np.asarray(robotx.get_jnt_values())

                    if animation_data.jaw_width < 0.07 and animation_data.current_jaw_width == 0:
                        # print("close_griper")
                        rbt_socket.close()
                        robotx.close_gripper(speed_percentage=100, force_percentage=50, finger_distance=20)
                        robotx._arm.send_program(robotx._modern_driver_urscript)
                        rbt_socket, rbt_socket_addr = robotx._pc_server_socket.accept()
                        observation_state = np.append(animation_data.current_jnt_values, animation_data.current_jaw_width)
                        animation_data.current_jaw_width = 1
                        gripper_action = 1
                        action = np.append(animation_data.current_jnt_values, gripper_action)

                    elif animation_data.jaw_width >= 0.12 and animation_data.current_jaw_width == 1:
                        # print("open_griper")
                        rbt_socket.close()
                        robotx.open_gripper(speed_percentage=100, force_percentage=50, finger_distance=77)
                        robotx._arm.send_program(robotx._modern_driver_urscript)
                        rbt_socket, rbt_socket_addr = robotx._pc_server_socket.accept()
                        observation_state = np.append(animation_data.current_jnt_values, animation_data.current_jaw_width)
                        animation_data.current_jaw_width = 0
                        gripper_action = 0
                        action = np.append(animation_data.current_jnt_values, gripper_action)

                    else:
                        # print(f"move")
                        # ikからもとめた関節位置
                        jnt_values = robot.ik(animation_data.tgt_pos, animation_data.tgt_rotmat,
                                              seed_jnt_values=animation_data.current_jnt_values, toggle_dbg=False)
                        if jnt_values is None:
                            print("No IK solution found!")
                            animation_data.next_jnt_values = animation_data.current_jnt_values
                        else:
                            animation_data.next_jnt_values = jnt_values
                            # robot.change_jaw_width(animation_data.jaw_width)
                            jnt_list = [animation_data.current_jnt_values, animation_data.next_jnt_values]
                        robotx.move_jspace_path_realtime(rbt_socket, path=jnt_list)
                        data = rbt_socket.recv(4)
                        observation_state = np.append(jnt_list[0], animation_data.current_jaw_width)
                        action = np.append(jnt_list[1], gripper_action)
                    # print(f"observation_state:{observation_state}")
                    # print(f"action :{action}")
                # データの取得
                if recording is True:
                    this_episode_dict["observations"]["rgb_robot_wrist"].append(frame_robot_wrist_numpy)
                    this_episode_dict["observations"]["rgb_robot"].append(frame_robot_numpy)
                    this_episode_dict["observations"]["state"].append(observation_state)
                    this_episode_dict["actions"].append(action)
                    reward = 0
                    this_episode_dict["rewards"].append(reward)
                    done = False
                    if last_record is True:
                        done = True
                        last_record=False
                        print("last_record!")
                    this_episode_dict["dones"].append(done)


                if is_recorded is True and recording is True:
                    train_data["observations"]["rgb_robot"].extend(this_episode_dict["observations"]["rgb_robot"])
                    train_data["observations"]["rgb_robot_wrist"].extend(this_episode_dict["observations"]["rgb_robot_wrist"])
                    train_data["observations"]["state"].extend(this_episode_dict["observations"]["state"])
                    train_data["actions"].extend(this_episode_dict["actions"])
                    train_data["rewards"].extend(this_episode_dict["rewards"])
                    train_data["dones"].extend(this_episode_dict["dones"])
                    with file_path.open("wb") as f:
                        pickle.dump(train_data, f)
                    print(f"データを更新して保存しました")
                    is_recorded=False
                    recording=False

                precise_wait(t_cycle_end)
                iter_idx += 1
                # t2=time.time()
                # print(f"周期：{t2-t1}")




        except Empty:
            logger.error(f"Fail to fetch image from camera in 10 secs. Please check your web camera device.")
            buf = struct.pack('!iiiiiii', -2, -2, -2, -2, -2, -2, -2)
            # パイプラインを停止
            # pipeline_robot.stop()
            # pipeline_robot_wrist.stop()
            cv2.destroyAllWindows()
            rbt_socket.send(buf)
            rbt_socket.close()

        finally:
            buf = struct.pack('!iiiiiii', -2, -2, -2, -2, -2, -2, -2)
            rbt_socket.send(buf)
            rbt_socket.close()
            # pipeline_robot.stop()
            # pipeline_robot_wrist.stop()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    queue1 = multiprocessing.Queue(maxsize=1)
    queue2 = multiprocessing.Queue(maxsize=1)
    queue3 = multiprocessing.Queue(maxsize=1)

    process1 = multiprocessing.Process(target=wilor_to_wrs, args=(queue1,))
    process2 = multiprocessing.Process(target=get_frame_robot, args=(queue2,))
    process3= multiprocessing.Process(target=get_frame_robot_wrist, args=(queue3,))
    process4 = multiprocessing.Process(target=wrs, args=(queue1,queue2,queue3))

    process1.start()
    process2.start()
    process3.start()
    process4.start()

    process1.join()
    process2.join()
    process3.join()
    process4.join()

    print("done")
