"""
Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20241202 Osaka Univ.
"""

import os
import numpy as np
import sys
from tqdm import tqdm
sys.path.append("E:/Qin/wrs")
import time
import wrs.basis.robot_math as rm
import wrs.basis.constant as ct
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import pickle
from wrs.HuGroup_Qin.robot_sim.nova2_wrsv3gripper_regrasp_env import nova2_gripper_v3
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v3 as wrs_gripper_v3
from wrs.grasping.reasoner import GraspReasoner
from wrs.manipulation.placement.flatsurface import FSReferencePoses
from scipy.stats import qmc  # 添加LHS采样器导入
import wrs.modeling.constant as const
from panda3d.core import Filename, NodePath, CollisionBox, Point3, CollisionNode
from scipy.spatial.transform import Rotation as R


BASE_PATH = r"E:\Qin\wrs\wrs\HuGroup_Qin\Shared_grasp_project\grasps\Bottle"
GRASP_DATA_PREFIX = "bottle_grasp_"
SAVE_PREFIX = "feasible_grasp_obstacle"

OBJ_INIT_POS = np.array([0.0, 0.0, 0.0])
OBJ_INIT_ROT = rm.rotmat_from_euler(0, 0, 0)
SAVE_BATCH_SIZE = 500
TOTAL_ITERATIONS = int(1e3)
GRASP_IDS = [109]  # 添加固定的grasp_id列表


# 添加空间参数作为全局常量
SPACE_LENGTH = 0.5
SPACE_WIDTH = 0.5
SPACE_HEIGHT = 0.6
GRID_SIZE = 0.02

# 添加新的全局常量
SPACE_Z_OFFSET = 0.09  # 空间在z轴上的偏移量
NOISE_POINTS_PER_OBSTACLE = 20  # 每个障碍物周围的噪声点数量
NOISE_MAX_DISTANCE = 0.02      # 噪声点最大距离（米）
NOISE_MIN_DISTANCE = 0.0      # 噪声点最小距离（米）
NOISE_SIZE = 0.01              # 噪声点的大小

def get_file_paths(grasp_id):
    """根据grasp_id生成对应的文件路径"""
    grasp_data_path = os.path.join(BASE_PATH, f"{GRASP_DATA_PREFIX}{grasp_id}.pickle")
    save_path = os.path.join(BASE_PATH, f"{SAVE_PREFIX}_{grasp_id}.pickle")
    return grasp_data_path, save_path


def grasp_load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def env_setup():
    robot = nova2_gripper_v3(enable_cc=True)
    # init_jnv = np.array([180, -18.1839, 136.3675, -28.1078, -90.09, -350.7043]) * np.pi / 180
    # robot.goto_given_conf(jnt_values=init_jnv)
    # robot.gen_meshmodel(alpha=1, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    # robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    # robot.show_cdprim()
    return robot


def obj_setup(name, pos, rotmat, rgb=None, alpha=None):
    """设置物体"""
    obj_cmodel = mcm.CollisionModel(name=name, rgb=rgb, alpha=alpha, 
                                   initor=r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl")
    obj_cmodel.pos = pos
    obj_cmodel.rotmat = rotmat
    obj_cmodel.attach_to(base)
    obj_cmodel.show_cdprim()
    obj_cmodel.show_local_frame()
    return obj_cmodel


def show_grasp(robot, grasp_collection, obj_pos, obj_rotmat):
    for grasp in grasp_collection:
        robot.end_effector.grip_at_by_pose(jaw_center_pos=grasp.ac_pos, jaw_center_rotmat=grasp.ac_rotmat,
                                jaw_width=grasp.ee_values)
        robot.end_effector.gen_meshmodel(rgb=ct.carrot_orange, alpha=.4, toggle_cdprim=True).attach_to(base)
    # robot.show_cdprim()
    # obj_setup(name='obj_{}'.format(time.time()), pos=obj_pos, rotmat=obj_rotmat)


def append_save(data, path):
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        return

    with open(path, 'rb') as f:
        existing_data = pickle.load(f)
    existing_data.extend(data)

    with open(path, 'wb') as f:
        pickle.dump(existing_data, f)


def obstacle_setup(length, width, height, pos):
    """设置障碍物"""
    planecm = mcm.gen_box(xyz_lengths=[length, width, height],
                         pos=pos,
                         rgb=ct.chinese_red, alpha=.8)
    planecm.attach_to(base)
    # planecm.show_local_frame()
    return planecm


def local_space_show(length=0.4, width=0.4, height=0.6):
    """显示采样空间的边界框，考虑z轴偏移"""
    # 考虑z轴偏移，调整顶点坐标
    vertices_array = np.array([[length/2, width/2, -height/2 + SPACE_Z_OFFSET],
                              [length/2, -width/2, -height/2 + SPACE_Z_OFFSET],
                              [-length/2, -width/2, -height/2 + SPACE_Z_OFFSET],
                              [-length/2, width/2, -height/2 + SPACE_Z_OFFSET],
                              [length/2, width/2, -height/2 + SPACE_Z_OFFSET],
                              [length/2, width/2, height/2 + SPACE_Z_OFFSET],
                              [length/2, -width/2, height/2 + SPACE_Z_OFFSET],
                              [-length/2, -width/2, height/2 + SPACE_Z_OFFSET],
                              [-length/2, width/2, height/2 + SPACE_Z_OFFSET]])
    
    def vertices(idx):
        return vertices_array[idx]
    
    spacecm = mgm.gen_wireframe(vertices=vertices,
                                edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0],  # 底部边缘
                                              [5, 6], [6, 7], [7, 8], [8, 5],  # 顶部边缘
                                              [0, 5], [1, 6], [2, 7], [3, 8]   # 连接边
                                             ]),
                                thickness=0.001,
                                rgb=np.array([0, 0, 0]),
                                alpha=0.3)
    spacecm.attach_to(base)
    return spacecm


def generate_noise_around_obstacle(obstacle_pos, obstacle_size):
    """在障碍物表面生成随机噪声点"""
    noise_points = []
    
    for _ in range(NOISE_POINTS_PER_OBSTACLE):
        # 随机选择一个面（0-5分别代表：前、后、左、右、上、下）
        face = np.random.randint(0, 6)
        
        # 计算在选定面上的随机位置
        if face < 4:  # 侧面
            # 随机选择y或z（对于前后面）或x或z（对于左右面）
            length = obstacle_size[0]
            width = obstacle_size[1]
            height = obstacle_size[2]
            
            if face < 2:  # 前后面
                x = (length/2) * (1 if face == 0 else -1)  # 前面为正，后面为负
                y = np.random.uniform(-width/2, width/2)
                z = np.random.uniform(-height/2, height/2)
            else:  # 左右面
                x = np.random.uniform(-length/2, length/2)
                y = (width/2) * (1 if face == 2 else -1)  # 左面为正，右面为负
                z = np.random.uniform(-height/2, height/2)
        else:  # 上下面
            x = np.random.uniform(-obstacle_size[0]/2, obstacle_size[0]/2)
            y = np.random.uniform(-obstacle_size[1]/2, obstacle_size[1]/2)
            z = (obstacle_size[2]/2) * (1 if face == 4 else -1)  # 上面为正，下面为负
        
        # 添加小的随机偏移（沿表面法向量方向）
        normal_offset = np.random.uniform(NOISE_MIN_DISTANCE, NOISE_MAX_DISTANCE)
        if face < 4:
            if face < 2:
                x += normal_offset * (1 if face == 0 else -1)
            else:
                y += normal_offset * (1 if face == 2 else -1)
        else:
            z += normal_offset * (1 if face == 4 else -1)
        
        # 计算最终噪声点位置（相对于障碍物中心）
        relative_pos = np.array([x, y, z])
        noise_pos = obstacle_pos + relative_pos
        
        # 创建噪声点的碰撞模型
        noise_point = obstacle_setup(
            length=NOISE_SIZE,
            width=NOISE_SIZE,
            height=NOISE_SIZE,
            pos=noise_pos
        )
        noise_points.append(noise_point)
    
    return noise_points


def generate_grid_obstacles(n_samples=10):
    """生成网格化的障碍物和噪声点"""
    # 使用全局常量
    space_length = SPACE_LENGTH
    space_width = SPACE_WIDTH
    space_height = SPACE_HEIGHT
    grid_size = GRID_SIZE
    
    # 计算网格数量
    nx = int(space_length / grid_size)
    ny = int(space_width / grid_size)
    nz = int(space_height / grid_size)

    # 设置最大尺寸限制（以网格数量为单位）
    max_length_grids = min(nx // 2, 15)  # 最大长度为空间的1/2或10个网格
    max_width_grids = min(ny // 2, 15)   # 最大宽度为空间的1/2或10个网格
    max_height_grids = min(nz // 2, 10)   # 最大高度为空间的1/2或8个网格

    obstacle_list = []
    all_noise_points = []
    
    for _ in range(n_samples):
        length_grids = np.random.randint(1, max_length_grids)
        width_grids = np.random.randint(1, max_width_grids)
        height_grids = np.random.randint(1, max_height_grids)
        
        length = length_grids * grid_size
        width = width_grids * grid_size
        height = height_grids * grid_size
        
        max_x = (space_length - length) / 2
        max_y = (space_width - width) / 2
        max_z = space_height/2 - height + SPACE_Z_OFFSET
        min_z = -space_height/2 + SPACE_Z_OFFSET
        
        if max_x < -max_x or max_y < -max_y:
            continue
            
        pos_x = np.random.uniform(-max_x, max_x)
        pos_y = np.random.uniform(-max_y, max_y)
        pos_z = np.random.uniform(min_z + height/2, max_z)
        
        pos = np.array([pos_x, pos_y, pos_z])
        obstacle = obstacle_setup(length=length, width=width, height=height, pos=pos)
        obstacle_list.append(obstacle)
        
        # 为每个障碍物生成噪声点
        noise_points = generate_noise_around_obstacle(pos, np.array([length, width, height]))
        all_noise_points.extend(noise_points)

    return obstacle_list + all_noise_points


def voxelize_scene_in_memory(voxel_size=GRID_SIZE):
    """通过碰撞模型的Box primitive来分别体素化物体、障碍物和噪声点"""
    # print("\n开始收集场景中的碰撞模型...")
    
    # 修改场景边界，考虑z轴偏移
    scene_min = np.array([-SPACE_LENGTH/2, -SPACE_WIDTH/2, -SPACE_HEIGHT/2 + SPACE_Z_OFFSET])
    scene_max = np.array([SPACE_LENGTH/2, SPACE_WIDTH/2, SPACE_HEIGHT/2 + SPACE_Z_OFFSET])
    
    object_boxes = []
    obstacle_boxes = []
    noise_boxes = []
    
    # 1. 优化：预先分配corners数组
    corners = np.empty((8, 3))
    
    # 收集所有碰撞模型
    for node in base.render.getChildren():
        collision_model = node.getPythonTag("collision_model")
        if collision_model is not None and isinstance(collision_model, mcm.CollisionModel):
            mesh = collision_model.trm_mesh
            mesh_min, mesh_max = mesh.bounds
            
            # 2. 优化：直接使用numpy操作填充corners数组
            corners[0] = [mesh_min[0], mesh_min[1], mesh_min[2]]
            corners[1] = [mesh_max[0], mesh_min[1], mesh_min[2]]
            corners[2] = [mesh_min[0], mesh_max[1], mesh_min[2]]
            corners[3] = [mesh_max[0], mesh_max[1], mesh_min[2]]
            corners[4] = [mesh_min[0], mesh_min[1], mesh_max[2]]
            corners[5] = [mesh_max[0], mesh_min[1], mesh_max[2]]
            corners[6] = [mesh_min[0], mesh_max[1], mesh_max[2]]
            corners[7] = [mesh_max[0], mesh_max[1], mesh_max[2]]
            
            # 3. 优化：一次性转换所有角点
            transform_mat = collision_model.pdndp.getMat()
            transformed_corners = np.array([transform_mat.xformPoint(Point3(*corner)) for corner in corners])
            
            min_point = np.maximum(np.min(transformed_corners, axis=0), scene_min)
            max_point = np.minimum(np.max(transformed_corners, axis=0), scene_max)
            
            # 4. 优化：提前计算size判断
            size = np.abs(max_point - min_point)
            is_noise = np.all(size <= NOISE_SIZE * 1.1)
            
            if 'obj_' in collision_model.name:
                object_boxes.append((min_point, max_point))
            elif is_noise:
                noise_boxes.append((min_point, max_point))
            else:
                obstacle_boxes.append((min_point, max_point))
    
    if not (object_boxes or obstacle_boxes or noise_boxes):
        print("没有找到任何碰撞模型")
        return None, None, None, None
    
    # 创建体素网格
    voxel_dims = np.ceil((scene_max - scene_min) / voxel_size).astype(int)
    object_voxels = np.zeros(voxel_dims, dtype=bool)
    obstacle_voxels = np.zeros(voxel_dims, dtype=bool)
    noise_voxels = np.zeros(voxel_dims, dtype=bool)
    
    # 5. 优化：预计算体素索引范围
    def get_voxel_range(box_min, box_max):
        start_idx = np.floor((box_min - scene_min) / voxel_size).astype(int)
        end_idx = np.ceil((box_max - scene_min) / voxel_size).astype(int)
        start_idx = np.maximum(start_idx, 0)
        end_idx = np.minimum(end_idx, voxel_dims)
        return start_idx, end_idx
    
    # 6. 优化：只遍历包围盒覆盖的体素范围
    def process_boxes(boxes, voxels):
        for box_min, box_max in boxes:
            start_idx, end_idx = get_voxel_range(box_min, box_max)
            for x in range(start_idx[0], end_idx[0]):
                for y in range(start_idx[1], end_idx[1]):
                    for z in range(start_idx[2], end_idx[2]):
                        voxel_min = scene_min + np.array([x, y, z]) * voxel_size
                        voxel_max = voxel_min + voxel_size
                        if check_box_overlap(voxel_min, voxel_max, box_min, box_max):
                            voxels[x, y, z] = True
    
    # 7. 优化：并行处理不同类型的体素
    if object_boxes:
        process_boxes(object_boxes, object_voxels)
    if obstacle_boxes:
        process_boxes(obstacle_boxes, obstacle_voxels)
    if noise_boxes:
        process_boxes(noise_boxes, noise_voxels)
    
    return object_voxels, obstacle_voxels, noise_voxels, scene_min


def check_box_overlap(box1_min, box1_max, box2_min, box2_max):
    """检查两个AABB盒是否有重叠"""
    return np.all(box1_max >= box2_min) and np.all(box1_min <= box2_max)


def visualize_voxels_separately(object_voxels, obstacle_voxels, noise_voxels, voxel_origin, voxel_size):
    """分别可视化物体、障碍物和噪声点的体素"""
    
    # 设置显示的体素盒子比实际体素稍小，创建间隙效果
    display_size = voxel_size * 0.85  # 显示尺寸为实际体素的85%
    
    if voxel_origin is not None:
        # 可视化物体体素
        if object_voxels is not None:
            # print("可视化物体体素...")
            for x, y, z in tqdm(zip(*np.where(object_voxels))):
                pos = voxel_origin + (np.array([x, y, z]) * voxel_size + voxel_size/2)
                mgm.gen_box(xyz_lengths=[display_size] * 3,
                        pos=pos,
                        rgb=ct.blue,
                        alpha=1).attach_to(base)
    
        # 可视化障碍物体素
        if obstacle_voxels is not None:
            # print("可视化障碍物体素...")
            for x, y, z in tqdm(zip(*np.where(obstacle_voxels))):
                pos = voxel_origin + (np.array([x, y, z]) * voxel_size + voxel_size/2)
                mgm.gen_box(xyz_lengths=[display_size] * 3,
                        pos=pos,
                        rgb=ct.red,
                        alpha=1).attach_to(base)
        
        # 可视化噪声点体素
        if noise_voxels is not None:
            # print("可视化噪声点体素...")
            for x, y, z in tqdm(zip(*np.where(noise_voxels))):
                pos = voxel_origin + (np.array([x, y, z]) * voxel_size + voxel_size/2)
                mgm.gen_box(xyz_lengths=[display_size] * 3,
                        pos=pos,
                        rgb=ct.green,  # 使用黄色表示噪声点
                        alpha=.9).attach_to(base)
    else:
        print("voxel_origin is None")


def save_batch_data(batch_data, base_path=BASE_PATH):
    """保存数据，如果文件存在则追加数据"""
    save_path = os.path.join(base_path, "obstacle_grasp_data.npz")

    # 检查是否有数据
    if not batch_data:
        print("❌ No data to save!")
        return

    # 确保 batch_data 是可用的
    new_voxels = np.array([item['voxels'] for item in batch_data], dtype=object)
    new_grasp_ids = np.array([item['grasp_ids'] if item['grasp_ids'] is not None else [] for item in batch_data], dtype=object)

    # 如果文件已存在，读取并追加数据
    if os.path.exists(save_path):
        try:
            existing_data = np.load(save_path, allow_pickle=True)
            existing_voxels = existing_data.get('voxel_data', np.empty((0,), dtype=object))
            existing_grasp_ids = existing_data.get('grasp_ids', np.empty((0,), dtype=object))

            # 确保 existing 数据类型正确
            if not isinstance(existing_voxels, np.ndarray) or not isinstance(existing_grasp_ids, np.ndarray):
                print("⚠️ 现有数据格式不正确，可能损坏，重置为新数据。")
                voxel_data = new_voxels
                grasp_ids = new_grasp_ids
            else:
                voxel_data = np.concatenate([existing_voxels, new_voxels], axis=0)
                grasp_ids = np.concatenate([existing_grasp_ids, new_grasp_ids], axis=0)

        except Exception as e:
            print(f"❌ 读取 {save_path} 失败，错误: {e}")
            print("📌 重新创建文件...")
            voxel_data = new_voxels
            grasp_ids = new_grasp_ids
    else:
        # 如果文件不存在，直接存储
        voxel_data = new_voxels
        grasp_ids = new_grasp_ids

    # 保存数据
    try:
        np.savez(save_path, voxel_data=voxel_data, grasp_ids=grasp_ids)
        print(f"✅ 数据成功追加到 {save_path}")
    except Exception as e:
        print(f"❌ 保存数据失败，错误: {e}")



if __name__ == '__main__':
    # 设置世界
    base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    # np.random.seed(22)

    # 设置场景
    robot = env_setup()
    obj_cmodel = mcm.CollisionModel(initor=r"E:\Qin\wrs\wrs\HuGroup_Qin\objects\meshes\bottle.stl")

    # grasp configuration
    GRASP_DATA_PATH, GRASP_ID_SAVE_PATH = get_file_paths(str(109))
    fs_reference_poses = FSReferencePoses(obj_cmodel=obj_cmodel)
    object_feasible_grasps = grasp_load(GRASP_DATA_PATH)
    RegraspReasoner = GraspReasoner(robot, object_feasible_grasps)

    
    # 使用LHS替代随机采样
    n_sampler = qmc.LatinHypercube(d=1)
    n_samples = n_sampler.random(n=TOTAL_ITERATIONS)
    n_samples_indices = qmc.scale(n_samples, [3], [10])
    n_samples_indice = np.floor(n_samples_indices).astype(int).flatten()


    # 初始化数据收集列表
    current_batch = []
    batch_number = 0
    
    # LHS sampling
    # generate grid obstacles map
    print("start to collect data")
    tic = time.time()
    with tqdm(total=TOTAL_ITERATIONS, desc="Processing All Iterations") as pbar:
        for sample_idx in range(TOTAL_ITERATIONS - 1):
            obstacle_list = generate_grid_obstacles(n_samples=n_samples_indice[sample_idx])
            # 修改：将物体放置在空间中心
            obj_pos = np.array([0, 0, 0])  # 设置物体在空间中心
            obj_rotmat = rm.rotmat_from_euler(0, 0, 0)  # 使用默认旋转矩阵

            available_gids, available_grasps = RegraspReasoner.find_feasible_gids(goal_pose=[obj_pos, obj_rotmat],
                                                                                 consider_robot=False,
                                                                                 obstacle_list=obstacle_list,
                                                                                 toggle_dbg=False)
            if available_gids:
                available_gids = list(set(available_gids))
                available_grasps = list(set(available_grasps))
            else:
                available_gids = None
                available_grasps = None
                
            # 体素化场景
            # tic = time.time()
            object_voxels, obstacle_voxels, noise_voxels, voxel_origin = voxelize_scene_in_memory(voxel_size=0.02)
            # tic = time.time() - tic
            # print(f"体素化场景耗时: {tic:.5f}秒")

            # 合并障碍物和噪声的体素数据
            combined_voxels = np.logical_or(obstacle_voxels, noise_voxels)
            
            # 收集当前样本的数据
            sample_data = {
                'voxels': combined_voxels,
                'grasp_ids': available_gids
            }
            current_batch.append(sample_data)
            
            # 每1000个样本保存一次
            if len(current_batch) >= SAVE_BATCH_SIZE:
                save_batch_data(current_batch)
                current_batch = []  # 清空当前批次

            # # 可视化体素
            # if object_voxels is not None:
            #     visualize_voxels_separately(object_voxels, obstacle_voxels, noise_voxels, voxel_origin, voxel_size=0.02)
            #     local_space_show(length=0.5, width=0.5, height=0.6)

            # if available_grasps is not None:
            #     show_grasp(robot, available_grasps, obj_pos=obj_pos, obj_rotmat=obj_rotmat)
                # base.run()

            # 去除所有障碍物
            for item in obstacle_list:
                item.detach()

            pbar.update(1)
    tic = time.time() - tic
    print(f"finish collect data, {tic:.5f}秒")