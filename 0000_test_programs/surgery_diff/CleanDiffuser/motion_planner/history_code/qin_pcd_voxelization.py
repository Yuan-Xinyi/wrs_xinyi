import numpy as np
import cv2
import time
from typing import Tuple
import pyrealsense2 as rs
import sys
sys.path.append("E:/Qin/wrs")
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import wrs.basis.constant as ct

class RealTimeVoxelizer:
    def __init__(self, 
                 voxel_size: float = 0.02,
                 space_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 space_center: Tuple[float, float, float] = (0, 0, 0.5)):
        """
        初始化实时体素化器
        :param voxel_size: 体素大小（米）
        :param space_size: 体素化空间大小 (length, width, height)
        :param space_center: 体素化空间中心点
        """
        # 体素化参数
        self.voxel_size = voxel_size
        self.space_size = np.array(space_size)
        self.space_center = np.array(space_center)
        
        # 计算场景边界
        self.scene_min = self.space_center - self.space_size/2
        self.scene_max = self.space_center + self.space_size/2
        
        # 计算体素网格维度
        self.voxel_dims = np.ceil((self.space_size) / self.voxel_size).astype(int)
        
        # 初始化RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 配置深度和彩色流
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 启动流
        self.profile = self.pipeline.start(self.config)
        
        # 创建点云对象
        self.pc = rs.pointcloud()
        
        # 获取深度传感器的深度标尺
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        
        # 设置过滤器
        self.decimate = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()
        self.threshold = rs.threshold_filter()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        
        # 配置Decimation Filter (下采样滤波器)
        self.decimate.set_option(rs.option.filter_magnitude, 3)  # 2-8之间，越大下采样越多
        
        # 配置Threshold Filter (阈值滤波器)
        self.threshold.set_option(rs.option.min_distance, 0.0)  # 最小深度距离（米）
        self.threshold.set_option(rs.option.max_distance, 2.0)  # 最大深度距离（米）
        
        # 配置Spatial Filter (空间滤波器)
        self.spatial.set_option(rs.option.filter_magnitude, 1)    # 过滤器的强度 (1-5)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)  # 平滑度 (0.25-1)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)   # 深度差异阈值
        
        # 配置Temporal Filter (时间滤波器)
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.1)  # 平滑度 alpha
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)   # 深度差异阈值
        self.temporal.set_option(rs.option.holes_fill, 3)    # 过滤持续性 (1-8)
        
        # 存储可视化的模型列表
        self.voxel_model_list = []
        
        # 等待相机预热
        time.sleep(2)

    def process_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """处理一帧数据并返回体素化结果"""
        # 等待一帧数据
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
        
        # 应用滤波器链
        filtered_depth = self.decimate.process(depth_frame)
        filtered_depth = self.threshold.process(filtered_depth)
        filtered_depth = self.depth_to_disparity.process(filtered_depth)
        filtered_depth = self.spatial.process(filtered_depth)
        filtered_depth = self.temporal.process(filtered_depth)
        filtered_depth = self.disparity_to_depth.process(filtered_depth)
        filtered_depth = self.hole_filling.process(filtered_depth)
        
        # 获取点云
        self.pc.map_to(color_frame)
        points = self.pc.calculate(filtered_depth)
        
        # 转换点云数据为numpy数组
        v = points.get_vertices()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        
        # 过滤掉空间外的点
        mask = np.all((verts >= self.scene_min) & (verts <= self.scene_max), axis=1)
        verts = verts[mask]
        
        # 体素化
        voxels = np.zeros(self.voxel_dims, dtype=bool)
        if len(verts) > 0:
            # 将点云坐标转换为体素索引
            voxel_indices = ((verts - self.scene_min) / self.voxel_size).astype(int)
            
            # 确保索引在有效范围内
            valid_indices = np.all((voxel_indices >= 0) & (voxel_indices < self.voxel_dims), axis=1)
            voxel_indices = voxel_indices[valid_indices]
            
            # 设置体素
            if len(voxel_indices) > 0:
                voxels[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
        
        # 获取图像数据并进行热力图转换
        depth_image = np.asanyarray(filtered_depth.get_data())
        
        # 将深度图转换为热力图显示
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        
        color_image = np.asanyarray(color_frame.get_data())
        
        return voxels, depth_colormap, color_image


    def visualize_voxels(self, base, voxels: np.ndarray):
        """优化的体素可视化方法，颜色根据距离变化"""
        if voxels is None:
            return
            
        # 清除之前的可视化模型
        for model in self.voxel_model_list:
            base.detach_noupdate_model(model)
        self.voxel_model_list.clear()
            
        # 批量创建体素模型
        display_size = self.voxel_size * 0.85
        voxel_positions = np.array(np.where(voxels)).T
        positions = self.scene_min + voxel_positions * self.voxel_size + self.voxel_size/2
        
        # 计算每个体素到相机的距离
        # 相机位置在原点(0,0,0)
        distances = np.linalg.norm(positions, axis=1)
        
        # 归一化距离到[0,1]范围
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        normalized_distances = (distances - min_dist) / (max_dist - min_dist + 1e-6)
        
        # 每次最多显示1000个体素以保持性能
        max_voxels = 5000
        if len(positions) > max_voxels:
            indices = np.random.choice(len(positions), max_voxels, replace=False)
            positions = positions[indices]
            normalized_distances = normalized_distances[indices]
        
        # 使用HSV颜色空间来生成渐变色
        for pos, dist in zip(positions, normalized_distances):
            # 使用HSV颜色空间：色调从240(蓝色)到0(红色)
            hue = (1 - dist) * 240  # 远处蓝色(240)，近处红色(0)
            # 转换HSV到RGB (色调值需要除以360因为cv2的H范围是0-180)
            rgb = cv2.cvtColor(np.uint8([[[hue/2, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0] / 255.0
            
            box_model = mgm.gen_box(xyz_lengths=[display_size]*3,
                                  pos=pos,
                                  rgb=rgb,
                                  alpha=1)
            base.attach_noupdate_model(box_model)
            self.voxel_model_list.append(box_model)


    def show_space_boundary(self, base):
        """显示采样空间的边界框"""
        vertices_array = np.array([
            [self.space_size[0]/2, self.space_size[1]/2, -self.space_size[2]/2],
            [self.space_size[0]/2, -self.space_size[1]/2, -self.space_size[2]/2],
            [-self.space_size[0]/2, -self.space_size[1]/2, -self.space_size[2]/2],
            [-self.space_size[0]/2, self.space_size[1]/2, -self.space_size[2]/2],
            [self.space_size[0]/2, self.space_size[1]/2, self.space_size[2]/2],
            [self.space_size[0]/2, -self.space_size[1]/2, self.space_size[2]/2],
            [-self.space_size[0]/2, -self.space_size[1]/2, self.space_size[2]/2],
            [-self.space_size[0]/2, self.space_size[1]/2, self.space_size[2]/2]
        ]) + self.space_center

        def vertices(idx):
            return vertices_array[idx]
        
        spacecm = mgm.gen_wireframe(vertices=vertices,
                                  edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0],  # 底部边缘
                                                [4, 5], [5, 6], [6, 7], [7, 4],  # 顶部边缘
                                                [0, 4], [1, 5], [2, 6], [3, 7]   # 连接边
                                               ]),
                                  thickness=0.001,
                                  rgb=np.array([0, 0, 0]),
                                  alpha=0.3)
        spacecm.attach_to(base)
        return spacecm


    def run(self, base):
        """运行实时体素化循环"""
        # 显示空间边界
        space_boundary = self.show_space_boundary(base)
        
        while True:
            # 处理一帧数据
            voxels, depth_colormap, color_image = self.process_frame()
            if voxels is None:
                continue
            
            # 显示深度图和彩色图
            cv2.imshow('Depth Image (JET colormap)', depth_colormap)
            cv2.imshow('Color Image', color_image)
            
            # 可视化体素
            self.visualize_voxels(base, voxels)
            
            # 更新场景
            base.task_mgr.step()
            
            # 检查退出条件
            if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
                break
        
            if base.inputmgr.keymap['w']:  # 检测w键
                break

        while True:
            base.taskMgr.step()  # 注意这里使用taskMgr而不是task_mgr
            time.sleep(0.01)

            if base.inputmgr.keymap['a']:  # 检测a键
                break

        self.pipeline.stop()
        cv2.destroyAllWindows()
        space_boundary.detach()


    def visualize_region_voxels(self, base, position, rotmat, size, color=None):
        """
        可视化指定区域内的体素
        
        Args:
            base: 场景基础对象
            position (np.ndarray): 区域中心位置 [x, y, z]
            rotmat (np.ndarray): 3x3旋转矩阵
            size (tuple): 区域的长宽高 (length, width, height)
            color (np.ndarray, optional): RGB颜色值，默认为None使用距离渐变色
        """
        # 获取当前帧的体素数据
        voxels, _, _ = self.process_frame()
        if voxels is None:
            return
        
        # 计算区域的8个顶点（在局部坐标系中）
        half_size = np.array(size) / 2
        local_corners = np.array([
            [ half_size[0],  half_size[1],  half_size[2]],
            [ half_size[0],  half_size[1], -half_size[2]],
            [ half_size[0], -half_size[1],  half_size[2]],
            [ half_size[0], -half_size[1], -half_size[2]],
            [-half_size[0],  half_size[1],  half_size[2]],
            [-half_size[0],  half_size[1], -half_size[2]],
            [-half_size[0], -half_size[1],  half_size[2]],
            [-half_size[0], -half_size[1], -half_size[2]]
        ])
        
        # 将顶点转换到世界坐标系
        world_corners = (rotmat @ local_corners.T).T + position
        
        # 计算包围盒的最小和最大点
        min_corner = np.min(world_corners, axis=0)
        max_corner = np.max(world_corners, axis=0)
        
        # 将世界坐标转换为体素索引
        min_voxel_idx = ((min_corner - self.scene_min) / self.voxel_size).astype(int)
        max_voxel_idx = ((max_corner - self.scene_min) / self.voxel_size).astype(int)
        
        # 确保索引在有效范围内
        min_voxel_idx = np.maximum(min_voxel_idx, 0)
        max_voxel_idx = np.minimum(max_voxel_idx, self.voxel_dims - 1)
        
        # 提取区域内的体素
        region_voxels = voxels[
            min_voxel_idx[0]:max_voxel_idx[0]+1,
            min_voxel_idx[1]:max_voxel_idx[1]+1,
            min_voxel_idx[2]:max_voxel_idx[2]+1
        ]
        
        # 创建一个新的体素数组，只包含指定区域
        full_voxels = np.zeros_like(voxels)
        full_voxels[
            min_voxel_idx[0]:max_voxel_idx[0]+1,
            min_voxel_idx[1]:max_voxel_idx[1]+1,
            min_voxel_idx[2]:max_voxel_idx[2]+1
        ] = region_voxels
        
        # 清除之前的可视化模型
        for model in self.voxel_model_list:
            base.detach_noupdate_model(model)
        self.voxel_model_list.clear()
        
        # 获取体素位置
        voxel_positions = np.array(np.where(full_voxels)).T
        positions = self.scene_min + voxel_positions * self.voxel_size + self.voxel_size/2
        
        if len(positions) == 0:
            return
        
        # 设置显示参数
        display_size = self.voxel_size * 0.85
        
        if color is None:
            # 使用距离渐变色
            distances = np.linalg.norm(positions, axis=1)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            normalized_distances = (distances - min_dist) / (max_dist - min_dist + 1e-6)
            
            # 每次最多显示5000个体素以保持性能
            max_voxels = 5000
            if len(positions) > max_voxels:
                indices = np.random.choice(len(positions), max_voxels, replace=False)
                positions = positions[indices]
                normalized_distances = normalized_distances[indices]
            
            # 使用HSV颜色空间来生成渐变色
            for pos, dist in zip(positions, normalized_distances):
                hue = (1 - dist) * 240  # 远处蓝色(240)，近处红色(0)
                rgb = cv2.cvtColor(np.uint8([[[hue/2, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0] / 255.0
                
                box_model = mgm.gen_box(xyz_lengths=[display_size]*3,
                                      pos=pos,
                                      rgb=rgb,
                                      alpha=1)
                base.attach_noupdate_model(box_model)
                self.voxel_model_list.append(box_model)
        else:
            # 使用指定颜色
            for pos in positions:
                box_model = mgm.gen_box(xyz_lengths=[display_size]*3,
                                      pos=pos,
                                      rgb=color,
                                      alpha=1)
                base.attach_noupdate_model(box_model)
                self.voxel_model_list.append(box_model)
        
        # 显示区域边界框
        vertices_array = world_corners
        def vertices(idx):
            return vertices_array[idx]
        
        boundary = mgm.gen_wireframe(vertices=vertices,
                                   edges=np.array([[0, 1], [1, 3], [3, 2], [2, 0],  # 顶面
                                                    [4, 5], [5, 7], [7, 6], [6, 4],  # 底面
                                                    [0, 4], [1, 5], [2, 6], [3, 7]   # 连接边
                                                ]),
                                   thickness=0.001,
                                   rgb=np.array([1, 0, 0]),  # 红色边界
                                   alpha=0.5)
        base.attach_noupdate_model(boundary)
        self.voxel_model_list.append(boundary)


    def __del__(self):
        """析构函数"""
        if hasattr(self, 'pipeline') and self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass


if __name__ == "__main__":
    # 创建3D可视化窗口
    base = wd.World(cam_pos=[0, 0, -1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)  # 添加坐标系参考框架
    
    # 创建实时体素化器实例
    voxelizer = RealTimeVoxelizer(
        voxel_size=0.01,  # 1cm的体素大小
        space_size=(0.6, 0.6, 0.6),  # 60cm x 60cm x 60cm的空间
        space_center=(0, 0, 0.5)  # 空间中心点在相机前方0.5m处
    )
    
    # # 运行实时体素化
    # voxelizer.run(base)

    # 定义要观察的区域
    position = np.array([0.1, 0.1, 0.3])  # 区域中心位置
    rotmat = np.eye(3)  # 或其他旋转矩阵
    size = (0.2, 0.2, 0.2)  # 区域大小
    
    # 可视化指定区域的体素
    voxelizer.visualize_region_voxels(
        base,
        position,
        rotmat,
        size,
        color=None  # 可选：指定颜色（这里是绿色）
    )
    
    # 更新场景
    while True:
        base.task_mgr.step()
        if base.inputmgr.keymap['a']:  # 按a键退出
            break
    
    voxelizer.pipeline.stop()