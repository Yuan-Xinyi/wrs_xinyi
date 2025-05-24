import numpy as np
from wrs import mcm
import wrs.visualization.panda.world as wd
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka

def generate_cabinet(base, width=0.6, depth=0.4, layer_height=0.3, layers=4, thickness=0.02,
                     rgb=[0.4, 0.3, 0.2], pos_offset=np.zeros(3), rot_theta_deg=0.0):
    """
    生成一个多层柜子（按每层高度构建），支持整体旋转和平移。
    :param base: 可视化 base
    :param width: 柜体宽度 沿 x
    :param depth: 柜体深度 沿 y
    :param layer_height: 每层高度
    :param layers: 层数
    :param thickness: 板材厚度
    :param rgb: 颜色
    :param pos_offset: 位置偏移 np.array([x, y, z])
    :param rot_theta_deg: 绕 Z 轴的旋转角度（单位：度）
    """
    height = layer_height * layers
    theta = np.radians(rot_theta_deg)
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    def transform(pos_local):
        return np.dot(rot_z, pos_local) + pos_offset

    def attach_box(size, local_pos):
        box = mcm.gen_box(size, pos=transform(local_pos), rotmat=rot_z, rgb=rgb)
        box.attach_to(base)

    # 各部分组件
    attach_box([width, depth, thickness], [0, 0, thickness / 2])                     # bottom
    attach_box([width, depth, thickness], [0, 0, height - thickness / 2])            # top
    attach_box([thickness, depth, height], [-width / 2 + thickness / 2, 0, height / 2])  # left
    attach_box([thickness, depth, height], [ width / 2 - thickness / 2, 0, height / 2])  # right
    attach_box([width - 2 * thickness, thickness, height],
               [0, -depth / 2 + thickness / 2, height / 2])                          # back

    for i in range(1, layers):
        z = i * layer_height
        attach_box([width - 2 * thickness, depth - thickness, thickness], [0, 0, z])  # shelves


if __name__ == "__main__":
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    generate_cabinet(
        base,
        width=1.0,
        depth=0.4,
        layer_height=0.2,
        layers=5,
        pos_offset=np.array([0.8, 0.2, 0.0]),
        rot_theta_deg=90
    )

    robot = franka.FrankaResearch3(enable_cc=True)
    robot.gen_meshmodel(alpha=1.).attach_to(base)
    base.run()
