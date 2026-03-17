import numpy as np
import samply
from pathlib import Path

import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim

def generate_cvt_kernels(robot, n_kernels=10000):
    """
    仅生成CVT核，不进行物理合法性检查
    """
    print(f"正在 6 维空间中计算 {n_kernels} 个 CVT 核...")
    
    # 1. 在归一化 [0, 1]^6 空间生成均匀分布的核
    # samply.hypercube.cvt 是目前处理高维均匀分布最稳健的库之一
    normalized_kernels = samply.hypercube.cvt(n_kernels, robot.n_dof)
    
    # 2. 将核从 [0, 1] 映射到机器人的实际关节量程 [min, max]
    jnt_mins = robot.jnt_ranges[:, 0]
    jnt_maxs = robot.jnt_ranges[:, 1]
    
    # 线性插值映射
    kernels_q = jnt_mins + normalized_kernels * (jnt_maxs - jnt_mins)
    
    print(f"核生成完毕。形状: {kernels_q.shape}")
    return kernels_q

if __name__ == "__main__":
    # 初始化机器人（仅用于获取关节限位参数）
    robot = xarm6_sim.XArmLite6Miller()

    # 执行生成
    n_kernels = 50000
    kernels = generate_cvt_kernels(robot, n_kernels=n_kernels)

    # 保存数据
    # 建议使用 .npy 格式，读写速度最快，且保留高精度浮点数
    save_path = Path("0000_test_programs/surgery_diff/CleanDiffuser/FlowMaxStraightLine/cvt_kernels_raw.npy")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, kernels)

    print("-" * 50)
    print(f"数据已保存至: {save_path}")
    print(f"每个核的维度: {kernels.shape[1]} (对应 xArm 6-DOF)")
    print("-" * 50)