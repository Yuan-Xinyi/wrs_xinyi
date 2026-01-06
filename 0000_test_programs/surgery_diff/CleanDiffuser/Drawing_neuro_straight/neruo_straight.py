'''wrs reliance'''
from wrs import wd, rm, mcm
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
import wrs.modeling.geometric_model as mgm

'''global variables'''
import time
import pickle
import numpy as np
import torch
torch.autograd.set_detect_anomaly(False)

import matplotlib.pyplot as plt


def make_rotmat_z_down(z_dir=np.array([0,0,-1]), x_hint=np.array([1,0,0])):
    z_dir = z_dir / np.linalg.norm(z_dir)
    x_axis = x_hint - np.dot(x_hint, z_dir) * z_dir
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_dir, x_axis)
    rotmat = np.stack([x_axis, y_axis, z_dir], axis=1)
    return rotmat

def optimize_trajectory(
    xarm_gpu,
    pos_target_batch,
    steps=1000,
    lr=1e-2,
    lambda_pos=50.0,
    lambda_smooth=5.0,
    lambda_cone=2.0,
    visualize=False
):
    device = pos_target_batch.device
    num_points = pos_target_batch.shape[0]
    n_dof = xarm_gpu.robot.n_dof

    # ========== Initialization ==========
    q_traj = xarm_gpu.robot.rand_conf_batch(num_points)  # (N,6)
    q_traj = q_traj.clone().detach().to(device).requires_grad_(True)

    optimizer = torch.optim.Adam([q_traj], lr=lr)
    z_target = torch.tensor([0, 0, -1], dtype=torch.float32, device=device)  # TCP朝下方向（写字笔）

    for step in range(steps):
        optimizer.zero_grad()
        pos_fk, rot_fk = xarm_gpu.robot.fk_batch(q_traj)  # pos:(N,3), rot:(N,3,3)
        pos_loss = torch.mean((pos_fk - pos_target_batch) ** 2)
        smooth_loss = torch.mean((q_traj[1:] - q_traj[:-1]) ** 2)
        # 3. Orientation loss (末端z轴朝向)
        z_target = torch.tensor([0, 0, -1], dtype=torch.float32, device=pos_fk.device)
        z_fk = rot_fk[:, :, 2]  # 末端z轴
        ori_loss = torch.mean((z_fk - z_target) ** 2)
        lambda_ori = 10.0
        loss = lambda_pos * pos_loss + lambda_smooth * smooth_loss + lambda_ori * ori_loss
        loss.backward()
        optimizer.step()

        # ---------- Logging ----------
        if visualize and (step % 50 == 0 or step == steps - 1):
            print(f"Step {step:03d}: "
                  f"loss={loss.item():.6f}, "
                  f"pos={pos_loss.item():.4f}, "
                  f"smooth={smooth_loss.item():.4f}, "
                  f"ori={ori_loss.item():.4f}")

    return q_traj.detach()

# initialize robot models and simulation
rbt_gpu = xarm6_gpu.XArmLite6GPU()
rbt_sim = xarm6_sim.XArmLite6Miller(enable_cc=True)
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)
# xarm_sim.goto_given_conf([0, 0, 0, 0, 0, 0])
# xarm_sim.gen_meshmodel().attach_to(base)

table_size = np.array([1.5, 1.5, 0.03])
table_pos  = np.array([0.2, 0, -0.025])
table = mcm.gen_box(xyz_lengths=table_size, pos=table_pos, rgb=np.array([0.6, 0.4, 0.2]), alpha=1)
table.attach_to(base)

paper_size = np.array([1.2, 1.2, 0.002])
paper_pos = table_pos.copy()
paper_pos[2] = table_pos[2] + table_size[2]/2 + paper_size[2]/2
print("paper pos:", paper_pos)
paper = mcm.gen_box(xyz_lengths=paper_size, pos=paper_pos, rgb=np.array([1, 1, 1]), alpha=1)
paper.attach_to(base)

device = rbt_gpu.device


if __name__ == "__main__":
    
    # ======================== 直线最大可行长度搜索 ==============================
    def sample_line(xc, d, L, num_points):
        s_list = np.linspace(-L/2, L/2, num_points)
        return np.stack([xc + s*d for s in s_list], axis=0)

    def check_line_feasibility_fk(rbt_gpu, rbt_sim, pos_path, joint_thresh=0.2, steps=2000):
        # 用中点IK解初始化，优先垂直向下（正交矩阵）
        idx_mid = len(pos_path)//2
        pos_mid = pos_path[idx_mid]
        rotmat_down = make_rotmat_z_down()
        q_init = rbt_sim.ik(tgt_pos=pos_mid, tgt_rotmat=rotmat_down)
        # 若无解，则在旋转锥内尝试
        if q_init is None:
            cone_angle = np.deg2rad(30)
            found = False
            for theta in np.linspace(0, 2*np.pi, 12, endpoint=False):
                for phi in np.linspace(0, cone_angle, 4):
                    dz = -np.cos(phi)
                    dx = np.sin(phi)*np.cos(theta)
                    dy = np.sin(phi)*np.sin(theta)
                    z_dir = np.array([dx, dy, dz])
                    rotmat = make_rotmat_z_down(z_dir)
                    q_init = rbt_sim.ik(tgt_pos=pos_mid, tgt_rotmat=rotmat)
                    if q_init is not None:
                        found = True
                        break
                if found:
                    break
            if not found:
                return False

        pos_target_batch = torch.tensor(pos_path, dtype=torch.float32, device=rbt_gpu.device)
        q_traj = optimize_trajectory(
            rbt_gpu,
            pos_target_batch,
            steps=steps,
            lr=1e-2,
            lambda_pos=50.0,
            lambda_smooth=5.0,
            visualize=False
        )

        q_np = q_traj.cpu().numpy()
        if np.any(np.linalg.norm(np.diff(q_np, axis=0), axis=1) > joint_thresh):
            return False

        for q in q_np:
            if not rbt_sim.are_jnts_in_ranges(q):
                return False
        return True

    xc = np.array([0.3, 0.0, 0])  # 线段中心
    d = np.array([0.0, -1.0, 0.0])  # 沿-y轴
    d = d / np.linalg.norm(d)
    L_max = 0.1
    joint_thresh = 0.2 
    num_points = 30

    pos_path = sample_line(xc, d, L_max, num_points)
    pos_target_batch = torch.tensor(pos_path, dtype=torch.float32, device=rbt_gpu.device)
    q_traj = optimize_trajectory(
        rbt_gpu,
        pos_target_batch,
        steps=10000,
        lr=1e-2,
        lambda_pos=50.0,
        lambda_smooth=5.0,
        visualize=True
    )

    def visualize_pos_batch(pos_batch):
        pos_batch_np = pos_batch.cpu().numpy()
        for t in range(pos_batch_np.shape[0]):
            mgm.gen_sphere(pos_batch_np[t], 0.005, [0,1,0]).attach_to(base)

    def visulize_jnt_batch(jnt_batch):
        jnt_batch_np = jnt_batch.cpu().numpy()
        for t in range(jnt_batch_np.shape[0]):
            rbt_sim.goto_given_conf(jnt_values=jnt_batch_np[t])
            rbt_sim.gen_meshmodel(alpha=0.2).attach_to(base)

    visualize_pos_batch(pos_target_batch)
    visulize_jnt_batch(q_traj)
    base.run()

