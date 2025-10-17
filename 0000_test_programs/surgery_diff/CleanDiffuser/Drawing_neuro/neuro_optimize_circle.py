'''wrs reliance'''
from wrs import wd, rm, mcm
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
import wrs.modeling.geometric_model as mgm

'''self modules'''
import rotation_cone_constraint as cone_constraint

'''global variables'''
import time
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.autograd.set_detect_anomaly(True)


# ======================== Initialization ============================
'''initialize robot and scene'''
xarm_gpu = xarm6_gpu.XArmLite6GPU()
xarm_sim = xarm6_sim.XArmLite6Miller(enable_cc=True) 
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)

'''if you want to test the fk consistency between sim and gpu'''
# jnt = xarm_sim.rand_conf()
# tgt_sim, rotmat_sim = xarm_sim.fk(jnt_values=jnt)
# tgt_gpu, rotmat_gpu = xarm_gpu.robot.fk(jnt_values=torch.tensor(jnt, dtype=torch.float32, device=xarm_gpu.device))
# print("Sim FK Pos:", tgt_sim, "Rotmat:\n", rotmat_sim)
# print("GPU FK Pos:", tgt_gpu.cpu().numpy(), "Rotmat:\n", rotmat_gpu.cpu().numpy())
# xarm_sim.goto_given_conf(jnt_values=jnt)
# xarm_sim.gen_meshmodel().attach_to(base)

'''table'''
table_size = np.array([0.9, 0.9, 0.05]) 
table_pos  = np.array([0.37, 0, -0.025])

table = mcm.gen_box(xyz_lengths=table_size,
                    pos=table_pos,
                    rgb=np.array([0.6, 0.4, 0.2]),
                    alpha=1)
table.attach_to(base)

'''paper'''
paper_size = np.array([0.5, 0.5, 0.002])
paper_pos = table_pos.copy()
paper_pos[2] = table_pos[2] + table_size[2]/2 + paper_size[2]/2
paper = mcm.gen_box(xyz_lengths=paper_size,
                    pos=paper_pos,
                    rgb=np.array([1, 1, 1]),
                    alpha=1)
paper.attach_to(base)

# ======================== Helper Functions ============================
def randomize_circle(paper_pos, paper_size, num_points=50, margin=0.05, visualize=False):
    """randomly generate a circle on the paper surface"""
    cx, cy, cz = paper_pos
    lx, ly, lz = paper_size
    z = cz + lz / 2
    x_min, x_max = cx - lx/2 + margin, cx + lx/2 - margin
    y_min, y_max = cy - ly/2 + margin, cy + ly/2 - margin

    # Randomly generate circle center and maximum radius constraints
    center = np.array([np.random.uniform(x_min, x_max),
                       np.random.uniform(y_min, y_max),
                       z])
    max_r = min(center[0] - x_min, x_max - center[0],
                center[1] - y_min, y_max - center[1])
    r = np.random.uniform(0.1, max_r)
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    pos_list = np.stack([center[0] + r*np.cos(theta),
                         center[1] + r*np.sin(theta),
                         np.full(num_points, z)], axis=1)

    if visualize:
        [mgm.gen_sphere(p, 0.005, [0,1,0]).attach_to(base) for p in pos_list]
        mgm.gen_sphere(center, 0.005, [0,1,0]).attach_to(base)
        for i in range(num_points):
            mgm.gen_stick(pos_list[i], pos_list[(i+1)%num_points], 0.0015, [0,0,-1]).attach_to(base)

    return pos_list, {'center': center, 'radius': r, 'normal': [0,0,-1], 'plane_z': z}


def randomize_circles_batch(paper_pos, paper_size, batch_size=16, num_points=50, 
                            margin=0.12, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # unpack paper info
    px, py, pz = paper_pos
    lx, ly, lz = paper_size
    z = pz + lz / 2

    # ⚠️ 用纸面的几何边界（左下角/右上角）
    x_min, x_max = (px - lx/2) + margin, (px + lx/2) - margin
    y_min, y_max = (py - ly/2) + margin, (py + ly/2) - margin

    '''visualize paper boundary'''
    # mgm.gen_stick(spos=np.array([x_min, y_min, z]),
    #             epos=np.array([x_max, y_min, z]),
    #             radius=0.001, rgb=np.array([1,0,0])).attach_to(base)

    # mgm.gen_stick(spos=np.array([x_max, y_min, z]),
    #             epos=np.array([x_max, y_max, z]),
    #             radius=0.001, rgb=np.array([1,0,0])).attach_to(base)

    # mgm.gen_stick(spos=np.array([x_max, y_max, z]),
    #             epos=np.array([x_min, y_max, z]),
    #             radius=0.001, rgb=np.array([1,0,0])).attach_to(base)

    # mgm.gen_stick(spos=np.array([x_min, y_max, z]),
    #             epos=np.array([x_min, y_min, z]),
    #             radius=0.001, rgb=np.array([1,0,0])).attach_to(base)

    # random centers
    centers_x = torch.empty(batch_size, device=device).uniform_(x_min, x_max)
    centers_y = torch.empty(batch_size, device=device).uniform_(y_min, y_max)
    centers_z = torch.full((batch_size,), z, device=device)
    centers = torch.stack([centers_x, centers_y, centers_z], dim=1)

    # radius computation
    left   = centers_x - x_min
    right  = x_max - centers_x
    bottom = centers_y - y_min
    top    = y_max - centers_y
    max_r  = torch.min(torch.min(left, right), torch.min(bottom, top))
    min_r  = torch.full_like(max_r, 0.1)
    max_r  = torch.clamp(max_r, min=min_r)
    radii  = torch.rand(batch_size, device=device) * (max_r - min_r) + min_r

    # circle sampling
    theta = torch.linspace(0, 2 * torch.pi, num_points + 1, device=device)[:-1]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    x_offsets = radii[:, None] * cos_t[None, :]
    y_offsets = radii[:, None] * sin_t[None, :]
    z_offsets = torch.zeros_like(x_offsets, device=device) + z

    pos_tensor = torch.stack([
        centers_x[:, None] + x_offsets,
        centers_y[:, None] + y_offsets,
        z_offsets
    ], dim=-1)

    info = {
        'center': centers,
        'radius': radii,
        'normal': torch.tensor([0, 0, 1], device=device).repeat(batch_size, 1),
        'plane_z': torch.full((batch_size,), z, device=device)
    }

    return pos_tensor, info


def visualize_pos_batch(pos_batch):
    pos_batch_np = pos_batch.cpu().numpy()
    for b in range(pos_batch_np.shape[0]):
        for t in range(pos_batch_np.shape[1]):
            mgm.gen_sphere(pos_batch_np[b, t], 0.005, [0,1,0]).attach_to(base)

def visulize_jnt_batch(jnt_batch):
    jnt_batch_np = jnt_batch.cpu().numpy()
    for b in range(jnt_batch_np.shape[0]):
        xarm_sim.goto_given_conf(jnt_values=jnt_batch_np[b])
        xarm_sim.gen_meshmodel(alpha=0.2).attach_to(base)

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

        # ---------- Forward Kinematics ----------
        pos_fk, rot_fk = xarm_gpu.robot.fk_batch(q_traj)  # pos:(N,3), rot:(N,3,3)

        # ---------- Loss terms ----------
        # 1. Position loss
        pos_loss = torch.mean((pos_fk - pos_target_batch) ** 2)

        # 2. Smoothness loss
        smooth_loss = torch.mean((q_traj[1:] - q_traj[:-1]) ** 2)

        # ---------- Total loss ----------
        loss = lambda_pos * pos_loss + lambda_smooth * smooth_loss

        # ---------- Backward & Update ----------
        loss.backward()
        optimizer.step()

        # ---------- Logging ----------
        if visualize and (step % 50 == 0 or step == steps - 1):
            print(f"Step {step:03d}: "
                  f"loss={loss.item():.6f}, "
                  f"pos={pos_loss.item():.4f}, "
                  f"smooth={smooth_loss.item():.4f}, ")

    return q_traj.detach()



if __name__ == "__main__":
    batch_size = 1
    pos_batch, info_batch = randomize_circles_batch(paper_pos, paper_size, batch_size=batch_size, num_points=25)

    # ======================== Test Visualization ============================
    # xarm_sim.goto_given_conf(jnt_values=xarm_sim.rand_conf())
    # xarm_sim.gen_meshmodel().attach_to(base)
    # visualize_pos_batch(pos_batch)
    # base.run()
    
    # ======================== Optimization Test ============================
    pos_target_batch = pos_batch[0].to(xarm_gpu.device)

    # 优化得到轨迹
    q_traj = optimize_trajectory(
        xarm_gpu,
        pos_target_batch,
        steps=5000,
        lr=1e-2,
        lambda_pos=50.0,
        lambda_smooth=5.0,
        visualize=True
    )

    # ======================= Visualization of Result ============================
    visualize_pos_batch(pos_batch)
    visulize_jnt_batch(q_traj)
    base.run()

