import torch
import numpy as np
import time
from wrs import wd, mcm
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
import wrs.modeling.geometric_model as mgm

# ------------------------------------------------------------
# environment setup
# ------------------------------------------------------------
xarm = xarm6_gpu.XArmLite6GPU()
robot = xarm.robot
device = xarm.device
base = wd.World(cam_pos=[1.2, 0.5, 0.5], lookat_pos=[0.3, 0, 0])
mgm.gen_frame().attach_to(base)

table = mcm.gen_box(xyz_lengths=[1.5, 1.5, 0.03], pos=[0.2, 0, -0.025], rgb=[0.6, 0.4, 0.2])
table.attach_to(base)
paper = mcm.gen_box(xyz_lengths=[1.2, 1.2, 0.002], pos=[0.3, 0, 0.001], rgb=[1, 1, 1])
paper.attach_to(base)

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def get_batch_rot_error(rotmat_batch):
    """calculate rotation difference (angle) between consecutive rotation matrices in a batch"""
    src_rot = rotmat_batch[:-1]
    tgt_rot = rotmat_batch[1:]
    
    # R_rel = R_tgt @ R_src^T
    delta_rotmat = torch.matmul(tgt_rot, src_rot.transpose(-1, -2))
    
    # calculate theta
    tr = delta_rotmat.diagonal(dim1=-2, dim2=-1).sum(-1)
    theta = torch.acos(torch.clamp((tr - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6))
    
    # calculate rotation vector
    r_diff = torch.stack([
        delta_rotmat[:, 2, 1] - delta_rotmat[:, 1, 2],
        delta_rotmat[:, 0, 2] - delta_rotmat[:, 2, 0],
        delta_rotmat[:, 1, 0] - delta_rotmat[:, 0, 1]
    ], dim=-1)
    
    # Taylor expansion for small angles
    sin_theta = torch.sin(theta)
    scale = torch.where(sin_theta > 1e-5, theta / (2.0 * sin_theta), 0.5 + (theta**2 / 12.0))
    
    delta_w = r_diff * scale.unsqueeze(-1)
    return torch.norm(delta_w, dim=-1)

def sample_line(xc, d, L, num_points):
    s_list = torch.linspace(-L/2, L/2, num_points, device=device)
    return torch.stack([xc + s*d for s in s_list], dim=0)


def optimize_path(pos_targets, steps=50):
    num_pts = pos_targets.shape[0]
    num_jnts = robot.n_dof

    q_traj = torch.zeros((num_pts, num_jnts), device=device, requires_grad=True)
    with torch.no_grad():
        q_traj += torch.randn_like(q_traj) * 0.1

    optimizer = torch.optim.LBFGS([q_traj], lr=1, max_iter=20, history_size=10)
    
    jnt_min = torch.tensor(robot.jnt_ranges[:,0], dtype=torch.float32, device=device)
    jnt_max = torch.tensor(robot.jnt_ranges[:,1], dtype=torch.float32, device=device)

    def closure():
        optimizer.zero_grad()
    
        pos_fk, rot_fk = robot.fk_batch(q_traj)

        ## position loss
        loss_pos = torch.mean(torch.sum((pos_fk - pos_targets)**2, dim=-1))
        vel = q_traj[1:] - q_traj[:-1]
        acc = q_traj[2:] - 2*q_traj[1:-1] + q_traj[:-2]
        loss_smooth = torch.mean(vel**2) * 1.0 + torch.mean(acc**2) * 0.5
        
        # rotation loss
        rot_errors = get_batch_rot_error(rot_fk)
        loss_rot = torch.mean(rot_errors**2)
        
        # loss towards joint limits
        loss_limit = torch.sum(torch.relu(jnt_min - q_traj)**2) + \
                     torch.sum(torch.relu(q_traj - jnt_max)**2)


        total_loss = 1000.0 * loss_pos + 10.0 * loss_smooth + 50.0 * loss_rot + \
                                         100.0 * loss_limit  
        total_loss.backward()
        return total_loss

    print("starting optimization...")
    for i in range(steps):
        current_loss = optimizer.step(closure)
        if i % 5 == 0:
            print(f"Step {i:02d} | Loss: {current_loss.item():.6f}")

    return q_traj.detach()

# ------------------------------------------------------------
# visualization and execution
# ------------------------------------------------------------
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
rbt_sim = xarm6_sim.XArmLite6Miller(enable_cc=True)

if __name__ == "__main__":
    xc = torch.tensor([0.4, 0.0, 0.02], device=device)
    d = torch.tensor([0.0, 1.0, 0.0], device=device) # 沿Y轴
    pos_path = sample_line(xc, d, 0.3, num_points=40)

    start_t = time.time()
    q_optimized = optimize_path(pos_path)
    print(f"total time: {time.time()-start_t:.2f}s")

    q_np = q_optimized.cpu().numpy()
    for i in range(len(q_np)):
        mgm.gen_sphere(pos_path[i].cpu().numpy(), radius=0.003, rgb=[1,0,0]).attach_to(base)
        if i % 4 == 0:
            rbt_sim.goto_given_conf(q_np[i])
            rbt_sim.gen_meshmodel(alpha=0.3).attach_to(base)
            pass
    base.run()