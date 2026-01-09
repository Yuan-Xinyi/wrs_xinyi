import torch
import numpy as np
import time
from matplotlib.path import Path
import pickle
from wrs import wd, mcm
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
import wrs.modeling.geometric_model as mgm
import warnings
import numpy as np
import pickle
import torch
from matplotlib.path import Path
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

from wrs.robot_sim.robots.xarmlite6_wg.sphere_collision_checker import SphereCollisionChecker 
import jax2torch
import jax

class LineSampler:
    def __init__(self, contour_path, z_value=0.0, device='cuda'):
        with open(contour_path, 'rb') as f:
            self.contour = pickle.load(f)
        
        self.path = Path(self.contour)
        self.z_value = z_value
        self.device = device
        
        # self.min_x, self.min_y = np.min(self.contour, axis=0)
        # self.max_x, self.max_y = np.max(self.contour, axis=0)
        self.min_y, self.max_y = 0, 0 # -0.1, 0.1
        self.min_x, self.max_x = 0.3, 0.3 # 0.2, 0.4

    def is_in_workspace(self, points_xy):
        return self.path.contains_points(points_xy)

    def sample_tasks(self, batch_size=1, L_range=(0.05, 0.5)):
        valid_tasks = []
        
        while len(valid_tasks) < batch_size:
            tmp_xc_xy = np.random.uniform([self.min_x, self.min_y], 
                                          [self.max_x, self.max_y], size=(1, 2))
            
            if not self.is_in_workspace(tmp_xc_xy)[0]:
                continue
            
            theta = np.random.uniform(0, 2 * np.pi)
            d_xy = np.array([np.cos(theta), np.sin(theta)])
            L = np.random.uniform(L_range[0], L_range[1])
            
            xs_xy = tmp_xc_xy[0] - (L / 2) * d_xy
            xe_xy = tmp_xc_xy[0] + (L / 2) * d_xy
        
            check_points = np.stack([tmp_xc_xy[0], xs_xy, xe_xy])
            if np.all(self.is_in_workspace(check_points)):
                task = {
                    'xc': np.append(tmp_xc_xy[0], self.z_value),
                    'd': np.append(d_xy, 0.0),
                    'L': L
                }
                valid_tasks.append(task)
                
        P_xc = torch.tensor([t['xc'] for t in valid_tasks], dtype=torch.float32, device=self.device)
        P_d = torch.tensor([t['d'] for t in valid_tasks], dtype=torch.float32, device=self.device)
        P_L = torch.tensor([t['L'] for t in valid_tasks], dtype=torch.float32, device=self.device)
        
        return P_xc, P_d, P_L

# ------------------------------------------------------------
# environment setup
# ------------------------------------------------------------
xarm = xarm6_gpu.XArmLite6GPU()
robot = xarm.robot
device = xarm.device
base = wd.World(cam_pos=[1.2, 0.5, 0.5], lookat_pos=[0.3, 0, 0])
mgm.gen_frame().attach_to(base)

table = mcm.gen_box(xyz_lengths=[2.0, 2.0, 0.03], pos=[0.2, 0, -0.014], rgb=[0.6, 0.4, 0.2])
table.attach_to(base)
paper = mcm.gen_box(xyz_lengths=[1.8, 1.8, 0.002], pos=[0.3, 0, 0.001], rgb=[1, 1, 1])
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

def sample_line_batch(xc, d, L, num_points, visulize=False):
    B = xc.shape[0]
    device = xc.device
    t = torch.linspace(-0.5, 0.5, num_points, device=device)
    if L.dim() == 1:
        L = L.unsqueeze(-1)
    s_grid = L * t.unsqueeze(0) 
    points = xc.unsqueeze(1) + s_grid.unsqueeze(-1) * d.unsqueeze(1)

    if visulize:
        mgm.gen_sphere([0.3,0,0.002], radius=0.003, rgb=[0,1,0]).attach_to(base)
        for i in range(points.shape[0]):
            mgm.gen_stick(points[i,0].cpu().numpy(),
                        points[i,-1].cpu().numpy(),
                        radius=0.0025,
                        rgb=[0,0,1]).attach_to(base)
        # base.run()
    return points


def optimize_path_batch(pos_targets, cc=None, steps=50):
    B, N, _ = pos_targets.shape # B=100, N=32
    num_jnts = robot.n_dof
    
    q_traj = (torch.randn((B, N, num_jnts), device=device) * 0.1).detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS([q_traj], lr=1, max_iter=20)

    ## collision checker setup
    vmap_jax_cost = jax.jit(jax.vmap(cc.self_collision_cost, in_axes=(0, None, None)))
    torch_collision_vmap = jax2torch.jax2torch(lambda q_batch: vmap_jax_cost(q_batch, 100.0, 0.01))
    
    jnt_min = torch.tensor(robot.jnt_ranges[:,0], device=device, dtype=torch.float32)
    jnt_max = torch.tensor(robot.jnt_ranges[:,1], device=device, dtype=torch.float32)

    def closure():
        optimizer.zero_grad()
        q_flat = q_traj.view(-1, num_jnts)
        
        ## fk distance loss
        pos_fk, rot_fk = robot.fk_batch(q_flat)
        pos_fk = pos_fk.view(B, N, 3)
        rot_fk = rot_fk.view(B, N, 3, 3)

        loss_pos = torch.mean(torch.sum((pos_fk - pos_targets)**2, dim=-1))
        
        vel = q_traj[:, 1:] - q_traj[:, :-1]
        acc = q_traj[:, 2:] - 2*q_traj[:, 1:-1] + q_traj[:, :-2]
        loss_smooth = torch.mean(vel**2) * 1.0 + torch.mean(acc**2) * 0.5
        
        rot_errors = get_batch_rot_error(rot_fk.view(-1, 3, 3)) 
        loss_rot = torch.mean(rot_errors**2)
        
        loss_limit = torch.mean(torch.relu(jnt_min - q_traj)**2 + torch.relu(q_traj - jnt_max)**2)

        loss_cc = torch_collision_vmap(q_flat).mean()

        total_loss = 1000.0 * loss_pos + 10.0 * loss_smooth + 50.0 * loss_rot + 100.0 * loss_limit + 10.0 * loss_cc
        total_loss.backward()
        return total_loss

    for _ in range(steps):
        optimizer.step(closure)
    return q_traj.detach()

# ------------------------------------------------------------
# visualization and execution
# ------------------------------------------------------------
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
import helper_functions as helpers
import jax.numpy as jnp
rbt_sim = xarm6_sim.XArmLite6Miller(enable_cc=True)

if __name__ == "__main__":
    '''if manually specify line:'''
    # xc = torch.tensor([0.4, 0.0, 0.02], device=device)
    # d = torch.tensor([0.0, 1.0, 0.0], device=device) # along y axis
    # pos_path = sample_line(xc, d, 1.4, num_points=40)
    # start_t = time.time()
    # q_optimized = optimize_path(pos_path)
    # print(f"total time: {time.time()-start_t:.2f}s")

    # q_np = q_optimized.cpu().numpy()
    # for i in range(len(q_np)):
    #     mgm.gen_sphere(pos_path[i].cpu().numpy(), radius=0.003, rgb=[1,0,0]).attach_to(base)

    # helpers.visualize_anime_path(base, rbt_sim, q_np)

    '''if randomly sample line within workspace:'''
    ## task sampler
    sampler = LineSampler(contour_path='0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl')
    xcs, ds, Ls = sampler.sample_tasks(batch_size=10, L_range=(0.8, 1.6))
    pos_paths = sample_line_batch(xcs, ds, Ls, num_points=32, visulize=True)
    print("[INFO] optimizing sampled lines...")

    ## collision checker setup
    cc_model = SphereCollisionChecker('wrs/robot_sim/robots/xarmlite6_wg/xarm6_sphere_visuals.urdf')
    _ = cc_model.update(jnp.array(np.zeros(robot.n_dof)))  # to warm up jax
    
    ## optimization with collision checking
    start_t = time.time()
    q_optimized_batch = optimize_path_batch(pos_paths, cc=cc_model, steps=100)
    print(f"[INFO] total time for batch optimization: {time.time()-start_t:.2f}s")
    print("[INFO] visualizing...")
    q_optimized_batch = q_optimized_batch.reshape(-1, 6).cpu().numpy()

    helpers.visualize_anime_path(base, rbt_sim, q_optimized_batch)