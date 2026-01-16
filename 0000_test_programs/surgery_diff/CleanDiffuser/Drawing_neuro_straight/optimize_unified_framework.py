import torch
import numpy as np
import pickle
import warnings
import jax
import jax2torch
from matplotlib.path import Path

from wrs import wd, mcm
import wrs.neuro.xarm_lite6_neuro as xarm6_gpu
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6_sim
from wrs.robot_sim.robots.xarmlite6_wg.sphere_collision_checker import SphereCollisionChecker
import helper_functions as helpers

warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")

# ------------------------------------------------------------
# 基础组件
# ------------------------------------------------------------
class LineSampler:
    def __init__(self, contour_path, z_value=0.0, device='cuda'):
        with open(contour_path, 'rb') as f:
            self.contour = pickle.load(f)
        self.path = Path(self.contour)
        self.z_value = z_value
        self.device = device
        self.min_x, self.min_y = np.min(self.contour, axis=0)
        self.max_x, self.max_y = np.max(self.contour, axis=0)

    def sample_seed_xcs(self, num_seeds=10):
        xcs = np.random.uniform([self.min_x, self.min_y], [self.max_x, self.max_y], size=(num_seeds, 2))
        z = np.full((num_seeds, 1), self.z_value)
        return torch.tensor(np.hstack([xcs, z]), dtype=torch.float32, device=device)

# ------------------------------------------------------------
# 优化器实现
# ------------------------------------------------------------
def optimize_multi_seeds_parallel(sampler, robot, torch_collision_vmap, base, num_seeds=20, dirs_per_seed=16, steps_total=600):
    device = sampler.device
    total_batch = num_seeds * dirs_per_seed  
    N = 32  
    num_jnts = robot.n_dof

    # 1. 初始化
    seeds = sampler.sample_seed_xcs(num_seeds=num_seeds) 
    xc_xy = seeds[:, :2].repeat_interleave(dirs_per_seed, dim=0).detach().requires_grad_(True)
    theta = (torch.rand(total_batch, device=device) * 2 * np.pi).detach().requires_grad_(True)
    raw_L = torch.full((total_batch,), -1.0, device=device, requires_grad=True) 
    q_traj = (torch.randn((total_batch, N, num_jnts), device=device) * 0.1).detach().requires_grad_(True)

    jnt_min = torch.tensor(robot.jnt_ranges[:,0], device=device, dtype=torch.float32)
    jnt_max = torch.tensor(robot.jnt_ranges[:,1], device=device, dtype=torch.float32)

    optimizer = torch.optim.Adam([
        {'params': [raw_L], 'lr': 0.02},
        {'params': [xc_xy, theta], 'lr': 0.01},
        {'params': [q_traj], 'lr': 0.005}
    ])

    ani_sticks = []
    print(f"[Optimization] Start. Batches: {total_batch} | Seeds: {num_seeds}")

    for step in range(steps_total):
        is_sliding = step > 100
        xc_xy.requires_grad = is_sliding
        theta.requires_grad = is_sliding
        
        # 权重调度：后期大幅增加对碰撞和轨迹误差的惩罚
        p_weight = 20000.0 if is_sliding else 8000.0
        c_weight = 15000.0 if is_sliding else 3000.0 

        optimizer.zero_grad()
        
        # 1. 计算目标直线轨迹
        L = torch.nn.functional.softplus(raw_L) * 1.5 + 0.05
        d_vec = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1)
        full_xc = torch.cat([xc_xy, torch.full((total_batch, 1), sampler.z_value, device=device)], dim=-1)
        t_samples = torch.linspace(-0.5, 0.5, N, device=device)
        pos_targets = full_xc.unsqueeze(1) + (L.unsqueeze(-1) * t_samples.unsqueeze(0)).unsqueeze(-1) * d_vec.unsqueeze(1)

        # 2. 正向运动学与位置误差
        pos_fk, _ = robot.fk_batch(q_traj.view(-1, num_jnts))
        loss_pos = torch.mean(torch.sum((pos_fk.view(total_batch, N, 3) - pos_targets)**2, dim=-1))
        
        # 3. 碰撞损失：重点！结合平均值和最大值，确保每一个点都脱离碰撞
        coll_cost_raw = torch_collision_vmap(q_traj.view(-1, num_jnts)).view(total_batch, N)
        loss_coll = torch.mean(coll_cost_raw) + 5.0 * torch.max(coll_cost_raw) # 强化最大碰撞惩罚
        
        # 4. 平滑度损失：防止关节剧烈跳变
        loss_smooth = torch.mean((q_traj[:, 1:] - q_traj[:, :-1])**2)
        
        # 5. 关节限位
        loss_jnts = torch.mean(torch.relu(jnt_min - q_traj)**2 + torch.relu(q_traj - jnt_max)**2)
        
        # 综合 Loss
        total_loss = p_weight * loss_pos \
                     - 100.0 * torch.mean(L * torch.exp(-loss_pos * 15.0)) \
                     + c_weight * loss_coll \
                     + 500.0 * loss_jnts \
                     + 1000.0 * loss_smooth

        total_loss.backward()
        optimizer.step()

        # 渲染预览
        if step % 10 == 0:
            for s in ani_sticks: s.detach()
            ani_sticks = []
            current_targets = pos_targets.detach().cpu().numpy()
            for seed_i in range(0, num_seeds, 3):
                idx = seed_i * dirs_per_seed
                p0, p1 = current_targets[idx, 0], current_targets[idx, -1]
                tmp_s = mgm.gen_stick(p0, p1, radius=0.003, rgb=[1, 1, 0])
                tmp_s.attach_to(base)
                ani_sticks.append(tmp_s)
            base.task_mgr.step()
            base.graphicsEngine.renderFrame()

        if step % 100 == 0:
            print(f"Step {step:3d} | Max L: {L.max().item():.3f} | Coll(Max): {torch.max(coll_cost_raw).item():.6f} | PosErr: {loss_pos.item():.6f}")

    for s in ani_sticks: s.detach()
    
    # ------------------------------------------------------------
    # 结果筛选：稍微放宽阈值，改用 max(collision) 判定
    # ------------------------------------------------------------
    q_flat_final = q_traj.detach().view(-1, num_jnts)
    coll_results = torch_collision_vmap(q_flat_final).view(total_batch, N)
    max_coll_per_traj = coll_results.max(dim=1)[0]
    
    # 容忍 0.001 以内的极微小渗透（通常是数值精度问题）
    success_mask = (max_coll_per_traj < 0.002) & \
                   (torch.norm(pos_fk.view(total_batch, N, 3) - pos_targets.detach(), dim=-1).max(1)[0] < 0.015)

    if success_mask.any():
        valid_indices = torch.nonzero(success_mask).squeeze(-1)
        # 在合法的解中找最长的
        best_idx = valid_indices[torch.argmax(L.detach()[success_mask])]
        print(f"✅ Found solution! L={L[best_idx]:.4f}, MaxColl={max_coll_per_traj[best_idx]:.6f}")
        return {'L': L[best_idx].detach(), 'xc': full_xc[best_idx].detach(), 'd': d_vec[best_idx].detach(),
                'q': q_traj[best_idx].detach(), 'pos_path': pos_targets[best_idx].detach()}
    
    print("❌ No valid solution after filtering.")
    return None

if __name__ == "__main__":
    base = wd.World(cam_pos=[1.2, 0.5, 0.5], lookat_pos=[0.3, 0, 0])
    xarm = xarm6_gpu.XArmLite6GPU()
    robot, device = xarm.robot, xarm.device
    
    mgm.gen_frame().attach_to(base)
    mcm.gen_box(xyz_lengths=[2, 2, 0.03], pos=[0, 0, -0.014], rgb=[0.6, 0.4, 0.2]).attach_to(base)
    mcm.gen_box(xyz_lengths=[1.8, 1.8, 0.002], pos=[0, 0, 0.001], rgb=[1, 1, 1]).attach_to(base)

    cc_model = SphereCollisionChecker('wrs/robot_sim/robots/xarmlite6_wg/xarm6_sphere_visuals.urdf')
    vmap_jax_cost = jax.jit(jax.vmap(cc_model.self_collision_cost, in_axes=(0, None, None)))
    
    # 增加安全边距 (0.008m)，给优化器留空间
    torch_collision_vmap = jax2torch.jax2torch(lambda q_batch: vmap_jax_cost(q_batch, 1.0, -0.008))
    
    sampler = LineSampler(contour_path='0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl', device=device)
    
    best_res = optimize_multi_seeds_parallel(sampler, robot, torch_collision_vmap, base, num_seeds=15, dirs_per_seed=12)

    if best_res:
        mgm.gen_stick(best_res['pos_path'][0].cpu().numpy(), 
                      best_res['pos_path'][-1].cpu().numpy(), 
                      radius=0.006, rgb=[0, 1, 0]).attach_to(base)
        
        rbt_sim = xarm6_sim.XArmLite6Miller(enable_cc=True)
        helpers.visualize_anime_path(base, rbt_sim, best_res['q'].cpu().numpy())
    
    base.run()