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
# 基础组件：直线采样器
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
        return torch.tensor(np.hstack([xcs, z]), dtype=torch.float32, device=self.device)

# ------------------------------------------------------------
# 核心优化器：强化位置约束
# ------------------------------------------------------------
def optimize_multi_seeds_parallel(sampler, robot, torch_collision_vmap, base, num_seeds=20, dirs_per_seed=16, steps_total=1200):
    device = sampler.device
    total_batch = num_seeds * dirs_per_seed  
    N = 32  
    num_jnts = robot.n_dof

    # 1. 变量初始化
    seeds = sampler.sample_seed_xcs(num_seeds=num_seeds) 
    xc_xy = seeds[:, :2].repeat_interleave(dirs_per_seed, dim=0).detach().requires_grad_(True)
    theta = (torch.rand(total_batch, device=device) * 2 * np.pi).detach().requires_grad_(True)
    # L 初始化为较小值 (-2.0 经过 softplus 后约等于 0.12m)
    raw_L = torch.full((total_batch,), -2.0, device=device, requires_grad=True) 
    q_traj = (torch.randn((total_batch, N, num_jnts), device=device) * 0.1).detach().requires_grad_(True)

    jnt_min = torch.tensor(robot.jnt_ranges[:,0], device=device, dtype=torch.float32)
    jnt_max = torch.tensor(robot.jnt_ranges[:,1], device=device, dtype=torch.float32)

    optimizer = torch.optim.Adam([
        {'params': [raw_L], 'lr': 0.02},
        {'params': [xc_xy, theta], 'lr': 0.01},
        {'params': [q_traj], 'lr': 0.005}
    ])

    ani_sticks = []
    print(f"[Optimization] Focused on Accuracy and Length. Batches: {total_batch}")

    for step in range(steps_total):
        is_sliding = step > 100
        xc_xy.requires_grad = is_sliding
        theta.requires_grad = is_sliding
        
        optimizer.zero_grad()
        
        # --- 几何计算 ---
        L = torch.nn.functional.softplus(raw_L) * 1.5 + 0.02
        d_vec = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1)
        full_xc = torch.cat([xc_xy, torch.full((total_batch, 1), sampler.z_value, device=device)], dim=-1)
        t_samples = torch.linspace(-0.5, 0.5, N, device=device)
        # 生成目标路径： xc + L * t * d
        pos_targets = full_xc.unsqueeze(1) + (L.unsqueeze(-1) * t_samples.unsqueeze(0)).unsqueeze(-1) * d_vec.unsqueeze(1)

        # --- 位置误差 ---
        pos_fk, _ = robot.fk_batch(q_traj.view(-1, num_jnts))
        pos_fk = pos_fk.view(total_batch, N, 3)
        dist_sq = torch.sum((pos_fk - pos_targets)**2, dim=-1) # (batch, N)
        
        loss_pos_mean = torch.mean(dist_sq)
        loss_pos_max = torch.mean(torch.max(dist_sq, dim=1)[0]) # 强制最差的点也要对齐

        # --- 碰撞损失 ---
        coll_cost_raw = torch_collision_vmap(q_traj.view(-1, num_jnts)).view(total_batch, N)
        loss_coll = torch.mean(coll_cost_raw) + 10.0 * torch.max(coll_cost_raw)
        
        # --- 长度激励 (带门控机制) ---
        # 核心：只有当位置误差降到极低时，accuracy_gate 才会变大，允许 L 增长
        # 200.0 是灵敏度系数，误差在 5mm 以上时激励几乎为 0
        accuracy_gate = torch.exp(-loss_pos_mean * 200.0) 
        loss_L = -150.0 * torch.mean(L) * accuracy_gate 
        
        # 物理限制：防止 L 增长到机械臂触及不到的区域
        loss_L_limit = 2000.0 * torch.mean(torch.relu(L - 0.7)**2)

        # --- 其他约束 ---
        loss_smooth = torch.mean((q_traj[:, 1:] - q_traj[:, :-1])**2)
        loss_jnts = torch.mean(torch.relu(jnt_min - q_traj)**2 + torch.relu(q_traj - jnt_max)**2)
        
        # --- 综合损失 ---
        total_loss = 60000.0 * loss_pos_mean \
                     + 25000.0 * loss_pos_max \
                     + loss_L \
                     + loss_L_limit \
                     + 15000.0 * loss_coll \
                     + 500.0 * loss_jnts \
                     + 2000.0 * loss_smooth

        total_loss.backward()
        optimizer.step()

        # 可视化预览
        if step % 20 == 0:
            for s in ani_sticks: s.detach()
            ani_sticks = []
            current_targets = pos_targets.detach().cpu().numpy()
            for seed_i in range(0, num_seeds, 4):
                idx = seed_i * dirs_per_seed
                p0, p1 = current_targets[idx, 0], current_targets[idx, -1]
                tmp_s = mgm.gen_stick(p0, p1, radius=0.003, rgb=[1, 1, 0])
                tmp_s.attach_to(base)
                ani_sticks.append(tmp_s)
            base.task_mgr.step()
            base.graphicsEngine.renderFrame()

        if step % 100 == 0:
            avg_err_mm = torch.sqrt(loss_pos_mean).item() * 1000 
            print(f"Step {step:4d} | Max L: {L.max().item():.3f} | Avg Err: {avg_err_mm:.2f}mm | Coll: {torch.max(coll_cost_raw).item():.6f}")

    for s in ani_sticks: s.detach()
    
    # --- 最终筛选 ---
    q_flat_final = q_traj.detach().view(-1, num_jnts)
    coll_results = torch_collision_vmap(q_flat_final).view(total_batch, N)
    max_coll_per_traj = coll_results.max(dim=1)[0]
    final_pos_err = torch.norm(pos_fk.detach() - pos_targets.detach(), dim=-1).max(dim=1)[0]
    
    # 筛选条件：最大碰撞 < 0.002 且 轨迹上最偏的点误差 < 1.2cm
    success_mask = (max_coll_per_traj < 0.002) & (final_pos_err < 0.012)

    if success_mask.any():
        valid_indices = torch.nonzero(success_mask).squeeze(-1)
        best_idx = valid_indices[torch.argmax(L.detach()[success_mask])]
        print(f"✅ Found! L={L[best_idx]:.4f}, MaxErr={final_pos_err[best_idx]*1000:.2f}mm")
        return {'L': L[best_idx].detach(), 'xc': full_xc[best_idx].detach(), 'd': d_vec[best_idx].detach(),
                'q': q_traj[best_idx].detach(), 'pos_path': pos_targets[best_idx].detach()}
    
    print("❌ No valid solution.")
    return None

# ------------------------------------------------------------
# 主程序入口
# ------------------------------------------------------------
if __name__ == "__main__":
    base = wd.World(cam_pos=[1.2, 0.5, 0.5], lookat_pos=[0.3, 0, 0])
    xarm = xarm6_gpu.XArmLite6GPU()
    robot, device = xarm.robot, xarm.device
    
    mgm.gen_frame().attach_to(base)
    mcm.gen_box(xyz_lengths=[2, 2, 0.03], pos=[0, 0, -0.014], rgb=[0.6, 0.4, 0.2]).attach_to(base)
    mcm.gen_box(xyz_lengths=[1.8, 1.8, 0.002], pos=[0, 0, 0.001], rgb=[1, 1, 1]).attach_to(base)

    # 碰撞模型
    cc_model = SphereCollisionChecker('wrs/robot_sim/robots/xarmlite6_wg/xarm6_sphere_visuals.urdf')
    vmap_jax_cost = jax.jit(jax.vmap(cc_model.self_collision_cost, in_axes=(0, None, None)))
    torch_collision_vmap = jax2torch.jax2torch(lambda q_batch: vmap_jax_cost(q_batch, 1.0, 0.005))
    
    # 路径采样器
    sampler = LineSampler(contour_path='0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl', device=device)
    
    # 执行优化
    best_res = optimize_multi_seeds_parallel(sampler, robot, torch_collision_vmap, base, num_seeds=15, dirs_per_seed=12)

    if best_res:
        # 绘制最终绿线
        mgm.gen_stick(best_res['pos_path'][0].cpu().numpy(), 
                      best_res['pos_path'][-1].cpu().numpy(), 
                      radius=0.006, rgb=[0, 1, 0]).attach_to(base)
        
        # 模拟展示
        rbt_sim = xarm6_sim.XArmLite6Miller(enable_cc=True)
        helpers.visualize_anime_path(base, rbt_sim, best_res['q'].cpu().numpy())
    
    base.run()