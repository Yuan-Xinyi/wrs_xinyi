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
        return torch.tensor(np.hstack([xcs, z]), dtype=torch.float32, device=self.device)

# ------------------------------------------------------------
# 强化渲染的并行优化器
# ------------------------------------------------------------
def optimize_multi_seeds_parallel(sampler, robot, torch_collision_vmap, base, num_seeds=20, dirs_per_seed=16, steps_total=600):
    device = sampler.device
    total_batch = num_seeds * dirs_per_seed  
    N = 32  
    num_jnts = robot.n_dof

    # 1. 变量初始化
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

    # 关键：用于存储每帧生成的几何体引用，方便删除
    ani_sticks = []

    print(f"[Visual Optimization] {num_seeds} seeds are converging...")

    for step in range(steps_total):
        is_sliding = step > 100
        xc_xy.requires_grad = is_sliding
        theta.requires_grad = is_sliding
        p_weight = 12000.0 if is_sliding else 6000.0

        optimizer.zero_grad()
        
        L = torch.nn.functional.softplus(raw_L) * 1.5 + 0.05
        d_vec = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1)
        full_xc = torch.cat([xc_xy, torch.full((total_batch, 1), sampler.z_value, device=device)], dim=-1)
        t_samples = torch.linspace(-0.5, 0.5, N, device=device)
        pos_targets = full_xc.unsqueeze(1) + (L.unsqueeze(-1) * t_samples.unsqueeze(0)).unsqueeze(-1) * d_vec.unsqueeze(1)

        pos_fk, _ = robot.fk_batch(q_traj.view(-1, num_jnts))
        loss_pos = torch.mean(torch.sum((pos_fk.view(total_batch, N, 3) - pos_targets)**2, dim=-1))
        
        # Loss 组合
        total_loss = p_weight * loss_pos - 60.0 * torch.mean(L * torch.exp(-loss_pos * 20.0)) + \
                     400.0 * torch.mean(torch.relu(jnt_min - q_traj)**2 + torch.relu(q_traj - jnt_max)**2)

        total_loss.backward()
        optimizer.step()

        # --- 核心修复：强制可视化过程 ---
        if step % 2 == 0:  # 提高刷新频率，每 2 步刷新一次
            # 1. 清理旧线
            for s in ani_sticks:
                s.detach()
            ani_sticks = []
            
            # 2. 绘制所有种子点的当前状态（取每个 seed 组的第一条线作为代表）
            # 转为 numpy 提高绘图速度
            current_targets = pos_targets.detach().cpu().numpy()
            for seed_i in range(num_seeds):
                idx = seed_i * dirs_per_seed
                p0 = current_targets[idx, 0]
                p1 = current_targets[idx, -1]
                
                # 创建预览线
                tmp_s = mgm.gen_stick(p0, p1, radius=0.004, rgb=[1, 1, 0])
                tmp_s.attach_to(base)
                ani_sticks.append(tmp_s)
            
            # 3. 强制刷新 Panda3D 渲染引擎
            base.task_mgr.step()  # 执行任务管理器
            base.graphicsEngine.renderFrame() # 强制显卡渲染当前帧内容

        if step % 100 == 0:
            print(f"Step {step:3d} | Max L: {L.max().item():.3f} | Avg Error: {loss_pos.item():.6f}")

    # 循环结束后清理预览线
    for s in ani_sticks: s.detach()
    base.graphicsEngine.renderFrame()
    
    # 结果筛选与评估
    q_flat = q_traj.detach().view(-1, num_jnts)
    success_mask = (torch_collision_vmap(q_flat).reshape(total_batch, N).sum(1) == 0) & \
                   (torch.norm(pos_fk.view(total_batch, N, 3) - pos_targets.detach(), dim=-1).max(1)[0] < 0.015)

    if success_mask.any():
        idx = torch.nonzero(success_mask).squeeze(-1)[torch.argmax(L.detach()[success_mask])]
        return {'L': L[idx].detach(), 'xc': full_xc[idx].detach(), 'd': d_vec[idx].detach(),
                'q': q_traj[idx].detach(), 'pos_path': pos_targets[idx].detach()}
    return None

if __name__ == "__main__":
    base = wd.World(cam_pos=[1.2, 0.5, 0.5], lookat_pos=[0.3, 0, 0])
    xarm = xarm6_gpu.XArmLite6GPU()
    robot, device = xarm.robot, xarm.device
    
    # 环境展示
    mgm.gen_frame().attach_to(base)
    mcm.gen_box(xyz_lengths=[2, 2, 0.03], pos=[0, 0, -0.014], rgb=[0.6, 0.4, 0.2]).attach_to(base)
    mcm.gen_box(xyz_lengths=[1.8, 1.8, 0.002], pos=[0, 0, 0.001], rgb=[1, 1, 1]).attach_to(base)

    cc_model = SphereCollisionChecker('wrs/robot_sim/robots/xarmlite6_wg/xarm6_sphere_visuals.urdf')
    vmap_jax_cost = jax.jit(jax.vmap(cc_model.self_collision_cost, in_axes=(0, None, None)))
    torch_collision_vmap = jax2torch.jax2torch(lambda q_batch: vmap_jax_cost(q_batch, 1.0, -0.005))
    
    sampler = LineSampler(contour_path='0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl')
    
    # 开始优化
    best_res = optimize_multi_seeds_parallel(sampler, robot, torch_collision_vmap, base, num_seeds=20, dirs_per_seed=16)

    if best_res:
        # 画出最终选定的最长绿色直线
        mgm.gen_stick(best_res['pos_path'][0].cpu().numpy(), 
                      best_res['pos_path'][-1].cpu().numpy(), 
                      radius=0.006, rgb=[0, 1, 0]).attach_to(base)
        
        rbt_sim = xarm6_sim.XArmLite6Miller(enable_cc=True)
        helpers.visualize_anime_path(base, rbt_sim, best_res['q'].cpu().numpy())
    else:
        print("Optimization failed to find a valid solution.")