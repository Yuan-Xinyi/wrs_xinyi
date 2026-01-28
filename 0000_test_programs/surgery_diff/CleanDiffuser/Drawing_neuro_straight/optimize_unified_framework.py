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
warnings.filterwarnings("ignore")

MAX_L = 1.5
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


def get_bspline_matrix(N, M, device):
    t = torch.linspace(0.0, 1.0, N, device=device)
    knots = torch.linspace(0.0, 1.0, M, device=device)
    sigma = 1.0 / max(M - 1, 1)
    phi = torch.exp(-0.5 * ((t.unsqueeze(1) - knots.unsqueeze(0)) / sigma) ** 2)
    phi = phi / (phi.sum(dim=1, keepdim=True) + 1e-12)
    return phi


def optimize_multi_seeds_parallel(
    sampler,
    robot,
    torch_collision_vmap,
    base,
    num_seeds=64,
    dirs_per_seed=32,
    steps_total=1000,
    N=32,
    M=16,
    coll_thresh=0.002,
    pos_thresh=0.01,
    vis_every=20,
    print_every=100
):
    device = sampler.device
    num_jnts = robot.n_dof
    total_batch = num_seeds * dirs_per_seed

    raw_seeds = sampler.sample_seed_xcs(num_seeds=num_seeds * 2)
    perm = torch.randperm(raw_seeds.shape[0], device=raw_seeds.device)
    seeds = raw_seeds[perm[:num_seeds]]

    xc_xy = seeds[:, :2].repeat_interleave(dirs_per_seed, dim=0).detach().requires_grad_(True)
    theta = (torch.rand(total_batch, device=device) * 2.0 * np.pi).detach().requires_grad_(True)
    raw_L = torch.zeros(total_batch, device=device, requires_grad=True)

    q_weights = (torch.randn((total_batch, M, num_jnts), device=device) * 0.1).detach().requires_grad_(True)
    Phi = get_bspline_matrix(N, M, device)
    t_samples = torch.linspace(-0.5, 0.5, N, device=device)

    jnt_min = torch.tensor(robot.jnt_ranges[:, 0], device=device, dtype=torch.float32)
    jnt_max = torch.tensor(robot.jnt_ranges[:, 1], device=device, dtype=torch.float32)

    optimizer = torch.optim.Adam([
        {'params': [raw_L], 'lr': 0.03},
        {'params': [xc_xy, theta], 'lr': 0.01},
        {'params': [q_weights], 'lr': 0.02},
    ])

    init_pos = None
    init_coll = None

    rho_coll = 8000.0
    rho_min, rho_max = 2000.0, 400000.0

    warmup_steps = 300
    allow_L_coll_margin = 1.5 * coll_thresh
    allow_L_pos_mse = 0.002
    # allow_L_pos_mse = 1e-4

    ani_sticks = []

    print(f"[Optimization] ProMP/Basis trajectory | batch={total_batch} | N={N} M={M}")

    for step in range(steps_total):
        optimizer.zero_grad(set_to_none=True)

        L_max_physical = MAX_L
        L = 0.05 + (L_max_physical - 0.05) * torch.sigmoid(raw_L)

        d_vec = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1)
        full_xc = torch.cat([xc_xy, torch.full((total_batch, 1), sampler.z_value, device=device)], dim=-1)

        q_traj = torch.einsum('nm,bmj->bnj', Phi, q_weights)

        pos_targets = full_xc.unsqueeze(1) + (L.unsqueeze(-1) * t_samples.unsqueeze(0)).unsqueeze(-1) * d_vec.unsqueeze(1)

        pos_fk, _ = robot.fk_batch(q_traj.reshape(-1, num_jnts))
        pos_fk = pos_fk.view(total_batch, N, 3)

        # for lengeth loss gating
        mse_pos = torch.mean(torch.sum((pos_fk - pos_targets) ** 2, dim=-1))
        # for real position loss on gradients
        per_point_err = torch.norm(pos_fk - pos_targets, dim=-1)      # [B,N]
        err_mean = per_point_err.mean(dim=1)                          # [B]
        err_max  = per_point_err.max(dim=1).values                    # [B]
        loss_pos = (err_mean + 2.0 * err_max).mean()                  # ✅

        coll_cost_raw = torch_collision_vmap(q_traj.reshape(-1, num_jnts)).view(total_batch, N)
        max_coll_now = torch.max(coll_cost_raw).detach()
        # max_coll_now = coll_cost_raw.max(dim=1).values.detach()
        loss_coll = torch.mean(coll_cost_raw) + 10.0 * torch.max(coll_cost_raw)

        if init_pos is None:
            init_pos = loss_pos.detach().abs() + 1e-6
        if init_coll is None:
            init_coll = loss_coll.detach().abs() + 1e-6

        pos_term = loss_pos / init_pos
        coll_term = loss_coll / init_coll

        if step % 25 == 0 and step > 0:
            if max_coll_now > 2.0 * coll_thresh:
                rho_coll = min(rho_coll * 1.35, rho_max)
            elif max_coll_now > 1.2 * coll_thresh:
                rho_coll = min(rho_coll * 1.20, rho_max)
            elif max_coll_now < 0.4 * coll_thresh:
                rho_coll = max(rho_coll * 0.92, rho_min)

        # allow_L = (step >= warmup_steps) and (max_coll_now < allow_L_coll_margin) and (loss_pos.detach() < allow_L_pos_mse)
        allow_L = (
            step >= warmup_steps
            and max_coll_now < allow_L_coll_margin
            and mse_pos.detach() < allow_L_pos_mse
        )


        if step < warmup_steps:
            loss_length = torch.tensor(0.0, device=device)
        else:
            # gate = 1.0 / (1.0 + (loss_pos.detach() / gate_eps))
            gate = 1.0 / (1.0 + (mse_pos.detach() / allow_L_pos_mse))
            loss_length = -torch.mean(L) * gate

        total_loss = 1.0 * pos_term + (rho_coll / 1000.0) * coll_term + (0.15 * loss_length if allow_L else 0.0)

        total_loss.backward()

        if step < warmup_steps or (not allow_L):
            raw_L.grad = None

        optimizer.step()

        with torch.no_grad():
            q_weights.clamp_(jnt_min, jnt_max)

        if vis_every is not None and step % vis_every == 0:
            for s in ani_sticks:
                s.detach()
            ani_sticks = []
            current_targets = pos_targets.detach().cpu().numpy()
            stride = max(1, num_seeds // 10)
            for seed_i in range(0, num_seeds, stride):
                idx = seed_i * dirs_per_seed
                p0, p1 = current_targets[idx, 0], current_targets[idx, -1]
                tmp_s = mgm.gen_stick(p0, p1, radius=0.003, rgb=[1, 1, 0])
                tmp_s.attach_to(base)
                ani_sticks.append(tmp_s)
            base.task_mgr.step()
            base.graphicsEngine.renderFrame()

        if step % print_every == 0:
            print(
                f"Step {step:4d} | "
                f"L_max={L.max().item():.3f} | "
                f"CollMax={max_coll_now.item():.6f} | "
                f"loss_pos={loss_pos.item():.6f} | "
                f"mse_pos={mse_pos.item():.6f} | "
                f"rho={rho_coll:.1f} | "
                f"allow_L={int(allow_L)}"
            )

    for s in ani_sticks:
        s.detach()

    with torch.no_grad():
        L_max_physical = MAX_L
        L = 0.05 + (L_max_physical - 0.05) * torch.sigmoid(raw_L)
        d_vec = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1)
        full_xc = torch.cat([xc_xy, torch.full((total_batch, 1), sampler.z_value, device=device)], dim=-1)
        pos_targets = full_xc.unsqueeze(1) + (L.unsqueeze(-1) * t_samples.unsqueeze(0)).unsqueeze(-1) * d_vec.unsqueeze(1)

        final_q_traj = torch.einsum('nm,bmj->bnj', Phi, q_weights)

        pos_fk, _ = robot.fk_batch(final_q_traj.reshape(-1, num_jnts))
        pos_fk = pos_fk.view(total_batch, N, 3)

        coll_results = torch_collision_vmap(final_q_traj.reshape(-1, num_jnts)).view(total_batch, N)
        max_coll = coll_results.max(dim=1)[0]

        pos_err = torch.norm(pos_fk - pos_targets, dim=-1).max(dim=1)[0]

        success_mask = (max_coll < coll_thresh) & (pos_err < pos_thresh)

        if success_mask.any():
            valid_idx = torch.nonzero(success_mask).squeeze(-1)
            best_idx = valid_idx[torch.argmax(L[success_mask])]
            print(
                f"[Result] Found solution! "
                f"L={L[best_idx].item():.4f}, "
                f"MaxColl={max_coll[best_idx].item():.6f}, "
                f"MaxPosErr={pos_err[best_idx].item():.6f}"
            )
            return {
                'L': L[best_idx].detach(),
                'xc': full_xc[best_idx].detach(),
                'd': d_vec[best_idx].detach(),
                'q': final_q_traj[best_idx].detach(),
                'pos_path': pos_targets[best_idx].detach(),
            }

    print("[Result] No valid solution after filtering.")
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
    torch_collision_vmap = jax2torch.jax2torch(lambda q_batch: vmap_jax_cost(q_batch, 1.0, -0.005))

    sampler = LineSampler(
        contour_path='0000_test_programs/surgery_diff/CleanDiffuser/Drawing_neuro_straight/xarm_contour_z0.pkl',
        device=device
    )

    best_res = optimize_multi_seeds_parallel(
        sampler,
        robot,
        torch_collision_vmap,
        base,
        num_seeds=32,
        dirs_per_seed=16,
        steps_total=6000,
        vis_every=20,
        print_every=100
    )

    if best_res:
        mgm.gen_stick(
            best_res['pos_path'][0].cpu().numpy(),
            best_res['pos_path'][-1].cpu().numpy(),
            radius=0.006,
            rgb=[0, 1, 0]
        ).attach_to(base)

        rbt_sim = xarm6_sim.XArmLite6Miller(enable_cc=True)
        helpers.visualize_anime_path(base, rbt_sim, best_res['q'].cpu().numpy())

    base.run()
