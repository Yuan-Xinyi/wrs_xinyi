# planar3d_max_circle_strict.py
# 3-DOF arm in 3D (q0: yaw about Z; q1,q2: pitch about Y)
# Loss-only circle fitting on a PAPER (XY plane at z=z_paper), no projection
# Paper center placed so that the robot base (0,0,0) lies at the MIDPOINT of a chosen edge
# 4-stage schedule: q_only -> unlock_r -> unlock_c -> refit_radius
# Exports a 3D animation (MP4 or GIF)

import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------------- Device / dtype -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
torch.manual_seed(0)

# ----------------------- 3-DOF kinematics ---------------------
# Link lengths
L1, L2, L3 = 0.35, 0.35, 0.20
REACH_MAX = L1 + L2 + L3
REACH_MIN = max(abs(L1 - L2) - L3, 0.0)  # conservative inner sphere

def Rz(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.zeros((*theta.shape, 3, 3), device=device, dtype=DTYPE)
    R[..., 0,0] =  c; R[..., 0,1] = -s; R[..., 0,2] = 0
    R[..., 1,0] =  s; R[..., 1,1] =  c; R[..., 1,2] = 0
    R[..., 2,0] =  0; R[..., 2,1] =  0; R[..., 2,2] = 1
    return R

def Ry(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.zeros((*theta.shape, 3, 3), device=device, dtype=DTYPE)
    R[..., 0,0] =  c; R[..., 0,1] = 0; R[..., 0,2] =  s
    R[..., 1,0] =  0; R[..., 1,1] = 1; R[..., 1,2] =  0
    R[..., 2,0] = -s; R[..., 2,1] = 0; R[..., 2,2] =  c
    return R

def fk_3d(q):  # q: (M,3) -> p: (M,3)
    """Base at origin. Chain: Rz(q0) Tx(L1) Ry(q1) Tx(L2) Ry(q2) Tx(L3). End-eff at last frame."""
    q0, q1, q2 = q[:,0], q[:,1], q[:,2]
    # Frame after first link tip:
    p1 = torch.einsum('mij,mj->mi', Rz(q0), torch.tensor([L1,0.0,0.0], device=device, dtype=DTYPE).repeat(q.shape[0],1))
    # Second link:
    R01 = Rz(q0)
    p2 = p1 + torch.einsum('mij,mj->mi', R01 @ Ry(q1), torch.tensor([L2,0.0,0.0], device=device, dtype=DTYPE).repeat(q.shape[0],1))
    # Third link:
    p3 = p2 + torch.einsum('mij,mj->mi', R01 @ Ry(q1) @ Ry(q2), torch.tensor([L3,0.0,0.0], device=device, dtype=DTYPE).repeat(q.shape[0],1))
    return p3

def fk_links(q_np):  # q_np: (3,) -> (4,3) base, joint1, joint2, tip (for animation)
    q0, q1, q2 = float(q_np[0]), float(q_np[1]), float(q_np[2])
    # build with numpy
    c0, s0 = np.cos(q0), np.sin(q0)
    Rz0 = np.array([[c0,-s0,0],[s0,c0,0],[0,0,1]])
    def Ry_np(a):
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])
    p0 = np.zeros(3)
    p1 = Rz0 @ np.array([L1,0,0])
    p2 = p1 + (Rz0 @ Ry_np(q1)) @ np.array([L2,0,0])
    p3 = p2 + (Rz0 @ Ry_np(q1) @ Ry_np(q2)) @ np.array([L3,0,0])
    return np.stack([p0,p1,p2,p3], axis=0)

# -------------------------- Utils ----------------------------
def inv_softplus(y: torch.Tensor):
    y = torch.clamp(y, min=1e-6)
    return torch.log(torch.expm1(y))

def unit2(v2, eps=1e-9):
    """Normalize 2D vectors along last dim."""
    n = v2.norm(dim=-1, keepdim=True).clamp_min(eps)
    return v2 / n

# -------------------------- Config ----------------------------
@dataclass
class Cfg:
    M: int = 192

    # joint limits (yaw around Z; two pitches around Y)
    q_min: tuple = (math.radians(-90), math.radians(-90), math.radians(-90))
    q_max: tuple = (math.radians( 90), math.radians(90), math.radians(90))

    # circle losses (on PAPER plane)
    lambda_dir: float = 80.0     # direction match in-plane (XY)
    lambda_rad: float = 40.0     # radius consistency
    lambda_rvar: float = 20.0    # radius variance (helper)
    lambda_center: float = 20.0  # mean(p_xy) ~ c_world_xy (helper)
    lambda_plane: float = 200.0  # keep z close to paper z

    # regularizers
    lambda_vel: float = 0.5
    lambda_acc: float = 6.0
    lambda_jl:  float = 1.0
    lambda_reach: float = 200.0
    eps_reach: float = 1e-2
    lambda_cycle: float = 2.0    # closure in joint space

    # radius reward
    lambda_r: float = 1.0

    # schedule & lr
    lr: float = 1e-2
    steps_stage1: int = 2500
    steps_stage2: int = 1500
    steps_stage3: int = 1500
    steps_stage4: int = 1200

    # init
    r_min: float = 0.02
    r_init: float = 0.08
    # c is PAPER-local (x,y on paper)
    c_init: tuple = (0.15, 0.00)

    # stability
    dq_clip: float = 0.35
    r_growth_clip: float = 0.003

    # -------- Paper settings --------
    # Paper is a rectangle on the XY plane at z=z_paper.
    # The robot base (0,0,0) sits at the MIDPOINT of the chosen edge.
    paper_w: float = 0.60
    paper_h: float = 0.40
    z_paper: float = 0.00
    paper_edge: str = "bottom"  # "bottom" | "top" | "left" | "right"
    paper_margin: float = 0.01
    lambda_paper: float = 200.0
    enforce_paper: bool = True

    # animation
    fps: int = 30
    repeat_cycles: int = 2
    out_name: str = "planar3d_circle.mp4"

cfg = Cfg()

q_min = torch.tensor(cfg.q_min, device=device, dtype=DTYPE)
q_max = torch.tensor(cfg.q_max, device=device, dtype=DTYPE)

def reparam_q(phi):
    return 0.5*(q_min + q_max) + 0.5*(q_max - q_min)*torch.tanh(phi)

def softplus_r(rho):
    return F.softplus(rho) + cfg.r_min

# ---------- Paper helpers ----------
def paper_center_world(cfg: Cfg) -> torch.Tensor:
    """Return PAPER center in WORLD so base is at midpoint of chosen edge."""
    z = cfg.z_paper
    if cfg.paper_edge == "bottom":
        return torch.tensor([0.0, cfg.paper_h/2, z], device=device, dtype=DTYPE)
    if cfg.paper_edge == "top":
        return torch.tensor([0.0, -cfg.paper_h/2, z], device=device, dtype=DTYPE)
    if cfg.paper_edge == "left":
        return torch.tensor([cfg.paper_w/2, 0.0, z], device=device, dtype=DTYPE)
    if cfg.paper_edge == "right":
        return torch.tensor([-cfg.paper_w/2, 0.0, z], device=device, dtype=DTYPE)
    raise ValueError("paper_edge must be one of: bottom/top/left/right")

def circle_inside_paper_penalty(c_local_xy: torch.Tensor, r: torch.Tensor, cfg: Cfg):
    if not cfg.enforce_paper:
        return torch.tensor(0.0, device=device, dtype=DTYPE)
    half_w = cfg.paper_w/2 - cfg.paper_margin
    half_h = cfg.paper_h/2 - cfg.paper_margin
    cx, cy = c_local_xy[0], c_local_xy[1]
    viol_x = F.relu(torch.abs(cx) + r - half_w)
    viol_y = F.relu(torch.abs(cy) + r - half_h)
    return viol_x**2 + viol_y**2

# ---------- Reachability (spherical annulus) ----------
def reachability_penalty(c_world: torch.Tensor, r: torch.Tensor):
    d = torch.linalg.norm(c_world, ord=2)                 # center distance to base
    outer = F.relu(d + r - (REACH_MAX - cfg.eps_reach))
    inner = F.relu((REACH_MIN + cfg.eps_reach) - torch.abs(d - r))
    return outer**2 + inner**2

# ------------------------- Optimization -------------------------
def optimize():
    M = cfg.M
    Δ = 2*math.pi / M
    PAPER_C = paper_center_world(cfg)  # (3,)

    # Variables
    phi  = torch.zeros((M, 3), device=device, dtype=DTYPE, requires_grad=True)
    rho  = torch.tensor(math.log(math.exp(cfg.r_init - cfg.r_min) - 1.0),
                        device=device, dtype=DTYPE, requires_grad=True)
    c_local = torch.tensor(cfg.c_init, device=device, dtype=DTYPE, requires_grad=True)  # (2,) paper-local xy
    phi0 = torch.zeros(1, device=device, dtype=DTYPE, requires_grad=True)

    # Heuristic q init: small circle near paper center, yaw facing +x
    with torch.no_grad():
        th = torch.linspace(0, 2*math.pi, M+1, device=device, dtype=DTYPE)[:-1]
        r0 = cfg.r_init
        c0_world = PAPER_C + torch.tensor([c_local[0], c_local[1], 0.0], device=device, dtype=DTYPE)
        px = c0_world[0] + r0*torch.cos(th)
        py = c0_world[1] + r0*torch.sin(th)
        pz = torch.full_like(th, cfg.z_paper)
        # yaw roughly points to x
        q0_guess = torch.atan2(py, px) * 0.2
        # pitches small
        q1_guess = torch.zeros_like(th)
        q2_guess = torch.zeros_like(th)
        phi.data = torch.stack([q0_guess, q1_guess, q2_guess], dim=-1)

    r_fixed = torch.tensor(cfg.r_init, device=device, dtype=DTYPE)
    c_fixed_local = torch.tensor(cfg.c_init, device=device, dtype=DTYPE)

    opt = torch.optim.Adam([phi, rho, c_local, phi0], lr=cfg.lr)

    def losses(q, p, r_var, c_loc_xy, add_radius_term):
        # PAPER center and world circle center
        c_world = PAPER_C + torch.tensor([c_loc_xy[0], c_loc_xy[1], 0.0], device=device, dtype=DTYPE)
        # In-plane XY vectors
        v = p - c_world[None, :]
        v_xy = v[:, :2]
        u = unit2(v_xy)
        idx = torch.arange(M, device=device, dtype=DTYPE)
        d = torch.stack([torch.cos(phi0 + idx*Δ), torch.sin(phi0 + idx*Δ)], dim=-1)

        L_dir = ((u - d)**2).mean()
        radii = v_xy.norm(dim=1)
        L_rad = ((radii - r_var)**2).mean()
        L_rvar = radii.var(unbiased=False)
        L_center = ((p[:, :2].mean(dim=0) - c_world[:2])**2).sum()
        L_plane = ((p[:, 2] - cfg.z_paper)**2).mean()  # keep on paper plane

        # regularizers
        L_vel = ((q[1:] - q[:-1])**2).mean()
        L_acc = ((q[2:] - 2*q[1:-1] + q[:-2])**2).mean()
        L_jl  = (F.relu(q_min - q)**2 + F.relu(q - q_max)**2).mean()
        L_reach = reachability_penalty(c_world, r_var)
        # simple closure (could add 2π periodicity for yaw if needed)
        L_cycle = (q[0] - q[-1]).pow(2).mean()

        L_paper = circle_inside_paper_penalty(c_loc_xy, r_var, cfg)

        total = (cfg.lambda_dir   * L_dir
                 + cfg.lambda_rad * L_rad
                 + cfg.lambda_rvar* L_rvar
                 + cfg.lambda_center*L_center
                 + cfg.lambda_plane * L_plane
                 + cfg.lambda_vel  * L_vel
                 + cfg.lambda_acc  * L_acc
                 + cfg.lambda_jl   * L_jl
                 + cfg.lambda_reach* L_reach
                 + cfg.lambda_cycle* L_cycle
                 + cfg.lambda_paper* L_paper)

        if add_radius_term:
            total = total - cfg.lambda_r * r_var

        return total, dict(L_dir=L_dir, L_rad=L_rad, L_rvar=L_rvar, L_center=L_center,
                           L_plane=L_plane, L_vel=L_vel, L_acc=L_acc, L_jl=L_jl,
                           L_reach=L_reach, L_cycle=L_cycle, L_paper=L_paper)

    def step_once(stage, prev_r=None):
        q = reparam_q(phi)
        r = softplus_r(rho)
        c_loc = c_local

        add_radius = stage in ("unlock_r", "unlock_c")
        if stage == "q_only":
            r = r_fixed; c_loc = c_fixed_local
        elif stage == "unlock_r":
            c_loc = c_fixed_local

        p = fk_3d(q)
        total, d = losses(q, p, r, c_loc, add_radius)

        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_([phi], max_norm=1.0)
        opt.step()

        # temporal trust region
        if cfg.dq_clip is not None:
            with torch.no_grad():
                dq = phi[1:] - phi[:-1]
                norms = dq.norm(dim=1).clamp_min(1e-9)
                scale = torch.clamp(cfg.dq_clip / norms, max=1.0).unsqueeze(-1)
                phi[1:] = phi[:-1] + dq*scale

        # radius growth limiter
        if add_radius and cfg.r_growth_clip is not None and prev_r is not None:
            with torch.no_grad():
                r_curr = softplus_r(rho)
                growth = float((r_curr - prev_r).detach().cpu())
                if growth > cfg.r_growth_clip:
                    target_r = prev_r + cfg.r_growth_clip
                    val = torch.tensor(target_r - cfg.r_min, device=device, dtype=DTYPE)
                    rho.copy_(inv_softplus(val))

        with torch.no_grad():
            cw = (paper_center_world(cfg) + torch.tensor([c_local[0], c_local[1], 0.0], device=device, dtype=DTYPE)).detach().cpu().numpy()
            out = {
                "r": float(r.detach().cpu()),
                "cx": float(cw[0]),
                "cy": float(cw[1]),
                "cz": float(cw[2]),
                "total": float(total.detach().cpu()),
                **{k: float(v.detach().cpu()) for k, v in d.items()},
            }
        return out

    # --------- 4-stage schedule (weights only) ---------
    stages = [("q_only", cfg.steps_stage1),
              ("unlock_r", cfg.steps_stage2),
              ("unlock_c", cfg.steps_stage3),
              ("refit_radius", cfg.steps_stage4)]
    prev_r = None
    lambda_r_backup = cfg.lambda_r
    lr_backup = cfg.lr

    for stage, steps in stages:
        if stage == "q_only":
            cfg.lambda_r   = 0.0
            cfg.lambda_dir = 100.0
            cfg.lambda_rad = 60.0
            cfg.lambda_vel = 0.3
            cfg.lambda_acc = 4.0
            for g in opt.param_groups: g["lr"] = lr_backup

        elif stage == "unlock_r":
            cfg.lambda_r   = 0.2
            cfg.lambda_dir = 80.0
            cfg.lambda_rad = 40.0
            cfg.lambda_vel = 0.5
            cfg.lambda_acc = 6.0
            for g in opt.param_groups: g["lr"] = lr_backup

        elif stage == "unlock_c":
            cfg.lambda_r   = lambda_r_backup
            cfg.lambda_dir = 60.0
            cfg.lambda_rad = 30.0
            cfg.lambda_vel = 1.0
            cfg.lambda_acc = 10.0
            for g in opt.param_groups: g["lr"] = lr_backup * 0.7

        else:  # refit_radius
            cfg.lambda_r   = 0.0
            cfg.lambda_dir = 100.0
            cfg.lambda_rad = 80.0
            cfg.lambda_vel = 1.0
            cfg.lambda_acc = 10.0
            for g in opt.param_groups: g["lr"] = lr_backup * 0.5

        for i in range(steps):
            out = step_once(stage, prev_r)
            if stage in ("unlock_r","unlock_c"):
                prev_r = out["r"]
            if i % 200 == 0 or i == steps - 1:
                print(f"[{stage:>12} {i:4d}] r={out['r']:.4f} c=({out['cx']:.3f},{out['cy']:.3f},{out['cz']:.3f}) "
                      f"dir={out['L_dir']:.3e} rad={out['L_rad']:.3e} rvar={out['L_rvar']:.3e} "
                      f"ctr={out['L_center']:.3e} plane={out['L_plane']:.3e} reach={out['L_reach']:.3e} "
                      f"cyc={out['L_cycle']:.3e} paper={out['L_paper']:.3e} total={out['total']:.3e}")

    # export
    with torch.no_grad():
        q_final = reparam_q(phi)
        r_final = float(softplus_r(rho).detach().cpu())
        c_world_final = (paper_center_world(cfg) + torch.tensor([float(c_local[0]), float(c_local[1]), 0.0], device=device, dtype=DTYPE)).detach().cpu().numpy()
        p_final = fk_3d(q_final).detach().cpu().numpy()
    return {"q": q_final, "r": r_final, "c_world": c_world_final, "p": p_final}

# --------------------------- Animation ---------------------------
def render_animation(res: dict, cfg: Cfg):
    import os, numpy as np, matplotlib.pyplot as plt
    q = res["q"].detach().cpu().numpy() if hasattr(res["q"], "detach") else res["q"]
    p = res["p"]                        # (M,3) numpy
    r = res["r"]; cx, cy, cz = res["c_world"]
    M = q.shape[0]

    q_anim = np.concatenate([q] * cfg.repeat_cycles, axis=0)
    p_anim = np.concatenate([p] * cfg.repeat_cycles, axis=0)
    total_frames = q_anim.shape[0]

    fig = plt.figure(figsize=(7.2, 6.6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3-DOF Arm – Drawing a Circle on Paper (XY plane)")
    lim = REACH_MAX + 0.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(cz-0.2, cz+0.2)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")

    # Paper rectangle at z = z_paper
    pc = paper_center_world(cfg).detach().cpu().numpy()
    hw, hh, z = cfg.paper_w/2, cfg.paper_h/2, cfg.z_paper
    rect_x = [pc[0]-hw, pc[0]+hw, pc[0]+hw, pc[0]-hw, pc[0]-hw]
    rect_y = [pc[1]-hh, pc[1]-hh, pc[1]+hh, pc[1]+hh, pc[1]-hh]
    rect_z = [z, z, z, z, z]
    ax.plot(rect_x, rect_y, rect_z, linestyle="-.", alpha=0.6, label="Paper")

    # Target circle (on paper)
    th = np.linspace(0, 2*np.pi, 720, endpoint=False)
    circle = np.stack([cx + r*np.cos(th), cy + r*np.sin(th), np.full_like(th, cz)], axis=-1)
    circle_line, = ax.plot(circle[:,0], circle[:,1], circle[:,2], lw=1.5, alpha=0.8, label="Target circle")

    # Arm & EE trace
    arm_line, = ax.plot([], [], [], lw=3, marker='o', label="Arm")
    trace_line, = ax.plot([], [], [], lw=1.5, alpha=0.9, label="EE trace")

    ax.scatter([0],[0],[0], c='k', s=25, label="Base")
    ax.legend(loc="upper right")

    trace_pts = []

    def init():
        arm_line.set_data_3d([], [], [])
        trace_line.set_data_3d([], [], [])
        return arm_line, trace_line, circle_line

    def update(frame):
        js = fk_links(q_anim[frame])  # (4,3)
        arm_line.set_data_3d(js[:,0], js[:,1], js[:,2])
        trace_pts.append(p_anim[frame])
        tp = np.array(trace_pts)
        trace_line.set_data_3d(tp[:,0], tp[:,1], tp[:,2])
        return arm_line, trace_line, circle_line

    ani = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=total_frames, interval=1000/cfg.fps, blit=True)

    _, ext = os.path.splitext(cfg.out_name.lower())
    if ext == ".gif":
        try:
            ani.save(cfg.out_name, writer='pillow', fps=cfg.fps)
        except Exception:
            ani.save("fallback.mp4", writer=animation.FFMpegWriter(fps=cfg.fps, bitrate=2400))
            print("GIF export failed; saved fallback.mp4")
    else:
        writer = animation.FFMpegWriter(fps=cfg.fps, bitrate=2400)
        ani.save(cfg.out_name, writer=writer, dpi=160)
    plt.close(fig)
    print(f"Animation saved to: {cfg.out_name}")

# ------------------------------ Main ------------------------------
if __name__ == "__main__":
    res = optimize()

    # quick static check (optional)
    pxy = res["p"]
    r = res["r"]; cx, cy, cz = res["c_world"]
    print(f"\nFinal: r* = {r:.4f}, c_world* = ({cx:.3f}, {cy:.3f}, {cz:.3f}), samples = {len(pxy)}")

    # export animation
    render_animation(res, cfg)
