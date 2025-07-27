import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =========================
# 0. Kinematics: 3-DoF planar arm (redundant for 2D position)
# =========================
def fk_3dof(q, L=(0.4, 0.3, 0.25)):
    """Forward kinematics: 3-link planar arm end-effector position (x,y)."""
    q1, q2, q3 = q
    c1, s1 = np.cos(q1), np.sin(q1)
    c12, s12 = np.cos(q1+q2), np.sin(q1+q2)
    c123, s123 = np.cos(q1+q2+q3), np.sin(q1+q2+q3)
    x = L[0]*c1 + L[1]*c12 + L[2]*c123
    y = L[0]*s1 + L[1]*s12 + L[2]*s123
    return np.array([x, y], dtype=float)

def jacobian_3dof(q, L=(0.4, 0.3, 0.25)):
    """Jacobian 2x3 (x,y wrt q1,q2,q3)."""
    q1, q2, q3 = q
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1+q2), np.cos(q1+q2)
    s123, c123 = np.sin(q1+q2+q3), np.cos(q1+q2+q3)
    J = np.array([
        [-L[0]*s1 - L[1]*s12 - L[2]*s123,  -L[1]*s12 - L[2]*s123,  -L[2]*s123],
        [ L[0]*c1 + L[1]*c12 + L[2]*c123,   L[1]*c12 + L[2]*c123,   L[2]*c123]
    ], dtype=float)
    return J

# =========================
# 1. Line utilities & IK helpers
# =========================
def project_to_line(pt, p0, v_total):
    """Project point 'pt' onto line p0 + sigma * v_total, clamp sigma to [0,1]."""
    v = v_total
    denom = np.dot(v, v)
    if denom < 1e-12:
        return p0.copy(), 0.0
    sigma = np.dot(pt - p0, v) / denom
    sigma_clamped = np.clip(sigma, 0.0, 1.0)
    proj = p0 + sigma_clamped * v
    return proj, sigma_clamped

def ik_iterate_to_target(q_init, x_target, fk_func, jac_func,
                         max_iter=10, eps=1e-6):
    """Small GN/servo IK iterations to hit x_target."""
    q = q_init.copy()
    for _ in range(max_iter):
        x_now = fk_func(q)
        err = x_target - x_now
        if np.linalg.norm(err) < eps:
            break
        J = jac_func(q)
        dq = np.linalg.pinv(J) @ err
        q = q + dq
    return q

def null_space_matrix(q, jac_func):
    J = jac_func(q)
    J_pinv = np.linalg.pinv(J)
    return np.eye(len(q)) - J_pinv @ J

# =========================
# 2. Strict decoder with null-space projection
# =========================
def decode_with_nullspace(Y,
                          fk_func,
                          jac_func,
                          p0,
                          v_total,
                          eps_line=1e-6,
                          sub_steps=3,
                          ik_iter=5):
    """
    Y = [q0 (n), d_sigma (T), z (T*(n-r))]
    For 3 DoF planar, n=3, r=2 -> n-r=1, so z is scalar per step.
    """
    # split
    n = 3
    # You must know T in advance; we infer from shape:
    # total length = n + T + T*(n-r)
    # let z_dim_per_step = n - r = 1
    z_dim = 1

    total_len = Y.shape[0]
    # solve T from total_len = n + T + T*z_dim => T = (total_len - n) / (1 + z_dim)
    T = (total_len - n) // (1 + z_dim)
    assert n + T + T*z_dim == total_len

    q0_diff = Y[:n]
    d_sigma = Y[n:n+T]
    z_all   = Y[n+T:].reshape(T, z_dim)

    # post-process d_sigma
    d_sigma = np.clip(d_sigma, 1e-6, None)
    d_sigma = d_sigma / d_sigma.sum()

    # 1) align q0 to p0 (only position constraint)
    q0 = ik_iterate_to_target(q0_diff, p0, fk_func, jac_func, max_iter=20, eps=1e-7)

    q_list = [q0.copy()]
    sigma  = 0.0
    for t in range(T):
        # optional: projection of current q to line (remove numerical drift)
        x_cur = fk_func(q_list[-1])
        x_proj, sigma_back = project_to_line(x_cur, p0, v_total)
        err_perp = x_cur - x_proj
        if np.linalg.norm(err_perp) > eps_line:
            dq_corr = np.linalg.pinv(jac_func(q_list[-1])) @ (-err_perp)
            q_corr  = q_list[-1] + dq_corr
        else:
            q_corr  = q_list[-1]

        # forward along line
        ds = d_sigma[t]
        for _ in range(sub_steps):
            ds_small = ds / sub_steps
            sigma += ds_small
            sigma = min(sigma, 1.0)
            x_des = p0 + sigma * v_total

            # main task
            dq_main = np.linalg.pinv(jac_func(q_corr)) @ (x_des - fk_func(q_corr))

            # null-space term
            N = null_space_matrix(q_corr, jac_func)
            z_t = z_all[t]  # shape (1,)
            dq_null = (N @ np.array([z_t[0], 0.0, 0.0]))  # simple mapping: only first axis in null
            # you can design a better mapping for multi-dim null space

            q_corr = q_corr + dq_main + dq_null
            # re-project again to be safe
            x_cur2 = fk_func(q_corr)
            x_proj2, sigma_back2 = project_to_line(x_cur2, p0, v_total)
            if np.linalg.norm(x_cur2 - x_proj2) > eps_line:
                q_corr = ik_iterate_to_target(q_corr, x_proj2, fk_func, jac_func,
                                              max_iter=ik_iter, eps=eps_line)
                sigma = sigma_back2

        q_list.append(q_corr)

    return np.array(q_list), d_sigma

# =========================
# 3. Diffusion model (very small MLP)
# =========================
class TinyDiffusion(nn.Module):
    def __init__(self, dim, cond_dim, hidden=256):
        super().__init__()
        self.fc_cond = nn.Linear(cond_dim, hidden)
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, y_noisy, cond_vec):
        h = self.fc_cond(cond_vec)
        x = torch.cat([y_noisy, h], dim=1)
        return self.net(x)

def get_schedule(T_diff=400, beta_start=1e-4, beta_end=0.02, device='cpu'):
    betas = torch.linspace(beta_start, beta_end, T_diff, device=device)
    alphas = 1.0 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cum

def q_sample_ddpm(y0, t, sqrt_alphas_cum, sqrt_one_minus, noise=None):
    if noise is None:
        noise = torch.randn_like(y0)
    return sqrt_alphas_cum[t][:,None]*y0 + sqrt_one_minus[t][:,None]*noise, noise

@torch.no_grad()
def p_sample_ddpm(model, y_t, t, cond, betas, alphas, alphas_cum):
    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = torch.prod(alphas[:t+1])
    eps_hat = model(y_t, cond)
    mean = (1/torch.sqrt(alpha_t))*(y_t - beta_t/torch.sqrt(1-alpha_bar_t)*eps_hat)
    if t > 0:
        z = torch.randn_like(y_t)
    else:
        z = torch.zeros_like(y_t)
    return mean + torch.sqrt(beta_t)*z

# =========================
# 4. Build dataset
# =========================
def build_dataset(num_traj=500, T=40, device='cpu'):
    # line setup
    L_total = 0.25
    v_dir = np.array([1.0, 0.0], dtype=np.float32)
    v_total = L_total * v_dir

    # randomly choose a feasible q0 (around a "home" pose) then project to p0
    def random_q0():
        return np.array([
            np.random.uniform(-math.pi/2, math.pi/2),
            np.random.uniform(-math.pi/2, math.pi/2),
            np.random.uniform(-math.pi/2, math.pi/2)
        ], dtype=np.float32)

    X_list = []
    cond_list = []
    for _ in range(num_traj):
        q0_guess = random_q0()
        p0 = fk_3dof(q0_guess)  # we let start position be whatever q0 suggests
        # build random d_sigma
        base = np.random.uniform(0.01, 0.05, size=T).astype(np.float32)
        base = base / base.sum()

        # random z
        z_dim = 1
        z = np.random.normal(0, 0.1, size=(T, z_dim)).astype(np.float32)

        # pack Y0
        Y0 = np.concatenate([q0_guess, base, z.reshape(-1)])  # dims: n + T + T*z_dim

        # cond: [p0(2), v_total(2)]  (你也可以加环境特征、障碍embedding等)
        cond_vec = np.concatenate([p0, v_total]).astype(np.float32)

        X_list.append(Y0)
        cond_list.append(cond_vec)

    X = torch.tensor(np.stack(X_list), device=device)
    C = torch.tensor(np.stack(cond_list), device=device)
    return X, C, v_total

# =========================
# 5. Train diffusion
# =========================
def train_diffusion(X, C, epochs=2000, batch=64, T_diff=400, lr=1e-3, device='cpu'):
    N, D = X.shape
    cond_dim = C.shape[1]
    model = TinyDiffusion(D, cond_dim).to(device)

    betas, alphas, alphas_cum = get_schedule(T_diff=T_diff, device=device)
    sqrt_alphas_cum = torch.sqrt(alphas_cum)
    sqrt_one_minus = torch.sqrt(1 - alphas_cum)

    optimz = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        idx = torch.randint(0, N, (batch,), device=device)
        y0 = X[idx]
        cond = C[idx]

        t = torch.randint(0, T_diff, (batch,), device=device)
        y_t, noise = q_sample_ddpm(y0, t, sqrt_alphas_cum, sqrt_one_minus)
        pred_noise = model(y_t, cond)
        loss = ((noise - pred_noise)**2).mean()

        optimz.zero_grad()
        loss.backward()
        optimz.step()

        if (ep+1) % 200 == 0:
            print(f"[train] epoch={ep+1} loss={loss.item():.6f}")
    return model, (betas, alphas, alphas_cum)

# =========================
# 6. Sample and decode
# =========================
@torch.no_grad()
def sample_and_decode(model, sched, cond_vec, fk_func, jac_func,
                      p0, v_total, device='cpu'):
    betas, alphas, alphas_cum = sched
    T_diff = len(betas)
    D = model.net[-1].out_features  # dimension of Y

    y_t = torch.randn(1, D, device=device)
    cond_t = torch.tensor(cond_vec[None,:], device=device)

    for t in reversed(range(T_diff)):
        y_t = p_sample_ddpm(model, y_t, t, cond_t, betas, alphas, alphas_cum)

    Y0 = y_t.cpu().numpy().reshape(-1)
    q_traj, d_sigma = decode_with_nullspace(
        Y0, fk_func, jac_func, p0, v_total,
        eps_line=1e-7, sub_steps=3, ik_iter=5
    )
    return q_traj, d_sigma, Y0

# =========================
# 7. Main
# =========================
def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    T = 40
    X, C, v_total = build_dataset(num_traj=600, T=T, device=device)

    model, sched = train_diffusion(X, C, epochs=2000, batch=64, T_diff=400, lr=1e-3, device=device)

    # pick one sample condition
    # We'll define start pose as the p0 from the condition: cond = [p0(2), v_total(2)]
    cond_vec = C[0].cpu().numpy()
    p0 = cond_vec[:2]

    q_traj, d_sigma, Y0 = sample_and_decode(model, sched, cond_vec,
                                            fk_3dof, jacobian_3dof,
                                            p0, v_total, device=device)

    # Plot
    ee_path = np.array([fk_3dof(q) for q in q_traj])
    line_x = np.linspace(p0[0], p0[0]+v_total[0], len(q_traj))
    line_y = np.linspace(p0[1], p0[1]+v_total[1], len(q_traj))

    plt.figure(figsize=(5,5))
    plt.plot(ee_path[:,0], ee_path[:,1], 'o-', label='EE path (sampled)')
    plt.plot(line_x, line_y, 'r--', label='Target line')
    plt.axis('equal'); plt.legend(); plt.title('End-effector vs Line')
    plt.grid(True); plt.show()

    print("sum(d_sigma) =", d_sigma.sum())
    print("q_traj shape =", q_traj.shape)

if __name__ == "__main__":
    main()
