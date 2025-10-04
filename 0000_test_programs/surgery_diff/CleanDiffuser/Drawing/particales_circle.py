import numpy as np
import json, time, os, atexit
from pathlib import Path
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka

# ============ Logging ============
start_time = time.time()
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LOG_PATH = SCRIPT_DIR / "opt_log.jsonl"
print(f"[log] writing to: {LOG_PATH}")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
log_file = open(LOG_PATH, "w", encoding="utf-8")
atexit.register(log_file.close)

def log_json_line(data: dict):
    json.dump(data, log_file, ensure_ascii=False)
    log_file.write("\n")
    log_file.flush()

# ============ Scene Setup ============
robot = franka.FrankaResearch3(enable_cc=True)
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)

table_size = np.array([1.5, 1.5, 0.05])
table_pos = np.array([0.6, 0, -0.025])
table = mcm.gen_box(xyz_lengths=table_size, pos=table_pos, rgb=np.array([0.6, 0.4, 0.2]), alpha=1)
table.attach_to(base)

paper_size = np.array([1.0, 1.0, 0.002])
paper_pos = table_pos.copy()
paper_pos[0] = paper_size[0] / 2.0
paper_pos[1] = 0.0
paper_pos[2] = table_pos[2] + table_size[2]/2 + paper_size[2]/2
paper = mcm.gen_box(xyz_lengths=paper_size, pos=paper_pos, rgb=np.array([1,1,1]), alpha=1)
paper.attach_to(base)

OBSTACLES = [table, paper]
R_DEFAULT = np.array([[1,0,0],[0,1,0],[0,0,-1]])
paper_surface_z = paper_pos[2] + paper_size[2]/2

# ============ Utils ============
def is_collided(robot, obstacles):
    res = robot.is_collided(obstacle_list=obstacles, toggle_contacts=False, toggle_dbg=False)
    return bool(res[0]) if isinstance(res, (list, tuple)) else bool(res)

def rot_distance(R1, R2):
    return np.linalg.norm(rm.delta_w_between_rotmat(R1, R2))

# ============ Evaluation ============
def evaluate_circle(robot, cx, cy, radius, num_points=80,
                    w_pos=1.0, w_smooth=1e-2, w_rot_smooth=1e-1):
    center = np.array([cx, cy, paper_surface_z + 0.010])
    thetas = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    pos_list = [center + radius * np.array([np.cos(t), np.sin(t), 0.0]) for t in thetas]

    prev_j, prev_R = None, None
    loss = 0.0
    success, fail_ik, fail_coll = 0, 0, 0

    for pos in pos_list:
        j = robot.ik(tgt_pos=pos, tgt_rotmat=R_DEFAULT, seed_jnt_values=prev_j)
        if j is None:
            fail_ik += 1
            continue
        if is_collided(robot, OBSTACLES):
            fail_coll += 1
            continue

        success += 1
        x_curr, R_curr = robot.fk(jnt_values=j)
        loss += w_pos * np.linalg.norm(x_curr - pos)**2

        if prev_j is not None and prev_R is not None:
            loss += w_smooth * np.linalg.norm(j - prev_j)**2
            loss += w_rot_smooth * rot_distance(prev_R, R_curr)**2

        prev_j, prev_R = j, R_curr

    success_ratio = success / num_points
    return success_ratio, loss, fail_ik, fail_coll

# ============ Objective ============
def objective(params):
    cx, cy, r = params
    succ, loss, fail_ik, fail_coll = evaluate_circle(robot, cx, cy, r)
    alpha, beta, threshold = 10.0, 50.0, 0.9

    # 软约束惩罚
    penalty = 0.0
    if succ < threshold:
        penalty = 1e5 * (threshold - succ)**2

    obj = loss - alpha * r - beta * (succ * r) + penalty

    log_json_line({
        "cx": round(cx, 4),
        "cy": round(cy, 4),
        "radius": round(r, 4),
        "success_ratio": round(succ, 3),
        "loss": float(f"{loss:.6f}"),
        "objective": float(f"{obj:.6f}"),
        "fail_ik": fail_ik,
        "fail_collision": fail_coll,
        "time_since_start": float(f"{time.time() - start_time:.2f}")
    })
    return obj, succ

# ============ Particle Swarm Optimization ============
def pso_optimize(n_particles=60, n_iters=100):
    # 搜索范围
    bounds = np.array([
        [0.05, 0.55],   # cx
        [-0.25, 0.25],  # cy
        [0.2, 0.40]    # radius
    ])

    # 初始化
    X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_particles, 3))
    V = np.zeros_like(X)
    p_best = X.copy()
    p_best_scores = np.full(n_particles, np.inf)
    g_best = None
    g_best_score = np.inf
    g_best_succ = 0.0

    w, c1, c2 = 0.7, 1.4, 1.4

    for it in range(n_iters):
        for i in range(n_particles):
            score, succ = objective(X[i])
            if score < p_best_scores[i]:
                p_best_scores[i] = score
                p_best[i] = X[i]
            if (score < g_best_score) or (succ >= 0.9 and succ > g_best_succ):
                g_best_score = score
                g_best = X[i].copy()
                g_best_succ = succ

        # 更新速度和位置
        r1, r2 = np.random.rand(*X.shape), np.random.rand(*X.shape)
        V = w * V + c1 * r1 * (p_best - X) + c2 * r2 * (g_best - X)
        X += V
        X = np.clip(X, bounds[:, 0], bounds[:, 1])

        print(f"[iter {it+1:03d}] best obj={g_best_score:.3e}, succ={g_best_succ:.3f}, params={g_best}")

    return g_best, g_best_succ

# ============ Main ============
if __name__ == "__main__":
    best_params, best_succ = pso_optimize()
    cx, cy, r = best_params
    print("\n[RESULT]")
    print(f"Best center: ({cx:.3f}, {cy:.3f}), radius={r:.3f}, success_ratio={best_succ:.3f}")

    # 可视化最优解
    center = np.array([cx, cy, paper_surface_z + 0.010])
    mgm.gen_sphere(radius=0.006, pos=center, rgb=[0,1,0], alpha=1).attach_to(base)

    thetas = np.linspace(0, 2*np.pi, 80, endpoint=False)
    for theta in thetas:
        pos = center + r * np.array([np.cos(theta), np.sin(theta), 0.0])
        mgm.gen_sphere(radius=0.004, pos=pos, rgb=[1,0,0], alpha=1).attach_to(base)

    base.run()
