import numpy as np
import json, time, atexit
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
def evaluate_circle(robot, cx, cy, radius, num_points=80):
    center = np.array([cx, cy, paper_surface_z + 0.010])
    thetas = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    pos_list = [center + radius * np.array([np.cos(t), np.sin(t), 0.0]) for t in thetas]

    prev_j, prev_R = None, None
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
        prev_j, prev_R = j, robot.fk(jnt_values=j)[1]

    success_ratio = success / num_points
    return success_ratio, fail_ik, fail_coll

# ============ Pareto dominance ============
def dominates(sol1, sol2):
    """Return True if sol1 dominates sol2"""
    s1, r1 = sol1
    s2, r2 = sol2
    return (s1 >= s2 and r1 > r2) or (s1 > s2 and r1 >= r2)

# ============ Multi-objective PSO ============
def pso_multiobjective(n_particles=50, n_iters=80):
    bounds = np.array([
        [0.05, 0.55],   # cx
        [-0.25, 0.25],  # cy
        [0.05, 0.40]    # radius
    ])
    X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_particles, 3))
    V = np.zeros_like(X)
    p_best = X.copy()
    p_best_obj = np.zeros((n_particles, 2))
    g_best = None

    w, c1, c2 = 0.7, 1.4, 1.4

    for it in range(n_iters):
        pareto_front = []

        for i in range(n_particles):
            cx, cy, r = X[i]
            succ, fail_ik, fail_coll = evaluate_circle(robot, cx, cy, r)
            obj = (succ, r)  # 双目标: (success_ratio, radius)

            # 记录所有解
            log_json_line({
                "iter": it + 1,
                "cx": round(cx, 4),
                "cy": round(cy, 4),
                "radius": round(r, 4),
                "success_ratio": round(succ, 3),
                "fail_ik": fail_ik,
                "fail_collision": fail_coll,
                "time_since_start": float(f"{time.time() - start_time:.2f}")
            })

            # 更新个体最优
            if dominates(obj, p_best_obj[i]):
                p_best[i] = X[i]
                p_best_obj[i] = obj

            # 更新 Pareto 前沿
            pareto_front = [p for p in pareto_front if not dominates(obj, p[1])]
            if not any(dominates(p[1], obj) for p in pareto_front):
                pareto_front.append((X[i], obj))

        # 选择全局最优（成功率≥0.9 且半径最大）
        feasible = [p for p in pareto_front if p[1][0] >= 0.9]
        if feasible:
            g_best = max(feasible, key=lambda p: p[1][1])[0]
        else:
            g_best = max(pareto_front, key=lambda p: p[1][0])[0]

        # 粒子更新
        r1, r2 = np.random.rand(*X.shape), np.random.rand(*X.shape)
        V = w * V + c1 * r1 * (p_best - X) + c2 * r2 * (g_best - X)
        X += V
        X = np.clip(X, bounds[:, 0], bounds[:, 1])

        print(f"[iter {it+1:03d}] Pareto size={len(pareto_front)} best succ={max([p[1][0] for p in pareto_front]):.3f}")

    return pareto_front

# ============ Main ============
if __name__ == "__main__":
    pareto_solutions = pso_multiobjective()

    feasible = [p for p in pareto_solutions if p[1][0] >= 0.9]
    best = max(feasible, key=lambda p: p[1][1]) if feasible else max(pareto_solutions, key=lambda p: p[1][0])
    cx, cy, r = best[0]
    succ, _, _ = evaluate_circle(robot, cx, cy, r)
    print(f"\n✅ Best solution: center=({cx:.3f},{cy:.3f}), radius={r:.3f}, success_ratio={succ:.3f}")

    # 可视化最优解
    center = np.array([cx, cy, paper_surface_z + 0.010])
    mgm.gen_sphere(radius=0.006, pos=center, rgb=[0,1,0], alpha=1).attach_to(base)
    for theta in np.linspace(0, 2*np.pi, 80, endpoint=False):
        pos = center + r * np.array([np.cos(theta), np.sin(theta), 0.0])
        mgm.gen_sphere(radius=0.004, pos=pos, rgb=[1,0,0], alpha=1).attach_to(base)

    base.run()
