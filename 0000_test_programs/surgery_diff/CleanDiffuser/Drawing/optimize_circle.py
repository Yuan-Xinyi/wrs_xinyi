import numpy as np
from scipy.optimize import minimize
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.xarmlite6_wg.xarm6_drill as xarm6
import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka

import json, time, os, atexit
from pathlib import Path

start_time = time.time()

# Determine script directory robustly
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

LOG_PATH = SCRIPT_DIR / "opt_log.jsonl"
print(f"[log] writing to: {LOG_PATH}")  
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

log_file = open(LOG_PATH, "w", encoding="utf-8")
atexit.register(log_file.close)

def log_json_line(data: dict):
    try:
        json.dump(data, log_file, ensure_ascii=False)
        log_file.write("\n")
        log_file.flush()
    except Exception as e:
        print("[log error]", e, data)

# ================= Scene Setup =================
# robot = xarm6.XArmLite6Miller(enable_cc=True)
robot = franka.FrankaResearch3(enable_cc=True)
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)

table_size = np.array([1.5, 1.5, 0.05])
table_pos = np.array([0.6, 0, -0.025])
table = mcm.gen_box(xyz_lengths=table_size, pos=table_pos,
                    rgb=np.array([0.6, 0.4, 0.2]), alpha=1)
table.attach_to(base)

paper_size = np.array([1.0, 1.0, 0.002])
paper_pos = table_pos.copy()
paper_pos[0] = paper_size[0] / 2.0
paper_pos[1] = 0.0
paper_pos[2] = table_pos[2] + table_size[2]/2 + paper_size[2]/2
paper = mcm.gen_box(xyz_lengths=paper_size, pos=paper_pos,
                    rgb=np.array([1,1,1]), alpha=1)
paper.attach_to(base)

OBSTACLES = [table, paper]
R_DEFAULT = np.array([[1,0,0],[0,1,0],[0,0,-1]])

# ================= Utils =================
def is_collided(robot, obstacles):
    res = robot.is_collided(obstacle_list=obstacles, toggle_contacts=False, toggle_dbg=False)
    if isinstance(res, (list, tuple)):
        return bool(res[0])
    return bool(res)

def rot_distance(R1, R2):
    d = rm.delta_w_between_rotmat(R1, R2)
    return np.linalg.norm(d)

# ================= Evaluation =================
def evaluate_radius(robot, center, radius, num_points=80,
                    w_pos=1.0, w_smooth=1e-2, w_rot_smooth=1e-1):
    thetas = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    pos_list = [center + radius * np.array([np.cos(t), np.sin(t), 0.0]) for t in thetas]

    jnts_ok = []
    prev_j = None
    prev_R = None
    loss = 0.0
    success = 0
    fail_ik = 0
    fail_coll = 0

    for pos in pos_list:
        j = robot.ik(tgt_pos=pos, tgt_rotmat=R_DEFAULT, seed_jnt_values=prev_j)
        if j is None:
            fail_ik += 1
            continue
        # avoid rendering call during eval
        # robot.goto_given_conf(j)
        if is_collided(robot, OBSTACLES):
            fail_coll += 1
            continue
        success += 1
        jnts_ok.append(j)

        x_curr, R_curr = robot.fk(jnt_values=j)
        loss += w_pos * np.linalg.norm(x_curr - pos)**2

        if prev_j is not None and prev_R is not None:
            loss += w_smooth * np.linalg.norm(j - prev_j)**2
            loss += w_rot_smooth * rot_distance(prev_R, R_curr)**2

        prev_j = j
        prev_R = R_curr

    success_ratio = success / num_points
    return success_ratio, loss, jnts_ok, pos_list, fail_ik, fail_coll

# ================= Objective & Constraint =================
def objective(radius, robot, center, num_points):
    r = float(radius)
    succ, loss, _, _, fail_ik, fail_coll = evaluate_radius(robot, center, r, num_points)
    alpha = 10.0
    beta = 50.0
    threshold = 0.9
    penalty = 0.0
    if succ < threshold:
        penalty = 1e5 * (threshold - succ)**2 * num_points
    obj = loss - alpha * r - beta * (succ * r) + penalty

    log_json_line({
        "radius": r,
        "success_ratio": succ,
        "loss": float(loss),
        "objective": float(obj),
        "fail_ik": fail_ik,
        "fail_collision": fail_coll,
        "time_since_start": time.time() - start_time
    })
    return obj

def constraint_success(radius, robot, center, num_points):
    succ, _, _, _, _, _ = evaluate_radius(robot, center, radius, num_points)
    return succ - 0.9

# ================= Optimization =================
def optimize_radius(robot, center, num_points=80, r_bounds=(0.05, 0.40), x0=0.15):
    # find initial feasible
    for r0 in np.linspace(r_bounds[1], r_bounds[0], 20):
        succ, _, _, _, _, _ = evaluate_radius(robot, center, r0, num_points)
        if succ > 0.3:
            x0 = r0
            print("Initial feasible guess:", x0)
            break

    res = minimize(
        fun=lambda r: objective(r, robot, center, num_points),
        x0=np.array([x0]),
        method="SLSQP",
        bounds=[r_bounds],
        constraints=[{'type': 'ineq', 'fun': lambda r: constraint_success(r, robot, center, num_points)}],
        options={'disp': True, 'maxiter': 200, 'ftol': 1e-6}
    )
    return float(res.x[0])

# ================= Main =================
if __name__ == "__main__":
    num_points = 80
    paper_surface_z = paper_pos[2] + paper_size[2]/2
    circle_center = np.array([0.35, 0.0, paper_surface_z + 0.010])
    mgm.gen_sphere(radius=0.006, pos=circle_center, rgb=[0,1,0], alpha=1).attach_to(base)

    best_r = optimize_radius(robot, circle_center, num_points, r_bounds=(0.05, 0.40))
    print("Optimized radius:", best_r)

    succ, loss, jnt_list, pos_list, _, _ = evaluate_radius(robot, circle_center, best_r, num_points)
    print("Success rate:", succ, "Num feasible:", len(jnt_list))

    for pos in pos_list:
        mgm.gen_sphere(radius=0.004, pos=pos, rgb=[1,0,0], alpha=1).attach_to(base)

    if jnt_list:
        import helper_functions as helper
        helper.visualize_anime_path(base, robot, jnt_list)
    base.run()
