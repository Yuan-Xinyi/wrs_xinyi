import os
import numpy as np
import pickle
from tqdm import tqdm
import scipy.spatial
from scipy.spatial.transform import Rotation
import wrs.basis.robot_math as rm
from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt

def normalize_joint_angles(jnt_values, jnt_ranges):
    jnt_mins = jnt_ranges[:, 0]
    jnt_maxs = jnt_ranges[:, 1]
    return (jnt_values - jnt_mins) / (jnt_maxs - jnt_mins + 1e-8)

def build_kd_trees(jlc, cache_dir="0708_test"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "cobotta_ddik_data.pkl")

    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return data

    sampled_jnts = []
    n_intervals = np.linspace(8, 6, jlc.n_dof, endpoint=False)
    print(f"Buidling Data for DDIK using the following joint granularity: {n_intervals.astype(int)}...")
    for i in range(jlc.n_dof):
        sampled_jnts.append(
            np.linspace(jlc.jnt_ranges[i][0], jlc.jnt_ranges[i][1], int(n_intervals[i] + 2))[1:-1])
    grid = np.meshgrid(*sampled_jnts)
    sampled_qs = np.vstack([x.ravel() for x in grid]).T

    query_data = []
    q_query_data = []
    jnt_data = []
    jinv_data = []
    pos_data = []
    rotmat_data = []
    jmat_data = []

    for jnt_values in tqdm(sampled_qs):
        pos, rotmat, j_mat = jlc.fk(jnt_values=jnt_values, toggle_jacobian=True)
        jinv = np.linalg.pinv(j_mat, rcond=1e-4)
        rel_pos, rel_rotmat = rm.rel_pose(jlc.pos, jlc.rotmat, pos, rotmat)
        rel_rotvec = Rotation.from_matrix(rel_rotmat).as_rotvec()

        query_data.append(np.concatenate((rel_pos, rel_rotvec)))
        norm_jnt_values = normalize_joint_angles(jnt_values, np.asarray(jlc.jnt_ranges))
        q_query_data.append(norm_jnt_values)
        jnt_data.append(jnt_values)
        jinv_data.append(jinv)
        pos_data.append(pos)
        rotmat_data.append(rotmat)
        jmat_data.append(j_mat)

    x_tree = scipy.spatial.cKDTree(query_data)
    q_tree = scipy.spatial.cKDTree(q_query_data)

    result = (
        x_tree,
        q_tree,
        np.array(jnt_data),
        np.array(query_data),
        np.array(jinv_data),
        np.array(pos_data),
        np.array(rotmat_data),
        np.array(jmat_data),
    )

    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
        print(f"Saved DDIK data to {cache_file}")

    return result

# --- 使用逻辑 ---

robot = cbt.Cobotta(pos=rm.vec(0.1, .3, .5), enable_cc=True)
x_tree, q_query_tree, jnt_data, tcp_feats, jinv, pos_data, rotmat_data, jmat_data = build_kd_trees(robot.manipulator.jlc)

jnt = robot.rand_conf()
gth_pos, ght_rotmat, gth_j_mat = robot.manipulator.jlc.fk(jnt_values=jnt, toggle_jacobian=True)
norm_jnt = normalize_joint_angles(jnt, np.asarray(robot.manipulator.jnt_ranges))
jnt_idx = q_query_tree.query(norm_jnt, k=1, workers=-1)[1]
nearest_jnt = jnt_data[jnt_idx]
pos, rotmat, j_mat = pos_data[jnt_idx], rotmat_data[jnt_idx], jmat_data[jnt_idx]

print(f"Ground truth joint configuration: {jnt}")
print(f"Nearest joint configuration: {nearest_jnt}")
print(f"Ground truth normalized joint configuration: {norm_jnt}")
print(f"Nearest joint normalized configuration: {normalize_joint_angles(np.array(nearest_jnt), np.asarray(robot.manipulator.jnt_ranges))}")

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

robot.goto_given_conf(jnt)
robot.gen_meshmodel(alpha=0.3, rgb=[0, 1, 0]).attach_to(base)

robot.goto_given_conf(nearest_jnt)
robot.gen_meshmodel(alpha=0.3, rgb=[0, 0, 1]).attach_to(base)

base.run()
