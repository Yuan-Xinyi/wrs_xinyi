from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.rs007l.rs007l as rs007l
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.manipulators.ur3e.ur3e as ur3e
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi
import wrs.robot_sim.robots.cobotta_pro1300.cobotta_pro1300 as cbtpro1300
import wrs.robot_sim.robots.cobotta_pro900.cobotta_pro900_spine as cbtpro900

import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

# robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
# file_name = "workspace_refined_cbt.npz"

# robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), enable_cc=True)
# file_name = f"workspace_refined_ur3.npz"

# robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
# file_name = "workspace_refined_yumi.npz"

robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)
file_name = "workspace_refined_cbtpro.npz"

jnt_values = robot.rand_conf()
tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)

# ===== 参数 =====
N_SAMPLES = 200000           # 关节空间随机采样数量
VOXEL_H = 0.5               # 位置体素边长（米），如 0.02~0.03m
ICO_LEVEL = 1                # icosphere 细分等级：2≈162轴，3≈642轴
ANG_STEP_DEG = 30            # 旋转角分箱步长（度），10~15 常用
PREFER = "margin"            # "first" 或 "margin"（同格保留关节余量更大者）
RNG = np.random.default_rng(0)

# ===== icosphere 顶点（均匀轴） =====
def icosphere_vertices(level=2):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0],
        [ 1,  t,  0],
        [-1, -t,  0],
        [ 1, -t,  0],
        [ 0, -1,  t],
        [ 0,  1,  t],
        [ 0, -1, -t],
        [ 0,  1, -t],
        [ t,  0, -1],
        [ t,  0,  1],
        [-t,  0, -1],
        [-t,  0,  1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
    ], dtype=int)

    for _ in range(level):
        mid_cache = {}

        def midpoint(i, j, vlist):
            key = (min(i, j), max(i, j))
            if key in mid_cache:
                return mid_cache[key]
            # 这里确保是 numpy 向量运算
            vv = (vlist[i] + vlist[j]) * 0.5
            vv = vv / np.linalg.norm(vv)
            vlist.append(vv)
            idx = len(vlist) - 1
            mid_cache[key] = idx
            return idx

        # 关键修改：保持为“ndarray 的列表”，而不是 list 的列表
        vlist = [verts[k].copy() for k in range(verts.shape[0])]

        new_faces = []
        for f in faces:
            a = midpoint(f[0], f[1], vlist)
            b = midpoint(f[1], f[2], vlist)
            c = midpoint(f[2], f[0], vlist)
            new_faces += [
                [f[0], a, c],
                [f[1], b, a],
                [f[2], c, b],
                [a, b, c]
            ]
        verts = np.array(vlist, dtype=float)
        faces = np.array(new_faces, dtype=int)

    # 再归一化一遍，去除数值误差
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    return verts


AXES = icosphere_vertices(ICO_LEVEL)              # (M,3)
ANG_STEP = np.deg2rad(ANG_STEP_DEG)
AXES_T = AXES.T                                   # 便于向量化点乘

# ===== 旋转矩阵 -> 轴角 =====
def rotmat_to_axis_angle(R, eps=1e-12):
    tr = np.trace(R)
    c = np.clip((tr-1.0)/2.0, -1.0, 1.0)
    theta = np.arccos(c)  # [0,pi]
    if theta < 1e-7:
        return np.array([1.0,0.0,0.0]), 0.0
    rx = R[2,1]-R[1,2]; ry = R[0,2]-R[2,0]; rz = R[1,0]-R[0,1]
    axis = np.array([rx,ry,rz])/(2.0*np.sin(theta)+eps)
    n = np.linalg.norm(axis); 
    if n < eps: axis = np.array([1.0,0.0,0.0])
    else: axis = axis/n
    return axis, theta  # theta in [0,pi]

# ===== 位置体素ID（原点参考，稀疏哈希网格） =====
def voxel_id(pos, h=VOXEL_H):
    return tuple(np.floor(pos/h).astype(int))

# ===== 旋转分箱（icosphere最近轴 + 角度分箱） =====
def rot_bin(R):
    axis, theta = rotmat_to_axis_angle(R)
    # 最近轴（最大点乘）
    idx = int(np.argmax(AXES @ axis))
    b = int(np.floor(theta/ANG_STEP))
    if b > int(np.floor(np.pi/ANG_STEP)):
        b = int(np.floor(np.pi/ANG_STEP))
    return (idx, b)

# ===== 最小关节余量（可选择优使用） =====
# 需要你的关节限位：下例用 Cobotta 的 robot.jlc.jnt_ranges（若接口不同请替换）
try:
    JNT_LIMS = np.array(robot.jlc.jnt_ranges)  # 形如 [[low,upp],...]
except:
    JNT_LIMS = None

def min_margin(q):
    if JNT_LIMS is None: return None
    low = JNT_LIMS[:,0]; upp = JNT_LIMS[:,1]
    return float(np.min(np.minimum(q-low, upp-q)))

# ===== 采样 + 网格化精炼 =====
kept = {}  # key -> dict(sample)
cnt = 0
for _ in tqdm(range(N_SAMPLES), desc="Sampling + FK + binning"):
    # 1) 关节随机采样（均匀）
    q = robot.rand_conf()  # 若要独立均匀每关节，可用 low+(upp-low)*RNG.random(...)
    # 2) FK
    p, R = robot.fk(q)
    p = np.asarray(p); R = np.asarray(R)
    # 3) 位置体素 + 旋转分箱
    vid = voxel_id(p, VOXEL_H)
    rid = rot_bin(R)
    key = (vid, rid)
    # 4) 去重/择优
    if key not in kept:
        kept[key] = {"q":q, "p":p, "R":R, "score":(min_margin(q) or 0.0)}
    else:
        if PREFER == "margin":
            new_score = (min_margin(q) or 0.0)
            if new_score > kept[key]["score"]:
                kept[key] = {"q":q, "p":p, "R":R, "score":new_score}
        # else: PREFER=="first" 不替换
    cnt += 1

refined = list(kept.values())
print(f"[Refine] raw={N_SAMPLES}, kept={len(refined)} (voxel={VOXEL_H}m, icoL={ICO_LEVEL}, dθ={ANG_STEP_DEG}°)")

# =====（可选）把 refined 的目标画出来看看覆盖情况 =====
# 仅画位置点云
pts = np.stack([it["p"] for it in refined], axis=0)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=2)
ax.set_title("Workspace after voxel+rotation bin refine")
ax.set_box_aspect([1,1,1])
plt.show()

# ===== refined 集合可用于 IK 验证 =====
# 举例：对每个目标位姿跑一次数值 IK
# for it in refined:
#     tgt_pos, tgt_rotmat = it["p"], it["R"]
#     q_sol = your_numik_solver.solve(tgt_pos, tgt_rotmat, seed=robot.rand_conf())
#     # 记录成功/失败、时间、步数……

# ===== 保存 refined 样本到 .npz =====
out_path = f"{file_name}".format(
    VOXEL_H, ICO_LEVEL, ANG_STEP_DEG
)

Q   = np.stack([it["q"] for it in refined], axis=0).astype(np.float32)      # (N, dof)
POS = np.stack([it["p"] for it in refined], axis=0).astype(np.float32)      # (N, 3)
ROT = np.stack([it["R"] for it in refined], axis=0).astype(np.float32)      # (N, 3, 3)

# 可选：把离散 key 也存一下，便于复现/分析
KEY_VOX = np.array([k[0] for k in kept.keys()], dtype=np.int32)             # (N, 3) 体素索引
KEY_ROT = np.array([k[1] for k in kept.keys()], dtype=np.int32)             # (N, 2) (axis_id, angle_bin)

# 可选：存一点元数据，便于追踪实验配置
meta = {
    "voxel_h_m": float(VOXEL_H),
    "ico_level": int(ICO_LEVEL),
    "ang_step_deg": int(ANG_STEP_DEG),
    "prefer": str(PREFER),
    "n_raw": int(N_SAMPLES),
    "n_kept": int(len(refined)),
    "jnt_lims": (JNT_LIMS.astype(np.float32).tolist() if JNT_LIMS is not None else None),
}
# 关节限位（若可用）
if JNT_LIMS is not None:
    meta["jnt_lims"] = JNT_LIMS.astype(np.float32).tolist()

np.savez_compressed(
    out_path,
    q=Q, pos=POS, rot=ROT,
    key_vox=KEY_VOX, key_rot=KEY_ROT,
    meta=json.dumps(meta)  # dict 转成 json 字符串存进去
)

print(f"Saved refined set to {out_path}  |  q:{Q.shape} pos:{POS.shape} rot:{ROT.shape}")

