"""
RRT with R‑Tree–powered nearest‑neighbor search
author: weiwei  (fixed by ChatGPT 2025‑07‑23)
"""

import time, math, uuid, random
import numpy as np
import networkx as nx
from rtree import index                       # pip install Rtree
import wrs.basis.robot_math as rm
import wrs.motion.motion_data as motd

# ------------------------------------------------------------------ #
# 1. R‑Tree + NetworkX Roadmap                                       #
# ------------------------------------------------------------------ #
class RTreeNxRoadmap:
    def __init__(self, dim: int):
        self.dim = dim
        self._init_structures()

    # -------- internal helpers --------
    def _init_structures(self):
        p = index.Property()
        p.dimension = self.dim
        self.idx = index.Index(interleaved=True, properties=p)
        self.g = nx.Graph()
        self._nid_to_rid = {}   # nid → rid
        self._rid_to_nid = {}   # rid → nid
        self._rid_to_conf = {}  # rid → conf ndarray
        self._next_rid = 0

    def _conf_to_bbox(self, conf_nd):
        # R‑Tree 需要 (min... , max...)
        v = conf_nd.astype(np.float32)
        return tuple(v.tolist() + v.tolist())

    # -------- public API (NN part) --------
    def add_conf(self, nid, conf):
        conf = np.asarray(conf, dtype=np.float32)
        rid = self._next_rid
        self._next_rid += 1
        self.idx.insert(rid, self._conf_to_bbox(conf))
        self._nid_to_rid[nid] = rid
        self._rid_to_nid[rid] = nid
        self._rid_to_conf[rid] = conf

    def nearest(self, conf):
        conf = np.asarray(conf, dtype=np.float32)
        rid = next(self.idx.nearest(self._conf_to_bbox(conf), 1))
        return self._rid_to_nid[rid]

    def get_conf(self, nid):
        rid = self._nid_to_rid[nid]
        return self._rid_to_conf[rid]

    # -------- public API (Graph proxy) --------
    def add_node(self, nid, conf):
        conf = np.asarray(conf, dtype=np.float32)
        self.g.add_node(nid, conf=conf)
        self.add_conf(nid, conf)

    def add_edge(self, u, v):
        self.g.add_edge(u, v)

    @property
    def nodes(self):
        return self.g.nodes

    def shortest_path(self, src, dst):
        return nx.shortest_path(self.g, source=src, target=dst)

    def clear(self):
        self._init_structures()

# ------------------------------------------------------------------ #
# 2. RRT Planner                                                     #
# ------------------------------------------------------------------ #
class RRT:
    def __init__(self, robot):
        self.robot = robot
        self.roadmap = RTreeNxRoadmap(dim=robot.n_dof)
        self.toggle_keep = True   # 备份 / 恢复机器人状态

    # ---- 备份/恢复装饰器 ----
    @staticmethod
    def keep_states(func):
        def wrapper(self, *args, **kw):
            if self.toggle_keep:
                self.robot.backup_state()
                res = func(self, *args, **kw)
                self.robot.restore_state()
                return res
            return func(self, *args, **kw)
        return wrapper

    # ---- 工具函数 ----
    def _is_collided(self, conf, obstacles=(), others=()):
        if not self.robot.are_jnts_in_ranges(conf):
            return True
        self.robot.goto_given_conf(conf)
        return self.robot.is_collided(obstacles, others)

    @staticmethod
    def _interpolate(src, dst, step, include_dst=True):
        L, vec = rm.unit_vector(dst - src, toggle_length=True)
        n = max(1, (math.floor if include_dst else math.ceil)(L / step))
        segs = np.linspace(src, src + n * step * vec, n, dtype=float)
        if include_dst and not np.allclose(segs[-1], dst):
            segs = np.vstack([segs, dst])
        return list(segs)

    # ---- roadmap 扩展 ----
    def _extend_roadmap(self, q_rand, ext_dist, q_goal, obstacles, others):
        nearest_nid = self.roadmap.nearest(q_rand)
        q_near = self.roadmap.get_conf(nearest_nid)

        # 沿 (near → rand) 每 ext_dist 插值
        for q_new in self._interpolate(q_near, q_rand, ext_dist)[1:]:
            if self._is_collided(q_new, obstacles, others):
                return nearest_nid            # 碰撞，停止扩展
            new_nid = uuid.uuid4()
            self.roadmap.add_node(new_nid, q_new)
            self.roadmap.add_edge(nearest_nid, new_nid)
            nearest_nid = new_nid

            # 到达目标半径
            if np.linalg.norm(q_new - q_goal) <= ext_dist:
                self.roadmap.add_node("goal", q_goal)
                self.roadmap.add_edge(nearest_nid, "goal")
                return "goal"
        return nearest_nid

    # ---- 主入口 ----
    @keep_states
    def plan(self, q_start, q_goal, *,
             ext_dist=.2, rand_rate=70,
             max_iter=1000, timeout=15.0,
             smooth_iter=80,
             obstacles=(), others=()):

        # 初始化
        self.roadmap.clear()
        self.roadmap.add_node("start", q_start)

        if self._is_collided(q_start, obstacles, others):
            print("Start in collision"); return None
        if self._is_collided(q_goal, obstacles, others):
            print("Goal in collision");  return None

        tic = time.time()
        for _ in range(max_iter):
            if timeout > 0 and time.time() - tic > timeout:
                print("Timeout"); return None

            q_rand = self.robot.rand_conf() if random.randint(0,99) < rand_rate else q_goal
            res = self._extend_roadmap(q_rand, ext_dist, q_goal, obstacles, others)
            if res == "goal":
                raw = [self.roadmap.get_conf(nid)
                       for nid in self.roadmap.shortest_path("start", "goal")]
                smoothed = self._smooth(raw, obstacles, others, ext_dist, smooth_iter)
                mot = motd.MotionData(self.robot); mot.extend(smoothed)
                return mot
        print("Fail"); return None

    # ---- 路径平滑 ----
    def _smooth(self, path, obstacles, others, step, n_iter):
        out = path
        for _ in range(n_iter):
            if len(out) <= 2: break
            i, j = sorted(random.sample(range(len(out)), 2))
            if j == i + 1: continue
            shortcut = self._interpolate(out[i], out[j], step, include_dst=(j == len(out)-1))
            if all(not self._is_collided(p, obstacles, others) for p in shortcut):
                out = out[:i] + shortcut + out[j+1:]
        return out

# ------------------------------------------------------------------ #
# 3. Quick demo (xArm‑Lite6)                                          #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import wrs.visualization.panda.world as wd
    import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm

    base = wd.World(cam_pos=[2,0,1], lookat_pos=[0,0,0])
    robot = xarm.XArmLite6(enable_cc=True)
    planner = RRT(robot)

    q_start = np.array([-0.7, -2.6, 2.0, 1.78, 1.54, -2.59])
    q_goal  = np.array([ 2.94, -0.40, 3.92, -2.15, 1.00, -0.35])

    robot.goto_given_conf(q_start); robot.gen_meshmodel(rgb=[.2,.2,.8],alpha=.3).attach_to(base)
    robot.goto_given_conf(q_goal);  robot.gen_meshmodel(rgb=[.2,.8,.2],alpha=.3).attach_to(base)

    path = planner.plan(q_start, q_goal, ext_dist=.25, timeout=20)
    if path:
        for jv in path.jv_list:
            robot.goto_given_conf(jv)
            robot.gen_meshmodel(rgb=[1,0,0], alpha=.05).attach_to(base)

    base.run()
