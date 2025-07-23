"""
RRT with FAISS‑powered nearest‑neighbor search
author: weiwei (refactored by ChatGPT)
date: 2025‑07‑23
"""

import time, math, uuid, random
from operator import itemgetter

import numpy as np
import faiss                      # pip install faiss‑cpu
import networkx as nx
import matplotlib.pyplot as plt
import scipy.spatial              # 只剩下平滑时可能还会用到

import wrs.basis.robot_math as rm
import wrs.motion.motion_data as motd
import wrs.modeling.geometric_model as mgm


# --------------------------------------------------------------------- #
#                       1. Roadmap 结构：Faiss + Nx                     #
# --------------------------------------------------------------------- #
class FaissNxRoadmap:
    """
    - 用 FAISS.IndexFlatL2 做最近邻搜索（add / search O(1)）
    - 用 networkx.Graph 存储拓扑（add_node / add_edge / shortest_path）
    - 对外暴露 .nodes / .edges 与 NetworkX 一致的接口
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.g = nx.Graph()                       # 拓扑关系
        self.idx = faiss.IndexFlatL2(dim)         # 精确 L2 NN
        self._nid_to_row = []                     # nid -> row id
        self._rows_conf = []                      # row id -> conf (np.ndarray)

    # ---------- NN 接口 ----------
    def add_conf(self, nid, conf):
        conf = np.asarray(conf, dtype=np.float32).reshape(1, -1)
        self.idx.add(conf)
        self._nid_to_row.append(nid)
        self._rows_conf.append(conf[0])

    def nearest(self, conf):
        conf = np.asarray(conf, dtype=np.float32).reshape(1, -1)
        D, I = self.idx.search(conf, 1)
        row = int(I[0, 0])
        return self._nid_to_row[row]

    def get_conf(self, nid):
        row = self._nid_to_row.index(nid)
        return self._rows_conf[row]

    # ---------- Graph proxy ----------
    # 直接把 networkx.Graph 的常用方法复用一下

    def add_node(self, nid, conf):
        self.g.add_node(nid, conf=conf)
        self.add_conf(nid, conf)

    def add_edge(self, u, v):
        self.g.add_edge(u, v)

    @property
    def nodes(self):
        return self.g.nodes

    @property
    def edges(self):
        return self.g.edges

    def clear(self):
        self.__init__(self.dim)      # 重新初始化

    # 让现有算法还能用 nx 的 shortest_path
    def shortest_path(self, source, target):
        return nx.shortest_path(self.g, source=source, target=target)


# --------------------------------------------------------------------- #
#                                2. RRT                                 #
# --------------------------------------------------------------------- #
class RRT:
    def __init__(self, robot):
        self.robot = robot
        self.roadmap = FaissNxRoadmap(dim=robot.n_dof)
        self.start_conf = None
        self.goal_conf = None
        self.toggle_keep = True     # 是否备份 / 恢复机器人状态

    # -------- decorator: 自动备份 / 恢复机器人状态 --------
    @staticmethod
    def keep_states(method):
        def wrapper(self, *args, **kwargs):
            if self.toggle_keep:
                self.robot.backup_state()
                res = method(self, *args, **kwargs)
                self.robot.restore_state()
                return res
            return method(self, *args, **kwargs)
        return wrapper

    # ------------------ 工具函数 ------------------
    def _is_collided(self, conf,
                     obstacle_list=(), other_robot_list=(),
                     toggle_contacts=False, toggle_dbg=False):
        if not self.robot.are_jnts_in_ranges(conf):
            return True
        self.robot.goto_given_conf(conf)
        return self.robot.is_collided(obstacle_list, other_robot_list,
                                      toggle_contacts)

    def _sample_conf(self, rand_rate, default_conf):
        return self.robot.rand_conf() if random.randint(0, 99) < rand_rate else default_conf

    # 取最近节点 —— **已经用 FAISS 实现**
    def _nearest_nid(self, conf):
        return self.roadmap.nearest(conf)

    # 按步长产生中间插值
    @staticmethod
    def _interpolate(src, dst, step, include_dst=True):
        L, vec = rm.unit_vector(dst - src, toggle_length=True)
        n = max(1, (math.floor if include_dst else math.ceil)(L / step))
        path = np.linspace(src, src + n * step * vec, n, dtype=float)
        if include_dst and not np.allclose(path[-1], dst):
            path = np.vstack([path, dst])
        return list(path)

    # ---------- roadmap 扩展 ----------
    def _extend_roadmap(self, conf_rand, ext_dist, goal_conf,
                        obstacle_list, other_robot_list, animation=False):
        nearest = self._nearest_nid(conf_rand)
        near_conf = self.roadmap.get_conf(nearest)

        for new_conf in self._interpolate(near_conf, conf_rand, ext_dist, include_dst=True)[1:]:
            if self._is_collided(new_conf, obstacle_list, other_robot_list):
                return nearest
            new_nid = uuid.uuid4()
            self.roadmap.add_node(new_nid, new_conf)
            self.roadmap.add_edge(nearest, new_nid)
            nearest = new_nid

            # 到终点判定
            if np.linalg.norm(new_conf - goal_conf) <= ext_dist:
                self.roadmap.add_node("goal", goal_conf)
                self.roadmap.add_edge(nearest, "goal")
                return "goal"
        return nearest

    # ------------------ 主要入口 ------------------
    @keep_states
    def plan(self, start_conf, goal_conf,
             obstacle_list=(), other_robot_list=(),
             ext_dist=.2, rand_rate=70,
             max_n_iter=1000, max_time=15.0,
             smoothing_n_iter=50, animation=False):

        self.roadmap.clear()
        self.start_conf, self.goal_conf = start_conf, goal_conf
        self.roadmap.add_node("start", start_conf)

        # 起终检查
        if self._is_collided(start_conf, obstacle_list, other_robot_list):
            print("Start in collision"); return None
        if self._is_collided(goal_conf,  obstacle_list, other_robot_list):
            print("Goal in collision");  return None
        if np.linalg.norm(start_conf - goal_conf) <= ext_dist:
            mot = motd.MotionData(self.robot); mot.extend([start_conf, goal_conf]); return mot

        tic = time.time()
        for _ in range(max_n_iter):
            if max_time > 0 and time.time() - tic > max_time:
                print("Timeout"); return None
            q_rand = self._sample_conf(rand_rate, goal_conf)
            last = self._extend_roadmap(q_rand, ext_dist, goal_conf,
                                        obstacle_list, other_robot_list, animation)
            if last == "goal":
                raw_path = [data["conf"] for _, data in
                            self.roadmap.nodes(data=True) if _ in
                            self.roadmap.shortest_path("start", "goal")]
                smoothed = self._smooth_path(raw_path, obstacle_list,
                                             other_robot_list, ext_dist,
                                             smoothing_n_iter)
                mot = motd.MotionData(self.robot); mot.extend(smoothed)
                return mot
        print("Fail"); return None

    # ------------------ 路径平滑 ------------------
    def _smooth_path(self, path, obstacle_list, other_robot_list,
                     granularity, n_iter):
        out = path
        for _ in range(n_iter):
            if len(out) <= 2: break
            i, j = sorted(random.sample(range(len(out)), 2))
            if j == i + 1: continue
            shortcut = self._interpolate(out[i], out[j], granularity,
                                         include_dst=(j == len(out) - 1))
            if all(not self._is_collided(p, obstacle_list, other_robot_list)
                   for p in shortcut):
                out = out[:i] + shortcut + out[j + 1:]
        return out


# --------------------------------------------------------------------- #
#                             3. demo / test                            #
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm
    import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    robot = xarm.XArmLite6(enable_cc=True)
    planner = RRT(robot)

    start_conf = np.array([-0.7, -2.6, 2.0, 1.78, 1.54, -2.59])
    goal_conf  = np.array([ 2.94, -0.40, 3.92, -2.15, 1.00, -0.35])

    robot.goto_given_conf(start_conf)
    robot.gen_meshmodel(rgb=rm.const.steel_blue, alpha=.3).attach_to(base)
    robot.goto_given_conf(goal_conf)
    robot.gen_meshmodel(rgb=[0,1,0], alpha=.3).attach_to(base)

    path_mot = planner.plan(start_conf, goal_conf,
                            ext_dist=.2, max_time=20)      # 20 秒超时
    if path_mot:
        for jv in path_mot.jv_list:
            robot.goto_given_conf(jv)
            robot.gen_meshmodel(rgb=[1,0,0], alpha=.05).attach_to(base)

    base.run()
