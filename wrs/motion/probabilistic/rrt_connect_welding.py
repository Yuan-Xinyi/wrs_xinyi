import time
import uuid
import networkx as nx
import numpy as np
import wrs.motion.probabilistic.rrt as rrt


class RRTConnectWelding(rrt.RRT):

    def __init__(self, robot):
        super().__init__(robot)
        self.roadmap_start = nx.Graph()
        self.roadmap_goal = nx.Graph()
        self.z_target = None
        self.z_tol = 0.01

    def _sample_conf(self, rand_rate=100, default_conf=None):
        while True:
            conf = self.robot.rand_conf() if default_conf is None else np.array(default_conf)
            if self.z_target is None:
                return conf
            tgt_pos, _ = self.robot.fk(jnt_values=conf)
            if abs(tgt_pos[2] - self.z_target) < self.z_tol:
                return conf

    def _extend_roadmap(self,
                        roadmap,
                        conf,
                        ext_dist,
                        goal_conf,
                        obstacle_list=[],
                        other_robot_list=[],
                        animation=False):
        nearest_nid = self._get_nearest_nid(roadmap, conf)
        new_conf_list = self._extend_conf(roadmap.nodes[nearest_nid]["conf"], conf, ext_dist, exact_end=False)[1:]
        for new_conf in new_conf_list:
            if self.z_target is not None:
                tgt_pos, _ = self.robot.fk(jnt_values=new_conf)
                if abs(tgt_pos[2] - self.z_target) >= self.z_tol:
                    continue
            if self._is_collided(new_conf, obstacle_list, other_robot_list):
                return -1
            else:
                new_nid = uuid.uuid4()
                roadmap.add_node(new_nid, conf=new_conf)
                roadmap.add_edge(nearest_nid, new_nid)
                nearest_nid = new_nid
                if animation:
                    self.draw_wspace([self.roadmap_start, self.roadmap_goal], self.start_conf, self.goal_conf,
                                     obstacle_list, [roadmap.nodes[nearest_nid]["conf"], conf], new_conf,
                                     "^c")
                if self._is_goal_reached(conf=roadmap.nodes[new_nid]["conf"], goal_conf=goal_conf, threshold=ext_dist):
                    roadmap.add_node("connection", conf=goal_conf)
                    roadmap.add_edge(new_nid, "connection")
                    return "connection"
        return nearest_nid

    @rrt.RRT.keep_states_decorator
    def plan(self,
             start_conf,
             goal_conf,
             obstacle_list=[],
             other_robot_list=[],
             ext_dist=.2,
             max_n_iter=10000,
             max_time=15.0,
             smoothing_n_iter=500,
             animation=False,
             toggle_dbg=False,
             z_target=None,
             z_tol=0.01):
        self.roadmap.clear()
        self.roadmap_start.clear()
        self.roadmap_goal.clear()
        self.start_conf = start_conf
        self.goal_conf = goal_conf
        self.z_target = z_target
        self.z_tol = z_tol

        if toggle_dbg:
            print("RRT: Checking start robot configuration...")
        if self._is_collided(start_conf, obstacle_list, other_robot_list, toggle_dbg=toggle_dbg):
            print("RRT: The start robot configuration is in collision!")
            return None
        if toggle_dbg:
            print("RRT: Checking goal robot configuration...")
        if self._is_collided(goal_conf, obstacle_list, other_robot_list, toggle_dbg=toggle_dbg):
            print("RRT: The goal robot configuration is in collision!")
            return None
        if self._is_goal_reached(conf=start_conf, goal_conf=goal_conf, threshold=ext_dist):
            mot_data = rrt.motd.MotionData(self.robot)
            mot_data.extend(jv_list=[start_conf, goal_conf])
            return mot_data

        self.roadmap_start.add_node("start", conf=start_conf)
        self.roadmap_goal.add_node("goal", conf=goal_conf)

        tic = time.time()
        tree_a = self.roadmap_start
        tree_b = self.roadmap_goal
        tree_a_goal_conf = self.roadmap_goal.nodes["goal"]["conf"]
        tree_b_goal_conf = self.roadmap_start.nodes["start"]["conf"]

        for _ in range(max_n_iter):
            toc = time.time()
            if max_time > 0.0 and toc - tic > max_time:
                print("Failed to find a path in the given max_time!")
                return None

            rand_conf = self._sample_conf(rand_rate=100)
            last_nid = self._extend_roadmap(tree_a, rand_conf, ext_dist, tree_a_goal_conf,
                                            obstacle_list, other_robot_list, animation)
            if last_nid != -1:
                goal_nid = last_nid
                tree_b_goal_conf = tree_a.nodes[goal_nid]["conf"]
                last_nid = self._extend_roadmap(tree_b, tree_a.nodes[last_nid]["conf"], ext_dist,
                                                tree_b_goal_conf, obstacle_list, other_robot_list, animation)
                if last_nid == "connection":
                    self.roadmap = nx.compose(tree_a, tree_b)
                    self.roadmap.add_edge(last_nid, goal_nid)
                    break
                elif last_nid != -1:
                    goal_nid = last_nid
                    tree_a_goal_conf = tree_b.nodes[goal_nid]["conf"]
            if tree_a.number_of_nodes() > tree_b.number_of_nodes():
                tree_a, tree_b = tree_b, tree_a
                tree_a_goal_conf, tree_b_goal_conf = tree_b_goal_conf, tree_a_goal_conf
        else:
            print("Failed to find a path with the given max_n_iter!")
            return None

        path = self._path_from_roadmap()
        smoothed_path = self._smooth_path(path, obstacle_list, other_robot_list,
                                          granularity=ext_dist, n_iter=smoothing_n_iter, animation=animation)
        mot_data = rrt.motd.MotionData(self.robot)
        mot_data.extend(jv_list=smoothed_path)
        return mot_data