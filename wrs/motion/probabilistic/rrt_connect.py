import time
import uuid
import networkx as nx
import wrs.motion.probabilistic.rrt as rrt


class RRTConnect(rrt.RRT):

    def __init__(self, robot):
        super().__init__(robot)
        self.roadmap_start = nx.Graph()
        self.roadmap_goal = nx.Graph()

    def _extend_roadmap(self,
                        roadmap,
                        conf,
                        ext_dist,
                        goal_conf,
                        obstacle_list=[],
                        other_robot_list=[],
                        animation=False):
        """
        find the nearest point between the given roadmap and the conf and then extend towards the conf
        :return:
        author: weiwei
        date: 20201228
        """
        nearest_nid = self._get_nearest_nid(roadmap, conf)
        new_conf_list = self._extend_conf(roadmap.nodes[nearest_nid]["conf"], conf, ext_dist, exact_end=False)[1:]
        for new_conf in new_conf_list:
            if self._is_collided(new_conf, obstacle_list, other_robot_list):
                return -1
            else:
                new_nid = uuid.uuid4()
                roadmap.add_node(new_nid, conf=new_conf)
                roadmap.add_edge(nearest_nid, new_nid)
                nearest_nid = new_nid
                # all_sampled_confs.append([new_node.point, False])
                if animation:
                    self.draw_wspace([self.roadmap_start, self.roadmap_goal], self.start_conf, self.goal_conf,
                                     obstacle_list, [roadmap.nodes[nearest_nid]["conf"], conf], new_conf,
                                     "^c")
                # check goal
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
             toggle_dbg=False):
        self.roadmap.clear()
        self.roadmap_start.clear()
        self.roadmap_goal.clear()
        self.start_conf = start_conf
        self.goal_conf = goal_conf
        # check start and goal
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
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Failed to find a path in the given max_time!")
                    return None
            # one tree grown using random target
            rand_conf = self._sample_conf(rand_rate=100,
                                          default_conf=None)
            last_nid = self._extend_roadmap(roadmap=tree_a,
                                            conf=rand_conf,
                                            ext_dist=ext_dist,
                                            goal_conf=tree_a_goal_conf,
                                            obstacle_list=obstacle_list,
                                            other_robot_list=other_robot_list,
                                            animation=animation)
            if last_nid != -1:  # not trapped:
                goal_nid = last_nid
                tree_b_goal_conf = tree_a.nodes[goal_nid]["conf"]
                last_nid = self._extend_roadmap(roadmap=tree_b,
                                                conf=tree_a.nodes[last_nid]["conf"],
                                                ext_dist=ext_dist,
                                                goal_conf=tree_b_goal_conf,
                                                obstacle_list=obstacle_list,
                                                other_robot_list=other_robot_list,
                                                animation=animation)
                if last_nid == "connection":
                    self.roadmap = nx.compose(tree_a, tree_b)
                    self.roadmap.add_edge(last_nid, goal_nid)
                    break
                elif last_nid != -1:
                    goal_nid = last_nid
                    tree_a_goal_conf = tree_b.nodes[goal_nid]["conf"]
            if tree_a.number_of_nodes() > tree_b.number_of_nodes():  # always extend the smaller tree
                tree_a, tree_b = tree_b, tree_a
                tree_a_goal_conf, tree_b_goal_conf = tree_b_goal_conf, tree_a_goal_conf
        else:
            print("Failed to find a path with the given max_n_ter!")
            return None
        path = self._path_from_roadmap()
        smoothed_path = self._smooth_path(path=path,
                                          obstacle_list=obstacle_list,
                                          other_robot_list=other_robot_list,
                                          granularity=ext_dist,
                                          n_iter=smoothing_n_iter,
                                          animation=animation)
        mot_data = rrt.motd.MotionData(self.robot)
        if getattr(base, "toggle_mesh", True):
            mot_data.extend(jv_list=smoothed_path)
        else:
            mot_data.extend(jv_list=smoothed_path, mesh_list=[])
        return mot_data


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm
    # import wrs.robot_sim.robots.xybot.xybot as robot
    import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as robot
    import wrs.basis.robot_math as rm

    # ====Search Path with RRT====
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot = robot.XArmLite6(enable_cc=True)
    rrtc = RRTConnect(robot)
    # start_conf = robot.rand_conf()
    # goal_conf = robot.rand_conf()
    start_conf = [-0.69984654, -2.61460858,  2.00115672,  1.78559269,  1.53977208,
       -2.59826454]
    goal_conf = [ 2.9438924 , -0.40144992,  3.91926001, -2.14635809,  1.00116669,
       -0.35189169]
    print(repr(start_conf))
    print(repr(goal_conf))

    robot.goto_given_conf(jnt_values=start_conf)
    robot.gen_meshmodel(rgb=rm.const.steel_blue, alpha=.3).attach_to(base)
    robot.goto_given_conf(jnt_values=goal_conf)
    robot.gen_meshmodel(rgb=[0,1,0], alpha=.3).attach_to(base)
    # base.run()


    path = rrtc.plan(start_conf=start_conf,
                     goal_conf=goal_conf,
                     ext_dist=.1,
                     max_time=300,
                     animation=True)
    # Draw final path
    print(path)
    
    for _ in path.jv_list:
        robot.goto_given_conf(jnt_values=_)
        robot.gen_meshmodel(rgb=[1,0,0], alpha=.3).attach_to(base)
