import time
import math
import uuid
import random
import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter
import wrs.basis.robot_math as rm
import wrs.motion.motion_data as motd
import wrs.modeling.geometric_model as mgm

import faiss


class FaissRoadmap:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.conf_list = []
        self.id_to_index = {}
        self.index_to_id = []
        self.counter = 0

    def add_node(self, conf):
        conf = np.asarray(conf).astype('float32')
        if conf.ndim == 1:
            conf = conf.reshape(1, -1)

        self.index.add(conf)
        self.conf_list.append(conf[0])
        node_id = f"node_{self.counter}"
        self.id_to_index[node_id] = len(self.conf_list) - 1
        self.index_to_id.append(node_id)
        self.counter += 1
        return node_id

    def get_nearest(self, conf, k=1):
        conf = np.asarray(conf).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(conf, k)
        nearest_ids = [self.index_to_id[i] for i in indices[0]]
        return nearest_ids[0] if k == 1 else nearest_ids

    def get_conf(self, node_id):
        index = self.id_to_index[node_id]
        return self.conf_list[index]

    def __len__(self):
        return len(self.conf_list)

    def items(self):
        return [(node_id, {"conf": self.conf_list[i]}) for i, node_id in enumerate(self.index_to_id)]

class RRT(object):
    """
    author: weiwei
    date: 20230807
    """

    def __init__(self, robot):
        self.robot = robot
        self.roadmap = FaissRoadmap(dim=len(start_conf))
        start_id = self.roadmap.add_node(start_conf)
        self.start_conf = None
        self.goal_conf = None
        # define data type
        self.toggle_keep = True

    @staticmethod
    def keep_states_decorator(method):
        """
        decorator function for save and restore robot's joint values
        applicable to both single or multi-arm sgl_arm_robots
        :return:
        author: weiwei
        date: 20220404
        """

        def wrapper(self, *args, **kwargs):
            if self.toggle_keep:
                self.robot.backup_state()
                result = method(self, *args, **kwargs)
                self.robot.restore_state()
                return result
            else:
                result = method(self, *args, **kwargs)
                return result

        return wrapper

    def _is_collided(self,
                     conf,
                     obstacle_list=[],
                     other_robot_list=[],
                     toggle_contacts=False,
                     toggle_dbg=False):
        """
        The function first examines if joint values of the given conf are in ranges.
        It will promptly return False if any joint value is out of range.
        Or else, it will compute fk and carry out collision checking.
        :param conf:
        :param obstacle_list:
        :param other_robot_list:
        :param toggle_contacts: for debugging collisions at start/goal
        :param toggle_dbg: for debugging
        :return:
        author: weiwei
        date: 20220326, 20240314
        """
        if self.robot.are_jnts_in_ranges(jnt_values=conf):
            self.robot.goto_given_conf(jnt_values=conf)
            # # toggle off the following code to consider object pose constraints
            # if len(self.robot.oiee_list)>0:
            #     angle = rm.angle_between_vectors(self.robot.oiee_list[-1].gl_rotmat[:,2], np.array([0,0,1]))
            #     if angle > np.radians(10):
            #         return True
            collision_info = self.robot.is_collided(obstacle_list=obstacle_list, other_robot_list=other_robot_list,
                                                    toggle_contacts=toggle_contacts)
            # if toggle_contacts:
            #     if collision_info[0]:
            #         for pnt in collision_info[1]:
            #             print(pnt)
            #             mgm.gen_sphere(pos=pnt, radius=.01).attach_to(base)
            #         self.robot.gen_meshmodel(toggle_cdprim=True).attach_to(base)
            #         for obs in obstacle_list:
            #             obs.rgb=np.array([1,1,1])
            #             obs.show_cdprim()
            #             obs.attach_to(base)
            #         base.run()
            return collision_info
        else:
            # print("The given joint angles are out of joint limits.")
            return (True, []) if toggle_contacts else True

    def _sample_conf(self, rand_rate, default_conf):
        if random.randint(0, 99) < rand_rate:
            return self.robot.rand_conf()
        else:
            return default_conf

    def _get_nearest_nid(self, roadmap, new_conf):
        """
        convert to numpy to accelerate access
        :param roadmap:
        :param new_conf:
        :return:
        author: weiwei
        date: 20210523
        """
        nodes_dict = dict(roadmap.nodes(data="conf"))
        nodes_key_list = list(nodes_dict.keys())  # use python > 3.7, or else there is no guarantee on the order
        nodes_value_list = list(nodes_dict.values())  # attention, correspondence is not guanranteed in python
        # ===============
        # the following code computes euclidean distances. it is decprecated and replaced using cdtree
        # ***** date: 20240304, correspondent: weiwei *****
        # conf_array = np.array(nodes_value_list)
        # diff_conf_array = np.linalg.norm(conf_array - new_conf, axis=1)
        # min_dist_nid = np.argmin(diff_conf_array)
        # return nodes_key_list[min_dist_nid]
        # ===============
        querry_tree = scipy.spatial.cKDTree(nodes_value_list)
        dist_value, indx = querry_tree.query(new_conf, k=1, workers=-1)
        return nodes_key_list[indx]

    def _extend_conf(self, src_conf, end_conf, ext_dist, exact_end=True):
        """
        :param src_conf:
        :param end_conf:
        :param ext_dist:
        :param exact_end:
        :return: a list of 1xn nparray
        """
        len, vec = rm.unit_vector(end_conf - src_conf, toggle_length=True)
        # ===============
        # one step extension: not used because it is slower than full extensions
        # ***** date: 20210523, correspondent: weiwei *****
        # return [src_conf + ext_dist * vec]
        # switch to the following code for ful extensions
        # ===============
        if not exact_end:
            nval = math.ceil(len / ext_dist)
            nval = 1 if nval == 0 else nval  # at least include itself
            conf_array = np.linspace(src_conf, src_conf + nval * ext_dist * vec, nval)
        else:
            nval = math.floor(len / ext_dist)
            nval = 1 if nval == 0 else nval  # at least include itself
            conf_array = np.linspace(src_conf, src_conf + nval * ext_dist * vec, nval)
            conf_array = np.vstack((conf_array, end_conf))
        return list(conf_array)

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
        # nearest_nid = self._get_nearest_nid(roadmap, conf)
        nearest_id = self.roadmap.get_nearest(conf)
        nearest_conf = self.roadmap.get_conf(nearest_id)

        new_conf_list = self._extend_conf(roadmap.nodes[nearest_nid]["conf"], conf, ext_dist)[1:]
        for new_conf in new_conf_list:
            if self._is_collided(new_conf, obstacle_list, other_robot_list):
                return nearest_nid
            else:
                new_nid = uuid.uuid4()
                roadmap.add_node(new_nid, conf=new_conf)
                roadmap.add_edge(nearest_nid, new_nid)
                nearest_nid = new_nid
                # all_sampled_confs.append([new_node.point, False])
                if animation:
                    self.draw_wspace([roadmap], self.start_conf, self.goal_conf,
                                     obstacle_list, [roadmap.nodes[nearest_nid]["conf"], conf],
                                     new_conf, '^c')
                # check goal
                if self._is_goal_reached(conf=roadmap.nodes[new_nid]["conf"], goal_conf=goal_conf, threshold=ext_dist):
                    roadmap.add_node("goal", conf=goal_conf)
                    roadmap.add_edge(new_nid, "goal")
                    return "goal"
        return nearest_nid

    def _is_goal_reached(self, conf, goal_conf, threshold):
        dist = np.linalg.norm(np.array(conf) - np.array(goal_conf))
        if dist <= threshold:
            # print("Goal reached!")
            return True
        else:
            return False

    def _path_from_roadmap(self):
        nid_path = nx.shortest_path(self.roadmap, source="start", target="goal")
        return list(itemgetter(*nid_path)(self.roadmap.nodes(data="conf")))

    def _smooth_path(self,
                     path,
                     obstacle_list=[],
                     other_robot_list=[],
                     granularity=.2,
                     n_iter=50,
                     animation=False):
        smoothed_path = path
        for _ in range(n_iter):
            if len(smoothed_path) <= 2:
                return smoothed_path
            i = random.randint(0, len(smoothed_path) - 1)
            j = random.randint(0, len(smoothed_path) - 1)
            if abs(i - j) <= 1:
                continue
            if j < i:
                i, j = j, i
            exact_end = True if j == len(smoothed_path) - 1 else False
            shortcut = self._extend_conf(src_conf=smoothed_path[i], end_conf=smoothed_path[j], ext_dist=granularity,
                                         exact_end=exact_end)
            if all(not self._is_collided(conf=conf,
                                         obstacle_list=obstacle_list,
                                         other_robot_list=other_robot_list)
                   for conf in shortcut):
                smoothed_path = smoothed_path[:i] + shortcut + smoothed_path[j + 1:]
            if animation:
                self.draw_wspace([self.roadmap], self.start_conf, self.goal_conf,
                                 obstacle_list, shortcut=shortcut, smoothed_path=smoothed_path)
            if i == 0 and exact_end:  # stop smoothing when shortcut was between start and end
                break
        return smoothed_path

    @keep_states_decorator
    def plan(self,
             start_conf,
             goal_conf,
             obstacle_list=[],
             other_robot_list=[],
             ext_dist=.2,
             rand_rate=70,
             max_n_iter=1000,
             max_time=15.0,
             smoothing_n_iter=50,
             animation=False,
             toggle_dbg=True):
        """
        :return: [path, all_sampled_confs]
        author: weiwei
        date: 20201226
        """
        self.roadmap.clear()
        self.start_conf = start_conf
        self.goal_conf = goal_conf
        # check start_conf and end_conf
        # if toggle_dbg:
        #     print("RRT: Checking start robot configuration...")
        if self._is_collided(start_conf, obstacle_list, other_robot_list, toggle_dbg=toggle_dbg):
            print("The start robot configuration is in collision!")
            return None
        # if toggle_dbg:
        #     print("RRT: Checking goal robot configuration...")
        if self._is_collided(goal_conf, obstacle_list, other_robot_list, toggle_dbg=toggle_dbg):
            print("The goal robot configuration is in collision!")
            return None
        if self._is_goal_reached(conf=start_conf, goal_conf=goal_conf, threshold=ext_dist):
            mot_data = motd.MotionData(self.robot)
            mot_data.extend(jv_list=[start_conf, goal_conf])
            return mot_data
        self.roadmap.add_node("start", conf=start_conf)
        tic = time.time()
        for _ in range(max_n_iter):
            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Failed to find a path in the given max_time!")
                    return None
            # Random Sampling
            rand_conf = self._sample_conf(rand_rate=rand_rate, default_conf=goal_conf)
            last_nid = self._extend_roadmap(roadmap=self.roadmap,
                                            conf=rand_conf,
                                            ext_dist=ext_dist,
                                            goal_conf=goal_conf,
                                            obstacle_list=obstacle_list,
                                            other_robot_list=other_robot_list,
                                            animation=animation)
            if last_nid == "goal":
                path = self._path_from_roadmap()
                smoothed_path = self._smooth_path(path=path,
                                                  obstacle_list=obstacle_list,
                                                  other_robot_list=other_robot_list,
                                                  granularity=ext_dist,
                                                  n_iter=smoothing_n_iter,
                                                  animation=animation)
                mot_data = motd.MotionData(self.robot)
                if getattr(base, "toggle_mesh", True):
                    mot_data.extend(jv_list=smoothed_path)
                else:
                    mot_data.extend(jv_list=smoothed_path, mesh_list=[])
                return mot_data
        else:
            print("Failed to find a path with the given max_n_ter!")
            return None

    @staticmethod
    def draw_wspace(roadmap_list,
                    start_conf,
                    goal_conf,
                    obstacle_list,
                    near_rand_conf_pair=None,
                    new_conf=None,
                    new_conf_mark='^r',
                    shortcut=None,
                    smoothed_path=None,
                    delay_time=.001):
        """
        Draw Graph
        """
        plt.clf()
        ax = plt.gca()
        ax.set_aspect("equal", "box")
        plt.grid(True)
        plt.xlim(-.4, 1.7)
        plt.ylim(-.4, 1.7)
        ax.add_patch(plt.Circle((start_conf[0], start_conf[1]), .05, color='r'))
        ax.add_patch(plt.Circle((goal_conf[0], goal_conf[1]), .05, color='g'))
        for (point, size) in obstacle_list:
            ax.add_patch(plt.Circle((point[0], point[1]), size / 2.0, color='k'))
        colors = "bgrcmykw"
        for i, roadmap in enumerate(roadmap_list):
            for (u, v) in roadmap.edges:
                plt.plot(roadmap.nodes[u]["conf"][0], roadmap.nodes[u]["conf"][1], 'o' + colors[i])
                plt.plot(roadmap.nodes[v]["conf"][0], roadmap.nodes[v]["conf"][1], 'o' + colors[i])
                plt.plot([roadmap.nodes[u]["conf"][0], roadmap.nodes[v]["conf"][0]],
                         [roadmap.nodes[u]["conf"][1], roadmap.nodes[v]["conf"][1]], '-' + colors[i])
        if near_rand_conf_pair is not None:
            plt.plot([near_rand_conf_pair[0][0], near_rand_conf_pair[1][0]],
                     [near_rand_conf_pair[0][1], near_rand_conf_pair[1][1]], "--k")
            ax.add_patch(plt.Circle((near_rand_conf_pair[1][0], near_rand_conf_pair[1][1]), .03, color='grey'))
        if new_conf is not None:
            plt.plot(new_conf[0], new_conf[1], new_conf_mark)
        if smoothed_path is not None:
            plt.plot([conf[0] for conf in smoothed_path], [conf[1] for conf in smoothed_path], linewidth=7,
                     linestyle='-', color='c')
        if shortcut is not None:
            plt.plot([conf[0] for conf in shortcut], [conf[1] for conf in shortcut], linewidth=4, linestyle='--',
                     color='r')
        if not hasattr(RRT, "img_counter"):
            RRT.img_counter = 0
        else:
            RRT.img_counter += 1
        # plt.savefig(str(RRT.img_counter)+'.jpg')
        if delay_time > 0:
            plt.pause(delay_time)
        # plt.waitforbuttonpress()


if __name__ == "__main__":
    # import wrs.robot_sim.robots.xybot.xybot as robot
    # # ====Search Path with RRT====
    # obstacle_list = [
    #     ((.5, .5), .3),
    #     ((.3, .6), .3),
    #     ((.3, .8), .3),
    #     ((.3, 1.0), .3),
    #     ((.7, .5), .3),
    #     ((.9, .5), .3),
    #     ((1.0, .5), .3)
    # ]  # [[x,y],size]
    # robot = robot.XYBot()
    # rrt = RRT(robot)
    # path = rrt.plan(start_conf=np.array([0, 0]), goal_conf=np.array([.6, .9]), obstacle_list=obstacle_list,
    #                 ext_dist=.1, rand_rate=70, max_time=300, animation=True)
    # # Draw final path
    # print(path)
    # rrt.draw_wspace(roadmap_list=[rrt.roadmap], start_conf=rrt.start_conf, goal_conf=rrt.goal_conf,
    #                 obstacle_list=obstacle_list)
    # plt.plot([conf[0] for conf in path], [conf[1] for conf in path], linewidth=4, color='y')
    # # plt.savefig(str(rrtc.img_counter)+'.jpg')
    # plt.pause(0.001)
    # plt.show()
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
    rrtc = RRT(robot)
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


    # path = rrtc.plan(start_conf=start_conf,
    #                  goal_conf=goal_conf,
    #                  ext_dist=.1,
    #                  max_time=300,
    #                  animation=True)
    # Draw final path
    # print(path)

    path = [
        [-0.69984654, -2.61460858, 2.00115672, 1.78559269, 1.53977208, -2.59826454],
        [-0.67556071, -2.54569066, 1.97052846, 1.7322922, 1.50533786, -2.60153753],
        [-0.67266307, -2.47447992, 2.00391268, 1.71375888, 1.4605327, -2.53988674],
        [-0.66976543, -2.40326918, 2.0372969, 1.69522555, 1.41572754, -2.47823595],
        [-0.66686779, -2.33205843, 2.07068112, 1.67669223, 1.37092238, -2.41658516],
        [-0.66360794, -2.25194635, 2.10823837, 1.65584224, 1.32051658, -2.34722802],
        [-0.6603481, -2.17183426, 2.14579562, 1.63499225, 1.27011077, -2.27787088],
        [-0.65708825, -2.09172218, 2.18335287, 1.61414226, 1.21970496, -2.20851373],
        [-0.6538284, -2.0116101, 2.22091012, 1.59329227, 1.16929916, -2.13915659],
        [-0.65145234, -1.95321729, 2.24828519, 1.57809495, 1.13255892, -2.08860295],
        [-0.64832289, -1.87630969, 2.28434015, 1.55807896, 1.08416935, -2.02202009],
        [-0.60311866, -1.87232501, 2.27134448, 1.51366705, 1.08168881, -1.98130798],
        [-0.55663464, -1.83977148, 2.27170343, 1.46170686, 1.06123307, -1.91576531],
        [-0.51015062, -1.80721795, 2.27206239, 1.40974667, 1.04077733, -1.85022264],
        [-0.46366659, -1.77466443, 2.27242134, 1.35778648, 1.02032159, -1.78467997],
        [-0.41718257, -1.7421109, 2.2727803, 1.30582628, 0.99986585, -1.71913731],
        [-0.37069855, -1.70955737, 2.27313925, 1.25386609, 0.97941011, -1.65359464],
        [-0.32421453, -1.67700384, 2.27349821, 1.2019059, 0.95895437, -1.58805197],
        [-0.27773051, -1.64445031, 2.27385716, 1.1499457, 0.93849863, -1.5225093],
        [-0.23124648, -1.61189679, 2.27421612, 1.09798551, 0.91804289, -1.45696663],
        [-0.18476246, -1.57934326, 2.27457507, 1.04602532, 0.89758715, -1.39142396],
        [-0.13827844, -1.54678973, 2.27493403, 0.99406513, 0.87713141, -1.3258813],
        [-0.09179442, -1.5142362, 2.27529298, 0.94210493, 0.85667567, -1.26033863],
        [-0.0453104, -1.48168267, 2.27565194, 0.89014474, 0.83621994, -1.19479596],
        [0.00117362, -1.44912915, 2.27601089, 0.83818455, 0.8157642, -1.12925329],
        [0.04765765, -1.41657562, 2.27636985, 0.78622435, 0.79530846, -1.06371062],
        [0.09414167, -1.38402209, 2.2767288, 0.73426416, 0.77485272, -0.99816795],
        [0.14062569, -1.35146856, 2.27708776, 0.68230397, 0.75439698, -0.93262528],
        [0.18710971, -1.31891503, 2.27744671, 0.63034377, 0.73394124, -0.86708262],
        [0.23359373, -1.28636151, 2.27780566, 0.57838358, 0.7134855, -0.80153995],
        [0.28007776, -1.25380798, 2.27816462, 0.52642339, 0.69302976, -0.73599728],
        [0.32656178, -1.22125445, 2.27852357, 0.4744632, 0.67257402, -0.67045461],
        [0.3730458, -1.18870092, 2.27888253, 0.422503, 0.65211828, -0.60491194],
        [0.41952982, -1.15614739, 2.27924148, 0.37054281, 0.63166254, -0.53936927],
        [0.46601384, -1.12359387, 2.27960044, 0.31858262, 0.6112068, -0.47382661],
        [0.51249786, -1.09104034, 2.27995939, 0.26662242, 0.59075106, -0.40828394],
        [0.56621805, -1.05652091, 2.28593378, 0.20801358, 0.57181703, -0.34352776],
        [0.61993824, -1.02200147, 2.29190817, 0.14940474, 0.552883, -0.27877158],
        [0.67365843, -0.98748204, 2.29788256, 0.09079589, 0.53394897, -0.2140154],
        [0.72737862, -0.95296261, 2.30385696, 0.03218705, 0.51501494, -0.14925923],
        [0.78109881, -0.91844318, 2.30983135, -0.0264218, 0.49608091, -0.08450305],
        [0.834819, -0.88392374, 2.31580574, -0.08503064, 0.47714688, -0.01974687],
        [0.88853919, -0.84940431, 2.32178013, -0.14363949, 0.45821285, 0.04500931],
        [0.94225938, -0.81488488, 2.32775452, -0.20224833, 0.43927882, 0.10976548],
        [0.99597957, -0.78036545, 2.33372891, -0.26085717, 0.42034479, 0.17452166],
        [1.04969976, -0.74584601, 2.3397033, -0.31946602, 0.40141076, 0.23927784],
        [1.08829575, -0.7468656, 2.39027695, -0.34958692, 0.42698353, 0.19431909],
        [1.1525943, -0.73136928, 2.44370906, -0.40774937, 0.44349725, 0.18034309],
        [1.21689285, -0.71587296, 2.49714117, -0.46591182, 0.46001098, 0.1663671],
        [1.2811914, -0.70037664, 2.55057328, -0.52407428, 0.4765247, 0.15239111],
        [1.34548995, -0.68488032, 2.60400539, -0.58223673, 0.49303843, 0.13841512],
        [1.4097885, -0.669384, 2.6574375, -0.64039918, 0.50955216, 0.12443913],
        [1.47408706, -0.65388768, 2.71086961, -0.69856163, 0.52606588, 0.11046314],
        [1.53838561, -0.63839136, 2.76430172, -0.75672409, 0.54257961, 0.09648714],
        [1.60268416, -0.62289504, 2.81773383, -0.81488654, 0.55909334, 0.08251115],
        [1.66698271, -0.60739872, 2.87116594, -0.87304899, 0.57560706, 0.06853516],
        [1.73128126, -0.5919024, 2.92459805, -0.93121144, 0.59212079, 0.05455917],
        [1.79557981, -0.57640608, 2.97803016, -0.9893739, 0.60863452, 0.04058318],
        [1.85987836, -0.56090976, 3.03146227, -1.04753635, 0.62514824, 0.02660718],
        [1.92417691, -0.54541344, 3.08489438, -1.1056988, 0.64166197, 0.01263119],
        [1.98847546, -0.52991712, 3.13832649, -1.16386125, 0.6581757, -0.0013448],
        [2.05277401, -0.5144208, 3.1917586, -1.22202371, 0.67468942, -0.01532079],
        [2.11707256, -0.49892448, 3.24519071, -1.28018616, 0.69120315, -0.02929678],
        [2.18137111, -0.48342815, 3.29862282, -1.33834861, 0.70771687, -0.04327278],
        [2.2393624, -0.47368154, 3.34646717, -1.39572501, 0.7264779, -0.06166187],
        [2.30227721, -0.46274833, 3.39840296, -1.45755507, 0.74650344, -0.08112115],
        [2.36519202, -0.45181512, 3.45033876, -1.51938514, 0.76652897, -0.10058043],
        [2.42810683, -0.44088191, 3.50227456, -1.58121521, 0.7865545, -0.1200397],
        [2.44152865, -0.44369597, 3.5124528, -1.60037737, 0.79599764, -0.13160537],
        [2.45661691, -0.38208949, 3.52195326, -1.60332132, 0.73803008, -0.18648319],
        [2.47170517, -0.32048301, 3.53145373, -1.60626528, 0.68006252, -0.24136102],
        [2.47210789, -0.27943586, 3.57631214, -1.55666748, 0.74725468, -0.26014832],
        [2.53332707, -0.29526851, 3.62081336, -1.63318626, 0.78020253, -0.27205302],
        [2.59454625, -0.31110116, 3.66531458, -1.70970503, 0.81315037, -0.28395772],
        [2.65882638, -0.32772545, 3.71204086, -1.79004975, 0.84774561, -0.29645766],
        [2.72310651, -0.34434973, 3.75876714, -1.87039446, 0.88234085, -0.3089576],
        [2.78738665, -0.36097402, 3.80549342, -1.95073918, 0.91693609, -0.32145753],
        [2.85166678, -0.37759831, 3.8522197, -2.03108389, 0.95153133, -0.33395747],
        [2.91594692, -0.39422259, 3.89894598, -2.11142861, 0.98612657, -0.3464574],
        [2.9438924, -0.40144992, 3.91926001, -2.14635809, 1.00116669, -0.35189169]
]
    
    for i in range(len(path)):
        robot.goto_given_conf(jnt_values=path[i])
        robot.gen_meshmodel(rgb=[1,0,0], alpha=.05).attach_to(base)
    
    base.run()
