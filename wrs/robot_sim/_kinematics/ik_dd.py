"""
Data driven ik solver
author: weiwei
date: 20231107
"""

import os
import pickle
import numpy as np
import scipy.spatial
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.ik_num as ikn
import wrs.robot_sim._kinematics.ik_opt as iko
import wrs.robot_sim._kinematics.ik_trac as ikt
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim._kinematics.model_generator as rkmg
import wrs.modeling.geometric_model as mgm
import wrs.basis.utils as bu
import time
import json
import joblib


# for seed data collection purpose
def append_to_logfile(filename, data):
    with open(filename, "a") as f: 
        json.dump(data, f) 
        f.write("\n")

class DDIKSolver(object):
    def __init__(self, jlc, path=None, identifier_str='test', backbone_solver='n', rebuild=False):
        """
        :param jlc:
        :param path:
        :param backbone_solver: 'n': num ik; 'o': opt ik; 't': trac ik
        :param rebuild:
        author: weiwei
        date: 20231111
        """
        self.jlc = jlc
        current_file_dir = os.path.dirname(__file__)
        if path is None:
            path = os.path.join(os.path.dirname(current_file_dir), "_data_files")
        self._fname_tree = os.path.join(path, f"{identifier_str}_ikdd_tree.pkl")
        self._fname_jnt = os.path.join(path, f"{identifier_str}_jnt_data.pkl")
        self._fname_jmat_inv = os.path.join(path, f"{identifier_str}_jmat_inv.pkl")
        self._fname_jmat = os.path.join(path, f"{identifier_str}_jmat.pkl")
        # self._k_bbs = 100  # number of nearest neighbours examined by the backbone solver
        # self._k_max = 200  # maximum nearest neighbours explored by the evolver
        self._k_bbs = 50  # number of nearest neighbours examined by the backbone solver
        self._k_max = 50  # maximum nearest neighbours explored by the evolver
        self._max_n_iter = 20  # max_n_iter of the backbone solver
        if backbone_solver == 'n':
            self._backbone_solver = ikn.NumIKSolver(self.jlc)
            print("NumIK is applied.")
        elif backbone_solver == 'o':
            self._backbone_solver = iko.OptIKSolver(self.jlc)
            print("OptIK is applied.")
        elif backbone_solver == 't':
            self._backbone_solver = ikt.TracIKSolver(self.jlc)
            print("TracIK is applied.")
        if rebuild:
            print("Rebuilding the database. It starts a new evolution and is costly.")
            y_or_n = bu.get_yesno()
            if y_or_n == 'y':
                self.query_tree, self.jnt_data = self._build_data()
                self.persist_data()
                self.evolve_data(n_times=100000)
        else:
            try:
                with open(self._fname_tree, 'rb') as f_tree:
                    self.query_tree = pickle.load(f_tree)
                with open(self._fname_jnt, 'rb') as f_jnt:
                    self.jnt_data = pickle.load(f_jnt)
                with open(self._fname_jmat_inv, 'rb') as f_jnt:
                    self.jmat_inv = pickle.load(f_jnt)
                with open(self._fname_jmat, 'rb') as f_jnt:
                    self.jmat = pickle.load(f_jnt)
            except FileNotFoundError:
                self.jmat, self.jmat_inv, self.query_tree, self.jnt_data = self._build_data()
                self.persist_data()
                self.evolve_data(n_times=100)

    def __call__(self,
                 tgt_pos,
                 tgt_rotmat,
                 best_sol_num,
                 seed_jnt_values=None,
                 max_n_iter=None,
                 toggle_evolve=True,
                 toggle_dbg=False):
        """
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param max_n_iter: use self._max_n_iter if None
        :param toggle_evolve: do we update the database file
        :param toggle_dbg:
        :return:
        """
        return self.ik(tgt_pos=tgt_pos,
                       tgt_rotmat=tgt_rotmat,
                       best_sol_num=best_sol_num,
                       seed_jnt_values=seed_jnt_values,
                       max_n_iter=max_n_iter,
                       toggle_evolve=toggle_evolve,
                       toggle_dbg=toggle_dbg)

    def _rotmat_to_vec(self, rotmat, method='v'):
        """
        convert a rotmat to vectors
        this will be used for computing the Minkowski p-norm required by KDTree query
        'f' or 'q' are recommended, they both have satisfying performance
        :param method: 'f': Frobenius; 'q': Quaternion; 'r': rpy; 'v': rotvec; '-': same value
        :return:
        author: weiwei
        date: 20231107
        """
        if method == 'f':
            return rotmat.ravel()
        if method == 'q':
            return Rotation.from_matrix(rotmat).as_quat()
        if method == 'r':
            return rm.rotmat_to_euler(rotmat)
        if method == 'v':
            return Rotation.from_matrix(rotmat).as_rotvec()
        if method == '-':
            return np.array([0])

    def _build_data(self):
        # margin = rm.pi / 18
        margin = 0 
        # gen sampled qs
        sampled_jnts = []
        n_intervals = np.linspace(8, 4, self.jlc.n_dof, endpoint=False)
        print(f"Buidling Data for DDIK using the following joint granularity: {n_intervals.astype(int)}...")
        for i in range(self.jlc.n_dof):
            sampled_jnts.append(
                np.linspace(self.jlc.jnt_ranges[i][0]+margin, self.jlc.jnt_ranges[i][1]-margin, int(n_intervals[i]+2))[1:-1])
        grid = np.meshgrid(*sampled_jnts)
        sampled_qs = np.vstack([x.ravel() for x in grid]).T
        # gen sampled qs and their correspondent flange poses
        query_data = []
        jnt_data = []
        jmat_inv_data = []
        jamt_data = []

        for id in tqdm(range(len(sampled_qs))):
            jnt_values = sampled_qs[id]
            flange_pos, flange_rotmat = self.jlc.fk(jnt_values=jnt_values, toggle_jacobian=False)
            # relative to base
            rel_pos, rel_rotmat = rm.rel_pose(self.jlc.pos, self.jlc.rotmat, flange_pos, flange_rotmat)
            rel_rotvec = self._rotmat_to_vec(rel_rotmat)

            query_data.append(np.concatenate((rel_pos, rel_rotvec)))
            jnt_data.append(jnt_values)
            jmat_inv = np.linalg.pinv(self.jlc.jacobian(jnt_values=jnt_values), rcond=1e-4)
            jmat_inv_data.append(jmat_inv)
            jamt_data.append(self.jlc.jacobian(jnt_values=jnt_values))
        query_tree = scipy.spatial.cKDTree(query_data)
        return jamt_data, jmat_inv_data, query_tree, jnt_data

    def multiepoch_evolve(self, n_times_per_epoch=10000, target_success_rate=.96):
        """
        calls evolve_data repeated based on user feedback
        :return:
        author: weiwei
        date: 20231111
        """
        print("Starting multi-epoch evolution.")
        current_success_rate = 0.0
        while current_success_rate < target_success_rate:
            self.evolve_data(n_times=n_times_per_epoch)
            current_success_rate = self.test_success_rate()
            # print("An epoch is done. Do you want to continue?")
            # y_or_n = bu.get_yesno()
            # if y_or_n == 'n':
            #     break
        self.persist_data()

    def evolve_data(self, n_times=100000, toggle_dbg=True):
        evolved_nns = []
        outer_progress_bar = tqdm(total=n_times, desc="new goals:", colour="red", position=0, leave=False)
        for i in range(n_times):
            outer_progress_bar.update(1)
            random_jnts = self.jlc.rand_conf()
            flange_pos, flange_rotmat = self.jlc.fk(jnt_values=random_jnts, update=False, toggle_jacobian=False)
            # relative to base
            rel_pos, rel_rotmat = rm.rel_pose(self.jlc.pos, self.jlc.rotmat, flange_pos, flange_rotmat)
            rel_rotvec = self._rotmat_to_vec(rel_rotmat)
            query_point = np.concatenate((rel_pos, rel_rotvec))
            dist_value_array, nn_indx_array = self.query_tree.query(query_point, k=self._k_max, workers=-1)
            is_solvable = False
            for nn_indx in nn_indx_array[:self._k_bbs]:
                seed_jnt_values = self.jnt_data[nn_indx]
                result = self._backbone_solver(tgt_pos=flange_pos,
                                               tgt_rotmat=flange_rotmat,
                                               seed_jnt_values=seed_jnt_values,
                                               max_n_iter=self._max_n_iter)
                if result is None:
                    continue
                else:
                    is_solvable = True
                    break
            if not is_solvable:
                # try solving the problem with additional nearest neighbours
                # inner_progress_bar = tqdm(total=self._k_max - self._k_bbs,
                #                           desc="    unvolsed. try extra nns:",
                #                           colour="green",
                #                           position=1,
                #                           leave=False)
                for id, nn_indx in enumerate(nn_indx_array[self._k_bbs:]):
                    # inner_progress_bar.update(1)
                    seed_jnt_values = self.jnt_data[nn_indx]
                    result = self._backbone_solver(tgt_pos=flange_pos,
                                                   tgt_rotmat=flange_rotmat,
                                                   seed_jnt_values=seed_jnt_values,
                                                   max_n_iter=self._max_n_iter)
                    if result is None:
                        continue
                    else:
                        # if solved, add the new jnts to the data and update the kd tree
                        tree_data = np.vstack((self.query_tree.data, query_point))
                        self.jmat_inv.append(np.linalg.pinv(self.jlc.jacobian(jnt_values=result)))
                        self.jmat.append(self.jlc.jacobian(jnt_values=result))
                        self.jnt_data.append(result)
                        self.query_tree = scipy.spatial.cKDTree(tree_data)
                        evolved_nns.append(self._k_bbs + id)
                        print(f"#### Previously unsolved ik solved using the {self._k_bbs + id}th nearest neighbour.")
                        break
                # inner_progress_bar.close()
        outer_progress_bar.close()
        if toggle_dbg:
            print("+++++++++++++++++++evolution details+++++++++++++++++++")
            if len(evolved_nns) > 0:
                evolved_nns = np.asarray(evolved_nns)
                print("Max nn id: ", evolved_nns.max())
                print("Min nn id: ", evolved_nns.min())
                print("Avg nn id: ", evolved_nns.mean())
                print("Std nn id: ", evolved_nns.std())
            else:
                print("No successful evolution.")
        self.persist_data()

    def persist_data(self):
        with open(self._fname_tree, 'wb') as f_tree:
            pickle.dump(self.query_tree, f_tree)
        with open(self._fname_jnt, 'wb') as f_jnt:
            pickle.dump(self.jnt_data, f_jnt)
        with open(self._fname_jmat_inv, 'wb') as f_jmat_inv:
            pickle.dump(self.jmat_inv, f_jmat_inv)
        with open(self._fname_jmat, 'wb') as f_jmat:
            pickle.dump(self.jmat, f_jmat)
        print("ddik data file saved.")

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           best_sol_num,
           seed_jnt_values=None,
           max_n_iter=None,
           toggle_evolve=True,
           toggle_dbg=False):
        """
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values: ignored
        :param toggle_dbg: ignored
        :return:
        author: weiwei
        date: 20231107
        """
        collect_successful_seed = False
        collect_seed_selection = False
        max_n_iter = self._max_n_iter if max_n_iter is None else max_n_iter
        if seed_jnt_values is not None:
            return self._backbone_solver(tgt_pos=tgt_pos,
                                         tgt_rotmat=tgt_rotmat,
                                         seed_jnt_values=seed_jnt_values,
                                         max_n_iter=max_n_iter,
                                         toggle_dbg=toggle_dbg)
        else:
            # relative to base
            rel_pos, rel_rotmat = rm.rel_pose(self.jlc.pos, self.jlc.rotmat, tgt_pos, tgt_rotmat)
            rel_rotvec = self._rotmat_to_vec(rel_rotmat)
            query_point = np.concatenate((rel_pos, rel_rotvec))
            # tic = time.time()
            dist_value_array, nn_indx_array = self.query_tree.query(query_point, k=self._k_max, workers=-1)
            # toc = time.time()
            # query_time = (toc - tic) * 1000
            # print(f"Querying the KDT-tree took {query_time:.3f} ms.")
            if type(nn_indx_array) is int:
                nn_indx_array = [nn_indx_array]
            
            '''xinyi: rank the jnt_seed by the delta_q'''
            seed_candidates = np.array([[nn_indx] + [0] * (self.jlc.n_dof+1) for nn_indx in nn_indx_array], dtype=float)
            jmat_pinv_batch = np.array([self.jmat_inv[idx] for idx in nn_indx_array])
            # jmat_pinv_batch = np.array([np.identity(6) for idx in nn_indx_array])
            tgt_candidates = np.array([self.query_tree.data[idx] for idx in nn_indx_array])  # target stored in the tree
            tgt_gth = np.tile(query_point, (len(nn_indx_array), 1))
            tgt_delta = np.abs(tgt_candidates - tgt_gth)
            
            # tgt_delta[:, 3:] = 0
            delta_q_batch = np.einsum('ijk,ik->ij', jmat_pinv_batch, tgt_delta)
            delta_q_l2norm = np.linalg.norm(delta_q_batch[:,:], axis=1)

            # delta_q_l2norm = np.mean(np.square(delta_q_batch), axis=1)/np.var(tgt_delta, axis=1)
            seed_candidates[:, -1] = delta_q_l2norm
            seed_candidates[:, 1:-1] = delta_q_batch

            # print("\n" + "*" * 150)
            # print(f'top ddik index is: {repr(nn_indx_array[0:10])}')
            
            '''xinyi: sort the seed_candidates by the delta_q'''
            sorted_indices = np.argsort(seed_candidates[:, -1])
            seed_candidates_sorted = seed_candidates[sorted_indices]
            nn_indx_array = seed_candidates_sorted[:, 0]
            nn_indx_array = nn_indx_array.astype(int)

            '''resort by how much the next jnt val is close to the limit'''
            cut_off_num = 5
            selected_idx_array = nn_indx_array[:cut_off_num]
            delta_q_selected = seed_candidates_sorted[:cut_off_num,1:-1]
            jnt_batch = np.array([self.jnt_data[idx] for idx in selected_idx_array])
            next_jnt_batch = jnt_batch + delta_q_selected

            lower_diff = next_jnt_batch - self.jlc.jnt_ranges[:, 0]
            upper_diff = self.jlc.jnt_ranges[:, 1] - next_jnt_batch
            dist_to_jnt_limit = np.minimum(lower_diff, upper_diff)
            dist_to_jnt_limit_norm = np.linalg.norm(dist_to_jnt_limit, axis=1)

            '''resorting'''
            # sorted_indices = np.argsort(dist_to_jnt_limit_norm)
            # selected_idx_array = selected_idx_array[sorted_indices]
            # selected_idx_array = selected_idx_array.astype(int)
            # nn_indx_array = selected_idx_array

            for id, nn_indx in enumerate(nn_indx_array[0:best_sol_num]):
                # seed_jnt_values = self.jnt_data[nn_indx]
                seed_jnt_values = self.jnt_data[40092]

                if toggle_dbg:
                    rkmg.gen_jlc_stick_by_jnt_values(self.jlc,
                                                     jnt_values=seed_jnt_values,
                                                     stick_rgba=rm.bc.red).attach_to(base)
                result = self._backbone_solver(tgt_pos=tgt_pos,
                                               tgt_rotmat=tgt_rotmat,
                                               seed_jnt_values=seed_jnt_values,
                                               max_n_iter=max_n_iter,
                                               toggle_dbg=toggle_dbg)
                if result is None:
                    # mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
                    # rkmg.gen_jlc_stick_by_jnt_values(self.jlc,
                    #                                  jnt_values=seed_jnt_values,
                    #                                  stick_rgba=rm.const.red).attach_to(base)
                    # print(result)
                    # if id == self._k_max-1:
                    #     base.run()

                    # print(f"Solution Not Found for id: {id}, KDT_indx: {nn_indx}.")
                    # print(f'jnt limit dist: {dist_to_jnt_limit_norm[id]}: {dist_to_jnt_limit[id]}')
                    # print(f"delta_q: {seed_candidates_sorted[id,-1]}: {seed_candidates_sorted[id,1:-1]}\n")

                    # if collect_seed_selection == True:
                    #     data = {"label": '0',
                    #             "query_point": query_point.tolist(),
                    #             "seed_jnt_value": seed_jnt_values.tolist()}
                    #     append_to_logfile("seed_selection_dataset.json", data)
                    if toggle_evolve:
                        continue
                    else:
                        return None
                else:
                    # if id > self._k_bbs:
                    #     tree_data = np.vstack((self.query_tree.data, query_point))
                    #     self.jnt_data.append(result)
                    #     self.jmat_inv.append(np.linalg.pinv(self.jlc.jacobian(jnt_values=result)))
                    #     self.query_tree = scipy.spatial.cKDTree(tree_data)
                    #     print(f"Updating query tree, {id} explored...")
                    #     self.persist_data()
                    #     break
                    if collect_successful_seed == True:
                        data = {"source": 'KDTree',"target": query_point.tolist(), "seed_jnt_value": seed_jnt_values.tolist(), "jnt_result": result.tolist()}
                        append_to_logfile("successful_seed_dataset.json", data)
                    
                    # print(f"Solution Found for id: {id}, KDT_indx: {nn_indx}")
                    # print(f'jnt limit dist: {dist_to_jnt_limit_norm[id]}: {dist_to_jnt_limit[id]}')
                    # print(f"delta_q: {seed_candidates_sorted[id,-1]}: {seed_candidates_sorted[id,1:-1]}")
                    # print(f"Successful Result.: {repr(result)}\n")

                    return result
            # failed to find a solution, use optimization methods to solve and update the database?
        return None


if __name__ == '__main__':
    import time
    import math
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.25, .75, .75], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)

    _jnt_safemargin = math.pi / 18.0
    jlc = rkjlc.JLChain(n_dof=7)
    jlc.anchor.pos = np.array([.0, .0, .3])
    jlc.anchor.rotmat = rm.rotmat_from_euler(np.pi / 3, 0, 0)
    jlc.jnts[0].loc_pos = np.array([.0, .0, .0])
    jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, np.pi)
    jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[0].motion_range = np.array([-2.94087978961 + _jnt_safemargin, 2.94087978961 - _jnt_safemargin])
    jlc.jnts[1].loc_pos = np.array([0.03, .0, .1])
    jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(np.pi / 2, 0.0, 0.0)
    jlc.jnts[1].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[1].motion_range = np.array([-2.50454747661 + _jnt_safemargin, 0.759218224618 - _jnt_safemargin])
    jlc.jnts[2].loc_pos = np.array([-0.03, 0.17283, 0.0])
    jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(-np.pi / 2, 0.0, 0.0)
    jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[2].motion_range = np.array([-2.94087978961 + _jnt_safemargin, 2.94087978961 - _jnt_safemargin])
    jlc.jnts[3].loc_pos = np.array([-0.04188, 0.0, 0.07873])
    jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(np.pi / 2, -np.pi / 2, 0.0)
    jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[3].motion_range = np.array([-2.15548162621 + _jnt_safemargin, 1.3962634016 - _jnt_safemargin])
    jlc.jnts[4].loc_pos = np.array([0.0405, 0.16461, 0.0])
    jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(-np.pi / 2, 0.0, 0.0)
    jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[4].motion_range = np.array([-5.06145483078 + _jnt_safemargin, 5.06145483078 - _jnt_safemargin])
    jlc.jnts[5].loc_pos = np.array([-0.027, 0, 0.10039])
    jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(np.pi / 2, 0.0, 0.0)
    jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[5].motion_range = np.array([-1.53588974176 + _jnt_safemargin, 2.40855436775 - _jnt_safemargin])
    jlc.jnts[6].loc_pos = np.array([0.027, 0.029, 0.0])
    jlc.jnts[6].loc_rotmat = rm.rotmat_from_euler(-np.pi / 2, 0.0, 0.0)
    jlc.jnts[6].loc_motion_ax = np.array([0, 0, 1])
    jlc.jnts[6].motion_range = np.array([-3.99680398707 + _jnt_safemargin, 3.99680398707 - _jnt_safemargin])
    jlc._loc_flange_pos = np.array([0, 0, .007])
    jlc.finalize(ik_solver='d', identifier_str="test")
    jlc._ik_solver.test_success_rate()

    goal_jnt_values = jlc.rand_conf()
    rkmg.gen_jlc_stick_by_jnt_values(jlc, jnt_values=goal_jnt_values, stick_rgba=rm.bc.blue).attach_to(base)

    tgt_pos, tgt_rotmat = jlc.fk(jnt_values=goal_jnt_values)
    tic = time.time()

    jnt_values = jlc.ik(tgt_pos=tgt_pos,
                        tgt_rotmat=tgt_rotmat,
                        toggle_dbg=False)
    toc = time.time()
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    print(toc - tic, jnt_values)
    base.run()
    if jnt_values is not None:
        jlc.goto_given_conf(jnt_values=jnt_values)
        rkmg.gen_jlc_stick(jlc, stick_rgba=rm.bc.navy_blue, toggle_flange_frame=True,
                           toggle_jnt_frames=False).attach_to(base)
        base.run()

    # jlc._ik_solver._test_success_rate()
    jlc._ik_solver.multiepoch_evolve(n_times_per_epoch=10000)
    # jlc._ik_solver.test_success_rate()
    base.run()
