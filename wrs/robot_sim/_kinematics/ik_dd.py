"""
Data driven ik solver
author: weiwei
date: 20231107
"""

import os
import pickle
import numpy as np
import scipy.spatial
from direct.showbase.Job import addTestJob
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
'''statistical analysis'''
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.distance import cdist
from sklearn.cluster import MeanShift, estimate_bandwidth

import numpy as np

import numpy as np

import numpy as np
import numpy as np

import numpy as np

def damped_pinv(J, damping=1e-4):
    m, n = J.shape
    if m >= n:
        # "Tall" or square Jacobian: (JᵗJ + λ²I)⁻¹ Jᵗ
        JTJ = J.T @ J
        damped_term = damping ** 2 * np.eye(n)
        return np.linalg.inv(JTJ + damped_term) @ J.T
    else:
        # "Wide" Jacobian: Jᵗ (JJᵗ + λ²I)⁻¹
        JJT = J @ J.T
        damped_term = damping ** 2 * np.eye(m)
        return J.T @ np.linalg.inv(JJT + damped_term)

def clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec):
    clamp_pos_err = .1
    clamp_rot_err = np.pi / 10.0
    clamped_vec = np.copy(f2t_err_vec)
    if f2t_pos_err >= clamp_pos_err:
        clamped_vec[:3] = clamp_pos_err * f2t_err_vec[:3] / f2t_pos_err
    if f2t_rot_err >= clamp_rot_err:
        clamped_vec[3:6] = clamp_rot_err * f2t_err_vec[3:6] / f2t_rot_err
    return clamped_vec

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
        if not os.path.exists(path):
            os.makedirs(path)
        self._fname_tree = os.path.join(path, f"{identifier_str}_ikdd_tree.pkl")
        self._fname_jnt = os.path.join(path, f"{identifier_str}_jnt_data.pkl")
        self._k_max = 1000  # maximum nearest neighbours examined by the backbone solver
        self._max_n_iter = 7  # max_n_iter of the backbone solver
        
        if backbone_solver == 'n':
            self._backbone_solver = ikn.NumIKSolver(self.jlc)
            print("Using NumIKSolver as the backbone solver.")
        elif backbone_solver == 'o':
            self._backbone_solver = iko.OptIKSolver(self.jlc)
            print("Using OptIKSolver as the backbone solver.")
        elif backbone_solver == 't':
            self._backbone_solver = ikt.TracIKSolver(self.jlc)
            print("Using TracIKSolver as the backbone solver.")
        if rebuild:
            print("Rebuilding the database. It starts a new evolution and is costly.")
            y_or_n = bu.get_yesno()
            if y_or_n == 'y':
                self.query_tree, self.jnt_data, self.tcp_data, self.jinv_data = self._build_data()
                self.persist_data()
        else:
            try:
                with open(self._fname_tree, 'rb') as f_tree:
                    self.query_tree = pickle.load(f_tree)
                with open(self._fname_jnt, 'rb') as f_jnt:
                    self.jnt_data, self.tcp_data, self.jinv_data = pickle.load(f_jnt)
            except FileNotFoundError:
                self.query_tree, self.jnt_data, self.tcp_data, self.jinv_data = self._build_data()
                self.persist_data()

    def __call__(self,
                 tgt_pos,
                 tgt_rotmat,
                 best_sol_num,
                 seed_jnt_values=None,
                 max_n_iter=None,
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
        # gen sampled qs
        sampled_jnts = []
        n_intervals = np.linspace(8, 4, self.jlc.n_dof, endpoint=False) # 6,8,10
        print(f"Buidling Data for DDIK using the following joint granularity: {n_intervals.astype(int)}...")
        for i in range(self.jlc.n_dof):
            sampled_jnts.append(
                np.linspace(self.jlc.jnt_ranges[i][0], self.jlc.jnt_ranges[i][1], int(n_intervals[i] + 2))[1:-1])
        grid = np.meshgrid(*sampled_jnts)
        sampled_qs = np.vstack([x.ravel() for x in grid]).T
        # gen sampled qs and their correspondent flange poses
        query_data = []
        jnt_data = []
        jinv_data = []
        for id in tqdm(range(len(sampled_qs))):
            jnt_values = sampled_qs[id]
            # pinv of jacobian
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=jnt_values, toggle_jacobian=True)
            jinv = np.linalg.pinv(j_mat, rcond=1e-4)
            # jinv = np.linalg.inv(j_mat.T @ j_mat + 1e-4 * np.eye(j_mat.shape[1])) @ j_mat.T
            # relative to base
            rel_pos, rel_rotmat = rm.rel_pose(self.jlc.pos, self.jlc.rotmat, flange_pos, flange_rotmat)
            rel_rotvec = self._rotmat_to_vec(rel_rotmat)

            '''baseline query data'''
            query_data.append(rel_pos.tolist() + rel_rotvec.tolist())
            jnt_data.append(jnt_values)
            jinv_data.append(jinv)
        query_tree = scipy.spatial.cKDTree(query_data)

        return query_tree, np.asarray(jnt_data), np.asarray(query_data), np.asarray(jinv_data)

    def persist_data(self):
        with open(self._fname_tree, 'wb') as f_tree:
            pickle.dump(self.query_tree, f_tree)
        with open(self._fname_jnt, 'wb') as f_jnt:
            pickle.dump([self.jnt_data, self.tcp_data, self.jinv_data], f_jnt)
        print("ddik data file saved.")

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           best_sol_num=1,
           seed_jnt_values=None,
           max_n_iter=None,
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
            dist_value_list, nn_indx_list = self.query_tree.query(query_point, k=self._k_max, workers=-1)
            if type(nn_indx_list) is int:
                nn_indx_list = [nn_indx_list]
            seed_jnt_array = self.jnt_data[nn_indx_list]
            seed_tcp_array = self.tcp_data[nn_indx_list]
            seed_jinv_array = self.jinv_data[nn_indx_list]
            seed_posrot_diff_array = query_point - seed_tcp_array
            
            '''original ranking by distance'''
            adjust_array = np.einsum('ijk,ik->ij', seed_jinv_array, seed_posrot_diff_array)
            square_sums = np.sum((adjust_array) ** 2, axis=1)
            sorted_indices = np.argsort(square_sums)
            seed_jnt_array_cad = seed_jnt_array[sorted_indices[:20]]

            '''first iteration'''
            # first_cad = seed_jnt_array_cad[0]
            # pos, rotmat, j_mat = self.jlc.fk(jnt_values=first_cad, toggle_jacobian=True)
            # f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_poses(src_pos=pos,
            #                                                               src_rotmat=rotmat,
            #                                                               tgt_pos=tgt_pos,
            #                                                               tgt_rotmat=tgt_rotmat)
            # clamped_err_vec = clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
            # # delta_jnt_values = np.linalg.lsts
            # delta_jnt_values = np.linalg.lstsq(j_mat, clamped_err_vec, rcond=1e-4)[0]
            # next_jnt_values = first_cad + delta_jnt_values
            # # adjust_array[sorted_indices[0]]
            
            '''next iteration'''
            # pos, rotmat, j_mat = self.jlc.fk(jnt_values=next_jnt_values, toggle_jacobian=True)
            # f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_poses(src_pos=pos,
            #                                                               src_rotmat=rotmat,
            #                                                               tgt_pos=tgt_pos,
            #                                                               tgt_rotmat=tgt_rotmat)
            # clamped_err_vec = clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
            # delta_jnt_values = np.linalg.lstsq(j_mat, clamped_err_vec, rcond=1e-4)[0]
            # next2_jnt_values = next_jnt_values + delta_jnt_values
            # print(f"delta_jnt_values: {delta_jnt_values}")
        
        
        
        
        
        #     top20_nn_index_list = [nn_indx_list[i] for i in sorted_indices[:20]]
        #     if type(top20_nn_index_list) is int:
        #         top20_nn_index_list = [top20_nn_index_list]


        #     '''add mid joint point evaluation'''
        #     # # relative to base
        #     # rel_pos, rel_rotmat = rm.rel_pose(self.jlc.pos, self.jlc.rotmat, tgt_pos, tgt_rotmat)
        #     # rel_rotvec = self._rotmat_to_vec(rel_rotmat)
        #     # query_point = np.concatenate((rel_pos, rel_rotvec))
        #     # dist_value_list, nn_indx_list = self.query_tree.query(query_point, k=self._k_max, workers=-1)

        #     # if isinstance(nn_indx_list, int):
        #     #     nn_indx_list = [nn_indx_list]

        #    # 原始种子点
        #     nn_indx_list = np.array(top20_nn_index_list)
        #     seed_jnt_array = self.jnt_data[nn_indx_list]
        #     seed_tcp_array = self.tcp_data[nn_indx_list]
        #     seed_jinv_array = self.jinv_data[nn_indx_list]

        #     # 差值 Δx（注意目标是 query_point = x_t）
        #     seed_dx_array = seed_tcp_array - query_point

        #     # e_s = J^+ * Δx
        #     adjust_array_1 = np.einsum('ijk,ik->ij', seed_jinv_array, seed_dx_array)

        #     # 中间点 joint = 原始 joint + 0.5 * Δq
        #     middle_jnt_array = seed_jnt_array + 0.5 * adjust_array_1
        #     # middle_jnt_array = seed_jnt_array + adjust_array_1
        #     middle_tcp_array = []
        #     middle_jinv_array = []

        #     for jnt in middle_jnt_array:
        #         pos, rotmat, jacobian = self.jlc.fk(jnt_values=jnt, toggle_jacobian=True)
        #         tcp = np.concatenate((pos, self._rotmat_to_vec(rotmat)))
        #         middle_tcp_array.append(tcp)
        #         # middle_jinv_array.append(np.linalg.pinv(jacobian, rcond=1e-4))
        #         middle_jinv_array.append(damped_pinv(jacobian, damping=1e-4))

        #     middle_tcp_array = np.array(middle_tcp_array)
        #     middle_jinv_array = np.array(middle_jinv_array)

        #     # 差值 Δx_mid
        #     middle_dx_array = middle_tcp_array - query_point

        #     # e_mid = J^+ * Δx_mid
        #     adjust_array_2 = np.einsum('ijk,ik->ij', middle_jinv_array, middle_dx_array)

        #     # 比例项 A_s = e / Δx
        #     ratio_array_1 = adjust_array_1 / seed_dx_array
        #     ratio_array_2 = adjust_array_2 / middle_dx_array

        #     # 线性性误差：均值差平方
        #     linear_error_array = (np.mean(np.abs(ratio_array_1), axis=1) - np.mean(np.abs(ratio_array_2), axis=1)) ** 2
        #     # linear_error_array = np.sum((ratio_array_1 - ratio_array_2) ** 2, axis=1)
        #     # linear_error_array = np.sum((ratio_array_2) ** 2, axis=1)

        #     # 根据误差排序（越小越线性）
        #     sorted_indices = np.argsort(linear_error_array)

        #     # 取 top-20 的 joint seed
        #     seed_jnt_array_cad = seed_jnt_array[sorted_indices[:20]]
        #     # seed_jnt_array_cad += adjust_array_2[sorted_indices[:20]] + adjust_array_1[sorted_indices[:20]]*1/2
        #     seed_jnt_array_cad += adjust_array_1[sorted_indices[:20]]


            for id, seed_jnt_values in enumerate(seed_jnt_array_cad):
                if id > best_sol_num:
                    return None
                if toggle_dbg:
                    rkmg.gen_jlc_stick_by_jnt_values(self.jlc,
                                                     jnt_values=seed_jnt_values,
                                                     stick_rgba=rm.const.red).attach_to(base)
                result = self._backbone_solver(tgt_pos=tgt_pos,
                                               tgt_rotmat=tgt_rotmat,
                                               seed_jnt_values=seed_jnt_values,
                                               max_n_iter=max_n_iter,
                                               toggle_dbg=toggle_dbg)
                if result is None:
                    nid = id+1
                    distances = np.linalg.norm(nid*seed_jnt_array_cad[nid:] - np.sum(seed_jnt_array_cad[:nid], axis=0), axis=1) # 和的差的平方
                    # distances = np.sum(np.sum((seed_jnt_array_cad[nid:, None, :] - seed_jnt_array_cad[:nid][None, :, :])**2, axis=2),axis=1)  # 差的平方和
                    sorted_cad_indices = np.argsort(-distances)
                    seed_jnt_array_cad[nid:] = seed_jnt_array_cad[nid:][sorted_cad_indices]
                    continue
                else:
                    return result
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
    jlc.finalize(ik_solver='dr', identifier_str="test_new")
    jlc.test_ik_success_rate()

    # goal_jnt_values = jlc.rand_conf()
    # rkmg.gen_jlc_stick_by_jnt_values(jlc, jnt_values=goal_jnt_values, stick_rgba=rm.bc.blue).attach_to(base)
    #
    # tgt_pos, tgt_rotmat = jlc.fk(jnt_values=goal_jnt_values)
    # tic = time.time()
    #
    # jnt_values = jlc.ik(tgt_pos=tgt_pos,
    #                     tgt_rotmat=tgt_rotmat,
    #                     toggle_dbg=False)
    # toc = time.time()
    # mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # print(toc - tic, jnt_values)
    # base.run()
    # if jnt_values is not None:
    #     jlc.goto_given_conf(jnt_values=jnt_values)
    #     rkmg.gen_jlc_stick(jlc, stick_rgba=rm.bc.navy_blue, toggle_flange_frame=True,
    #                        toggle_jnt_frames=False).attach_to(base)
    #     base.run()

    # jlc._ik_solver._test_success_rate()
    # jlc._ik_solver.multiepoch_evolve(n_times_per_epoch=10000)
    # jlc._ik_solver.test_success_rate()
    base.run()
