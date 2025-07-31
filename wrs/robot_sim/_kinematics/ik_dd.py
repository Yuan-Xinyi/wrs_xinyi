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

def cosine_similarity(a, b, eps=1e-8):
    dot = np.sum(a * b, axis=1)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return dot / (norm + eps)


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
        self._fname_x_tree = os.path.join(path, f"{identifier_str}_ikdd_x_tree.pkl")
        self._fname_q_tree = os.path.join(path, f"{identifier_str}_ikdd_q_tree.pkl")
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
                self.query_tree, self.q_query_tree, self.jnt_data, self.tcp_data, self.jinv_data = self._build_data()
                self.persist_data()
        else:
            try:
                with open(self._fname_x_tree, 'rb') as f_tree:
                    self.query_tree = pickle.load(f_tree)
                with open(self._fname_q_tree, 'rb') as f_tree:
                    self.q_query_tree = pickle.load(f_tree)
                with open(self._fname_jnt, 'rb') as f_jnt:
                    self.jnt_data, self.tcp_data, self.jinv_data, self.pos_data, self.rotmat_data, self.jmat_data = pickle.load(f_jnt)
            except FileNotFoundError:
                (self.query_tree, self.q_query_tree, 
                 self.jnt_data, self.tcp_data, self.jinv_data, 
                 self.pos_data, self.rotmat_data, self.jmat_data) = self._build_data()
                print("Database files not found. Rebuilding the database.")
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
        print("Building Data for DDIK...")

        # ----------- Part 1: Build x_query_tree (coarse) -----------
        query_data = []
        jnt_data = []
        jinv_data = []
        scale = 'scale5'  # 'scale9', 'scale7', 'largest'
        if self.jlc.name == 'cobotta_arm':
            rbt_name = 'cobotta' # [cobotta, cobotta_pro1300, yumi, ur3]
        elif self.jlc.name == 'cobotta_pro_1300_manipulator':
            rbt_name = 'cobotta_pro1300'
        elif self.jlc.name == 'irb14050_sglarm_yumi':
            rbt_name = 'yumi'
        elif self.jlc.name == 'ur3':
            rbt_name = 'ur3'
        else:
            raise ValueError(f"Unknown robot name: {self.jlc.robot_name}. Please use 'cobotta_arm', 'cobotta_pro_1300_manipulator', 'irb14050_sglarm_yumi', or 'ur3'.")
        sampled_qs_x = np.load(f'cvt_joint_samples_{rbt_name}_{scale}.npy') # [cobotta, cobotta_pro1300, yumi, ur3]

        for jnt_values in tqdm(sampled_qs_x, desc="x_query_tree sampling"):
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=jnt_values, toggle_jacobian=True)
            jinv = np.linalg.pinv(j_mat, rcond=1e-4)
            rel_pos, rel_rotmat = rm.rel_pose(self.jlc.pos, self.jlc.rotmat, flange_pos, flange_rotmat)
            rel_rotvec = self._rotmat_to_vec(rel_rotmat)
            query_data.append(rel_pos.tolist() + rel_rotvec.tolist())
            jnt_data.append(jnt_values)
            jinv_data.append(jinv)
        x_query_tree = scipy.spatial.cKDTree(query_data)

        # ----------- Part 2: Build q_query_tree (fine) -----------
        q_query_data = []
        pos_data = []
        rotmat_data = []
        jmat_data = []

        scale = 'largest'  # 'scale9', 'scale7', 'largest'
        if self.jlc.name == 'cobotta_arm':
            rbt_name = 'cobotta' # [cobotta, cobotta_pro1300, yumi, ur3]
        elif self.jlc.name == 'cobotta_pro_1300_manipulator':
            rbt_name = 'cobotta_pro1300'
        elif self.jlc.name == 'irb14050_sglarm_yumi':
            rbt_name = 'yumi'
        elif self.jlc.name == 'ur3':
            rbt_name = 'ur3'
        else:
            raise ValueError(f"Unknown robot name: {self.jlc.robot_name}. Please use 'cobotta_arm', 'cobotta_pro_1300_manipulator', 'irb14050_sglarm_yumi', or 'ur3'.")
        sampled_qs_q = np.load(f'cvt_joint_samples_{rbt_name}_{scale}.npy') # [cobotta, cobotta_pro1300, yumi, ur3]


        for jnt_values in tqdm(sampled_qs_q, desc="q_query_tree sampling"):
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=jnt_values, toggle_jacobian=True)
            normalized_jnt = (jnt_values - self.jlc.jnt_ranges[:, 0]) / (self.jlc.jnt_ranges[:, 1] - self.jlc.jnt_ranges[:, 0] + 1e-8)
            q_query_data.append(normalized_jnt.tolist())
            pos_data.append(flange_pos)
            rotmat_data.append(flange_rotmat)
            jmat_data.append(j_mat)

        q_query_tree = scipy.spatial.cKDTree(q_query_data)

        return (
            x_query_tree,
            q_query_tree,
            np.asarray(jnt_data),
            np.asarray(query_data),
            np.asarray(jinv_data),
            np.asarray(pos_data),
            np.asarray(rotmat_data),
            np.asarray(jmat_data),
        )



    def persist_data(self):
        with open(self._fname_x_tree, 'wb') as f_tree:
            pickle.dump(self.query_tree, f_tree)
        with open(self._fname_q_tree, 'wb') as f_tree:
            pickle.dump(self.q_query_tree, f_tree)
        with open(self._fname_jnt, 'wb') as f_jnt:
            pickle.dump([self.jnt_data, self.tcp_data, self.jinv_data, self.pos_data, self.rotmat_data, self.jmat_data], f_jnt)
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
            import time
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

            # === 第一次迭代 ===
            delta_q_list = []
            delta_pose_list = []
            for cad_id, jnt in enumerate(seed_jnt_array_cad):
                jnt_idx = self.q_query_tree.query(((jnt - self.jlc.jnt_ranges[:, 0]) / (self.jlc.jnt_ranges[:, 1] - self.jlc.jnt_ranges[:, 0] + 1e-8)), 
                                                  k=1, workers=-1)[1]
                pos, rotmat, j_mat = self.pos_data[jnt_idx], self.rotmat_data[jnt_idx], self.jmat_data[jnt_idx]
                pos1, rotmat1, j_mat1 = self.jlc.fk(jnt_values=jnt, toggle_jacobian=True)
                f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_poses(src_pos=pos,
                                                                              src_rotmat=rotmat,
                                                                              tgt_pos=tgt_pos,
                                                                              tgt_rotmat=tgt_rotmat)
                clamped_err_vec = clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
                delta_pose_list.append(clamped_err_vec)
                delta_jnt_values = np.linalg.lstsq(j_mat, clamped_err_vec, rcond=1e-4)[0]
                delta_q_list.append(delta_jnt_values)
            delta_jnt_values_array = np.array(delta_q_list)
            delta_pose_array = np.array(delta_pose_list)
            next_jnt_values_array = seed_jnt_array_cad + delta_jnt_values_array

            # === 第二次迭代 ===
            middle_jnt_array = seed_jnt_array_cad + 1.0 * delta_jnt_values_array
            mid_delta_q_list = []
            mid_delta_pose_list = []
            for cad_id, jnt in enumerate(middle_jnt_array):
                jnt_idx = self.q_query_tree.query(((jnt - self.jlc.jnt_ranges[:, 0]) / (self.jlc.jnt_ranges[:, 1] - self.jlc.jnt_ranges[:, 0] + 1e-8)), 
                                                  k=1, workers=-1)[1]
                pos, rotmat, j_mat = self.pos_data[jnt_idx], self.rotmat_data[jnt_idx], self.jmat_data[jnt_idx]
                f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_poses(src_pos=pos,
                                                                              src_rotmat=rotmat,
                                                                              tgt_pos=tgt_pos,
                                                                              tgt_rotmat=tgt_rotmat)
                clamped_err_vec = clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
                mid_delta_pose_list.append(clamped_err_vec)
                delta_jnt_values = np.linalg.lstsq(j_mat, clamped_err_vec, rcond=1e-4)[0]
                mid_delta_q_list.append(delta_jnt_values)
            mid_delta_jnt_values_array = np.array(mid_delta_q_list)
            mid_next_jnt_values_array = middle_jnt_array + mid_delta_jnt_values_array
            mid_delta_pose_array = np.array(mid_delta_pose_list)

            # === 第三次迭代 ===
            final_delta_q_list = []
            final_delta_pose_list = []
            for cad_id, jnt in enumerate(mid_next_jnt_values_array):
                jnt_idx = self.q_query_tree.query(((jnt - self.jlc.jnt_ranges[:, 0]) / (self.jlc.jnt_ranges[:, 1] - self.jlc.jnt_ranges[:, 0] + 1e-8)), 
                                                  k=1, workers=-1)[1]
                pos, rotmat, j_mat = self.pos_data[jnt_idx], self.rotmat_data[jnt_idx], self.jmat_data[jnt_idx]
                f2t_pos_err, f2t_rot_err, f2t_err_vec = rm.diff_between_poses(
                    src_pos=pos,
                    src_rotmat=rotmat,
                    tgt_pos=tgt_pos,
                    tgt_rotmat=tgt_rotmat
                )
                clamped_err_vec = clamp_tgt_err(f2t_pos_err, f2t_rot_err, f2t_err_vec)
                final_delta_pose_list.append(clamped_err_vec)
                delta_jnt_values = np.linalg.lstsq(j_mat, clamped_err_vec, rcond=1e-4)[0]
                final_delta_q_list.append(delta_jnt_values)

            final_delta_jnt_values_array = np.array(final_delta_q_list)
            final_delta_pose_array = np.array(final_delta_pose_list)
            final_jnt_values_array = mid_next_jnt_values_array + final_delta_jnt_values_array

            '''sum of three joint adjustment based ranking'''
            # delta_jnt_values_squared = np.sum(delta_jnt_values_array ** 2, axis=1)
            # mid_delta_jnt_values_squared = np.sum(mid_delta_jnt_values_array ** 2, axis=1)
            final_delta_jnt_values_squared = np.sum(final_delta_jnt_values_array ** 2, axis=1)
            sum_squared = final_delta_jnt_values_squared
            sorted_indices = np.argsort(sum_squared)
            seed_jnt_array_cad = final_jnt_values_array[sorted_indices[:20]]

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
