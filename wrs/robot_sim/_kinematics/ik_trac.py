"""
This trac ik solver is implemented following instructions from the trac ik paper.
P. Beeson and B. Ames, "TRAC-IK: An open-source library for improved solving of generic inverse _kinematics,"
IEEE-RAS International Conference on Humanoid Robots (Humanoids), Seoul, Korea (South), 2015, pp. 928-935,
doi: 10.1109/HUMANOIDS.2015.7363472.

Key differences: KDL_RR is implemented as PINV_CW
Known issues: The PINV_CW solver has a much lower success rate. Random restart does not improve performance.

author: weiwei
date: 20231107
"""

import numpy as np
import multiprocessing as mp
# from wrs import basis as rm, robot_sim as rkc
import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.constant as rkc
import scipy.optimize as sopt
import multiprocessing


class NumIKSolverProc(mp.Process):

    def __init__(self,
                 jlc,
                 wln_ratio,
                 param_queue,
                 state_queue,
                 result_queue):
        super(NumIKSolverProc, self).__init__()
        self.jlc = jlc
        self._param_queue = param_queue
        self._state_queue = state_queue
        self._result_queue = result_queue

        self.max_link_length = self._get_max_link_length()
        # self.clamp_pos_err = 2 * self.max_link_length
        # self.clamp_rot_err = np.pi / 3
        self.clamp_pos_err = .1
        self.clamp_rot_err = np.pi / 10.0
        self.jnt_wt_ratio = wln_ratio
        # maximum reach
        self.max_rng = 10.0
        self.min_jnt_values = self.jlc.jnt_ranges[:, 0]
        self.max_jnt_values = self.jlc.jnt_ranges[:, 1]
        self.jnt_range_values = self.max_jnt_values - self.min_jnt_values
        self.min_jnt_value_thresholds = self.min_jnt_values + self.jnt_range_values * self.jnt_wt_ratio
        self.max_jnt_value_thresholds = self.max_jnt_values - self.jnt_range_values * self.jnt_wt_ratio

    def _get_max_link_length(self):
        max_len = 0
        for i in range(1, self.jlc.n_dof):
            if self.jlc.jnts[i].type == rkc.JntType.REVOLUTE:
                tmp_vec = self.jlc.jnts[i].gl_pos_q - self.jlc.jnts[i - 1].gl_pos_q
                tmp_len = np.linalg.norm(tmp_vec)
                if tmp_len > max_len:
                    max_len = tmp_len
        return max_len

    def _jnt_wt_mat(self, jnt_values):
        """
        get the joint weight mat
        :param jnt_values:
        :return: W, W^(1/2)
        author: weiwei
        date: 20201126
        """
        jnt_wt = np.ones(self.jlc.n_dof)
        # min damping interval
        damping_selection = jnt_values < self.min_jnt_value_thresholds
        normalized_diff = (jnt_values - self.min_jnt_values) / (self.min_jnt_value_thresholds - self.min_jnt_values)
        damping_diff = normalized_diff[damping_selection]
        jnt_wt[damping_selection] = -2 * np.power(damping_diff, 3) + 3 * np.power(damping_diff, 2)
        cutting_selection = jnt_values <= self.min_jnt_values
        jnt_wt[cutting_selection] = 0.0
        # max damping interval
        damping_selection = jnt_values > self.max_jnt_value_thresholds
        normalized_diff = (self.max_jnt_values - jnt_values) / (self.max_jnt_values - self.max_jnt_value_thresholds)
        damping_diff = normalized_diff[damping_selection]
        jnt_wt[damping_selection] = -2 * np.power(damping_diff, 3) + 3 * np.power(damping_diff, 2)
        cutting_selection = jnt_values >= self.max_jnt_values
        jnt_wt[cutting_selection] = 0.0
        return np.diag(jnt_wt), np.diag(np.sqrt(jnt_wt))

    def _clamp_tcp_err(self, f2t_pos_err, f2t_rot_err, f2t_err_vec):
        clamped_vec = np.copy(f2t_err_vec)
        if f2t_pos_err >= self.clamp_pos_err:
            clamped_vec[:3] = self.clamp_pos_err * f2t_err_vec[:3] / f2t_pos_err
        if f2t_rot_err >= self.clamp_rot_err:
            clamped_vec[3:6] = self.clamp_rot_err * f2t_err_vec[3:6] / f2t_rot_err
        return clamped_vec

    def run(self):
        while True:
            tgt_pos, tgt_rotmat, seed_jnt_values, max_n_iter = self._param_queue.get()
            iter_jnt_vals = seed_jnt_values.copy()
            counter = 0
            while self._result_queue.empty():  # check if other solver succeeded in the beginning
                tcp_gl_pos, tcp_gl_rotmat, j_mat = self.jlc.fk(jnt_values=iter_jnt_vals,
                                                                toggle_jacobian=True,
                                                                update=False)
                tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_poses(src_pos=tcp_gl_pos,
                                                                                      src_rotmat=tcp_gl_rotmat,
                                                                                      tgt_pos=tgt_pos,
                                                                                      tgt_rotmat=tgt_rotmat)
                if tcp_pos_err_val < 1e-4 and tcp_rot_err_val < 1e-3:
                    if self._result_queue.empty():
                        self._result_queue.put(('n', iter_jnt_vals))
                        # print('n-','tgt_pos:', tgt_pos, 'jnt_values:', iter_jnt_vals)
                    break
                clamped_err_vec = self._clamp_tcp_err(tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec)
                wln, wln_sqrt = self._jnt_wt_mat(iter_jnt_vals)
                # weighted clamping
                k_phi = 0.1
                tmp_mm_jnt_values = self.max_jnt_values + self.min_jnt_values
                phi_q = ((2 * iter_jnt_vals - tmp_mm_jnt_values) / self.jnt_range_values) * k_phi
                clamping = -(np.identity(wln.shape[0]) - wln) @ phi_q
                # pinv with weighted clamping
                delta_jnt_values = clamping + wln_sqrt @ np.linalg.pinv(j_mat @ wln_sqrt, rcond=1e-4) @ (
                        clamped_err_vec - j_mat @ clamping)
                iter_jnt_vals = iter_jnt_vals + delta_jnt_values
                if counter > max_n_iter:
                    # optik failed
                    self._result_queue.put(None)
                    break
                counter += 1
            self._state_queue.put(1)


class OptIKSolverProc(mp.Process):
    def __init__(self,
                 jlc,
                 param_queue,
                 state_queue,
                 result_queue):
        super(OptIKSolverProc, self).__init__()
        self.jlc = jlc
        self._param_queue = param_queue
        self._result_queue = result_queue
        self._state_queue = state_queue


    def _rand_conf(self):
        """
        generate a random configuration
        author: weiwei
        date: 20200326
        """
        return np.multiply(np.random.rand(self.jlc.n_dof),
                           (self.jlc.jnt_ranges[:, 1] - self.jlc.jnt_ranges[:, 0])) + self.jlc.jnt_ranges[:, 0]

    def run(self):  # OptIKSolver.sqpss
        """
        sqpss is faster than sqp
        :return:
        author: weiwei
        date: 20231101
        """

        def _objective(x, tgt_pos, tgt_rotmat):
            flange_pos, flange_rotmat, j_mat = self.jlc.fk(jnt_values=x,
                                                            toggle_jacobian=True,
                                                            update=False)
            tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_poses(src_pos=flange_pos,
                                                                                  src_rotmat=flange_rotmat,
                                                                                  tgt_pos=tgt_pos,
                                                                                  tgt_rotmat=tgt_rotmat)
            return tcp_err_vec.dot(tcp_err_vec)

        def _call_back(x):
            """
            check if other solvers succeeded at the end of each iteration
            :param x:
            :return:
            """
            if not self._result_queue.empty():
                self._state_queue.put(1)
                raise StopIteration

        # sqpss with random restart
        while True:
            tgt_pos, tgt_rotmat, seed_jnt_values, max_n_iter = self._param_queue.get()
            options = {'maxiter': max_n_iter}
            counter = 0
            while True:
                counter += 1
                try:
                    result = sopt.minimize(fun=_objective,
                                           args=(tgt_pos, tgt_rotmat),
                                           x0=seed_jnt_values,
                                           method='SLSQP',
                                           bounds=self.jlc.jnt_ranges,
                                           options=options,
                                           callback=_call_back)
                except StopIteration:
                    break  # other solver succeeded
                if self._result_queue.empty():
                    if result.success and result.fun < 1e-4:
                        self._result_queue.put(('o', result.x))
                        # print('o-','tgt_pos:', tgt_pos, 'jnt_values:', result.x)
                        break
                    else:
                        if counter > 10:
                            self._result_queue.put(None)
                            break
                        else:
                            seed_jnt_values = self._rand_conf()
                            continue
                break
            self._state_queue.put(1)


class TracIKSolver(object):
    """
    author: weiwei
    date: 20231102
    """

    def __init__(self, jlc, wln_ratio=.05):
        self.jlc = jlc
        self._default_seed_jnt_values = self.jlc.get_jnt_values()
        self._nik_param_queue = mp.Queue()
        self._oik_param_queue = mp.Queue()
        self._nik_state_queue = mp.Queue()
        self._oik_state_queue = mp.Queue()
        self._result_queue = mp.Queue(maxsize=1)
        self.nik_solver_proc = NumIKSolverProc(self.jlc,
                                               wln_ratio,
                                               self._nik_param_queue,
                                               self._nik_state_queue,
                                               self._result_queue)
        self.oik_solver_proc = OptIKSolverProc(self.jlc,
                                               self._oik_param_queue,
                                               self._oik_state_queue,
                                               self._result_queue)
        self.nik_solver_proc.start()
        self.oik_solver_proc.start()
        # self._tcp_gl_pos, self._tcp_gl_rotmat = self.jlc.get_gl_tcp()
        self._tcp_gl_pos, self._tcp_gl_rotmat = self.jlc.gl_flange_pos, self.jlc.gl_flange_rotmat
        # run once to avoid long waiting time in the beginning
        self._oik_param_queue.put((self._tcp_gl_pos, self._tcp_gl_rotmat, self._default_seed_jnt_values, 10))
        self._oik_state_queue.get()
        self._result_queue.get()

    def __call__(self,
                 tgt_pos,
                 tgt_rotmat,
                 seed_jnt_values=None,
                 max_n_iter=100,
                 toggle_dbg=False):
        return self.ik(tgt_pos=tgt_pos,
                       tgt_rotmat=tgt_rotmat,
                       seed_jnt_values=seed_jnt_values,
                       max_n_iter=max_n_iter,
                       toggle_dbg=toggle_dbg)

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_values=None,
           max_n_iter=100,
           toggle_dbg=False):
        """
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param max_n_iter:
        :param toggle_dbg: the function will return a tuple like (solver, jnt_values); solver is 'o' (opt) or 'n' (num)
        :return:
        author: weiwei
        date: 20231107
        """
        if seed_jnt_values is None:
            seed_jnt_values = self._default_seed_jnt_values
        self._nik_param_queue.put((tgt_pos, tgt_rotmat, seed_jnt_values, max_n_iter))
        self._oik_param_queue.put((tgt_pos, tgt_rotmat, seed_jnt_values, max_n_iter))
        if self._nik_state_queue.get() and self._oik_state_queue.get():
            result = self._result_queue.get()
            if toggle_dbg:
                return result
            else:
                if result is None:
                    return None
                else:
                    return result[1]