import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.robot_sim.manipulators.cobotta.ikgeo as ikgeo


class IRB14050(mi.ManipulatorInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 name='irb14050',
                 enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=np.zeros(7), name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        # self.jlc.anchor.lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link_1.stl"))
        # self.jlc.anchor.lnk.cmodel.rgba = np.array([.5, .5, .5, 1])
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, np.pi)
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-2.94087978961, 2.94087978961])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link_1.stl"), name="irb14050_link_1")
        self.jlc.jnts[0].lnk.cmodel.rgba = rm.const.hug_gray
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([0.03, .0, .1])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(np.pi / 2, 0.0, 0.0)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[1].motion_range = np.array([-2.50454747661, 0.759218224618])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link_2.stl"), name="irb14050_link_2")
        self.jlc.jnts[1].lnk.cmodel.rgba = rm.const.hug_blue
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([-0.03, 0.17283, 0.0])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(-np.pi / 2, 0.0, 0.0)
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].motion_range = np.array([-2.94087978961, 2.94087978961])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link_3.stl"), name="irb14050_link_3")
        self.jlc.jnts[2].lnk.cmodel.rgba = rm.const.hug_gray
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([-0.04188, 0.0, 0.07873])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(np.pi / 2, -np.pi / 2, 0.0)
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[3].motion_range = np.array([-2.15548162621, 1.3962634016])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link_4.stl"), name="irb14050_link_4")
        self.jlc.jnts[3].lnk.cmodel.rgba = rm.const.hug_blue
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([0.0405, 0.16461, 0.0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(-np.pi / 2, 0.0, 0.0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-5.06145483078, 5.06145483078])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link_5.stl"), name="irb14050_link_5")
        self.jlc.jnts[4].lnk.cmodel.rgba = rm.const.hug_gray
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([-0.027, 0, 0.10039])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(np.pi / 2, 0.0, 0.0)
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].motion_range = np.array([-1.53588974176, 2.40855436775])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link_6.stl"), name="irb14050_link_6")
        self.jlc.jnts[5].lnk.cmodel.rgba = rm.const.hug_gray
        # seventh joint and link
        self.jlc.jnts[6].loc_pos = np.array([0.027, 0.029, 0.0])
        self.jlc.jnts[6].loc_rotmat = rm.rotmat_from_euler(-np.pi / 2, 0.0, 0.0)
        self.jlc.jnts[6].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[6].motion_range = np.array([-3.99680398707, 3.99680398707])
        self.jlc.jnts[6].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link_7.stl"), name="irb14050_link_7")
        self.jlc.jnts[6].lnk.cmodel.rgba = rm.const.hug_gray
        self.jlc._loc_flange_rotmat = rm.rotmat_from_euler(0, 0, np.pi / 2)
        self.jlc.finalize(ik_solver='d', identifier_str=name)
        # tcp
        self.loc_tcp_pos = np.array([0, 0, .007])
        self.loc_tcp_rotmat = np.eye(3)
        # set up cc
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        l6 = self.cc.add_cce(self.jlc.jnts[6].lnk)
        from_list = [l0]
        into_list = [l4, l5]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           best_sol_num,
           seed_jnt_values=None,
           option="single",
           toggle_dbg=False):
        """
        This ik solver uses ikgeo to find an initial solution and then uses numik(pinv) as a backbone for precise
        computation. IKGeo assumes the jlc root is at pos=0 and rotmat=I. Numik uses jlc fk and does not have this
        assumption. IKGeo will shift jlc root to zero. There is no need to do them on the upper level. (20241121)
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param option:
        :param toggle_dbg:
        :return:
        """
        toggle_update = False
        # directly use specified ik
        self.jlc._ik_solver._k_max = 200
        rel_rotmat = tgt_rotmat @ self.loc_tcp_rotmat.T
        rel_pos = tgt_pos - tgt_rotmat @ self.loc_tcp_pos
        result = self.jlc.ik(tgt_pos=rel_pos, tgt_rotmat=rel_rotmat, seed_jnt_values=seed_jnt_values, best_sol_num = best_sol_num)

        return result

        # mcm.mgm.gen_myc_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        # result = ikgeo.ik(jlc=self.jlc, tgt_pos=rel_pos, tgt_rotmat=rel_rotmat, seed_jnt_values=None)
        # if result is None:
        #     # print("No valid solutions found")
        #     return None
        # else:
        #     if toggle_update:
        #         rel_pos, rel_rotmat = rm.rel_pose(self.jlc.pos, self.jlc.rotmat, rel_pos, rel_rotmat)
        #         rel_rotvec = self.jlc._ik_solver._rotmat_to_vec(rel_rotmat)
        #         query_point = np.concatenate((rel_pos, rel_rotvec))
        #         # update dd driven file
        #         tree_data = np.vstack((self.jlc._ik_solver.query_tree.data, query_point))
        #         self.jlc._ik_solver.jnt_data.append(result)
        #         self.jlc._ik_solver.query_tree = scipy.spatial.cKDTree(tree_data)
        #         print(f"Updating query tree, {id} explored...")
        #         self.jlc._ik_solver.persist_data()
        #     return result


if __name__ == '__main__':
    import time
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.5, 0, 1.5], lookat_pos=[0, 0, .2])
    mcm.mgm.gen_frame().attach_to(base)
    arm = IRB14050(enable_cc=True)
    arm.gen_stickmodel().attach_to(base)
    arm.gen_meshmodel(alpha=1, toggle_flange_frame=True).attach_to(base)
    arm.show_cdprim()
    tic = time.time()
    print(arm.is_collided())
    toc = time.time()
    print(toc - tic)

    while True:
        tgt_pos, tgt_rotmat = arm.fk(jnt_values=arm.rand_conf())
        mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
        tic = time.time()
        jnt_values = arm.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
        toc = time.time()
        print(toc - tic)
        if jnt_values is not None:
            arm.goto_given_conf(jnt_values=jnt_values)
            arm.gen_meshmodel(toggle_flange_frame=True).attach_to(base)
            base.run()
