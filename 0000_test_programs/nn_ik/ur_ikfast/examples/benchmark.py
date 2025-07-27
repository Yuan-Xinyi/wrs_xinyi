from ur_ikfast import ur_kinematics
import numpy as np
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.basis.robot_math as rm
from wrs import wd, rm, mcm
import time
base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)
robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), enable_cc=True)

# org = [ 4.97551478, -0.16479593, -2.16665078,  2.33002626,  4.77466304,
#         5.68801327]
# solution = [ 3.58773524, -1.06290841, -2.16530871,  3.22238022,  3.38688738,  5.68226105]
# robot.goto_given_conf(jnt_values=org)
# robot.gen_meshmodel( alpha=.3, rgb=[0, 1, 0]).attach_to(base)
# tgt_pos, tgt_rotmat = robot.fk(jnt_values = org)
# print(f'tgt_pos: {tgt_pos}, tgt_rotmat: {tgt_rotmat}')
# mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

# robot.goto_given_conf(jnt_values=solution)
# robot.gen_meshmodel(alpha=.3, rgb=[1, 0, 0]).attach_to(base)
# tgt_pos, tgt_rotmat = robot.fk(jnt_values = solution)
# mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
# print(f'tgt_pos: {tgt_pos}, tgt_rotmat: {tgt_rotmat}')
# base.run()

robot_name = 'ur3'
arm = ur_kinematics.URKinematics(robot_name)
ikfast_res = 0
total = 10000
pos_err_list = []
rot_err_list = []
time_list = []

for i in range(total):
    # joint_angles = np.random.uniform(-1*np.pi, 1*np.pi, size=6)
    joint_angles = robot.rand_conf()
    pose = arm.forward(joint_angles, rotation_type='matrix')
    tgt_pos, tgt_rotmat = pose[:, 3], pose[:, :3]
    tic = time.time()
    ik_solution = arm.inverse(pose, False, q_guess=joint_angles)
    toc = time.time()
    time_list.append(toc-tic)
    
    if ik_solution is not None:
        ikfast_res += 1
        # if np.allclose(joint_angles, ik_solution, rtol=0.01):
        #     ikfast_res += 1
        # else:
        #     tgt_pos, tgt_rotmat = robot.fk(jnt_values = joint_angles)
        #     pred_pos, pred_rotmat = robot.fk(jnt_values = ik_solution)
        #     pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
        #     print(f'pos_err: {pos_err:.6f} mm, rot_err: {rot_err*180/np.pi:.2f} deg')
        #     ikfast_res += 0
        #     # print(f"IKFAST solution not close to the original joint angles: {repr(joint_angles)} vs {repr(ik_solution)}")
        # # ikfast_res += 1 if np.allclose(joint_angles, ik_solution, rtol=0.01) else 0
        pred_pose = arm.forward(ik_solution, rotation_type='matrix')
        pred_pos, pred_rotmat = pred_pose[:, 3], pred_pose[:, :3]
        pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
        pos_err_list.append(pos_err)
        rot_err_list.append(rot_err)

print('==========================================================')
print(f'current robot: {robot.__class__.__name__}')
print("IKFAST success rate %s of %s" % (ikfast_res, total))
print("percentage %.1f", ikfast_res/float(total)*100.)
print(f't mean: {np.mean(time_list) * 1000:.2f} ms')
print(f't std: {np.std(time_list) * 1000:.2f} ms')
print(f't min: {np.min(time_list) * 1000:.2f} ms')
print(f't max: {np.max(time_list) * 1000:.2f} ms')
print(f'pos err mean: {np.mean(pos_err_list):.6f} mm')
print(f'pos err std: {np.std(pos_err_list):.6f} mm')
print(f'pos err min: {np.min(pos_err_list):.2f} mm')
print(f'pos err max: {np.max(pos_err_list):.2f} mm')
print(f'rot err mean: {np.mean(rot_err_list)*180/np.pi:.2f} deg')
print(f'rot err std: {np.std(rot_err_list)*180/np.pi:.2f} deg')
print(f'rot err min: {np.min(rot_err_list)*180/np.pi:.2f} deg')
print(f'rot err max: {np.max(rot_err_list)*180/np.pi:.2f} deg')
print('==========================================================')