from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.rs007l.rs007l as rs007l
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.manipulators.ur3e.ur3e as ur3e
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi
import wrs.robot_sim.robots.cobotta_pro1300.cobotta_pro1300 as cbtpro1300
import wrs.robot_sim.robots.cobotta_pro900.cobotta_pro900_spine as cbtpro900

import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

nupdate = 10000
# best_sol_num_list = [3,5]
best_sol_num_list = [0, 1, 3, 5, 10, 20]
best_sol_num_list = [1]
# para_list = [10,20,50,80,100,150,200,300]
para_list = [2000]
# best_sol_num_list = np.arange(7, 21, 1).tolist() # [1,2,3,...,30]
robot_list = ['cbt','cbtpro1300', 'ur3', 'yumi']
json_file = "metrics_robot_result_1029.jsonl"

if __name__ == '__main__':
# while True:


    for robot in robot_list:
        if robot == 'yumi':
            robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
        elif robot == 'cbt':
            robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
        elif robot == 'ur3':
            robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), enable_cc=True)
            # robot = ur3e.UR3e(pos=rm.vec(0.1, .3, .5), enable_cc=True)
        elif robot == 'cbtpro1300':
            robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)
        else:
            print("Invalid robot name")
        best_sol_num = 1
        for para in para_list:            
            success_num = 0
            time_list = []
            pos_err_list = []
            rot_err_list = []

            for i in tqdm(range(nupdate)):
                jnt_values = robot.rand_conf()
                tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
                tic = time.time()
                result = robot.ik(tgt_pos, tgt_rotmat, best_sol_num = best_sol_num, para = para)
                toc = time.time()
                time_list.append(toc-tic)
                if result is not None:
                    success_num += 1

                    # calculate the pos error and rot error
                    pred_pos, pred_rotmat = robot.fk(jnt_values=result)
                    pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
                    pos_err_list.append(pos_err)
                    rot_err_list.append(rot_err)

            data_entry = {
                "robot": robot.__class__.__name__,
                "best_solution_number": best_sol_num,
                "theta": para,
                "success_rate": f"{success_num / nupdate * 100:.2f}%",

                "t_mean": f"{np.mean(time_list) * 1000:.2f} ms",
                "t_std": f"{np.std(time_list) * 1000:.2f} ms",
                "t_min": f"{np.min(time_list) * 1000:.2f} ms",
                "t_max": f"{np.max(time_list) * 1000:.2f} ms",
                
                'pos_err_mean': f"{np.mean(pos_err_list):.2f} mm",
                'pos_err_std': f"{np.std(pos_err_list):.2f} mm",
                'pos_err_min': f"{np.min(pos_err_list):.2f} mm",
                'pos_err_q1': f"{np.percentile(pos_err_list, 25):.2f} mm",
                'pos_err_q3': f"{np.percentile(pos_err_list, 75):.2f} mm",
                'pos_err_max': f"{np.max(pos_err_list):.2f} mm",

                'rot_err_mean': f"{np.mean(rot_err_list)*180/np.pi:.2f} deg",
                'rot_err_std': f"{np.std(rot_err_list)*180/np.pi:.2f} deg",
                'rot_err_min': f"{np.min(rot_err_list)*180/np.pi:.2f} deg",
                'rot_err_q1': f"{np.percentile(rot_err_list, 25)*180/np.pi:.2f} deg",
                'rot_err_q3': f"{np.percentile(rot_err_list, 75)*180/np.pi:.2f} deg",
                'rot_err_max': f"{np.max(rot_err_list)*180/np.pi:.2f} deg"
            }

            with open(json_file, "a") as f:
                f.write(json.dumps(data_entry) + "\n")
