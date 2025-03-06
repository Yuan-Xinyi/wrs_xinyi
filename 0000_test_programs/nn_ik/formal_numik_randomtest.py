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

'''define robot'''
# robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
# robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
# robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), ik_solver='d' ,enable_cc=True)
# robot = rs007l.RS007L(pos=rm.vec(0.1, .3, .5), enable_cc=True)
# robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)
# robot = cbtpro900.CobottaPro900Spine(pos=rm.vec(0.1, .3, .5), enable_cc=True)




nupdate = 10000
best_sol_num_list = [1] # [1,3,5,10,20]
# robot_list = ['yumi', 'cbt','ur3', 'cbtpro1300']
robot_list = ['cbt']
json_file = "metrics_robot_result.jsonl"

if __name__ == '__main__':
# while True:

    for robot in robot_list:
        if robot == 'yumi':
            robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
        elif robot == 'cbt':
            robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
        elif robot == 'ur3':
            robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), enable_cc=True)
        elif robot == 'cbtpro1300':
            robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)
        else:
            print("Invalid robot name")
        
        for best_sol_num in best_sol_num_list:
            success_num = 0
            time_list = []
            pos_err_list = []
            rot_err_list = []

            for i in tqdm(range(nupdate)):
                jnt_values = robot.rand_conf()
                # print("*" * 150 + "\n")
                # print('jnt', repr(jnt_values))
                tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
                # print('tgt_pos', tgt_pos)
                tic = time.time()
                result = robot.ik(tgt_pos, tgt_rotmat, best_sol_num = best_sol_num)
                toc = time.time()
                time_list.append(toc-tic)
                if result is not None:
                    success_num += 1

                    # calculate the pos error and rot error
                    pred_pos, pred_rotmat = robot.fk(jnt_values=result)
                    pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
                    pos_err_list.append(pos_err)
                    rot_err_list.append(rot_err)
                    
            # print('==========================================================')
            # print(f'current robot: {robot.__class__.__name__}')
            # print(f'best sol num: {best_sol_num}')
            # print(f'success rate: {success_num / nupdate * 100:.2f}%')
            # print(f't mean: {np.mean(time_list) * 1000:.2f} ms')
            # print(f't std: {np.std(time_list) * 1000:.2f} ms')
            # print(f't Coefficient of Variation: {np.std(time_list) / np.mean(time_list):.2f}')
            # print(f't 25 percentile: {np.percentile(time_list, 25) * 1000:.2f} ms')
            # print(f't 75 percentile: {np.percentile(time_list, 75) * 1000:.2f} ms')
            # print(f't Interquartile Range: {(np.percentile(time_list, 75) - np.percentile(time_list, 25)) * 1000:.2f} ms')
            # # print(f'Average position error: {np.mean(pos_err_list)}')
            # # print(f'Average rotation error: {np.mean(rot_err_list)*180/np.pi}')
            # print('==========================================================')
            # # plt.plot(range(nupdate), time_list)
            # # plt.show()

            data_entry = {
                "robot": robot.__class__.__name__,
                "best_solution_number": best_sol_num,
                "success_rate": f"{success_num / nupdate * 100:.2f}%",
                "time_statistics": {
                    "mean": f"{np.mean(time_list) * 1000:.2f} ms",
                    "std": f"{np.std(time_list) * 1000:.2f} ms",
                    "min": f"{np.min(time_list) * 1000:.2f} ms",
                    "max": f"{np.max(time_list) * 1000:.2f} ms",
                    "coefficient_of_variation": f"{np.std(time_list) / np.mean(time_list):.2f}",
                    "percentile_25": f"{np.percentile(time_list, 25) * 1000:.2f} ms",
                    "percentile_75": f"{np.percentile(time_list, 75) * 1000:.2f} ms",
                    "interquartile_range": f"{(np.percentile(time_list, 75) - np.percentile(time_list, 25)) * 1000:.2f} ms"
                }
            }

            with open(json_file, "a") as f:
                f.write(json.dumps(data_entry) + "\n")


            # arm_mesh = robot.gen_meshmodel(alpha=.3)
            # arm_mesh.attach_to(base)
            # tmp_arm_stick = robot.gen_stickmodel(toggle_flange_frame=True)
            # tmp_arm_stick.attach_to(base)
            # mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

            # if jnt_values is not None:
            #     print('jnt degree values: ',np.degrees(result))
            #     robot.goto_given_conf(jnt_values=result)
            # arm_mesh = robot.gen_meshmodel(alpha=.3)
            # arm_mesh.attach_to(base)
            # tmp_arm_stick = robot.gen_stickmodel(toggle_flange_frame=True)
            # tmp_arm_stick.attach_to(base)
            # base.run()

