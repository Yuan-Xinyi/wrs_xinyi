import os
import sys
 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from tqdm import tqdm
import wandb
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''cleandiffuser imports'''
from datetime import datetime
from cleandiffuser.diffusion.diffusionsde import BaseDiffusionSDE, DiscreteDiffusionSDE, BaseDiffusionSDE
from cleandiffuser.nn_condition import MultiImageObsCondition
from cleandiffuser.nn_diffusion import ChiTransformer, JannerUNet1d, ChiUNet1d
from cleandiffuser.utils import report_parameters
from cleandiffuser.diffusion.ddpm import DDPM
from cleandiffuser.dataset.dataset_utils import loop_dataloader
import json
# from cleandiffuser.dataset.dataset_utils import loop_dataloader

'''preparations'''
seed = 0

# diffuser parameters
backbone = 'unet1d' # ['transformer', 'unet']
mode = 'as_seed'  # ['train', 'inference', 'loop_inference']
train_batch_size = 64
test_batch_size = 1
solver = 'ddpm'
diffusion_steps = 20
predict_noise = False # [True, False]
obs_steps = 1
action_steps = 1
action_loss_weight = 10.0
dim_mult = [1, 4, 2]
model_dim = 32

# Training
device = 'cuda'
diffusion_gradient_steps = 1000000
log_interval = 100
save_interval = 50000
lr = 0.00001
horizon = 4

use_group_norm = True
ema_rate = 0.9999

# inference parameters
sampling_steps = 5
w_cg = 1.0 # 0.0001
temperature = 1.0
use_ema = False


from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.rs007l.rs007l as rs007l
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi
import wrs.robot_sim.robots.cobotta_pro1300.cobotta_pro1300 as cbtpro1300


base = wd.World(cam_pos=[1, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

'''define robot'''
# robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
# robot = cbt.Cobotta(pos=rm.vec(0,.0,.0), enable_cc=True)
# robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), ik_solver='d' ,enable_cc=True)
# robot = rs007l.RS007L(pos=rm.vec(0.1, .3, .5), enable_cc=True)
robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)

if __name__ == '__main__':

    '''inference the ik solution'''

    nupdate = 10000
    json_file = "diffusion_ik_as_seed.jsonl"
    infer_batch = 32
    robot_list = ['yumi','cbt', 'ur3', 'cbtpro1300']

    for robot in robot_list:
        if robot == 'yumi':
            robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
            predict_noise = True # [True, False]
            sampling_steps = 5
            model_path = '0000_test_programs/nn_ik/results/saved_model/yumi_diffusion_ik.pt'
        elif robot == 'cbt':
            robot = cbt.Cobotta(pos=rm.vec(0.0,.0,.0), enable_cc=True)
            predict_noise = False # [True, False]
            sampling_steps = 5
            model_path = '0000_test_programs/nn_ik/results/saved_model/cobotta_diffusion_ik.pt'
        elif robot == 'ur3':
            robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), enable_cc=True)
            predict_noise = True # [True, False]
            sampling_steps = 5
            model_path = '0000_test_programs/nn_ik/results/saved_model/ur3_diffusion_ik.pt'
        elif robot == 'cbtpro1300':
            robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)
            predict_noise = True # [True, False]
            sampling_steps = 10
            model_path = '0000_test_programs/nn_ik/results/0224_1614_cobotta_pro_1300_h4_steps20_train/diffusion_ckpt_latest.pt'
        else:
            print("Invalid robot name")
       
        action_dim = robot.n_dof
        obs_dim = 7

        # --------------- Network Architecture -----------------
        nn_diffusion = JannerUNet1d(obs_dim + action_dim, model_dim=model_dim, emb_dim=model_dim, dim_mult=dim_mult,
                                    timestep_emb_type="positional", attention=False, kernel_size=5)
        
        # ----------------- Masking -------------------
        fix_mask = torch.zeros((horizon, obs_dim + action_dim))
        fix_mask[0, action_dim:] = 1.
        loss_weight = torch.ones((horizon, obs_dim + action_dim))
        loss_weight[0, :action_dim] = action_loss_weight


        # --------------- Diffusion Model --------------------
        agent = DiscreteDiffusionSDE(nn_diffusion, 
                                    None, 
                                    fix_mask=fix_mask, 
                                    loss_weight=loss_weight, 
                                    ema_rate=ema_rate,
                                    device=device, 
                                    diffusion_steps=diffusion_steps, 
                                    predict_noise=predict_noise)

        diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=diffusion_gradient_steps)

        agent.load(model_path)
        agent.eval()
        prior = torch.zeros((1, horizon, obs_dim + action_dim)).to(device)

        with torch.no_grad():
            success_num = 0
            for i in tqdm(range(nupdate)):
                pos_err_list = []
                rot_err_list = []
                time_list = []
                
                '''provide the prior condition information'''
                jnt_values = robot.rand_conf()
                tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
                tgt_rotq = rm.rotmat_to_quaternion(tgt_rotmat)
                tgt = torch.tensor(np.concatenate((tgt_pos.flatten(), tgt_rotq.flatten())), dtype=torch.float32).to(device)
                
                tic = time.time()
                prior[0, :, action_dim:] = tgt
                
                trajectory, _ = agent.sample(prior, solver=solver, n_samples = infer_batch,
                                            sample_steps=sampling_steps,
                                            use_ema=use_ema, w_cg=w_cg, temperature=temperature)
                solutions = trajectory[:, 0, :action_dim]
                
                if infer_batch != 1:
                    error_array = np.zeros(solutions.shape[0])
                    solutions = solutions.cpu().numpy()
                    for id in range(solutions.shape[0]):
                        pos, rotmat = robot.fk(jnt_values = solutions[id])
                        rotq = rm.rotmat_to_quaternion(rotmat)
                        pos_rotq = torch.tensor(np.concatenate((pos.flatten(), rotq.flatten())), dtype=torch.float32).to(device)
                        error_array[id] = torch.norm(tgt - pos_rotq)
                    solution = solutions[np.argmin(error_array)]
                else:
                    solution = solutions[0].cpu().numpy()
                
                result = robot.ik(tgt_pos, tgt_rotmat, seed_jnt_values=solution, best_sol_num = 1)
                
                toc = time.time()
                time_list.append(toc-tic)
                if result is not None:
                    success_num += 1
                    pred_pos, pred_rotmat = robot.fk(jnt_values=result)
                    pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
                    pos_err_list.append(pos_err), rot_err_list.append(rot_err)
                
                    
                    
            data_entry = {
                "robot": robot.__class__.__name__,
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

    

    