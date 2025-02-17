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
# from cleandiffuser.dataset.dataset_utils import loop_dataloader

'''preparations'''
seed = 0

# diffuser parameters
backbone = 'unet1d' # ['transformer', 'unet']
mode = 'inference'  # ['train', 'inference', 'loop_inference']
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
w_cg = 0.0001
temperature = 0.5
use_ema = False


if __name__ == '__main__':
    # --------------- Data Loading -----------------
    '''prepare the save path'''
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f'{TimeCode}_{backbone}_h{horizon}_steps{diffusion_steps}_{mode}'
    current_file_dir = os.path.dirname(__file__)
    save_path = os.path.join(current_file_dir, 'results', rootpath)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    '''load the dataset from npy file'''
    dataset = np.load(f'0000_test_programs/nn_ik/datasets/formal/dataset_1M_loc_rotquat.npz', allow_pickle=True)
    jnt, tgt_pos, tgt_rot = dataset['jnt_values'], dataset['tgt_pos'], dataset['tgt_rotmat']
    traj = np.concatenate((jnt, tgt_pos, tgt_rot), axis=1)
    action_dim = jnt.shape[1]
    obs_dim = tgt_pos.shape[1] + tgt_rot.shape[1]
    traj = torch.tensor(traj, dtype=torch.float32).to(device)
    traj_loader = DataLoader(traj, batch_size=train_batch_size, shuffle=True, drop_last=True)

    
    # --------------- Network Architecture -----------------
    nn_diffusion = JannerUNet1d(obs_dim + action_dim, model_dim=model_dim, emb_dim=model_dim, dim_mult=dim_mult,
                                timestep_emb_type="positional", attention=False, kernel_size=5)
    
    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)


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



    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")

    diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=diffusion_gradient_steps)

    if mode == 'train':
        wandb.init(project="ik_diffusion", name=rootpath)
        # ---------------------- Training ----------------------
        agent.train()
        n_gradient_step = 0
        log = {'avg_loss_diffusion': 0.}
        start_time = time.time()


        for batch in loop_dataloader(traj_loader):
            x = batch.unsqueeze(1).expand(train_batch_size, horizon, action_dim + obs_dim)
            current_loss = agent.update(x)['loss']
            log["avg_loss_diffusion"] += current_loss  # BaseDiffusionSDE.update
            diffusion_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % log_interval == 0:
                log['gradient_steps'] = n_gradient_step + 1
                log["avg_loss_diffusion"] /= log_interval
                wandb.log(
                    {'step': log['gradient_steps'],
                    'avg_training_loss': log['avg_loss_diffusion'],
                    'total_time': time.time() - start_time}, commit = True)
                print(log)
                log = {"avg_loss_diffusion": 0.}

            # ----------- Saving ------------
            if (n_gradient_step + 1) % save_interval == 0:
                agent.save(save_path + f"/diffusion_ckpt_{n_gradient_step + 1}.pt")
                agent.save(save_path + f"/diffusion_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= diffusion_gradient_steps:
                break
        wandb.finish()
    
    elif mode == 'inference':
        from wrs import wd, rm, mcm
        import wrs.robot_sim.robots.cobotta.cobotta as cbt
        
        base = wd.World(cam_pos=[1, 1.7, 1.7], lookat_pos=[0, 0, .3])
        mcm.mgm.gen_frame().attach_to(base)

        '''define robot'''
        # robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
        robot = cbt.Cobotta(pos=rm.vec(0,.0,.0), enable_cc=True)
        # robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), ik_solver='d' ,enable_cc=True)
        # robot = rs007l.RS007L(pos=rm.vec(0.1, .3, .5), enable_cc=True)

        '''load the model'''
        model_path = '0000_test_programs/nn_ik/results/0217_1334_unet1d_h4_steps20_traindiffusion_ckpt_latest.pt'
        agent.load(model_path)
        agent.eval()
        prior = torch.zeros((1, horizon, obs_dim + action_dim)).to(device)

        '''inference the ik solution'''
        success_num = 0
        time_list = []
        pos_err_list = []
        rot_err_list = []
        plot = False
        nupdate = 1 if plot else 1000

        with torch.no_grad():
            for i in tqdm(range(nupdate)):
                
                '''provide the prior condition information'''
                jnt_values = robot.rand_conf()
                tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
                tgt_rotq = rm.rotmat_to_quaternion(tgt_rotmat)
                tgt = torch.tensor(np.concatenate((tgt_pos.flatten(), tgt_rotq.flatten())), dtype=torch.float32).to(device)
                tic = time.time()
                prior[0, :, action_dim:] = tgt

                trajectory, log = agent.sample(prior, solver=solver, n_samples = 1,
                                               sample_steps=sampling_steps,
                                               use_ema=use_ema, w_cg=w_cg, temperature=temperature)
            
                result = trajectory.mean(dim=1).squeeze(0)
                toc = time.time()
                time_list.append(toc-tic)

                if result is not None:
                    success_num += 1

                    # calculate the pos error and rot error
                    pred_pos, pred_rotmat = robot.fk(jnt_values=result)
                    pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
                    pos_err_list.append(pos_err)
                    rot_err_list.append(rot_err)

                if plot:
                    mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
                    robot.goto_given_conf(jnt_values=result)
                    arm_mesh = robot.gen_meshmodel(alpha=0.25, rgb=[0,0,1])
                    arm_mesh.attach_to(base)
                    base.run()
                
        print('==========================================================')
        print(f'Average time: {np.mean(time_list) * 1000:.2f} ms')
        print(f'success rate: {success_num / nupdate * 100:.2f}%')
        print(f'Average position error: {np.mean(pos_err_list)}')
        print(f'Average rotation error: {np.mean(rot_err_list)*180/np.pi}')
        print('==========================================================')

    else:
        raise ValueError(f"Not implemented mode: {mode}")
    

    