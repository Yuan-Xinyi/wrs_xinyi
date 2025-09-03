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
from cleandiffuser.nn_diffusion import ChiTransformer, JannerUNet1d, ChiUNet1d
from cleandiffuser.nn_condition import IdentityCondition, MLPCondition
from cleandiffuser.utils import report_parameters
from cleandiffuser.diffusion.ddpm import DDPM
from cleandiffuser.dataset.dataset_utils import loop_dataloader
# from cleandiffuser.dataset.dataset_utils import loop_dataloader

'''preparations'''
seed = 0

# diffuser parameters
backbone = 'unet' # ['transformer', 'unet']
mode = 'inference'  # ['train', 'inference', 'loop_inference']
train_batch_size = 64
test_batch_size = 1
solver = 'ddpm'
diffusion_steps = 50
predict_noise = True # [True, False]
obs_steps = 1
action_steps = 1
action_loss_weight = 10.0
dim_mult = [1, 4, 2]
model_dim = 32
use_norm = True
resume_training = True

# Training
device = 'cuda'
diffusion_gradient_steps = 1000000
log_interval = 100
save_interval = 50000
lr = 0.00001
horizon = 4
use_group_norm = True
ema_rate = 0.9999
cond_type = 'identity'  # ['identity', 'mlp', None]
cond_code = 'cfg' # [none, cfg, cg, gg]

# inference parameters
sampling_steps = 25
w_cfg = 1.0 # 0.0001
temperature = 0.0
use_ema = False


from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.rs007l.rs007l as rs007l
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi
import wrs.robot_sim.robots.cobotta_pro1300.cobotta_pro1300 as cbtpro1300

def reset_log():
    log = {
        'avg_time': [],
        'pos_err_mean': [],
        'pos_err_min': [],
        'pos_err_max': [],
        'pos_err_q1': [],
        'pos_err_q3': [],
        'rot_err_mean': [],
        'rot_err_min': [],
        'rot_err_max': [],
        'rot_err_q1': [],
        'rot_err_q3': []
    }

    return log

def update_log(log, time_list, pos_err_list, rot_err_list):
    log['avg_time'].append(np.mean(time_list) * 1000)
    
    log['pos_err_mean'].append(np.mean(pos_err_list))
    log['pos_err_min'].append(np.min(pos_err_list))
    log['pos_err_max'].append(np.max(pos_err_list))
    log['pos_err_q1'].append(np.percentile(pos_err_list, 25))
    log['pos_err_q3'].append(np.percentile(pos_err_list, 75))
    
    log['rot_err_mean'].append(np.mean(rot_err_list) * 180 / np.pi)
    log['rot_err_min'].append(np.min(rot_err_list) * 180 / np.pi)
    log['rot_err_max'].append(np.max(rot_err_list) * 180 / np.pi)
    log['rot_err_q1'].append(np.percentile(rot_err_list, 25) * 180 / np.pi)
    log['rot_err_q3'].append(np.percentile(rot_err_list, 75) * 180 / np.pi)


class MinMaxScaler:
    def __init__(self, X):
        """
        初始化时直接根据输入 X 计算每个维度的 (min, max)
        """
        X = np.asarray(X, dtype=float)
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        self.ranges = np.stack([min_val, max_val], axis=1)  # shape (n_features, 2)

    def normalize(self, X):
        """
        将 X 归一化到 [-1, 1]
        """
        X = np.asarray(X, dtype=float)
        min_val = self.ranges[:, 0]
        max_val = self.ranges[:, 1]
        
        X_01 = (X - min_val) / (max_val - min_val + 1e-8)
        return X_01 * 2 - 1

    def unnormalize(self, X_norm):
        """
        将 [-1, 1] 空间的 X 恢复到原始范围
        """
        X_norm = np.asarray(X_norm, dtype=float)
        min_val = self.ranges[:, 0]
        max_val = self.ranges[:, 1]
        
        return (X_norm + 1) / 2 * (max_val - min_val) + min_val



base = wd.World(cam_pos=[1, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

'''define robot'''
current_rbt = 'cbt'  # ['cbt','cbtpro1300', 'ur3', 'yumi']
if current_rbt == 'yumi':
    robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
    abv_name = 'yumi'
elif current_rbt == 'cbt':
    robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
    abv_name = 'cbt'
elif current_rbt == 'ur3':
    robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), enable_cc=True)
    abv_name = 'ur3'
elif current_rbt == 'cbtpro1300':
    robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)
    abv_name = 'cbtpro1300'
else:
    print("Invalid robot name")

if __name__ == '__main__':

    '''load the dataset from npy file'''
    dataset = np.load(f'0000_test_programs/nn_ik/datasets/ikdiff/{current_rbt}_ik_dataset_rotquat.npz')
    print('current dataset: ', f'0000_test_programs/nn_ik/datasets/ikdiff/{current_rbt}_ik_dataset_rotquat.npz')
    jnt_values, pos_rot = dataset['jnt'], dataset['pos_rotq']
    traj = np.concatenate((jnt_values, pos_rot), axis=1)
    action_dim, obs_dim = jnt_values.shape[1], pos_rot.shape[1]

    if use_norm == True:
        scaler = MinMaxScaler(traj)
        traj = scaler.normalize(traj)
        print('*'*100)
        print("Using min-max normalization on the dataset")
        print('*'*100)
    traj = torch.tensor(traj, dtype=torch.float32).to(device)
    traj_loader = DataLoader(traj, batch_size=train_batch_size, shuffle=True, drop_last=True)

    
    # --------------- Network Architecture -----------------
    if cond_type == "identity":
        nn_condition = IdentityCondition(dropout=0.0).to(device)
        print("Using Identity Condition")
    else:
        nn_condition = None
        print("Using No Condition")
    nn_diffusion = ChiUNet1d(
        action_dim, obs_dim, obs_steps, model_dim=256, emb_dim=256, dim_mult=dim_mult,
        obs_as_global_cond=True, timestep_emb_type="positional").to(device)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"===================================================================================")

    # ----------------- Masking -------------------
    # fix_mask = torch.zeros((horizon, obs_dim + action_dim))
    # fix_mask[0, action_dim:] = 1.
    # loss_weight = torch.ones((horizon, obs_dim + action_dim))
    # loss_weight[0, :action_dim] = action_loss_weight

    # --------------- Diffusion Model --------------------
    x_max = torch.ones((1, horizon, action_dim), device=device) * +1.0
    x_min = torch.ones((1, horizon, action_dim), device=device) * -1.0
    agent = DDPM(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition, loss_weight=None,
        device=device, diffusion_steps=diffusion_steps, x_max=x_max, x_min=x_min,
        optim_params={"lr": 1e-4}, predict_noise=predict_noise)
    diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=diffusion_gradient_steps)

    if mode == 'train':
        # --------------- Data Loading -----------------
        '''prepare the save path'''
        if not resume_training:
            TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
            rootpath = f'{TimeCode}_{current_rbt}_{use_norm}_{cond_code}'
            current_file_dir = os.path.dirname(__file__)
            save_path = os.path.join(current_file_dir, 'results', rootpath)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            wandb.init(project="ikdpg", name=rootpath)
        else:
            model_path = '0000_test_programs/nn_ik/results/0902_2154_cbt_True_cfg/diffusion_ckpt_latest.pt'
            agent.load(model_path)

        # ---------------------- Training ----------------------
        agent.train()
        n_gradient_step = 0
        log = {'avg_loss_diffusion': 0.}
        start_time = time.time()


        for batch in loop_dataloader(traj_loader):
            jnt = batch[:, :action_dim]
            tgt = batch[:, action_dim:]
            x = jnt.unsqueeze(1).expand(train_batch_size, horizon, action_dim)
            if condition == "identity":
                condition = batch[:, action_dim:]
                print("Using Identity Condition")
            current_loss = agent.update(x, condition)['loss']
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

        '''load the model'''
        model_path = '0000_test_programs/nn_ik/results/0902_2154_cbt_True_cfg/diffusion_ckpt_latest.pt'
        agent.load(model_path)
        agent.eval()
        prior = torch.zeros((1, horizon, action_dim)).to(device)

        '''inference the ik solution'''
        log = reset_log()
        
        plot = True
        nupdate = 1 if plot else 2000
        trails = 1  # repeat the experiment for n trails
        infer_batch = 32

        for _ in range(trails):
            with torch.no_grad():
                for i in tqdm(range(nupdate)):
                    pos_err_list = []
                    rot_err_list = []
                    time_list = []
                    
                    '''provide the prior condition information'''
                    jnt_values = robot.rand_conf()
                    tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
                    tgt_rotq = rm.rotmat_to_quaternion(tgt_rotmat)
                    tgt = torch.tensor(np.concatenate((tgt_pos.flatten(), tgt_rotq.flatten())), dtype=torch.float32).to(device)
                    if use_norm == True:
                        query = np.concatenate((jnt_values.flatten(), tgt.cpu().flatten()))
                        normalized_query = scaler.normalize(query)

                    for _ in range(infer_batch):
                        tic = time.time()
                        if cond_type == "identity":
                            condition = torch.tensor(normalized_query[action_dim:], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

                        else:
                            condition = None
                        prior = torch.zeros((1, horizon, action_dim)).to(device)
                        with torch.no_grad():
                            trajectory, _ = agent.sample(prior, solver=solver, n_samples = 1,
                                                         x_max = x_max, x_min = x_min,
                                                        sample_steps=sampling_steps, condition_cfg=condition,
                                                        use_ema=use_ema, w_cfg=w_cfg, temperature=temperature) 
                        result = trajectory[0, 0, :]
                        jnt = (result.cpu() + 1) / 2 * (robot.jnt_ranges[:, 1] - robot.jnt_ranges[:, 0]) + robot.jnt_ranges[:, 0]
                        toc = time.time()
                        time_list.append(toc-tic)

                        pred_pos, pred_rotmat = robot.fk(jnt_values=jnt)
                        pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
                        # print(f'pos err: {pos_err:.2f} mm, rot err: {rot_err:.2f} degree')
                        pos_err_list.append(pos_err), rot_err_list.append(rot_err)

                        robot.goto_given_conf(jnt_values=jnt)
                        arm_mesh = robot.gen_meshmodel(alpha=0.1, rgb=[1,0,0])
                        arm_mesh.attach_to(base)
                    
                    update_log(log, time_list, pos_err_list, rot_err_list)
                    
            print('==========================================================')
            print('current robot: ', robot.name)
            print(f'Average time: {np.mean(log["avg_time"]):.2f} ms')
            print('-'*50)
            print(f'pos err mean: {np.mean(log["pos_err_mean"]):.2f} mm')
            print(f'pos err min: {np.mean(log["pos_err_min"]):.2f} mm')
            print(f'pos err max: {np.mean(log["pos_err_max"]):.2f} mm')
            print(f'pos err q1: {np.mean(log["pos_err_q1"]):.2f} mm')
            print(f'pos err q3: {np.mean(log["pos_err_q3"]):.2f} mm')
            print('-'*50)
            print(f'rot err mean: {np.mean(log["rot_err_mean"]):.2f} degree')
            print(f'rot err min: {np.mean(log["rot_err_min"]):.2f} degree')
            print(f'rot err max: {np.mean(log["rot_err_max"]):.2f} degree')
            print(f'rot err q1: {np.mean(log["rot_err_q1"]):.2f} degree')
            print(f'rot err q3: {np.mean(log["rot_err_q3"]):.2f} degree')
            print('==========================================================')



            # if plot:
            #     for i in range(result.shape[0]):
            #         print(result.shape[0])
            #         mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
            #         robot.goto_given_conf(jnt_values=result[i])
            #         arm_mesh = robot.gen_meshmodel(alpha=0.1, rgb=[0,0,1])
            #         arm_mesh.attach_to(base)

            #     robot.goto_given_conf(jnt_values=jnt_values)
            #     arm_mesh = robot.gen_meshmodel(alpha=0.25, rgb=[1,0,0])
            #     arm_mesh.attach_to(base)

            #     base.run()
            robot.goto_given_conf(jnt_values=jnt_values)
            arm_mesh = robot.gen_meshmodel(alpha=0.25, rgb=[0,1,0])
            arm_mesh.attach_to(base)

            base.run()

    else:
        raise ValueError(f"Not implemented mode: {mode}")
    

    