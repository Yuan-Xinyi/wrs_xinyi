import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
import wandb
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import time
import torch.nn.functional as F
import json

from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.rs007l.rs007l as rs007l
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.manipulators.ur3e.ur3e as ur3e
import wrs.basis.robot_math as rm
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

'''define the world'''
base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

'''define robot'''
output_as_seed = False
# robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
# robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)  # rotv
# robot = ur3.UR3(pos=rm.vec(0.1, .3, .5),enable_cc=True)
robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)


'''define the initail paras'''
nupdate = 100
trail_num = 100
mode = 'as_seed' # ['train' random_ik_test, as_seed]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train paras
train_epoch = 5000
train_batch = 64
lr = 0.001
save_intervel = 500

# val and test paras
val_batch = train_batch
test_batch = 1


class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_mixtures):
        super(MDN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_mixtures = num_mixtures

        # A simple MLP feature extractor
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        # MDN output layers
        # pi: mixture coefficients of size M
        # mu: means of size M*D
        # sigma: standard deviations of size M*D
        self.fc_pi = nn.Linear(128, num_mixtures) 
        self.fc_mu = nn.Linear(128, num_mixtures * output_dim)
        self.fc_sigma = nn.Linear(128, num_mixtures * output_dim)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        
        pi = self.fc_pi(h)
        mu = self.fc_mu(h)
        sigma = self.fc_sigma(h)

        # Apply softmax to pi to ensure they sum to 1 and are in range (0,1)
        pi = F.softmax(pi, dim=-1)
        # Use exp to ensure sigma is always positive
        sigma = torch.exp(sigma)

        # Reshape mu and sigma to [batch_size, M, D]
        mu = mu.view(-1, self.num_mixtures, self.output_dim)
        sigma = sigma.view(-1, self.num_mixtures, self.output_dim)

        return pi, mu, sigma

def mdn_loss(pi, mu, sigma, y):

    y = y.unsqueeze(1)  # [batch_size, 1, D]
    normal_const = 1.0 / (torch.sqrt(torch.tensor(2.0 * 3.141592653589793)) * sigma)
    exponent = torch.exp(-0.5 * ((y - mu) / sigma)**2)
    gauss = torch.prod(normal_const * exponent, dim=2)  

    # Weight each Gaussian by its mixture coefficient
    weighted_gauss = pi * gauss
    # To avoid log(0), add a small epsilon
    epsilon = 1e-9
    # NLL = -log(sum(pi_k * N(y|mu_k, sigma_k)))
    loss = -torch.log(weighted_gauss.sum(dim=1) + epsilon)
    return loss.mean()


if __name__ == '__main__':
    # # dataset generation
    # dataset_generation_rotmat(10000000,'0000_test_programs/nn_ik/dataset_10M_loc_rotquat.npz', loc_coord=False, rot='quaternion')
    # exit()
    
    # dataset loading
    # dataset = np.load(f'0000_test_programs/nn_ik/datasets/formal/{robot.name}_ik_dataset.npz')
    dataset = np.load(f'0000_test_programs/nn_ik/datasets/formal/{robot.name}_ik_dataset_rotquat.npz')
    jnt_values, pos_rot = dataset['jnt'], dataset['pos_rotv']

    input_dim = pos_rot.shape[1]    # e.g., 3D position + 3D rotation vector
    output_dim = jnt_values.shape[1]   # e.g., a 6-DOF robot arm
    num_mixtures = 120 # number of mixture components

    # dataset pre-processing
    gth_jnt_values = torch.tensor(jnt_values, dtype=torch.float32).to(device)
    pos_rot = torch.tensor(pos_rot, dtype=torch.float32).to(device)

    # train-val-test split
    train_input, input, train_label, label = train_test_split(pos_rot, gth_jnt_values, test_size=0.3)
    val_input, test_input, val_label, test_label = train_test_split(input, label, test_size=0.5)

    # data loader
    train_dataset = TensorDataset(train_input, train_label)
    val_dataset = TensorDataset(val_input, val_label)
    test_dataset = TensorDataset(test_input, test_label)

    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=True, drop_last=False)


    # Initialize the model
    model = MDN(input_dim, output_dim, num_mixtures).to(device).float()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if mode == 'train':
        # prepare the save path
        wandb_name = f'MDN_{robot.name}_{num_mixtures}'
        wandb.init(project="ik", name=wandb_name)
        TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
        rootpath = f'{TimeCode}_{robot.name}_MDN_rotquat_{num_mixtures}'
        save_path = f'0000_test_programs/nn_ik/results/{rootpath}/'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        
        for epoch in tqdm(range(train_epoch)):
            model.train()
            train_loss = 0
            for pos_rot, gth_jnt in train_loader:
                optimizer.zero_grad()
                pi, mu, sigma = model(pos_rot)
                loss = mdn_loss(pi, mu, sigma, gth_jnt)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            wandb.log({"train_loss": train_loss})
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for pos_rot, gth_jnt in val_loader:
                    pi, mu, sigma = model(pos_rot)
                    loss = mdn_loss(pi, mu, sigma, gth_jnt)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            wandb.log({"val_loss": val_loss})
            print(f"Epoch {epoch + 1}/{train_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if epoch % save_intervel == 0:
                PATH = f"{save_path}/model{epoch}"
                print(f"model been saved in: {PATH}")
                torch.save(model.state_dict(), PATH)

        wandb.finish()

    elif mode == 'random_ik_test':

        model.eval()
        output_as_seed = False        
        nupdate = 2000
        num_mixtures = 120
        robot_list = ['sglarm_yumi','ur3']

        for robot_ in robot_list:
            if robot == 'yumi':
                robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
                model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/formal_0119_1911_sglarm_yumi_MDN_rotv/model200'))
            elif robot == 'cbt':
                robot = cbt.Cobotta(pos=rm.vec(0.0,.0,.0), enable_cc=True)
                predict_noise = False # [True, False]
                sampling_steps = 5
                model_path = '0000_test_programs/nn_ik/results/saved_model/cobotta_diffusion_ik.pt'
            elif robot == 'ur3':
                robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), enable_cc=True)
                model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/formal_0119_1912_ur3_MDN_rotv/model400'))
            elif robot == 'cbtpro1300':
                robot = cbtpro1300.CobottaPro1300WithRobotiq140(pos=rm.vec(0.1, .3, .5), enable_cc=True)
                predict_noise = True # [True, False]
                sampling_steps = 10
                model_path = '0000_test_programs/nn_ik/results/0224_1614_cobotta_pro_1300_h4_steps20_train/diffusion_ckpt_latest.pt'
            else:
                print("Invalid robot name")

            log = reset_log()
            infer_batch = 1
        
            with torch.no_grad():
                for i in tqdm(range(nupdate)):
                    pos_err_list = []
                    rot_err_list = []
                    time_list = []
                    
                    '''provide the prior condition information'''
                    jnt_values = robot.rand_conf()
                    tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
                    rotv = rm.rotmat_to_wvec(tgt_rotmat)
                    pos_rot = torch.tensor(np.concatenate((tgt_pos.flatten(), rotv.flatten()), axis=0), dtype=torch.float32).to(device).unsqueeze(0)
                    
                    for _ in range(infer_batch):
                        tic = time.time()
                        pi, mu, sigma  = model(pos_rot)
                        pi, mu, sigma = pi.squeeze(0), mu.squeeze(0), sigma.squeeze(0)
                        max_pi_idx = torch.argmax(pi).item()
                        result = mu[max_pi_idx].cpu().numpy()
                        toc = time.time()
                        time_list.append(toc-tic)

                        pred_pos, pred_rotmat = robot.fk(jnt_values=result)
                        pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
                        # print(f'pos err: {pos_err:.2f} mm, rot err: {rot_err:.2f} degree')
                        pos_err_list.append(pos_err), rot_err_list.append(rot_err)

                        # robot.goto_given_conf(jnt_values=result)
                        # arm_mesh = robot.gen_meshmodel(alpha=0.1, rgb=[0,0,1])
                        # arm_mesh.attach_to(base)
                    
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

    elif mode == 'as_seed':
        num_mixtures = 120
        robot_list = ['sglarm_yumi','ur3']
        if num_mixtures == 5:
            if robot.name == 'cobotta':
                model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/formal_0119_1911_cobotta_MDN_rotv/model200'))
            elif robot.name == 'sglarm_yumi':
                model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/formal_0119_1911_sglarm_yumi_MDN_rotv/model200'))
            elif robot.name == 'ur3':
                model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/formal_0119_1912_ur3_MDN_rotv/model400'))
            elif robot.name == 'khi_rs007l':
                model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/formal_0119_1912_khi_rs007l_MDN_rotv/model300'))
            else:
                raise ValueError('Invalid robot name!')
        elif num_mixtures == 120:
            if robot.name == 'cobotta':
                model_path = '0000_test_programs/nn_ik/results/0227_1028_cobotta_MDN_rotquat_120/model1000'  
                model.load_state_dict(torch.load(model_path))
            elif robot.name == 'sglarm_yumi':
                model_path = '0000_test_programs/nn_ik/results/saved_model/formal_0119_2106_sglarm_yumi_MDN_rotv_120_model2000'
                model.load_state_dict(torch.load(model_path))
            elif robot.name == 'ur3':
                model_path = '0000_test_programs/nn_ik/results/saved_model/formal_0119_2107_ur3_MDN_rotv_120_model3500'
                model.load_state_dict(torch.load(model_path))
            elif robot.name == 'cobotta_pro_1300':
                # dataset: rotquat
                # according to the dataset, the rotmat is in the form of quaternion
                model_path = '0000_test_programs/nn_ik/results/0227_1030_cobotta_pro_1300_MDN_rotquat_120/model1000'
                model.load_state_dict(torch.load(model_path))
            else:
                raise ValueError('Invalid robot name!')
        else:
            raise ValueError('Invalid num_mixtures!')
        
        model.eval()
        output_as_seed = False        
        json_file = 'MDN_result.json'
        plot = False
        nupdate = 1 if plot else 2000

        with torch.no_grad():
            log = reset_log()
            success_num = 0
            time_list = []
            pos_err_list = []
            rot_err_list = []
            for _ in tqdm(range(nupdate)):
                jnt_values = robot.rand_conf()
                # print(repr(jnt_values))
                # jnt_values = [-2.13804772, -1.04085843,  0.11936408, -0.40867239,  2.7227841 , 2.11849168, -1.40243529]
                tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)  # fk --> pos(3), rotmat(3x3)
                # rot format conversion
                # rotv = rm.rotmat_to_wvec(tgt_rotmat)
                rotv = rm.rotmat_to_quaternion(tgt_rotmat)
            
                pos_rot = torch.tensor(np.concatenate((tgt_pos.flatten(), rotv.flatten()), axis=0), dtype=torch.float32).to(device).unsqueeze(0)
                if output_as_seed:
                    tic = time.time()
                    pi, mu, sigma  = model(pos_rot)
                    pi, mu, sigma = pi.squeeze(0), mu.squeeze(0), sigma.squeeze(0)
                    max_pi_idx = torch.argmax(pi).item()
                    seed_jnt_mdn = mu[max_pi_idx].cpu().numpy()

                    result = robot.ik(tgt_pos, tgt_rotmat, seed_jnt_values=seed_jnt_mdn, best_sol_num = 1)
                    toc = time.time()
                    time_list.append(toc-tic)
                    if result is not None:
                        success_num += 1
                else:
                    tic = time.time()
                    pi, mu, sigma  = model(pos_rot)
                    pi, mu, sigma = pi.squeeze(0), mu.squeeze(0), sigma.squeeze(0)
                    max_pi_idx = torch.argmax(pi).item()
                    seed_jnt_mdn = mu[max_pi_idx].cpu().numpy()

                    toc = time.time()
                    time_list.append(toc-tic)
                    success_num += 1
                    result = seed_jnt_mdn

                    if plot:
                        mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
                        robot.goto_given_conf(jnt_values=result)
                        arm_mesh = robot.gen_meshmodel(alpha=0.1, rgb=[0,0,1])
                        arm_mesh.attach_to(base)

                        robot.goto_given_conf(jnt_values=jnt_values)
                        arm_mesh = robot.gen_meshmodel(alpha=0.25, rgb=[1,0,0])
                        arm_mesh.attach_to(base)

                        base.run()
                
                if result is not None:
                    pred_pos, pred_rotmat = robot.fk(jnt_values=result)
                    pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
                    # print(f'pos err: {pos_err:.2f} mm, rot err: {rot_err:.2f} degree')
                    pos_err_list.append(pos_err), rot_err_list.append(rot_err)

            update_log(log, time_list, pos_err_list, rot_err_list)
        
        print('==========================================================')
        print('current robot: ', robot.name)
        print(model_path)
        print('success rate: ', success_num / nupdate * 100)
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


                    
