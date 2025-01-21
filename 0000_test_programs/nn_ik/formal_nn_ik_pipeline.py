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

from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import wrs.robot_sim.manipulators.rs007l.rs007l as rs007l
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.manipulators.ur3e.ur3e as ur3e
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.yumi.yumi_single_arm as yumi

'''define the world'''
base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

'''define robot'''
output_as_seed = True
# robot = yumi.YumiSglArm(pos=rm.vec(0.1, .3, .5),enable_cc=True)
# robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
# robot = ur3.UR3(pos=rm.vec(0.1, .3, .5), ik_solver='d' ,enable_cc=True)
robot = rs007l.RS007L(pos=rm.vec(0.1, .3, .5), enable_cc=True)

'''define the initail paras'''
seed = 42
mode = 'random_ik_test' # ['train' or 'random_ik_test']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train paras
train_epoch = 2000
train_batch = 64
lr = 0.001
save_intervel = 200

# val and test paras
val_batch = train_batch
test_batch = 1


class IKMLPNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(IKMLPNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),  # input: tgt_pos(3) + tgt_rotmat(9)
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),    # output: jnt_values(6)
        )

    def forward(self, x):
        return self.network(x)

def dataset_generation_rotmat(size, file_name, loc_coord=False, rot='rotmat'):
    jnt_values_list = []
    pos_list = []
    rotmat_list = []

    for _ in tqdm(range(size)):
        jnt_values = robot.rand_conf()
        jnt_values_list.append(jnt_values)

        tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)  # world corrdinate
        if loc_coord:
            tgt_pos, tgt_rotmat = rm.real_pose(robot.pos, robot.rotmat, tgt_pos, tgt_rotmat)  # local corrdinate
        if rot == 'rotv':
            tgt_rotmat = rm.rotmat_to_wvec(tgt_rotmat)
        elif rot == 'quaternion':
            tgt_rotmat = rm.rotmat_to_quaternion(tgt_rotmat)

        pos_list.append(tgt_pos.flatten())
        rotmat_list.append(tgt_rotmat.flatten())
    
    jnt_values = np.array(jnt_values_list)
    pos = np.array(pos_list)
    rotmat = np.array(rotmat_list)

    np.savez(file_name, jnt_values=jnt_values, tgt_pos=pos, tgt_rotmat=rotmat)
    print('Dataset generated successfully!')


if __name__ == '__main__':
    # # dataset generation
    # dataset_generation_rotmat(10000000,'0000_test_programs/nn_ik/dataset_10M_loc_rotquat.npz', loc_coord=False, rot='quaternion')
    # exit()

    # dataset loading
    dataset = np.load(f'0000_test_programs/nn_ik/datasets/formal/{robot.name}_ik_dataset.npz')
    jnt_values, pos_rot = dataset['jnt'], dataset['pos_rotv']

    # dataset pre-processing
    gth_jnt_values = torch.tensor(jnt_values, dtype=torch.float32).to(device)
    pos_rot = torch.tensor(pos_rot, dtype=torch.float32).to(device)

    # train-val-test split
    train_input, input, train_label, label = train_test_split(pos_rot, gth_jnt_values, test_size=0.3, random_state=seed)
    val_input, test_input, val_label, test_label = train_test_split(input, label, test_size=0.5, random_state=seed)

    # data loader
    train_dataset = TensorDataset(train_input, train_label)
    val_dataset = TensorDataset(val_input, val_label)
    test_dataset = TensorDataset(test_input, test_label)

    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=True, drop_last=False)


    # Initialize the model
    model = IKMLPNet(input_dim=pos_rot.shape[1], output_dim=gth_jnt_values.shape[1]).to(device).float()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if mode == 'train':
        wandb.init(project="ik", name=robot.name)
        # prepare the save path
        TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
        rootpath = f'formal_{TimeCode}_{robot.name}_MLP_rotv'
        save_path = f'0000_test_programs/nn_ik/results/{rootpath}/'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        
        for epoch in tqdm(range(train_epoch)):
            model.train()
            train_loss = 0
            for pos_rotmat, gth_jnt in train_loader:
                optimizer.zero_grad()
                pred_jnt = model(pos_rotmat)
                loss = criterion(pred_jnt, gth_jnt)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            wandb.log({"train_loss": train_loss})
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for pos_rotmat, gth_jnt in val_loader:
                    pred_jnt = model(pos_rotmat)
                    loss = criterion(pred_jnt, gth_jnt)
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
        if robot.name == 'cobotta':
            model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/formal_0119_1842_cobotta_MLP_rotv/model400'))
        elif robot.name == 'sglarm_yumi':
            model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/formal_0119_1842_sglarm_yumi_MLP_rotv/model400'))
        elif robot.name == 'ur3':
            model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/formal_0119_1842_ur3_MLP_rotv/model400'))
        elif robot.name == 'khi_rs007l':
            model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/formal_0119_1843_khi_rs007l_MLP_rotv/model400'))
        else:
            raise ValueError('Invalid robot name!')
        
        model.eval()
        success_num = 0
        time_list = []
        pos_err_list = []
        rot_err_list = []

        nupdate = 10000
        with torch.no_grad():
            for _ in tqdm(range(nupdate)):
                jnt_values = robot.rand_conf()
                tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)  # fk --> pos(3), rotmat(3x3)
                # rot format conversion
                rotv = rm.rotmat_to_wvec(tgt_rotmat)
            
                pos_rot = torch.tensor(np.concatenate((tgt_pos.flatten(), rotv.flatten()), axis=0), dtype=torch.float32).to(device).unsqueeze(0)
                if output_as_seed:
                    tic = time.time()
                    nn_pred_jnt_values = model(pos_rot).cpu().numpy()[0]
                    result = robot.ik(tgt_pos, tgt_rotmat, seed_jnt_values=nn_pred_jnt_values, best_sol_num = 1)
                    toc = time.time()
                    time_list.append(toc-tic)
                    if result is not None:
                        success_num += 1
                else:
                        # calculate the pos error and rot error
                    tic = time.time()
                    nn_pred_jnt_values = model(pos_rot).cpu().numpy()[0]
                    toc = time.time()
                    time_list.append(toc-tic)
                    success_num += 1
                    result = nn_pred_jnt_values
                
                if result is not None:
                    pred_pos, pred_rotmat = robot.fk(jnt_values=result)
                    pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
                    pos_err_list.append(pos_err)
                    rot_err_list.append(rot_err)
                        
        print('=============================')
        print(f'current robot: {robot.name}')
        print(f'Average time: {np.mean(time_list) * 1000:.2f} ms')
        print(f'success rate: {success_num / nupdate * 100:.2f}%')
        print(f'Average position error: {np.mean(pos_err_list)}')
        print(f'Average rotation error: {np.mean(rot_err_list)*180/np.pi}')
        print('=============================')