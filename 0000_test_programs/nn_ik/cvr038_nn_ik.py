from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
nupdate = 100
trail_num = 3
seed = 42
mode = 'train' # ['train' or 'test']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train paras
train_epoch = 10000
train_batch = 64
lr = 0.001
save_intervel = 500
wandb.init(project="ik")
dataset_size = '1M'
backbone = 'IKMLPScaleNet'  # [IKMLPNet, IKMLPScaleNet]

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
            nn.Linear(64, output_dim)    # output: jnt_values(6)
        )
    
    def forward(self, x):
        return self.network(x)

class IKMLPScaleNet(IKMLPNet):
    def __init__(self, input_dim, output_dim, robot):
        super(IKMLPNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),  # input: tgt_pos(3) + tgt_rotmat(9)
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),    # output: jnt_values(6)
            nn.Tanh()
        )
        self.robot = robot
        self.jnt_ranges = torch.tensor(robot.jnt_ranges, dtype=torch.float32).to(device)
    
    def forward(self, x):
        output = self.network(x)  # output: (batch_size, 6)
        scaled_output = output * (self.jnt_ranges[:, 1] - self.jnt_ranges[:, 0]) + self.jnt_ranges[:, 0]
        return scaled_output

def ik_test():
    for _ in range(trail_num):
        success_rate = 0
        time_list = []
        for i in range(nupdate):
            jnt_values = robot.rand_conf()
            tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)  # fk --> pos(3), rotmat(3x3)
            tic = time.time()
            result = robot.ik(tgt_pos, tgt_rotmat)
            toc = time.time()
            time_list.append(toc-tic)
            if result is not None:
                success_rate += 1

        print('Success Rate: ',success_rate/nupdate)
        plt.plot(range(nupdate), time_list)
        plt.show()

def dataset_generation(size, file_name):
    jnt_values_list = []
    pos_list = []
    rotmat_list = []

    for _ in tqdm(range(size)):
        jnt_values = robot.rand_conf()
        jnt_values_list.append(jnt_values)

        tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
        pos_list.append(tgt_pos.flatten())
        rotmat_list.append(tgt_rotmat.flatten())
    
    jnt_values = np.array(jnt_values_list)
    pos = np.array(pos_list)
    rotmat = np.array(rotmat_list)

    np.savez(file_name, jnt_values=jnt_values, tgt_pos=pos, tgt_rotmat=rotmat)
    print('Dataset generated successfully!')


if __name__ == '__main__':
    # # dataset generation
    # dataset_generation(10000,'0000_test_programs/nn_ik/dataset_10k.npz')
    # exit()

    # dataset loading
    dataset = np.load(f'0000_test_programs/nn_ik/datasets/dataset_{dataset_size}.npz')
    jnt_values, tgt_pos, tgt_rotmat = dataset['jnt_values'], dataset['tgt_pos'], dataset['tgt_rotmat']

    # dataset pre-processing
    gth_jnt_values = torch.tensor(jnt_values, dtype=torch.float32).to(device)
    pos_rotmat = torch.tensor(np.concatenate((tgt_pos, tgt_rotmat), axis=1), dtype=torch.float32).to(device)  # pos(3), rotmat(3x3)

    # train-val-test split
    train_input, input, train_label, label = train_test_split(pos_rotmat, gth_jnt_values, test_size=0.3, random_state=seed)
    val_input, test_input, val_label, test_label = train_test_split(input, label, test_size=0.5, random_state=seed)

    # data loader
    train_dataset = TensorDataset(train_input, train_label)
    val_dataset = TensorDataset(val_input, val_label)
    test_dataset = TensorDataset(test_input, test_label)

    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=True, drop_last=False)


    # Initialize the model
    if backbone == 'IKMLPNet':
        model = IKMLPNet(input_dim=pos_rotmat.shape[1], output_dim=gth_jnt_values.shape[1]).to(device).float()
    elif backbone == 'IKMLPScaleNet':
        model = IKMLPScaleNet(input_dim=pos_rotmat.shape[1], output_dim=gth_jnt_values.shape[1], robot=robot).to(device).float()
    else:
        raise ValueError('Invalid backbone name!')
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if mode == 'train':
        # prepare the save path
        TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
        rootpath = f'{TimeCode}_IKMLPNet_{dataset_size}dataset'
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

    elif mode == 'test':
        model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/1202_2109_IKMLPNet_1M_dataset/model900'))
        model.eval()
        test_loss = 0
        time_list = []
        with torch.no_grad():
            for pos_rotmat, gth_jnt in test_loader:
                tic = time.time()
                pred_jnt = model(pos_rotmat)
                toc = time.time()
                loss = criterion(pred_jnt, gth_jnt)
                time_list.append(toc-tic)
                test_loss += loss.item()
                print('current time: ',toc-tic) 
                # print(f'pred_jnt: {pred_jnt}\ngth_jnt: {gth_jnt}\nloss: ',loss)
        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss:.4f}")