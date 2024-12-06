from wrs import wd, rm, mcm
import wrs.robot_sim.robots.cobotta.cobotta as cbt
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
from wrs import rm, mcm, wd

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

# robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
robot = cbt.Cobotta(pos=rm.vec(0.0,.0,.0), enable_cc=True)
nupdate = 100
trail_num = 100
seed = 42
mode = 'ik_test' # ['train' or 'test','inference','ik_test']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train paras
train_epoch = 10000
train_batch = 64
lr = 0.001
save_intervel = 1000
wandb.init(project="ik")
dataset_size = '1M' # [1M, 1M_loc_rotmat, 1M_loc_rotv, 1M_loc_rotquat]
backbone = 'IKMLPNet'  # [IKMLPNet, IKMLPScaleNet, IKLSTMNet]

# val and test paras
val_batch = train_batch
test_batch = 1

class IKLSTMNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, robot):
        super(IKLSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.robot = robot
        self.jnt_ranges = torch.tensor(robot.jnt_ranges, dtype=torch.float32).to(device)
        nn.init.normal_(self.linear.weight, 0, .02)
        nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        enc_x, (h_n, c_n) = self.lstm(x)
        # enc_x shape: (batch_size, seq_len, hidden_dim)
        # h_n shape: (num_layers, batch_size, hidden_dim)

        # Take the hidden state from the last layer
        h_n = h_n[-1]  # shape: (batch_size, hidden_dim)

        out = self.linear(h_n)  # shape: (batch_size, output_dim)
        scaled_output = out * (self.jnt_ranges[:, 1] - self.jnt_ranges[:, 0]) + self.jnt_ranges[:, 0]
        return scaled_output

class IKMLPNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(IKMLPNet, self).__init__()
        self.network = nn.Sequential(
            # nn.Linear(input_dim, 256),  # Increased neurons
            # nn.ReLU(),
            # nn.Linear(256, 512),       # New hidden layer with more neurons
            # nn.ReLU(),
            # nn.Linear(512, 256),       # Another hidden layer
            # nn.ReLU(),
            # nn.Linear(256, 128),       # Yet another hidden layer
            # nn.ReLU(),
            # nn.Linear(128, 64),        # Continue reducing dimensions
            # nn.ReLU(),
            # nn.Linear(64, output_dim)  # Final output layer
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
    elif backbone == 'IKLSTMNet':
        model = IKLSTMNet(input_dim=pos_rotmat.shape[1], output_dim=gth_jnt_values.shape[1], hidden_dim=64, num_layers=2, robot=robot).to(device).float()
    else:
        raise ValueError('Invalid backbone name!')
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if mode == 'train':
        # prepare the save path
        TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
        rootpath = f'{TimeCode}_{backbone}_{dataset_size}dataset'
        save_path = f'0000_test_programs/nn_ik/results/{rootpath}/'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        
        for epoch in tqdm(range(train_epoch)):
            model.train()
            train_loss = 0
            for pos_rotmat, gth_jnt in train_loader:
                optimizer.zero_grad()
                if backbone == 'IKLSTMNet':
                    pred_jnt = model(pos_rotmat.unsqueeze(1))
                else:
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
                    if backbone == 'IKLSTMNet':
                        pred_jnt = model(pos_rotmat.unsqueeze(1))
                    else:
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
        loss_list = []
        time_list = []

        with torch.no_grad():
            for pos_rotmat, gth_jnt in tqdm(test_loader):
                tic = time.time()
                pred_jnt = model(pos_rotmat)
                toc = time.time()
                loss = criterion(pred_jnt, gth_jnt)
                time_list.append(toc-tic)
                test_loss += loss.item()
                loss_list.append(loss.item())
                # print('current time: ',toc-tic, 'current loss: ',loss) 
                if loss > 2:
                    print(f'pred_jnt: {pred_jnt}\ngth_jnt: {gth_jnt}\nloss: ',loss)

        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss:.4f}")
        
        loss = np.array(loss_list)
        plt.figure(figsize=(10, 6))
        plt.hist(loss, bins=20, density=True, alpha=0.6, color='g', label="Histogram")

        sns.kdeplot(loss, color='b', label="KDE Curve")

        plt.title("Probability Distribution of MSE Loss Differences")
        plt.xlabel("MSE Loss Difference")
        plt.ylabel("Density")
        plt.legend()
        plt.grid()
        plt.show()

    elif mode == 'inference':
        model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/1202_2109_IKMLPNet_1M_dataset/model900'))
        model.eval()
        # base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
        # mcm.mgm.gen_frame().attach_to(base)
        # robot = cbt.Cobotta(pos=rm.vec(0.168, .3, 0), rotmat=rm.rotmat_from_euler(0, 0, rm.pi / 2), enable_cc=True)
        # # robot.jaw_to(.02)
        with torch.no_grad():
            robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
            # robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
            tgt_pos = np.array([0.3, .23, .58])
            # tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
            # tgt_rotmat = rm.rotmat_from_quaternion([0.707, -0.707, 0, 0])
            tgt_rotmat = rm.rotmat_from_quaternion([0.156, 0.988, 0, 0])
            mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
            # base.run()
            pos_rotmat = torch.tensor(np.concatenate((tgt_pos.flatten(), tgt_rotmat.flatten()), axis=0), dtype=torch.float32).to(device).unsqueeze(0)
            jnt_values_model = model(pos_rotmat)
            jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, toggle_dbg=False)
            print('nn jnt_values: ', jnt_values_model, 'ik jnt_values: ', jnt_values)
            if jnt_values is not None:
                robot.goto_given_conf(jnt_values=jnt_values)
                robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
                robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
            # robot.show_cdprim()
            # robot.unshow_cdprim()

            robot.goto_given_conf(jnt_values=jnt_values_model.squeeze(1).cpu().numpy()[0])
            robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
            # robot.show_cdprim()
            base.run()
            box = mcm.gen_box(xyz_lengths=np.array([0.1, .1, .1]), pos=tgt_pos, rgb=np.array([1, 1, 0]), alpha=.3)
            box.attach_to(base)
            tic = time.time()
            result, contacts = robot.is_collided(obstacle_list=[box], toggle_contacts=True)
            print(result)
            toc = time.time()
            print(toc - tic)
            # for pnt in contacts:
            #     mgm.gen_sphere(pnt).attach_to(base)

            base.run()

    elif mode == 'ik_test':
        rot = 'rotmat' # ['rotmat', 'rotv', 'quaternion']
        # model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/1205_1535_IKMLPNet_1M_loc_rotquatdataset/model1000'))
        model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/1202_2109_IKMLPNet_1M_dataset/model900'))
        model.eval()

        with torch.no_grad():
            for _ in range(trail_num):
                nn_success_rate = 0
                trad_success_rate = 0
                nn_faster_count = 0

                time_list_nn = []
                time_list_trad = []
                for i in range(nupdate):
                    jnt_values = robot.rand_conf()
                    tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)  # fk --> pos(3), rotmat(3x3)
                    # rot format conversion
                    if rot == 'rotv':
                        rotmat = rm.rotmat_to_wvec(tgt_rotmat)
                    elif rot == 'quaternion':
                        rotmat = rm.rotmat_to_quaternion(tgt_rotmat)
                    else:
                        rotmat = tgt_rotmat
                    pos_rotmat = torch.tensor(np.concatenate((tgt_pos.flatten(), rotmat.flatten()), axis=0), dtype=torch.float32).to(device).unsqueeze(0)
                    nn_pred_jnt_values = model(pos_rotmat).cpu().numpy()[0]
                    tic = time.time()
                    result_nn = robot.ik(tgt_pos, tgt_rotmat,seed_jnt_values=nn_pred_jnt_values)
                    toc = time.time()
                    nn_time = toc - tic
                    time_list_nn.append(nn_time)
                    
                    tic = time.time()
                    result_trad = robot.ik(tgt_pos, tgt_rotmat)
                    toc = time.time()
                    trad_time = toc - tic
                    time_list_trad.append(trad_time)

                    if result_nn is not None:
                        nn_success_rate += 1
                    if result_trad is not None:
                        trad_success_rate += 1
                    if nn_time < trad_time:
                        nn_faster_count += 1

                avg_time_nn = sum(time_list_nn) / nupdate
                avg_time_trad = sum(time_list_trad) / nupdate
                
                print('NN Success Rate: ',nn_success_rate/nupdate, 'Trad Success Rate: ',trad_success_rate/nupdate)
                print('Average NN Time: ', avg_time_nn, 'Average Traditional Time: ', avg_time_trad)
                print('NN Faster Ratio: ', nn_faster_count/nupdate)
                plt.plot(range(nupdate), time_list_nn, label='IK with NN seed')
                plt.plot(range(nupdate), time_list_trad, label='IK')
                for x in range(nupdate):
                            plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
                
                plt.xlabel('Update Step')
                plt.ylabel('Time (s)')
                plt.legend()
                plt.show()