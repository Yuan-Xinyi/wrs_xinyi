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
save_intervel = 1000
wandb.init(project="ik")
dataset_size = '1M'
backbone = 'IKTransformerNet'  # [IKMLPNet, IKMLPScaleNet, IKLSTMNet, IKTransformerNet]

# val and test paras
val_batch = train_batch
test_batch = 1

class IKTransformerNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, hidden_dim, robot):
        super(IKTransformerNet, self).__init__()
        self.robot = robot
        self.jnt_ranges = torch.tensor(robot.jnt_ranges, dtype=torch.float32).to(device)

        # Embedding layer to map input to `hidden_dim` for Transformer
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # Map input to `hidden_dim`
        x = self.transformer(x)  # Apply Transformer Encoder
        x = x[:, -1, :]  # Take the output of the last time step (batch_size, hidden_dim)
        out = self.output_layer(x)  # Final output layer
        
        # Scale output to joint ranges
        scaled_output = out * (self.jnt_ranges[:, 1] - self.jnt_ranges[:, 0]) + self.jnt_ranges[:, 0]
        return scaled_output


if __name__ == '__main__':
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
    if backbone == 'IKTransformerNet':
        model = IKTransformerNet(
            input_dim=pos_rotmat.shape[1], output_dim=gth_jnt_values.shape[1],
            num_heads=4, num_layers=3, hidden_dim=128, robot=robot
            ).to(device)
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
                if backbone == 'IKLSTMNet' or backbone == 'IKTransformerNet':
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
                    if backbone == 'IKLSTMNet' or backbone == 'IKTransformerNet':
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