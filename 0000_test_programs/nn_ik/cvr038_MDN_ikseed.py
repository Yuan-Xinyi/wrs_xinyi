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
import torch.nn.functional as F

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)

# robot = cbt.Cobotta(pos=rm.vec(0.1,.3,.5), enable_cc=True)
robot = cbt.Cobotta(pos=rm.vec(0.0,.0,.0), enable_cc=True)
nupdate = 100
trail_num = 100
seed = 42
mode = 'train' # ['train' or 'test','inference','ik_test']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train paras
train_epoch = 10000
train_batch = 64
lr = 0.001
save_intervel = 100
wandb.init(project="ik")
dataset_size = '1M' # [1M, 1M_loc_rotmat, 1M_loc_rotv, 1M_loc_rotquat]
backbone = 'MDN'

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
    # y: [batch_size, D]
    # pi: [batch_size, M]
    # mu: [batch_size, M, D]
    # sigma: [batch_size, M, D]

    # Expand y to [batch_size, 1, D] so it can be broadcasted against [M, D]
    y = y.unsqueeze(1)  # [batch_size, 1, D]

    # Compute the probability density of the data under each Gaussian component
    # For a diagonal Gaussian:
    # p(y|mu,sigma) = (1 / (sqrt(2*pi)*sigma)) * exp(-0.5 * ((y - mu) / sigma)^2)
    normal_const = 1.0 / (torch.sqrt(torch.tensor(2.0 * 3.141592653589793)) * sigma)
    exponent = torch.exp(-0.5 * ((y - mu) / sigma)**2)
    # gauss: [batch_size, M], product across D since we assume independence in dimensions
    gauss = torch.prod(normal_const * exponent, dim=2)  

    # Weight each Gaussian by its mixture coefficient
    weighted_gauss = pi * gauss
    # To avoid log(0), add a small epsilon
    epsilon = 1e-9
    # NLL = -log(sum(pi_k * N(y|mu_k, sigma_k)))
    loss = -torch.log(weighted_gauss.sum(dim=1) + epsilon)
    return loss.mean()


if __name__ == '__main__':
    # dataset loading
    dataset = np.load(f'0000_test_programs/nn_ik/datasets/effective_seedset_{dataset_size}.npz') # ['source', 'target', 'seed_jnt_value', 'jnt_result']
    jnt_result, tgt, jnt_seed = dataset['jnt_result'], dataset['target'], dataset['seed_jnt_value']

    input_dim = 6    # e.g., 3D position + 3D rotation vector
    output_dim = 6   # e.g., a 6-DOF robot arm
    num_mixtures = 5 # number of mixture components

    # dataset pre-processing
    model_in = torch.tensor(tgt, dtype=torch.float32).to(device)
    model_out = torch.tensor(jnt_seed, dtype=torch.float32).to(device)

    # train-val-test split
    train_input, input, train_label, label = train_test_split(model_in, model_out, test_size=0.3, random_state=seed)
    val_input, test_input, val_label, test_label = train_test_split(input, label, test_size=0.5, random_state=seed)

    # data loader
    train_dataset = TensorDataset(train_input, train_label)
    val_dataset = TensorDataset(val_input, val_label)
    test_dataset = TensorDataset(test_input, test_label)

    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=True, drop_last=False)


    # Initialize the model
    if backbone == 'MDN':
        model = MDN(input_dim, output_dim, num_mixtures).to(device).float()
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

    # elif mode == 'test':
    #     model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/1202_2109_IKMLPNet_1M_dataset/model900'))
    #     model.eval()
    #     test_loss = 0
    #     loss_list = []
    #     time_list = []

    #     with torch.no_grad():
    #         for pos_rotmat, gth_jnt in tqdm(test_loader):
    #             tic = time.time()
    #             pred_jnt = model(pos_rotmat)
    #             toc = time.time()
    #             loss = criterion(pred_jnt, gth_jnt)
    #             time_list.append(toc-tic)
    #             test_loss += loss.item()
    #             loss_list.append(loss.item())
    #             # print('current time: ',toc-tic, 'current loss: ',loss) 
    #             if loss > 2:
    #                 print(f'pred_jnt: {pred_jnt}\ngth_jnt: {gth_jnt}\nloss: ',loss)

    #     test_loss /= len(test_loader)
    #     print(f"Test Loss: {test_loss:.4f}")
        
    #     loss = np.array(loss_list)
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(loss, bins=20, density=True, alpha=0.6, color='g', label="Histogram")

    #     sns.kdeplot(loss, color='b', label="KDE Curve")

    #     plt.title("Probability Distribution of MSE Loss Differences")
    #     plt.xlabel("MSE Loss Difference")
    #     plt.ylabel("Density")
    #     plt.legend()
    #     plt.grid()
    #     plt.show()

    # elif mode == 'inference':
    #     model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/1202_2109_IKMLPNet_1M_dataset/model900'))
    #     model.eval()
    #     # base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    #     # mcm.mgm.gen_frame().attach_to(base)
    #     # robot = cbt.Cobotta(pos=rm.vec(0.168, .3, 0), rotmat=rm.rotmat_from_euler(0, 0, rm.pi / 2), enable_cc=True)
    #     # # robot.jaw_to(.02)
    #     with torch.no_grad():
    #         robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    #         # robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    #         tgt_pos = np.array([0.3, .23, .58])
    #         # tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    #         # tgt_rotmat = rm.rotmat_from_quaternion([0.707, -0.707, 0, 0])
    #         tgt_rotmat = rm.rotmat_from_quaternion([0.156, 0.988, 0, 0])
    #         mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    #         # base.run()
    #         pos_rotmat = torch.tensor(np.concatenate((tgt_pos.flatten(), tgt_rotmat.flatten()), axis=0), dtype=torch.float32).to(device).unsqueeze(0)
    #         jnt_values_model = model(pos_rotmat)
    #         jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, toggle_dbg=False)
    #         print('nn jnt_values: ', jnt_values_model, 'ik jnt_values: ', jnt_values)
    #         if jnt_values is not None:
    #             robot.goto_given_conf(jnt_values=jnt_values)
    #             robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    #             robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    #         # robot.show_cdprim()
    #         # robot.unshow_cdprim()

    #         robot.goto_given_conf(jnt_values=jnt_values_model.squeeze(1).cpu().numpy()[0])
    #         robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    #         # robot.show_cdprim()
    #         base.run()
    #         box = mcm.gen_box(xyz_lengths=np.array([0.1, .1, .1]), pos=tgt_pos, rgb=np.array([1, 1, 0]), alpha=.3)
    #         box.attach_to(base)
    #         tic = time.time()
    #         result, contacts = robot.is_collided(obstacle_list=[box], toggle_contacts=True)
    #         print(result)
    #         toc = time.time()
    #         print(toc - tic)
    #         # for pnt in contacts:
    #         #     mgm.gen_sphere(pnt).attach_to(base)

    #         base.run()

    # elif mode == 'ik_test':
    #     rot = 'rotmat' # ['rotmat', 'rotv', 'quaternion']
    #     # model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/1205_1535_IKMLPNet_1M_loc_rotquatdataset/model1000'))
    #     model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/1202_2109_IKMLPNet_1M_dataset/model900'))
    #     model.eval()

    #     with torch.no_grad():
    #         for _ in range(trail_num):
    #             nn_success_rate = 0
    #             trad_success_rate = 0
    #             nn_faster_count = 0

    #             time_list_nn = []
    #             time_list_trad = []
    #             for i in range(nupdate):
    #                 jnt_values = robot.rand_conf()
    #                 tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)  # fk --> pos(3), rotmat(3x3)
    #                 # rot format conversion
    #                 if rot == 'rotv':
    #                     rotmat = rm.rotmat_to_wvec(tgt_rotmat)
    #                 elif rot == 'quaternion':
    #                     rotmat = rm.rotmat_to_quaternion(tgt_rotmat)
    #                 else:
    #                     rotmat = tgt_rotmat
    #                 pos_rotmat = torch.tensor(np.concatenate((tgt_pos.flatten(), rotmat.flatten()), axis=0), dtype=torch.float32).to(device).unsqueeze(0)
    #                 nn_pred_jnt_values = model(pos_rotmat).cpu().numpy()[0]
    #                 tic = time.time()
    #                 result_nn = robot.ik(tgt_pos, tgt_rotmat,seed_jnt_values=nn_pred_jnt_values)
    #                 toc = time.time()
    #                 nn_time = toc - tic
    #                 time_list_nn.append(nn_time)
                    
    #                 tic = time.time()
    #                 result_trad = robot.ik(tgt_pos, tgt_rotmat)
    #                 toc = time.time()
    #                 trad_time = toc - tic
    #                 time_list_trad.append(trad_time)

    #                 if result_nn is not None:
    #                     nn_success_rate += 1
    #                 if result_trad is not None:
    #                     trad_success_rate += 1
    #                 if nn_time < trad_time:
    #                     nn_faster_count += 1

    #             avg_time_nn = sum(time_list_nn) / nupdate
    #             avg_time_trad = sum(time_list_trad) / nupdate
                
    #             print('NN Success Rate: ',nn_success_rate/nupdate, 'Trad Success Rate: ',trad_success_rate/nupdate)
    #             print('Average NN Time: ', avg_time_nn, 'Average Traditional Time: ', avg_time_trad)
    #             print('NN Faster Ratio: ', nn_faster_count/nupdate)
    #             plt.plot(range(nupdate), time_list_nn, label='IK with NN seed')
    #             plt.plot(range(nupdate), time_list_trad, label='IK')
    #             for x in range(nupdate):
    #                         plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
                
    #             plt.xlabel('Update Step')
    #             plt.ylabel('Time (s)')
    #             plt.legend()
    #             plt.show()