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
trail_num = 10
seed = 42
mode = 'ik_test' # ['train' or 'test','inference','ik_test']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train paras
train_epoch = 10000
train_batch = 64
lr = 0.001
save_intervel = 100
dataset_size = '1M' # [1M, 1M_loc_rotmat, 1M_loc_rotv, 1M_loc_rotquat]
backbone = 'MDN'

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

def rgb2color(rgb):
    return tuple(x / 255.0 for x in rgb)

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
        wandb.init(project="ik")
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


    elif mode == 'test':
        # Load the model checkpoint
        model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/1202_2109_MDN_1M_dataset/model900'))
        model.eval()  # Set the model to evaluation mode

        test_loss = 0.0  # Initialize cumulative test loss
        loss_list = []   # List to store individual losses for analysis
        time_list = []   # List to record inference time per test sample

        # Perform testing without gradient computation
        with torch.no_grad():
            for pos_rotmat, gth_jnt in tqdm(test_loader, desc="Testing MDN"):
                # Measure the time taken for the forward pass
                start_time = time.time()
                pi, mu, sigma = model(pos_rotmat)  # Forward pass through the MDN
                end_time = time.time()

                # Compute the MDN loss
                loss = mdn_loss(pi, mu, sigma, gth_jnt)
                time_list.append(end_time - start_time)  # Record inference time
                test_loss += loss.item()  # Accumulate loss
                loss_list.append(loss.item())  # Store individual loss for analysis

                # Log samples with high loss for debugging
                if loss > 2.0:
                    print(f"Ground Truth Joint Angles: {gth_jnt}")
                    print(f"Predicted Mixtures:\n- Pi: {pi}\n- Mu: {mu}\n- Sigma: {sigma}")
                    print(f"Loss: {loss.item()}")

        # Compute the average test loss
        test_loss /= len(test_loader)
        print(f"Average Test Loss: {test_loss:.4f}")

        # Convert the loss list to a numpy array for visualization
        loss_array = np.array(loss_list)

        # Plot the histogram and KDE curve for the loss distribution
        plt.figure(figsize=(10, 6))
        plt.hist(loss_array, bins=20, density=True, alpha=0.6, color='g', label="Histogram")
        sns.kdeplot(loss_array, color='b', label="KDE Curve")

        # Add plot titles and labels
        plt.title("Probability Distribution of MDN Loss")
        plt.xlabel("MDN Loss")
        plt.ylabel("Density")
        plt.legend()
        plt.grid()
        plt.show()


    elif mode == 'inference':
        model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/1216_1735_MDN_1Mdataset/model7200'))
        model.eval()

        with torch.no_grad():
            robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
            # base.run()
            jnt_values = robot.rand_conf()
            tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)
            mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)

            rotv = rm.rotmat_to_wvec(tgt_rotmat)
            pos_rot = torch.tensor(np.concatenate((tgt_pos.flatten(), rotv.flatten()), axis=0), dtype=torch.float32).to(device).unsqueeze(0)
            
            with torch.no_grad():
                pi, mu, sigma  = model(pos_rot)
            pi, mu, sigma = pi.squeeze(0), mu.squeeze(0), sigma.squeeze(0)
            max_pi_idx = torch.argmax(pi).item()
            seed_jnt_mdn = mu[max_pi_idx].cpu().numpy()

            result = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat,seed_jnt_values=seed_jnt_mdn, toggle_dbg=False)
            print('seed jnt: ', seed_jnt_mdn, 'ik jnt_values: ', result)
            
            if result is not None:
                robot.goto_given_conf(jnt_values=result)
                robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
                robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)

            robot.goto_given_conf(jnt_values=result)
            robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)

            base.run()
            box = mcm.gen_box(xyz_lengths=np.array([0.1, .1, .1]), pos=tgt_pos, rgb=np.array([1, 1, 0]), alpha=.3)
            box.attach_to(base)
            tic = time.time()
            result, contacts = robot.is_collided(obstacle_list=[box], toggle_contacts=True)
            print(result)
            toc = time.time()
            print(toc - tic)

            base.run()

    elif mode == 'ik_test':
        # # only compare the MDN and the ik
        # model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/1216_1735_MDN_1Mdataset/model7200'))
        # model.eval()

        # with torch.no_grad():
        #     for _ in range(trail_num):
        #         nn_success_rate = 0
        #         trad_success_rate = 0
        #         nn_faster_count = 0

        #         time_list_nn = []
        #         time_list_trad = []
        #         for i in range(nupdate):
        #             jnt_values = robot.rand_conf()
        #             tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)  # fk --> pos(3), rotmat(3x3)
        #             rotv = rm.rotmat_to_wvec(tgt_rotmat)
        #             pos_rot = torch.tensor(np.concatenate((tgt_pos.flatten(), rotv.flatten()), axis=0), dtype=torch.float32).to(device).unsqueeze(0)
                    
        #             with torch.no_grad():
        #                 pi, mu, sigma  = model(pos_rot)
        #             pi, mu, sigma = pi.squeeze(0), mu.squeeze(0), sigma.squeeze(0)
        #             max_pi_idx = torch.argmax(pi).item()
        #             seed_jnt_mdn = mu[max_pi_idx].cpu().numpy()

        #             tic = time.time()
        #             result_nn = robot.ik(tgt_pos, tgt_rotmat,seed_jnt_values=seed_jnt_mdn)
        #             toc = time.time()
        #             nn_time = toc - tic
        #             time_list_nn.append(nn_time)
                    
        #             tic = time.time()
        #             result_trad = robot.ik(tgt_pos, tgt_rotmat)
        #             toc = time.time()
        #             trad_time = toc - tic
        #             time_list_trad.append(trad_time)

        #             if result_nn is not None:
        #                 nn_success_rate += 1
        #             if result_trad is not None:
        #                 trad_success_rate += 1
        #             if nn_time < trad_time:
        #                 nn_faster_count += 1

        #         avg_time_nn = sum(time_list_nn) / nupdate
        #         avg_time_trad = sum(time_list_trad) / nupdate
                
        #         print('NN Success Rate: ',nn_success_rate/nupdate, 'Trad Success Rate: ',trad_success_rate/nupdate)
        #         print('Average NN Time: ', avg_time_nn, 'Average Traditional Time: ', avg_time_trad)
        #         print('NN Faster Ratio: ', nn_faster_count/nupdate)
        #         plt.plot(range(nupdate), time_list_nn, label='IK with MDN seed')
        #         plt.plot(range(nupdate), time_list_trad, label='IK')
        #         for x in range(nupdate):
        #                     plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
                
        #         plt.xlabel('Update Step')
        #         plt.ylabel('Time (s)')
        #         plt.legend()
        #         plt.show()

        # compare the MDN, the ik and the NN
        nn_model = IKMLPNet(input_dim=6, output_dim=6).to(device).float()
        nn_model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/1216_1412_IKMLPNet_1Mdataset/model1000'))
        model.eval()
        model.load_state_dict(torch.load('0000_test_programs/nn_ik/results/1216_1735_MDN_1Mdataset/model7200'))
        model.eval()
        
        fig_dir = "0000_test_programs/nn_ik/res_figs/mdn_nn_ik"
        os.makedirs(fig_dir, exist_ok=True)

        with torch.no_grad():
            for idx in range(trail_num):

                time_list_mdn = []
                time_list_trad = []
                time_list_nn = []
                
                for i in range(nupdate):
                    jnt_values = robot.rand_conf()
                    tgt_pos, tgt_rotmat = robot.fk(jnt_values = jnt_values)  # fk --> pos(3), rotmat(3x3)
                    rotv = rm.rotmat_to_wvec(tgt_rotmat)
                    pos_rot = torch.tensor(np.concatenate((tgt_pos.flatten(), rotv.flatten()), axis=0), dtype=torch.float32).to(device).unsqueeze(0)
                    
                    with torch.no_grad():
                        pi, mu, sigma  = model(pos_rot)
                    pi, mu, sigma = pi.squeeze(0), mu.squeeze(0), sigma.squeeze(0)
                    max_pi_idx = torch.argmax(pi).item()
                    seed_jnt_mdn = mu[max_pi_idx].cpu().numpy()

                    with torch.no_grad():
                        seed_jnt_nn = nn_model(pos_rot).cpu().numpy()[0]

                    tic = time.time()
                    result_mdn = robot.ik(tgt_pos, tgt_rotmat,seed_jnt_values=seed_jnt_mdn)
                    toc = time.time()
                    mdn_time = toc - tic
                    time_list_mdn.append(mdn_time)

                    tic = time.time()
                    result_nn = robot.ik(tgt_pos, tgt_rotmat,seed_jnt_values=seed_jnt_nn)
                    toc = time.time()
                    nn_time = toc - tic
                    time_list_nn.append(nn_time)
                    
                    tic = time.time()
                    result_trad = robot.ik(tgt_pos, tgt_rotmat)
                    toc = time.time()
                    trad_time = toc - tic
                    time_list_trad.append(trad_time)

                width = 1.5
                plt.figure(figsize=(20, 5))
                plt.plot(range(nupdate), time_list_nn, label='IK with NN seed', color=rgb2color((130, 176, 210)), linewidth=width)
                plt.plot(range(nupdate), time_list_trad, label='IK',  color=rgb2color((250, 127, 111)), linewidth=width)
                plt.plot(range(nupdate), time_list_mdn, label='IK with MDN seed', color=rgb2color((255, 190, 122)), linewidth=width)
                # for x in range(nupdate):
                #             plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                
                plt.xlabel('samples')
                plt.ylabel('Time (s)')
                plt.legend()
                # plt.show()

                save_path = os.path.join(fig_dir, f"ik_timing_plot_{idx + 1:03d}.png")
                plt.savefig(save_path, bbox_inches='tight')  # Save the figure with tight layout
                plt.close()  # Close the figure to free memory