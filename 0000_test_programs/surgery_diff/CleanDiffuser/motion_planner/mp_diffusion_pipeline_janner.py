import zarr
import os
import sys
import warnings
warnings.filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import zarr
import wandb
import gym
import pathlib
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from datetime import datetime
import random
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters
from motion_planner_dataset import MotionPlanningDataset
from torch.utils.data import random_split

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def validate(agent, val_dataloader, device):
    agent.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            x = batch['trajectory'].to(device)
            val_loss = agent.loss(x)
            total_val_loss += val_loss.item()
    return total_val_loss / len(val_dataloader)

'''load the config file'''
current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))
config_file = os.path.join(current_file_dir, 'config.yaml')
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

'''dataset loading'''
dataset_path = os.path.join(parent_dir, 'datasets/xarm_toppra_mp.zarr')

dataset = MotionPlanningDataset(dataset_path, horizon=config['horizon'], obs_keys=config['obs_keys'], 
                                pad_before=config['obs_steps']-1, pad_after=config['action_steps']-1, abs_action=config['abs_action'])

train_dataset, val_dataset = random_split(
    dataset, lengths=[0.95, 0.05],
    generator=torch.Generator().manual_seed(config['seed'])
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    num_workers=2,
    shuffle=False,
    pin_memory=True
)

# --------------- Create Diffusion Model -----------------
set_seed(config['seed'])
assert config["nn"] == "chi_unet"
assert config['diffusion'] == "ddpm"

from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import ChiUNet1d, JannerUNet1d
from cleandiffuser.diffusion.ddpm import DDPM
from cleandiffuser.diffusion.diffusionsde import DiscreteDiffusionSDE

# --------------- Network Architecture -----------------
nn_diffusion = JannerUNet1d(
    config['obs_dim'] + config['action_dim'], model_dim=config['model_dim'], emb_dim=config['model_dim'], 
    dim_mult=config['dim_mult'], timestep_emb_type="positional", attention=False, kernel_size=5)

print(f"======================= Parameter Report of Diffusion Model =======================")
report_parameters(nn_diffusion)

# ----------------- Masking -------------------
fix_mask = torch.zeros((config['horizon']-1, config['obs_dim'] + config['action_dim']))
fix_mask[0, config['action_dim']:] = 1.
fix_mask[-1:, config['action_dim']:] = 1.
loss_weight = torch.ones((config['horizon']-1, config['obs_dim'] + config['action_dim']))
loss_weight[0, :config['action_dim']] = config['action_loss_weight']

print(f"======================= Parameter Report of Diffusion Model =======================")
report_parameters(nn_diffusion)
print(f"==============================================================================")

import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm_s
from wrs import wd, rm, mcm
robot_s = xarm_s.XArmLite6(enable_cc=True)

'''define the robot joint limits'''
jnt_v_max = rm.np.asarray([rm.pi * 2 / 3] * robot_s.n_dof)
jnt_config_range = robot_s.jnt_ranges
x_max = torch.zeros((1, config['horizon']-1, config['action_dim']+config['obs_dim']), device=config['device'])
x_min = torch.zeros((1, config['horizon']-1, config['action_dim']+config['obs_dim']), device=config['device'])

x_max[:, :, :6] = torch.tensor(jnt_config_range[:, 1], device=config['device']) 
x_min[:, :, :6] = torch.tensor(jnt_config_range[:, 0], device=config['device'])
x_max[:, :, -6:] = torch.tensor(jnt_config_range[:, 1], device=config['device']) 
x_min[:, :, -6:] = torch.tensor(jnt_config_range[:, 0], device=config['device'])
x_max[:, :, 6:12] = torch.tensor(jnt_v_max, device=config['device'])
x_min[:, :, 6:12] = torch.tensor(-jnt_v_max, device=config['device'])

agent = DiscreteDiffusionSDE(nn_diffusion, None,
                             x_max=x_max, x_min=x_min,
                             fix_mask=fix_mask, 
                             loss_weight=loss_weight, 
                             ema_rate=config['ema_rate'],
                             device=config['device'], 
                             diffusion_steps=config['diffusion_steps'], 
                             predict_noise=config['predict_noise'])

lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=config['gradient_steps'])

if config['mode'] == "train":
    # --------------- Data Loading -----------------
    '''prepare the save path'''
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f"{TimeCode}_h{config['horizon']-1}_unnorm"
    current_file_dir = os.path.dirname(__file__)
    save_path = os.path.join(current_file_dir, 'results', rootpath)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project="mp_diff", name=rootpath)

    # ----------------- Training ----------------------
    agent.train()
    n_gradient_step = 0
    diffusion_loss_list = []
    log = {'avg_loss_diffusion': 0.}
    start_time = time.time()
    
    for batch in loop_dataloader(train_loader):
        # get condition
        x = batch['trajectory'].to(config['device'])

        # update diffusion
        diffusion_loss = agent.update(x)['loss']
        log["avg_loss_diffusion"] += diffusion_loss
        lr_scheduler.step()
        diffusion_loss_list.append(diffusion_loss)

        if n_gradient_step != 0 and n_gradient_step % config['log_freq'] == 0:
            log['gradient_steps'] = n_gradient_step
            log["avg_loss_diffusion"] /= config['log_freq']
            diffusion_loss_list = []
            wandb.log(
                {'step': log['gradient_steps'],
                'avg_training_loss': log['avg_loss_diffusion'],
                'total_time': time.time() - start_time}, commit = True)
            print(log)
            log = {"avg_loss_diffusion": 0.}
        
        if n_gradient_step != 0 and n_gradient_step % config['save_freq'] == 0:
            agent.save(save_path + f"/diffusion_ckpt_{n_gradient_step}.pt")
            agent.save(save_path + f"/diffusion_ckpt_latest.pt")
        
        if n_gradient_step != 0 and n_gradient_step % config['val_freq'] == 0:
            val_loss = validate(agent, val_dataloader, config['device'])
            print(f"Validation Loss: {val_loss}")
            wandb.log({'val_loss': val_loss}, commit=False)

        n_gradient_step += 1
        if n_gradient_step >= config['gradient_steps']:
            break
    wandb.finish()

elif config['mode'] == "inference":
    # ----------------- Inference ----------------------
    # model_path = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/results/0325_1601_h32/diffusion_ckpt_latest.pt'
    model_path = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/results/0325_1819_h32_unnorm/diffusion_ckpt_latest.pt'    
    agent.load(model_path)
    agent.model.eval()
    agent.model_ema.eval()

    '''capture the image'''
    sys.path.append('/home/lqin/wrs_xinyi/wrs')
    import wrs.robot_con.xarm_lite6.xarm_lite6_x as xarm_x
    import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm_s
    import wrs.visualization.panda.world as wd
    from wrs import wd, rm, mcm
    import wrs.modeling.geometric_model as mgm
    import mp_data_gen as mp_helper
    
    # init
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    robot_s = xarm_s.XArmLite6(enable_cc=True)

    # inference
    solver = config['inference_solver']
    inference_steps = config['inference_samples']
    prior = torch.zeros((1, config['horizon']-1, config['obs_dim'] + config['action_dim']), device=config['device'])
    
    start_conf, goal_conf = mp_helper.gen_collision_free_start_goal(robot_s)
    print(f"Start Conf: {start_conf}, Goal Conf: {goal_conf}")
    robot_s.goto_given_conf(jnt_values=goal_conf)
    robot_s.gen_meshmodel(rgb=[0,1,0], alpha=0.1).attach_to(base)
    robot_s.goto_given_conf(jnt_values=start_conf)
    robot_s.gen_meshmodel(rgb=[0,0,1], alpha=0.1).attach_to(base)

    if config['normalize']:
        n_goal_conf = dataset.normalizer['obs']['interp_confs'].normalize(goal_conf)
        update_counter = 0

        for _ in range(inference_steps):
            start_conf = robot_s.get_jnt_values()
            n_start_conf = dataset.normalizer['obs']['interp_confs'].normalize(start_conf)

            prior[:, 0, config['action_dim']:] = torch.tensor(n_start_conf).to(config['device'])
            prior[:, -1, config['action_dim']:] = torch.tensor(n_goal_conf).to(config['device'])

            trajectory, log = agent.sample(prior,
                                        solver=solver,
                                        n_samples = 1,
                                        sample_steps=config['sample_steps'],
                                        use_ema=True, w_cg=0.0, temperature=1.0)

            # sample actions and unnorm
            actions = trajectory[0, :, :config['action_dim']].cpu().detach().numpy() # 32,12
            jnt_cfgs_pred = dataset.normalizer['obs']['interp_confs'].unnormalize(actions[:, :6]) 
            jnt_spds_pred = dataset.normalizer['obs']['interp_spds'].unnormalize(actions[:, 6:]) 

            for idx in range(jnt_cfgs_pred.shape[0]):
                robot_s.goto_given_conf(jnt_values=jnt_cfgs_pred[idx])
                robot_s.gen_meshmodel(alpha=0.3).attach_to(base)
                update_counter += 1
                print(f"Step: {update_counter}, distance: {np.linalg.norm(robot_s.get_jnt_values() - goal_conf)}")
            
            if np.linalg.norm(robot_s.get_jnt_values() - goal_conf) < 1e-2 or update_counter > 10:
                break
                
        base.run()

    else:
        assert config['normalize'] == False
        update_counter = 0

        for _ in range(inference_steps):
            start_conf = robot_s.get_jnt_values()
            prior[:, 0, config['action_dim']:] = torch.tensor(start_conf).to(config['device'])
            prior[:, -1, config['action_dim']:] = torch.tensor(goal_conf).to(config['device'])

            trajectory, log = agent.sample(prior,
                                        solver=solver,
                                        n_samples = 1,
                                        sample_steps=config['sample_steps'],
                                        use_ema=True, w_cg=0.0, temperature=1.0)

            # sample actions and unnorm
            actions = trajectory[0, :, :config['action_dim']].cpu().detach().numpy() # 32,12
            jnt_cfgs_pred = actions[:, :6]

            for idx in range(jnt_cfgs_pred.shape[0]):
                robot_s.goto_given_conf(jnt_values=jnt_cfgs_pred[idx])
                robot_s.gen_meshmodel(alpha=0.3).attach_to(base)
                update_counter += 1
                print(f"Step: {update_counter}, distance: {np.linalg.norm(robot_s.get_jnt_values() - goal_conf)}")
            
            if np.linalg.norm(robot_s.get_jnt_values() - goal_conf) < 1e-2 or update_counter > 10:
                break
                
        base.run()

else:
    raise ValueError("Illegal mode")