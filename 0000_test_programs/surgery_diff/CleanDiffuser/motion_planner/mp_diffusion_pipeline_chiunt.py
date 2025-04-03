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
from motion_planner_dataset import MotionPlanningDataset, ObstaclePlanningDataset
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
dataset_path = os.path.join(parent_dir, config['dataset_name'])

dataset = ObstaclePlanningDataset(dataset_path, horizon=config['horizon'], obs_keys=config['obs_keys'], 
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
if config['mode'] == "train":
    set_seed(config['seed'])
assert config["nn"] == "chi_unet"
assert config['diffusion'] == "ddpm"

from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import ChiUNet1d, JannerUNet1d
from cleandiffuser.diffusion.ddpm import DDPM
from cleandiffuser.diffusion.diffusionsde import DiscreteDiffusionSDE

# --------------- Network Architecture -----------------
nn_diffusion = ChiUNet1d(
    config['action_dim'], config['obs_dim'], config['obs_steps'], model_dim=256, emb_dim=256, dim_mult=config['dim_mult'],
    obs_as_global_cond=True, timestep_emb_type="positional").to(config['device'])
nn_condition = IdentityCondition(dropout=0.0).to(config['device'])


print(f"======================= Parameter Report of Diffusion Model =======================")
report_parameters(nn_diffusion)


import wrs.robot_sim.manipulators.xarm_lite6.xarm_lite6 as xarm_s
from wrs import wd, rm, mcm
robot_s = xarm_s.XArmLite6(enable_cc=True)

'''define the robot joint limits'''
jnt_v_max = rm.np.asarray([rm.pi * 2 / 3] * robot_s.n_dof)
jnt_config_range = robot_s.jnt_ranges
x_max = torch.zeros((1, config['horizon']-1, config['action_dim']), device=config['device'])
x_min = torch.zeros((1, config['horizon']-1, config['action_dim']), device=config['device'])

x_max[:, :, :6] = torch.tensor(jnt_config_range[:, 1], device=config['device']) 
x_min[:, :, :6] = torch.tensor(jnt_config_range[:, 0], device=config['device'])
x_max[:, :, -6:] = torch.tensor(jnt_v_max, device=config['device']) 
x_min[:, :, -6:] = torch.tensor(-jnt_v_max, device=config['device'])

agent = DDPM(
    nn_diffusion=nn_diffusion, nn_condition=nn_condition, device=config['device'],
    diffusion_steps=config['sample_steps'], x_max=x_max, x_min=x_min,
    optim_params={"lr": config['lr']})

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
        obs = batch['obs'].to(config['device'])
        action = batch['action'].to(config['device'])

        condition = obs[:, [0, -1], :]
        condition = condition.flatten(start_dim=1) # (64,12)
        
        # update diffusion
        diffusion_loss = agent.update(action, condition)['loss']
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
    model_path = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/results/0325_1926_h32_unnorm/diffusion_ckpt_latest.pt'    
    # model_path = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/results/0403_1111_h128_unnorm/diffusion_ckpt_latest.pt'
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
    visualization = config['visualization']
    
    success_num = 0
    for _ in tqdm(range(config['episode_num'])):
        start_conf, goal_conf = mp_helper.gen_collision_free_start_goal(robot_s)
        tgt_pos, tgt_rotmat = robot_s.fk(jnt_values=goal_conf)
        # print(f"Start Conf: {start_conf}, Goal Conf: {goal_conf}")
        
        if visualization:
            robot_s.goto_given_conf(jnt_values=goal_conf)
            robot_s.gen_meshmodel(rgb=[0,1,0], alpha=1).attach_to(base)
            robot_s.goto_given_conf(jnt_values=start_conf)
            robot_s.gen_meshmodel(rgb=[0,0,1], alpha=1).attach_to(base)

        assert config['normalize'] == False
        update_counter = 0
        condition = torch.zeros((1, config['obs_dim']*config['obs_steps']), device=config['device'])

        for _ in range(inference_steps):
            start_conf = robot_s.get_jnt_values()
            condition[:, :config['obs_dim']] = torch.tensor(start_conf).to(config['device'])
            condition[:, config['obs_dim']:] = torch.tensor(goal_conf).to(config['device'])
            
            with torch.no_grad():
                prior = torch.zeros((1, config['horizon']-1, config['action_dim']), device=config['device'])
                action, _ = agent.sample(prior=prior, n_samples=1, sample_steps=config['sample_steps'],
                                        solver=solver, condition_cfg=condition, w_cfg=1.0, use_ema=True)

            # sample actions and unnorm
            jnt_cfgs_pred = action[0, :,:6].detach().to('cpu').numpy()

            for idx in range(jnt_cfgs_pred.shape[0]):
                robot_s.goto_given_conf(jnt_values=jnt_cfgs_pred[idx])
                pred_pos, pred_rotmat = robot_s.fk(jnt_values=jnt_cfgs_pred[idx])
                pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos*1000, tgt_rotmat, pred_pos*1000, pred_rotmat)
                rot_err = np.rad2deg(rot_err)
                # print(f"Step: {update_counter}, pos_err: {pos_err} mms, rot_err: {rot_err} degrees")

                if visualization:
                    robot_s.gen_meshmodel(alpha=0.2).attach_to(base)
                update_counter += 1
                # print(f"Step: {update_counter}, distance: {np.linalg.norm(robot_s.get_jnt_values() - goal_conf)}")

                if robot_s.cc.is_collided():
                    break

                if (pos_err < 5 and rot_err < 3) or update_counter > 200:
                    break
            
            if pos_err < 5 and rot_err < 3 and not robot_s.cc.is_collided():
                success_num += 1
                break
        
        if visualization:
            base.run()    

    print(f"Success rate: {success_num/config['episode_num']*100}%")

else:
    raise ValueError("Illegal mode")