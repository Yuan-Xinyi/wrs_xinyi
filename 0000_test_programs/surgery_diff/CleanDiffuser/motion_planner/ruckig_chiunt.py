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
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import CosineAnnealingLR
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters
from ruckig_dataset import MotionPlanningDataset, ObstaclePlanningDataset, FixedGoalPlanningDataset, PosPlanningDataset
from torch.utils.data import random_split

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

'''load the config file'''
# current_file_dir = os.path.dirname(__file__)
# current_file_dir = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/results/0413_1256_64h_128b_norm'
# current_file_dir = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/results/0413_1305_64h_128b_norm'
# current_file_dir = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/results/0413_2053_64h_128b_norm'
current_file_dir = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/results/0414_1716_128h_64b_norm_fh'

# parent_dir = os.path.dirname(os.path.dirname(__file__))
config_file = os.path.join(current_file_dir, 'ruckig_config.yaml')
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

'''dataset loading'''
dataset_path = os.path.join('/home/lqin', 'zarr_datasets', config['dataset_name'])

dataset = PosPlanningDataset(dataset_path, horizon=config['horizon'], obs_keys=config['obs_keys'], normalize=config['normalize'],
                                pad_before=config['obs_steps']-1, pad_after=config['action_steps']-1, abs_action=config['abs_action'])
print('dataset loaded in:', dataset_path)
if config['mode'] == "train":
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
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

if config['condition'] == "identity":
    nn_condition = IdentityCondition(dropout=0.0).to(config['device'])
    print("Using Identity Condition")
else:
    nn_condition = None
    print("Using No Condition")


print(f"======================= Parameter Report of Diffusion Model =======================")
report_parameters(nn_diffusion)


import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from wrs import wd, rm, mcm
robot_s = franka.FrankaResearch3(enable_cc=True)

'''define the robot joint limits'''
jnt_v_max = rm.np.asarray([rm.pi * 2 / 3] * robot_s.n_dof)
jnt_a_max = rm.np.asarray([rm.pi] * robot_s.n_dof)
jnt_config_range = torch.tensor(robot_s.jnt_ranges, device=config['device'])

x_max = (jnt_config_range[:,1]).repeat(1, config['horizon'], 1)
x_min = (jnt_config_range[:,0]).repeat(1, config['horizon'], 1)

# x_max = torch.tensor(jnt_config_range[1], device=config['device'])
# x_min = torch.tensor(jnt_config_range[0], device=config['device'])

# x_max[:, :, :7] = torch.tensor(jnt_v_max, device=config['device']) 
# x_max[:, :, -7:] = torch.tensor(jnt_a_max, device=config['device']) 
# x_min[:, :, :7] = torch.tensor(-jnt_v_max, device=config['device'])
# x_min[:, :, -7:] = torch.tensor(-jnt_a_max, device=config['device'])
fix_mask = torch.zeros((config['horizon'], config['action_dim']), device=config['device'])
fix_mask[0, :] = 1.
fix_mask[-1:, :] = 1.

agent = DDPM(
    nn_diffusion=nn_diffusion, nn_condition=nn_condition, fix_mask = fix_mask,
    device=config['device'], diffusion_steps=config['diffusion_steps'], x_max=x_max, x_min=x_min,
    optim_params={"lr": config['lr']}, predict_noise=config['predict_noise'])

lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=config['gradient_steps'])

if config['mode'] == "train":
    # --------------- Data Loading -----------------
    '''prepare the save path'''
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f"{TimeCode}_{config['horizon']}h_{config['batch_size']}b_norm_fh"
    # current_file_dir = os.path.dirname(__file__)
    save_path = os.path.join(current_file_dir, 'results', rootpath)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project="rrt_ruckig", name=rootpath)

    # ----------------- Training ----------------------
    agent.train()
    n_gradient_step = 0
    diffusion_loss_list = []
    log = {'avg_loss_diffusion': 0.}
    start_time = time.time()
    
    for batch in loop_dataloader(train_loader):
        # get condition
        condition = batch['cond'].to(config['device'])
        action = batch['action'].to(config['device']) # (batch,horizon,7)

        if config['condition'] == "identity":
            condition = condition.flatten(start_dim=1) # (batch,14)
        else:
            condition = None
        
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
        
        n_gradient_step += 1
        if n_gradient_step >= config['gradient_steps']:
            break
    wandb.finish()

elif config['mode'] == "pos_inference":
    # ----------------- Inference ----------------------
    model_path = os.path.join(current_file_dir, 'diffusion_ckpt_latest.pt')
    agent.load(model_path)
    agent.model.eval()
    agent.model_ema.eval()

    '''capture the image'''
    sys.path.append('/home/lqin/wrs_xinyi/wrs')
    import wrs.visualization.panda.world as wd
    from wrs import wd, rm, mcm
    import wrs.modeling.geometric_model as mgm
    import mp_datagen_ruckig_obstacle as mp_helper
    import copy
    
    # init
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    # inference
    solver = config['inference_solver']
    inference_steps = config['inference_steps']
    visualization = config['visualization']
    
    success_num = 0
    n_samples = 1

    root = zarr.open('/home/lqin/zarr_datasets/franka_ruckig_100hz_fixgoal.zarr', mode='r')
    start_list = []
    goal_list = []
    for id in tqdm(range(root['meta']['episode_ends'].shape[0]-1)):
        traj_start = int(np.sum(root['meta']['episode_ends'][:id]))
        traj_end = int(np.sum(root['meta']['episode_ends'][:id + 1]))
        # print(f"current traj start: {traj_start}, end: {traj_end}")
        start_list.append(root['data']['jnt_pos'][traj_start])
        goal_list.append(root['data']['jnt_pos'][traj_end])

    success_num = 0
    # for traj_id in tqdm(range(len(start_list))):
    for traj_id in tqdm(range(1)):
        counter = 0
        assert len(start_list) == len(goal_list) 
        jnt_pos_list = []

        # start_conf, goal_conf = mp_helper.gen_collision_free_start_goal(robot_s)
        start_conf = np.array(start_list[traj_id])
        goal_conf = np.array(goal_list[traj_id])
        start_conf = np.array([-1.8307296, -1.575878 , -1.6512135, -1.0037161,  0.3407542,
        2.7061534, -2.0410614])
        goal_conf = np.array([ 0.6153231 ,  0.13956377,  2.0086024 , -2.9329627 ,  1.4677436 ,
        1.3641189 , -2.990753  ])
        print(f"Start Conf: {start_conf}, Goal Conf: {goal_conf}")

        tgt_pos, tgt_rotmat = robot_s.fk(jnt_values=goal_conf)
        
        robot_s.goto_given_conf(jnt_values=goal_conf)
        robot_s.gen_meshmodel(alpha=0.2, rgb=[0,1,0]).attach_to(base)
        robot_s.goto_given_conf(jnt_values=start_conf)
        robot_s.gen_meshmodel(alpha=0.2, rgb=[0,0,1]).attach_to(base)
        
        for _ in range(20):
            if n_samples != 1:
                start_conf = np.tile(start_conf, (n_samples, 1))
                goal_conf = np.tile(goal_conf, (n_samples, 1))

            if config['normalize'] == None:
                condition = None
            
            if n_samples == 1:
                start_conf = robot_s.get_jnt_values()
            else:
                raise NotImplementedError("Not implemented for multiple samples")
            
            prior = torch.zeros((n_samples, config['horizon'], config['action_dim']), device=config['device'])
            prior[:, 0, :] = torch.tensor(start_conf).to(config['device'])
            prior[:, -1, :] = torch.tensor(goal_conf).to(config['device'])


            with torch.no_grad():
                # action, _ = agent.sample(prior=prior, n_samples=n_samples, sample_steps=config['sample_steps'], temperature=1.0,
                #                         solver=solver, condition_cfg=condition, w_cfg=7.0, use_ema=True)
                action, _ = agent.sample(prior=prior, n_samples=n_samples, sample_steps=config['sample_steps'], temperature=1.0,
                                        solver=solver, use_ema=True)

            # plt.figure(figsize=(10, 3 * robot_s.n_dof))
            # for i in range(robot_s.n_dof):
            #     plt.subplot(robot_s.n_dof, 1, i + 1)
            #     plt.plot(np.array(action[0,:,i].to('cpu')), label='Position')
            #     print(f"joint {i}: {action[0,:,i].to('cpu')}")
            #     plt.ylabel(f'DoF {i}')
            #     plt.legend()
            #     plt.grid(True)
            # plt.xlabel('Time [s]')
            # plt.tight_layout()
            # plt.show()
            # break

            # for idx in range(config['horizon']-1):
            for idx in range(32):
                if visualization:
                    for id in range(n_samples):
                        counter += 1
                        jnt_pos_list.append(action[id][idx+1])
                        robot_s.goto_given_conf(jnt_values=action[id][idx])
                        # robot_s.gen_meshmodel(alpha=0.2).attach_to(base)

                        pred_pos, pred_rotmat = robot_s.fk(jnt_values=action[id][idx])
                        pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos, tgt_rotmat, pred_pos, pred_rotmat)
                        # print(f"Step: {idx}, pos_err: {pos_err} m, rot_err: {rot_err} rad")

                        if pos_err < 0.05:
                            print(f"Goal reached! current traj id: {traj_id}, steps {counter}, pos_err: {pos_err}")
                            success_num += 1
                            break

                        s_pos, _ = robot_s.fk(jnt_values=action[id][idx])
                        e_pos, _ = robot_s.fk(jnt_values=action[id][idx+1])
                        mgm.gen_stick(spos=s_pos, epos=e_pos, radius=.0005, rgb=[0,0,0]).attach_to(base)
                        # print(f"Time: {idx}, sample: {id}")
                if pos_err < 0.05:
                    break
            if pos_err < 0.05:
                break
        print(f"Success rate: {success_num/(traj_id+1)*100}%")
        # plt.figure(figsize=(10, 3 * robot_s.n_dof))
        # for i in range(robot_s.n_dof):
        #     plt.subplot(robot_s.n_dof, 1, i + 1)
        #     plt.plot(np.array(jnt_pos_list[0,i].to('cpu')), label='Position')
        #     plt.ylabel(f'DoF {i}')
        #     plt.legend()
        #     plt.grid(True)
        # plt.xlabel('Time [s]')
        # plt.tight_layout()
        # plt.show()
        
        base.run()


elif config['mode'] == "acc_inference":
    # ----------------- Inference ----------------------
    model_path = os.path.join(current_file_dir, 'diffusion_ckpt_latest.pt')
    agent.load(model_path)
    agent.model.eval()
    agent.model_ema.eval()

    '''capture the image'''
    sys.path.append('/home/lqin/wrs_xinyi/wrs')
    import wrs.visualization.panda.world as wd
    from wrs import wd, rm, mcm
    import wrs.modeling.geometric_model as mgm
    import mp_datagen_ruckig_obstacle as mp_helper
    import copy
    
    # init
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    # inference
    solver = config['inference_solver']
    inference_steps = config['inference_steps']
    visualization = config['visualization']
    
    success_num = 0
    n_samples = 1
    for _ in tqdm(range(config['episode_num'])):
        jnt_pos_list = []
        jnt_vel_list = []
        jnt_acc_list = []
        delta_t = 0.01

        # start_conf, goal_conf = mp_helper.gen_collision_free_start_goal(robot_s)
        # start_conf = robot_s.rand_conf()
        start_conf = np.array([-1.8307296, -1.575878 , -1.6512135, -1.0037161,  0.3407542,
        2.7061534, -2.0410614])
        goal_conf = np.array([ 0.6153231 ,  0.13956377,  2.0086024 , -2.9329627 ,  1.4677436 ,
        1.3641189 , -2.990753  ])
        print(f"Start Conf: {start_conf}, Goal Conf: {goal_conf}")

        START_CONF = copy.deepcopy(start_conf)
        GOAL_CONF = copy.deepcopy(goal_conf)
        tgt_pos, tgt_rotmat = robot_s.fk(jnt_values=goal_conf)
        
        robot_s.goto_given_conf(jnt_values=goal_conf)
        robot_s.gen_meshmodel(alpha=0.2, rgb=[0,1,0]).attach_to(base)
        robot_s.goto_given_conf(jnt_values=start_conf)
        robot_s.gen_meshmodel(alpha=0.2, rgb=[0,0,1]).attach_to(base)

        if n_samples != 1:
            start_conf = np.tile(robot_s.get_jnt_values(), (n_samples, 1))

        update_counter = 0
        condition = torch.zeros((n_samples, config['obs_dim']*config['obs_steps']), device=config['device'])
        if config['normalize']:
            normed_goal_conf = dataset.normalizer['obs']['jnt_pos'].normalize(GOAL_CONF)
            condition[:, robot_s.n_dof:2*robot_s.n_dof] = torch.tensor(normed_goal_conf).to(config['device'])
        else:
            condition[:, robot_s.n_dof:2*robot_s.n_dof] = torch.tensor(GOAL_CONF).to(config['device'])
        prior = torch.zeros((n_samples, config['horizon'], config['action_dim']), device=config['device'])

        for _ in range(inference_steps):
            if n_samples == 1:
                start_conf = robot_s.get_jnt_values()
            elif n_samples > 1 and update_counter != 0:
                start_conf = jnt_pos_list[-1]

            if config['normalize']:
                normed_start_conf = dataset.normalizer['obs']['jnt_pos'].normalize(start_conf)
                condition[:, :robot_s.n_dof] = torch.tensor(normed_start_conf).to(config['device'])
            else:
                condition[:, :robot_s.n_dof] = torch.tensor(start_conf).to(config['device'])
            
            with torch.no_grad():
                # action, _ = agent.sample(prior=prior, n_samples=n_samples, sample_steps=config['sample_steps'], temperature=1.0,
                #                         solver=solver, condition_cfg=condition, w_cfg=7.0, use_ema=True)
                action, _ = agent.sample(prior=prior, n_samples=n_samples, sample_steps=config['sample_steps'], temperature=0.0,
                                        solver=solver, use_ema=True)

            if n_samples == 1:
                # sample actions and unnorm
                jnt_pos_list.append(start_conf)
                jnt_acc_pred = action[0, :,14:21].detach().to('cpu').numpy()
                jnt_vel_pred = action[0, :, 7:14].detach().to('cpu').numpy()
                
                if config['normalize']:
                    jnt_acc_pred = dataset.normalizer['obs']['jnt_acc'].unnormalize(jnt_acc_pred)
                    jnt_vel_pred = dataset.normalizer['obs']['jnt_vel'].unnormalize(jnt_vel_pred)

                jnt_vel_list.append(jnt_vel_pred[0])

                for idx in range(config['action_steps']):
                    # print(update_counter)
                    robot_s.goto_given_conf(jnt_values=jnt_pos_list[-1])
                    pred_pos, pred_rotmat = robot_s.fk(jnt_values=jnt_pos_list[-1])
                    pos_err, rot_err, _ = rm.diff_between_poses(tgt_pos, tgt_rotmat, pred_pos, pred_rotmat)
                    print(f"Step: {update_counter}, pos_err: {pos_err} m, rot_err: {rot_err} rad")

                    jnt_pos_list.append(jnt_pos_list[-1] + delta_t*jnt_vel_list[-1])  # p(t+1)
                    jnt_acc_list.append(jnt_acc_pred[idx]) # a(t)
                    jnt_vel_list.append(jnt_vel_list[-1] + delta_t*jnt_acc_pred[idx]) # v(t+1)

                    if visualization:
                        # robot_s.gen_meshmodel(alpha=0.2).attach_to(base)
                        s_pos, _ = robot_s.fk(jnt_values=jnt_pos_list[-2])
                        e_pos, _ = robot_s.fk(jnt_values=jnt_pos_list[-1])
                        mgm.gen_stick(spos=s_pos, epos=e_pos, radius=.001, rgb=[0,0,0]).attach_to(base)
                    update_counter += 1
                    # print(f"Step: {update_counter}, distance: {np.linalg.norm(robot_s.get_jnt_values() - goal_conf)}")

                    if robot_s.cc.is_collided(obstacle_list=[]):
                        print("Collision detected!")
                        # break

                    if (pos_err < config['max_pos_err']):
                        print("Goal reached!")
                        break

                    if update_counter > config['max_iter']:
                        print("Max iteration reached!")
                        break
                
                if robot_s.cc.is_collided(obstacle_list=[]):
                    print("Collision detected!")
                    # break

                if (pos_err < config['max_pos_err']):
                    print("Goal reached!")
                    break

                if update_counter > config['max_iter']:
                    print("Max iteration reached!")
                    break            
                
                if pos_err < config['max_pos_err'] and not robot_s.cc.is_collided(obstacle_list=[]):
                    success_num += 1
                    print("Success!")
                    break
            else:
                # sample actions and unnorm
                jnt_pos_list.append(start_conf)
                jnt_acc_pred = action[:, :,14:21].detach().to('cpu').numpy()
                jnt_vel_pred = action[:, :, 7:14].detach().to('cpu').numpy()
                
                if config['normalize']:
                    jnt_acc_pred = dataset.normalizer['obs']['jnt_acc'].unnormalize(jnt_acc_pred)
                    jnt_vel_pred = dataset.normalizer['obs']['jnt_vel'].unnormalize(jnt_vel_pred)

                jnt_vel_list.append(jnt_vel_pred[:,0,:])

                for idx in range(config['action_steps']):
                    jnt_pos_list.append(jnt_pos_list[-1] + delta_t*jnt_vel_list[-1])  # p(t+1)
                    jnt_acc_list.append(jnt_acc_pred[:,idx,:]) # a(t)
                    jnt_vel_list.append(jnt_vel_list[-1] + delta_t*jnt_acc_pred[:,idx,:]) # v(t+1)

                    if visualization:
                        for id in range(n_samples):
                            s_pos, _ = robot_s.fk(jnt_values=jnt_pos_list[-2][id])
                            e_pos, _ = robot_s.fk(jnt_values=jnt_pos_list[-1][id])
                            mgm.gen_stick(spos=s_pos, epos=e_pos, radius=.0015, rgb=[0,0,0]).attach_to(base)
                    update_counter += 1
                    print(f"Step: {update_counter}")
                
                if update_counter > config['max_iter']:
                    print("Max iteration reached!")
                    break  

            prior[:,0,7:14] = torch.tensor(dataset.normalizer['obs']['jnt_vel'].normalize(jnt_vel_list[-1]))
            prior[:,0,14:21] = torch.tensor(dataset.normalizer['obs']['jnt_acc'].normalize(jnt_acc_list[-1]))
            
            '''update the condition'''
            condition[:, 2*robot_s.n_dof:3*robot_s.n_dof] = torch.tensor(jnt_vel_list[-1]).to(config['device'])
            condition[:, 3*robot_s.n_dof:4*robot_s.n_dof] = torch.tensor(jnt_acc_list[-1]).to(config['device'])
        
        if visualization:
            # mp_datagen.visualize_anime_diffusion(robot=robot_s, path = jnt_pos_list, 
            #                                      start_conf=START_CONF, goal_conf=GOAL_CONF)
            jnt_pos_array, jnt_vel_arrat, jnt_acc_array = np.array(jnt_pos_list), np.array(jnt_vel_list), np.array(jnt_acc_list)
            np.savez('jnt_info.npz', jnt_pos=jnt_pos_array, jnt_vel=jnt_vel_arrat, jnt_acc=jnt_acc_array)
            mp_helper.plot_details(robot_s, jnt_pos_list, jnt_vel_list, jnt_acc_list)
            base.run()    

    print(f"Success rate: {success_num/config['episode_num']*100}%")
    print('strart conf:', repr(START_CONF))
    print('goal conf:', repr(GOAL_CONF))

else:
    raise ValueError("Illegal mode")