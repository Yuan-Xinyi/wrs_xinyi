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
import json

from torch.optim.lr_scheduler import CosineAnnealingLR
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.utils import report_parameters
from ruckig_dataset import CurveLineDataset
from torch.utils.data import random_split
from generate_complex_trajectory import ComplexTrajectoryGenerator
import helper_functions as hf

def AttachTraj2base(path, radius=0.005):
    for i in range(path.shape[0] - 1):
        spos = path[i]
        epos = path[i + 1]
        stick_sgm = mgm.gen_stick(spos=spos, epos=epos, radius=radius)
        stick_sgm.attach_to(base)


'''if want to test the directory'''
# import os
# print(os.getcwd())
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

'''load the config file'''
current_file_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(os.path.dirname(__file__))
config_file = os.path.join(current_file_dir,'config', 'fr3_mixline_config.yaml')
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

'''dataset loading'''
# dataset_path = os.path.join('/home/lqin', 'zarr_datasets', config['dataset_name'])

# dataset = CurveLineDataset(dataset_path, horizon=config['horizon'], obs_keys=config['obs_keys'], 
#                          normalize=config['normalize'], abs_action=config['abs_action'])
# print('dataset loaded in:', dataset_path)
# if config['mode'] == "train":
#     train_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=config["batch_size"],
#         num_workers=4,
#         shuffle=True,
#         pin_memory=True,
#         persistent_workers=True
#     )

# --------------- Create Diffusion Model -----------------
if config['mode'] == "train":
    set_seed(config['seed'])
assert config['diffusion'] == "ddpm"

from cleandiffuser.nn_condition import IdentityCondition, MLPCondition
from cleandiffuser.nn_diffusion import CloC1d
from cleandiffuser.diffusion.ddpm import DDPM

# --------------- Network Architecture -----------------
nn_diffusion = CloC1d(
                        state_dim=config['state_dim'],     # e.g. 165
                        action_dim=config['action_dim'],   # e.g. 69
                        d_model=512,
                        n_heads=8,
                        depth=6,
                        noise_emb_dim=64,
                        state_mlp_hidden=512,
                        max_tokens=256,         # 至少 >= 2*(state_horizon+action_horizon)
                        emphasis_c=2.0,
                        global_state_idx=None,  # 例如 root pos/orient/vel 的索引
                        timestep_emb_type="positional"
                    ).to(config['device'])


if config['condition'] == "identity":
    nn_condition = IdentityCondition(dropout=0.0).to(config['device'])
    print("Using Identity Condition")
elif config['condition'] == "mlp":
    nn_condition = MLPCondition(
        in_dim=16*3, out_dim=64, hidden_dims=256, dropout=0.0).to(config['device'])
    print("Using MLP Condition")
else:
    nn_condition = None
    print("Using No Condition")


print(f"======================= Parameter Report of Diffusion Model =======================")
report_parameters(nn_diffusion)


import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
robot_s = franka.FrankaResearch3(enable_cc=True)

'''define the robot joint limits'''
if config['normalize']:
    x_max = torch.ones((1, config['horizon'], config['action_dim']), device=config['device']) * +1.0
    x_min = torch.ones((1, config['horizon'], config['action_dim']), device=config['device']) * -1.0
    print('*'*100)
    print("Using Normalized Action Space. the action space is normalized to [-1, 1]")
    print('*'*100)
else:
    jnt_config_range = torch.tensor(robot_s.jnt_ranges, device=config['device'])
    x_max = (jnt_config_range[:,1]).repeat(1, config['horizon'], 1)
    x_min = (jnt_config_range[:,0]).repeat(1, config['horizon'], 1)
    print('*'*50)
    print("Using Absolute Action Space. the action space is absolute joint configuration")
    print('*'*50)

loss_weight = torch.ones((config['horizon'], config['action_dim']))
loss_weight[0, :] = config['action_loss_weight']

agent = DDPM(
    nn_diffusion=nn_diffusion, nn_condition=nn_condition, loss_weight=loss_weight,
    device=config['device'], diffusion_steps=config['diffusion_steps'], x_max=x_max, x_min=x_min,
    optim_params={"lr": config['lr']}, predict_noise=config['predict_noise'])

lr_scheduler = CosineAnnealingLR(agent.optimizer, T_max=config['gradient_steps'])

if config['mode'] == "train":
    # --------------- Data Loading -----------------
    '''prepare the save path'''
    TimeCode = ((datetime.now()).strftime("%m%d_%H%M")).replace(" ", "")
    rootpath = f"{TimeCode}gostraight_h{config['horizon']}_b{config['batch_size']}_norm{config['normalize']}"
    save_path = os.path.join(current_file_dir, 'results', rootpath)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wandb.init(project="curve_line", name=rootpath)

    # ----------------- Training ----------------------
    agent.train()
    n_gradient_step = 0
    diffusion_loss_list = []
    log = {'avg_loss_diffusion': 0.}
    start_time = time.time()
    
    for batch in loop_dataloader(train_loader):
        # get condition
        if config['condition'] == "identity":
            condition = batch['coef_cond'].to(config['device'])
            condition = condition.flatten(start_dim=1) # (batch,14)
        elif config['condition'] == "mlp":
            condition = batch['condition'].to(config['device'])
            condition = condition.flatten(start_dim=1) # (batch,14)
        else:
            condition = None
        action = batch['jnt_pos'].to(config['device']) # (batch,horizon,7)
        
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

def calculate_rot_error(rot1, rot2):
    delta = rm.delta_w_between_rotmat(rot1, rot2)
    return np.linalg.norm(delta)

def cost_fn(q_all, path_points, num_joints, weight_smooth=1e-2, weight_rot_smooth=1e-1):
    q_all = q_all.reshape(len(path_points), num_joints)
    loss = 0.0
    rot_prev = None

    for i, (q, x_desired) in enumerate(zip(q_all, path_points)):
        x, rot = robot_s.fk(jnt_values=q)
        loss += np.linalg.norm(x - x_desired)**2
        if i > 0:
            loss += weight_smooth * np.linalg.norm(q - q_all[i-1])**2
            rot_dist = calculate_rot_error(rot_prev, rot)
            loss += weight_rot_smooth * rot_dist**2
        rot_prev = rot

    return loss

if config['mode'] == "inference":
    model_path = '0000_test_programs/surgery_diff/CleanDiffuser/motion_planner/' \
    'results/0617_1913gostraight_h16_b64_normTrue/diffusion_ckpt_latest.pt'
    agent.load(model_path)
    agent.model.eval()
    agent.model_ema.eval()

    '''capture the image'''
    sys.path.append('/home/lqin/wrs_xinyi/wrs')
    import wrs.visualization.panda.world as wd
    from wrs import wd, rm, mcm
    import wrs.modeling.geometric_model as mgm
    import copy
    
    # init
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)

    # inference
    solver = config['inference_solver']
    inference_steps = config['inference_steps']
    n_samples = 1
    testing_num = 1000
    success = 0
    
    '''diffusion model inference'''
    # for id in tqdm(range(testing_num)):
    #     '''randomly generate a complex trajectory'''
    #     pos_list = []
    #     while len(pos_list) < 32:
    #         init_jnt = robot_s.rand_conf()
    #         pos_init, rotmat_init = robot_s.fk(jnt_values=init_jnt)
    #         scale = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
    #         # scale = np.random.choice([0.1, 0.2, 0.3])
    #         import mp_datagen_curveline as datagen
    #         workspace_points, coeffs = datagen.generate_random_cubic_curve(num_points=32, scale=scale, center=pos_init)
            
    #         pos_list, success_count = datagen.gen_jnt_list_from_pos_list(init_jnt=init_jnt,
    #             pos_list=workspace_points, robot=robot_s, obstacle_list=None, base=base,
    #             max_try_time=5.0, check_collision=True, visualize=False
    #         )

    #     '''inference the trajectory'''
    #     trajectory_window = torch.tensor(workspace_points[:config['horizon']]).to(config['device'])
    #     if config['condition'] == "mlp":
    #         condition = torch.tensor(trajectory_window, device=config['device']).unsqueeze(0).float()
    #         condition = condition.flatten(start_dim=1)
    #     prior = torch.zeros((n_samples, config['horizon'], config['action_dim']), device=config['device'])
    #     if n_samples != 1:
    #         condition = condition.repeat(n_samples, 1)
    #     with torch.no_grad():
    #         action, _ = agent.sample(prior=prior, n_samples=n_samples, sample_steps=config['sample_steps'], temperature=1.0,
    #                                 solver=solver, condition_cfg=condition, w_cfg = 1.0, use_ema=True)
    #     if config['normalize']:
    #         action = dataset.normalizer['obs']['jnt_pos'].unnormalize(np.array(action.to('cpu')))
    #     jnt_seed = action[0, 0, :]
    #     _, rot = robot_s.fk(jnt_values=jnt_seed)
    #     adjusted_jnt = robot_s.ik(tgt_pos=workspace_points[0], tgt_rotmat=rot, seed_jnt_values=jnt_seed)
    #     if adjusted_jnt is None:
    #         print("IK failed at initial position:", workspace_points[0])
        
    #     path = [adjusted_jnt] if adjusted_jnt is not None else []
    #     for _ in workspace_points:
    #         current_pos = np.array(_)
    #         adjusted_jnt = robot_s.ik(tgt_pos=np.array(_), tgt_rotmat=rot, 
    #                         seed_jnt_values=path[-1] if path else jnt_seed)
    #         if adjusted_jnt is not None:
    #             path.append(adjusted_jnt)
    #         else:
    #             print("IK failed at position:", current_pos)
    #             break
    #     if len(path) == 33:
    #         print(f"Trajectory {id} generated successfully.")
    #         success += 1
    
    # print(f"Total {testing_num} trajectories generated, {success} successful.")
    # print(f"Success rate: {success / testing_num * 100:.2f}%")

    '''randomly generate a complex trajectory and single visulization'''
    # pos_list = []
    # traj_length = 32
    # while len(pos_list) < traj_length:
    #     init_jnt = robot_s.rand_conf()
    #     pos_init, rotmat_init = robot_s.fk(jnt_values=init_jnt)
    #     # scale = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
    #     scale = np.random.choice([0.1, 0.2, 0.3])
    #     import mp_datagen_curveline as datagen
    #     workspace_points, coeffs = datagen.generate_random_cubic_curve(num_points=traj_length, scale=scale, center=pos_init)
        
    #     pos_list, success_count = datagen.gen_jnt_list_from_pos_list(init_jnt=init_jnt,
    #         pos_list=workspace_points, robot=robot_s, obstacle_list=None, base=base,
    #         max_try_time=5.0, check_collision=True, visualize=False
    #     )
    # # trajectory_generator = ComplexTrajectoryGenerator(num_segments=1, num_points_per_segment=32, 
    # #                                                   scale_range=(0.1,0.3), center_init=init_pos)
    # # test_trajectory = trajectory_generator.generate()
    # AttachTraj2base(workspace_points, radius=0.0025)

    # robot_s.goto_given_conf(pos_list[0])
    # robot_s.gen_meshmodel(rgb=[0,1,0], alpha=0.2).attach_to(base)
    # robot_s.goto_given_conf(pos_list[-1])
    # robot_s.gen_meshmodel(rgb=[0,1,0], alpha=0.2).attach_to(base)

    # '''inference the trajectory'''
    # visualization = 'dynamic' # 'static' or 'dynamic'

    # trajectory_window = torch.tensor(workspace_points[:config['horizon']]).to(config['device'])
    # if config['condition'] == "mlp":
    #     condition = torch.tensor(trajectory_window, device=config['device']).unsqueeze(0).float()
    #     condition = condition.flatten(start_dim=1)
    # prior = torch.zeros((n_samples, config['horizon'], config['action_dim']), device=config['device'])
    # if n_samples != 1:
    #     condition = condition.repeat(n_samples, 1)
    # with torch.no_grad():
    #     action, _ = agent.sample(prior=prior, n_samples=n_samples, sample_steps=config['sample_steps'], temperature=1.0,
    #                             solver=solver, condition_cfg=condition, w_cfg = 1.0, use_ema=True)
    # if config['normalize']:
    #     action = dataset.normalizer['obs']['jnt_pos'].unnormalize(np.array(action.to('cpu')))
    # jnt_seed = action[0, 0, :]
    # if visualization == 'static':
    #     robot_s.goto_given_conf(jnt_seed)
    #     robot_s.gen_meshmodel(rgb=[1,0,0], alpha=0.2).attach_to(base)
    # _, rot = robot_s.fk(jnt_values=jnt_seed)
    # adjusted_jnt = robot_s.ik(tgt_pos=workspace_points[0], tgt_rotmat=rot, seed_jnt_values=jnt_seed)
    # if adjusted_jnt is not None:
    #     if visualization == 'static':
    #         robot_s.goto_given_conf(adjusted_jnt)
    #         robot_s.gen_meshmodel(rgb=[0,0,1], alpha=0.2).attach_to(base)
    # else:
    #     print("IK failed at initial position:", workspace_points[0])
    
    # path = [adjusted_jnt] if adjusted_jnt is not None else []
    # for _ in workspace_points[1:]:
    #     current_pos = np.array(_)
    #     adjusted_jnt = robot_s.ik(tgt_pos=np.array(_), tgt_rotmat=rot, 
    #                     seed_jnt_values=path[-1] if path else jnt_seed)
    #     if adjusted_jnt is not None:
    #         path.append(adjusted_jnt)
    #         if visualization == 'static':
    #             robot_s.goto_given_conf(adjusted_jnt)
    #             robot_s.gen_meshmodel(rgb=[0,0,1], alpha=0.2).attach_to(base)
    #     else:
    #         print("IK failed at position:", current_pos)
    #         break
    # if len(path) == traj_length:
    #     print(f"Trajectory generated successfully.")
    # else:
    #     print(f"Trajectory generation failed.")
    #     print(f"Generated {len(path)} waypoints, expected {traj_length} waypoints.")
    # import helper_functions as hf
    # if visualization == 'dynamic':
    #     hf.visualize_anime_path(base, robot_s, path)
    # else:
    #     base.run()

    '''
    necessary part 1:
    generate trajectory parellelly
    '''
    # n_samples = 100
    # pos_list = []
    # traj_length = 16
    # while len(pos_list) < 16:
    #     init_jnt = robot_s.rand_conf()
    #     pos_init, rotmat_init = robot_s.fk(jnt_values=init_jnt)
    #     # scale = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
    #     scale = np.random.choice([0.1, 0.2, 0.3])
    #     import mp_datagen_curveline as datagen
    #     workspace_points, coeffs = datagen.generate_random_cubic_curve(num_points=16, scale=scale, center=pos_init)
        
    #     pos_list, success_count = datagen.gen_jnt_list_from_pos_list(init_jnt=init_jnt,
    #         pos_list=workspace_points, robot=robot_s, obstacle_list=None, base=base,
    #         max_try_time=5.0, check_collision=True, visualize=False
    #     )
    # # trajectory_generator = ComplexTrajectoryGenerator(num_segments=1, num_points_per_segment=32, 
    # #                                                   scale_range=(0.1,0.3), center_init=init_pos)
    # # test_trajectory = trajectory_generator.generate()
    # AttachTraj2base(workspace_points, radius=0.0025)

    # robot_s.goto_given_conf(pos_list[0])
    # robot_s.gen_meshmodel(rgb=[0,1,0], alpha=0.2).attach_to(base)
    # robot_s.goto_given_conf(pos_list[-1])
    # robot_s.gen_meshmodel(rgb=[0,1,0], alpha=0.2).attach_to(base)

    '''
    necessary part 2:
    inference the trajectory
    '''
    # visualization = 'none' # 'static' or 'dynamic' or 'none'

    # trajectory_window = torch.tensor(workspace_points[:config['horizon']]).to(config['device'])
    # if config['condition'] == "mlp":
    #     condition = torch.tensor(trajectory_window, device=config['device']).unsqueeze(0).float()
    #     condition = condition.flatten(start_dim=1)
    # prior = torch.zeros((n_samples, config['horizon'], config['action_dim']), device=config['device'])
    # if n_samples != 1:
    #     condition = condition.repeat(n_samples, 1)
    # with torch.no_grad():
    #     action, _ = agent.sample(prior=prior, n_samples=n_samples, sample_steps=config['sample_steps'], temperature=1.0,
    #                             solver=solver, condition_cfg=condition, w_cfg = 1.0, use_ema=True)
    # if config['normalize']:
    #     action = dataset.normalizer['obs']['jnt_pos'].unnormalize(np.array(action.to('cpu')))
    # jnt_seed = action[:, 0, :]

    '''if you need to visualize the seeds'''
    # for jnt in jnt_seed:
    #     robot_s.goto_given_conf(jnt)
    #     robot_s.gen_meshmodel(alpha=0.1).attach_to(base)
    # base.run()

    '''
    trial: pre-evaluate all the seeds
    for each seed, check if the start and end joint configurations are valid.
    If both are valid, then the seed is considered valid.
    This is to avoid the IK failure in the later stage.
    '''
    # valid_seeds = []
    # for seed in jnt_seed:
    #     _, rot = robot_s.fk(jnt_values=seed)
    #     start_jnt = robot_s.ik(tgt_pos=workspace_points[0], tgt_rotmat=rot, seed_jnt_values=seed)
    #     end_jnt = robot_s.ik(tgt_pos=workspace_points[-1], tgt_rotmat=rot, seed_jnt_values=seed)
    #     if start_jnt is not None and end_jnt is not None:
    #         print("Seed is valid, start and end joint configurations are valid.")
    #         valid_seeds.append(seed)
    # print(f"Total {len(valid_seeds)} valid seeds found out of {n_samples} samples.")

    # '''evaluate all the initial seeds'''
    # result_array = np.zeros((n_samples, traj_length, robot_s.n_dof))
    # for id, seed in enumerate(valid_seeds):
    #     _, rot = robot_s.fk(jnt_values=seed)
    #     adjusted_jnt = robot_s.ik(tgt_pos=workspace_points[0], tgt_rotmat=rot, seed_jnt_values=seed)
    #     if adjusted_jnt is None:
    #         print("IK failed for seed: id", id, "at initial position:", workspace_points[0])
    #         continue
    #     result_array[id, 0, :] = adjusted_jnt
    #     for i in range(1, traj_length):
    #         current_pos = np.array(workspace_points[i])
    #         adjusted_jnt = robot_s.ik(tgt_pos=current_pos, tgt_rotmat=rot, 
    #                         seed_jnt_values=result_array[id, i - 1, :])
    #         if adjusted_jnt is not None:
    #             result_array[id, i, :] = adjusted_jnt
    #         else:
    #             print("IK failed for seed: id", id, "at step:", i)
    #             break
    # print(f"Total {len(valid_seeds)} seeds evaluated, {np.sum(np.all(result_array != 0, axis=(1,2)))} successful.")
    # print(f"Success rate: {np.sum(np.all(result_array != 0, axis=(1,2))) / len(valid_seeds) * 100:.2f}%")

    # hf.visualize_anime_path(base, robot_s, result_array[0])

    '''
    Multiple example:
    trial: pre-select the seeds
    In this section, we individually check each seed and generate the trajectory.
    If the trajectory is valid, we save it.
    '''
    n_samples = 1
    testing_num = 1000
    time_list = []
    success = 0
    traj_length = 128

    opt_time_list = []
    opt_success = 0
    visualization = 'dynamic' # 'static' or 'dynamic' or 'none'

    '''optimization solver'''
    q_min = robot_s.jnt_ranges[:, 0]
    q_max = robot_s.jnt_ranges[:, 1]
    bounds = [(q_min[i % robot_s.n_dof], q_max[i % robot_s.n_dof]) for i in range(config['horizon'] * robot_s.n_dof)]
    
    for test_id in range(testing_num):
        pos_list = []
        while len(pos_list) < traj_length:
            init_jnt = robot_s.rand_conf()
            pos_init, rotmat_init = robot_s.fk(jnt_values=init_jnt)
            # scale = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5])
            scale = np.random.choice([0.1, 0.2, 0.3])
            import mp_datagen_curveline as datagen
            workspace_points, coeffs = datagen.generate_random_cubic_curve(num_points=traj_length, scale=scale, center=pos_init)
            
            pos_list, success_count = datagen.gen_jnt_list_from_pos_list(init_jnt=init_jnt,
                pos_list=workspace_points, robot=robot_s, obstacle_list=None, base=base,
                max_try_time=5.0, check_collision=True, visualize=False
            )

        trajectory_window = torch.tensor(workspace_points[:config['horizon']]).to(config['device'])

        '''seed from diffusion model'''
        if testing_num == 1 and visualization == 'dynamic':
            AttachTraj2base(workspace_points, radius=0.0025)
            robot_s.goto_given_conf(pos_list[0])
            robot_s.gen_meshmodel(rgb=[0,1,0], alpha=0.2).attach_to(base)
            robot_s.goto_given_conf(pos_list[-1])
            robot_s.gen_meshmodel(rgb=[0,1,0], alpha=0.2).attach_to(base)

        if config['condition'] == "mlp":
            condition = torch.tensor(trajectory_window, device=config['device']).unsqueeze(0).float()
            condition = condition.flatten(start_dim=1)
        prior = torch.zeros((n_samples, config['horizon'], config['action_dim']), device=config['device'])
        if n_samples != 1:
            condition = condition.repeat(n_samples, 1)
        with torch.no_grad():
            action, _ = agent.sample(prior=prior, n_samples=n_samples, sample_steps=config['sample_steps'], temperature=1.0,
                                    solver=solver, condition_cfg=condition, w_cfg = 1.0, use_ema=True)
        if config['normalize']:
            action = dataset.normalizer['obs']['jnt_pos'].unnormalize(np.array(action.to('cpu')))
        jnt_seed = action[:, 0, :]

        '''randomly generate the seeds'''
        # jnt_seed = []
        # for _ in range(n_samples):
        #     seed = robot_s.rand_conf()
        #     jnt_seed.append(seed)
        # jnt_seed = np.array(jnt_seed)

        tic = time.time()
        for id, seed in enumerate(jnt_seed):
            path = None
            _, rot = robot_s.fk(jnt_values=seed)
            start_jnt = robot_s.ik(tgt_pos=workspace_points[0], tgt_rotmat=rot, seed_jnt_values=seed)
            end_jnt = robot_s.ik(tgt_pos=workspace_points[-1], tgt_rotmat=rot, seed_jnt_values=seed)
            if start_jnt is None or end_jnt is None:
                # print("IK failed for seed: id", id, "at initial or final position:", workspace_points[0], workspace_points[-1])
                continue
            adjusted_jnt = robot_s.ik(tgt_pos=workspace_points[0], tgt_rotmat=rot, seed_jnt_values=seed)
            path = [adjusted_jnt]
            for _ in workspace_points[1:]:
                current_pos = np.array(_)
                adjusted_jnt = robot_s.ik(tgt_pos=current_pos, tgt_rotmat=rot, 
                                seed_jnt_values=adjusted_jnt if adjusted_jnt is not None else seed)
                if adjusted_jnt is not None:
                    path.append(adjusted_jnt)
                else:
                    # print("IK failed for seed: id", id, "at position:", current_pos)
                    break
            if len(path) == traj_length:    
                # print(f"Trajectory generation successful for seed {id}, generated {len(path)} waypoints.")
                break
            else:
                # print(f"Trajectory generation failed for seed {id}, generated {len(path)} waypoints.")
                continue
        toc = time.time()
        # print(f"Time taken to generate trajectory for seed {id}: {(toc - tic)*1000:.2f} ms, {len(path)} waypoints generated.")
        if path is not None and len(path) == traj_length:
            if len(path) == traj_length:
                time_list.append(toc - tic)
                success += 1
                print(f'success count: {success}, test_id: {test_id}, success_id: {id}, time: {(toc - tic)*1000:.2f} ms, waypoints: {len(path)}')
        if visualization == 'dynamic' and testing_num == 1:
            hf.visualize_anime_path(base, robot_s, path)

        '''
        use the optimization method to solve the trajectory
        '''
        # import time
        # from scipy.optimize import minimize
        # start_time = time.time()
        # q_init = np.zeros((config['horizon'], robot_s.n_dof))
        # res = minimize(
        #     cost_fn,
        #     q_init.flatten(),
        #     args=(workspace_points, robot_s.n_dof),
        #     method='L-BFGS-B',
        #     bounds=bounds,
        #     options={'disp': False, 'maxiter': 1000, 'gtol': 1e-4}
        # )
        # end_time = time.time()
        # q_traj = res.x.reshape(config['horizon'], robot_s.n_dof)
        # opt_time_list.append(end_time - start_time)
        # if res.success: opt_success += 1
        # print(f"Optimization took {end_time - start_time:.2f} seconds, success: {res.success}, message: {res.message}")
        # hf.visualize_anime_path(base, robot_s, q_traj)
        
    
    print('='*80)
    time_list = np.array(time_list) * 1000  # convert to milliseconds
    print(f"Inference Mode: Diffusion seed + IK")
    print(f'trajectory length: {traj_length}, n_samples: {n_samples}')
    print(f"Total {testing_num} trajectories generated, {success} successful.")
    print(f"Success rate: {success / testing_num * 100:.2f}%")
    print(f"Average time taken: {np.mean(time_list):.2f} ms, std: {np.std(time_list):.2f} ms, min: {np.min(time_list):.2f} ms, max: {np.max(time_list):.2f} ms")
    print('='*80)

    # print("Now start the optimization process...")
    # print('='*80)
    # print(f"Optimization Mode: L-BFGS-B")
    # print(f"Total {testing_num} trajectories optimized, {opt_success} successful.")
    # print(f"Success rate: {opt_success / testing_num * 100:.2f}%")
    # print(f"Average optimization time taken: {np.mean(opt_time_list):.2f} s, std: {np.std(opt_time_list):.2f} s, min: {np.min(opt_time_list):.2f} s, max: {np.max(opt_time_list):.2f} s")
    # print('='*80)

else:
    raise ValueError("Illegal mode")