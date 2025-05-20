from typing import Dict
import torch
import numpy as np
import copy
from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
from cleandiffuser.dataset.dataset_utils import SequenceSampler, MinMaxNormalizer, ImageNormalizer, dict_apply
import obstacle_utils as obstacle_utils


import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.basis.constant as ct
# Initialize 3D visualization environment


class MotionPlanningDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            obs_keys=[], 
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=False,
            normalize=False
        ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=obs_keys)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after)
        
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        if normalize:
            self.normalizer = self.get_normalizer(zarr_path)

    def get_normalizer(self, zarr_path):
        jnt_pos_min_val = np.array([-2.8973, -1.8326, -2.8972, -3.0718, -2.8798,  0.4364, -3.0543])
        jnt_pos_max_val = np.array([ 2.8973,  1.8326,  2.8972, -0.1222,  2.8798,  4.6251,  3.0543])
        jnt_vel_limit = np.asarray([np.pi * 2 / 3] * 7)
        jnt_acc_limit = np.asarray([np.pi] * 7)

        jnt_pos_normalizer = MinMaxNormalizer(np.array([jnt_pos_min_val, jnt_pos_max_val]))
        jnt_vel_normalizer = MinMaxNormalizer(np.array([-jnt_vel_limit, jnt_vel_limit]))
        jnt_acc_normalizer = MinMaxNormalizer(np.array([-jnt_acc_limit, jnt_acc_limit]))
        print('*' * 100)
        print('ATTENTION: Normalizer is used in the dataset.')
        print('*' * 100)
        
        return {
            "obs": {
                "jnt_pos": jnt_pos_normalizer,
                "jnt_vel": jnt_vel_normalizer,
                "jnt_acc": jnt_acc_normalizer
            }
        }

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        if not self.normalizer:
            jnt_pos = sample['jnt_pos']
            jnt_vel = sample['jnt_vel']
            jnt_acc = sample['jnt_acc']
            action = np.concatenate([jnt_pos, jnt_vel, jnt_acc], axis=-1)

            '''condition'''
            condition = np.concatenate([jnt_pos[0], jnt_pos[-1], jnt_vel[0], jnt_acc[0]], axis=-1)
        
        else:
            '''normalize'''
            jnt_pos = self.normalizer['obs']['jnt_pos'].normalize(sample['jnt_pos'])
            jnt_vel = self.normalizer['obs']['jnt_vel'].normalize(sample['jnt_vel'])
            jnt_acc = self.normalizer['obs']['jnt_acc'].normalize(sample['jnt_acc'])
            action = np.concatenate([jnt_pos, jnt_vel, jnt_acc], axis=-1)

            '''condition'''
            condition = np.concatenate([jnt_pos[0], jnt_pos[-1], jnt_vel[0], jnt_acc[0]], axis=-1)
        

        data = {
            'cond': condition,
            'action': action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data


class PosPlanningDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            obs_keys=[], 
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=False,
            normalize=False
        ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=obs_keys)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after)
        
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        if normalize:
            self.normalizer = self.get_normalizer()
        else:
            self.normalizer = None


    def get_normalizer(self):
        jnt_pos_min_val = np.array([-2.8973, -1.8326, -2.8972, -3.0718, -2.8798,  0.4364, -3.0543])
        jnt_pos_max_val = np.array([ 2.8973,  1.8326,  2.8972, -0.1222,  2.8798,  4.6251,  3.0543])
        jnt_pos_normalizer = MinMaxNormalizer(np.array([jnt_pos_min_val, jnt_pos_max_val]))

        print('*' * 100)
        print('ATTENTION: Normalizer is used in the dataset.')
        print('*' * 100)
        
        return {
            "obs": {
                "jnt_pos": jnt_pos_normalizer
            }
        }

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        jnt_pos = sample['jnt_pos']
        goal_conf = sample['goal_conf'][0]
        if self.normalizer:
            jnt_pos = self.normalizer['obs']['jnt_pos'].normalize(sample['jnt_pos'])

        '''condition'''
        condition = np.concatenate([jnt_pos[0], jnt_pos[-1]], axis=-1)
        # condition = np.concatenate([jnt_pos[0], jnt_pos[-1]], axis=-1)
        

        data = {
            'cond': condition,
            'action': jnt_pos
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data

class ObstaclePlanningDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            obs_keys=[], 
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=False,
            normalize=False
        ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=obs_keys)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after)
        
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        if normalize:
            self.normalizer = self.get_normalizer(zarr_path)

    def get_normalizer(self, zarr_path):
        jnt_pos_min_val = np.array([-2.8973, -1.8326, -2.8972, -3.0718, -2.8798,  0.4364, -3.0543])
        jnt_pos_max_val = np.array([ 2.8973,  1.8326,  2.8972, -0.1222,  2.8798,  4.6251,  3.0543])
        jnt_vel_limit = np.asarray([np.pi * 2 / 3] * 7)
        jnt_acc_limit = np.asarray([np.pi] * 7)

        jnt_pos_normalizer = MinMaxNormalizer(np.array([jnt_pos_min_val, jnt_pos_max_val]))
        jnt_vel_normalizer = MinMaxNormalizer(np.array([-jnt_vel_limit, jnt_vel_limit]))
        jnt_acc_normalizer = MinMaxNormalizer(np.array([-jnt_acc_limit, jnt_acc_limit]))
        print('*' * 100)
        print('ATTENTION: Normalizer is used in the dataset.')
        print('*' * 100)
        
        return {
            "obs": {
                "jnt_pos": jnt_pos_normalizer,
                "jnt_vel": jnt_vel_normalizer,
                "jnt_acc": jnt_acc_normalizer
            }
        }

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        if not self.normalizer:
            jnt_pos = sample['jnt_pos']
            jnt_vel = sample['jnt_vel']
            jnt_acc = sample['jnt_acc']
            action = np.concatenate([jnt_pos, jnt_vel, jnt_acc], axis=-1)

            '''condition'''
            condition = np.concatenate([jnt_pos[0], jnt_pos[-1], jnt_vel[0], jnt_acc[0]], axis=-1)
        
        else:
            '''normalize'''
            jnt_pos = self.normalizer['obs']['jnt_pos'].normalize(sample['jnt_pos'])
            jnt_vel = self.normalizer['obs']['jnt_vel'].normalize(sample['jnt_vel'])
            jnt_acc = self.normalizer['obs']['jnt_acc'].normalize(sample['jnt_acc'])
            action = np.concatenate([jnt_pos, jnt_vel, jnt_acc], axis=-1)

            '''generate obstacles'''
            import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
            from wrs import wd, rm, mcm
            robot_s = franka.FrankaResearch3(enable_cc=True)
            base = wd.World(cam_pos=[2, 2, 2], lookat_pos=[0, 0, 0.3])
            mgm.gen_frame().attach_to(base)  # Add reference frame
            obstacle_utils.visualize_space()

            n_samples = np.random.randint(8, 12)
            obstacles = obstacle_utils.generate_grid_obstacles(n_samples=n_samples, generate_noise=False)
            for jnt in sample['jnt_pos']:
                robot_s.goto_given_conf(jnt_values=jnt)
                for obstacle in obstacles:
                    if robot_s.cc.is_collided(obstacle_list=[obstacle]):
                        obstacles.remove(obstacle)
            # print('collision free obstacles:', len(obstacles))

            '''condition'''
            jnt_condition = np.concatenate([jnt_pos[0], jnt_pos[-1], jnt_vel[0], jnt_acc[0]], axis=-1)
            base.close_window()

        data = {
            'cond': {
                'jnt_info': jnt_condition,
                'obstacles': obstacles
            },
            'action': action
        }

        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data


class FixedGoalPlanningDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            obs_keys=[], 
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=False,
            normalize=False
        ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=obs_keys)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after)
        
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        if normalize:
            self.normalizer = self.get_normalizer(zarr_path)

    def get_normalizer(self, zarr_path):
        jnt_pos_min_val = np.array([-2.8973, -1.8326, -2.8972, -3.0718, -2.8798,  0.4364, -3.0543])
        jnt_pos_max_val = np.array([ 2.8973,  1.8326,  2.8972, -0.1222,  2.8798,  4.6251,  3.0543])
        jnt_vel_limit = np.asarray([np.pi * 2 / 3] * 7)
        jnt_acc_limit = np.asarray([np.pi] * 7)

        jnt_pos_normalizer = MinMaxNormalizer(np.array([jnt_pos_min_val, jnt_pos_max_val]))
        jnt_vel_normalizer = MinMaxNormalizer(np.array([-jnt_vel_limit, jnt_vel_limit]))
        jnt_acc_normalizer = MinMaxNormalizer(np.array([-jnt_acc_limit, jnt_acc_limit]))
        print('*' * 100)
        print('ATTENTION: Normalizer is used in the dataset.')
        print('*' * 100)
        
        return {
            "obs": {
                "jnt_pos": jnt_pos_normalizer,
                "jnt_vel": jnt_vel_normalizer,
                "jnt_acc": jnt_acc_normalizer
            }
        }

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        if not self.normalizer:
            jnt_pos = sample['jnt_pos']
            jnt_vel = sample['jnt_vel']
            jnt_acc = sample['jnt_acc']
            action = np.concatenate([jnt_pos, jnt_vel, jnt_acc], axis=-1)

            '''condition'''
            condition = np.concatenate([jnt_pos[0], jnt_pos[-1], jnt_vel[0], jnt_acc[0]], axis=-1)
        
        else:
            '''normalize'''
            jnt_pos = self.normalizer['obs']['jnt_pos'].normalize(sample['jnt_pos'])
            goal_conf = self.normalizer['obs']['jnt_pos'].normalize(sample['goal_conf'][0])
            jnt_vel = self.normalizer['obs']['jnt_vel'].normalize(sample['jnt_vel'])
            jnt_acc = self.normalizer['obs']['jnt_acc'].normalize(sample['jnt_acc'])
            action = np.concatenate([jnt_pos, jnt_vel, jnt_acc], axis=-1)

            '''condition'''
            condition = np.concatenate([jnt_pos[0], goal_conf, jnt_vel[0], jnt_acc[0]], axis=-1)
        

        data = {
            'cond': condition,
            'action': action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data
    

class BSplineDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            obs_keys=[], 
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=False,
            normalize=False
        ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=obs_keys)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after)
        
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        if normalize:
            self.normalizer = self.get_normalizer()
        else:
            self.normalizer = None


    def get_normalizer(self):
        jnt_pos_min_val = np.array([-2.8973, -1.8326, -2.8972, -3.0718, -2.8798,  0.4364, -3.0543])
        jnt_pos_max_val = np.array([ 2.8973,  1.8326,  2.8972, -0.1222,  2.8798,  4.6251,  3.0543])
        jnt_pos_normalizer = MinMaxNormalizer(np.array([jnt_pos_min_val, jnt_pos_max_val]))

        print('*' * 100)
        print('ATTENTION: Normalizer is used in the dataset.')
        print('*' * 100)
        
        return {
            "obs": {
                "jnt_pos": jnt_pos_normalizer
            }
        }

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        control_points = sample['control_points']
        start_conf = control_points[0]
        end_conf = control_points[-1]
        if self.normalizer:
            control_points = self.normalizer['obs']['jnt_pos'].normalize(control_points)
    
        data = {
            'start_conf': start_conf,
            'end_conf': end_conf,
            'control_points': control_points
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data
    
class PolynomialDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            obs_keys=[], 
            poly_coef_range=[],
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=False,
            normalize=False
        ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=obs_keys)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after)
        
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.poly_coef_range = poly_coef_range
        if normalize:
            # raise NotImplementedError('PolynomialDataset does not support normalization.')
            self.normalizer = self.get_normalizer()
        else:
            self.normalizer = None

    def get_normalizer(self):
        jnt_pos_min_val = np.array([-2.8973, -1.8326, -2.8972, -3.0718, -2.8798,  0.4364, -3.0543])
        jnt_pos_max_val = np.array([ 2.8973,  1.8326,  2.8972, -0.1222,  2.8798,  4.6251,  3.0543])
        jnt_pos_normalizer = MinMaxNormalizer(np.array([jnt_pos_min_val, jnt_pos_max_val]))

        poly_coef_range = self.poly_coef_range
        poly_coef_normalizer = MinMaxNormalizer(np.array(poly_coef_range))

        print('*' * 100)
        print('ATTENTION: Normalizer is used in the dataset.')
        print('*' * 100)
        
        return {
            "obs": {
                "jnt_pos": jnt_pos_normalizer
            },
            "action": {
                "poly_coef": poly_coef_normalizer
            }
        }

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        poly_coef = np.concatenate([sample['poly_coef'][:, :4]], axis=1)
        start_conf = sample['start_conf'][0]
        goal_conf = sample['goal_conf'][0]

        if self.normalizer:
            poly_coef = self.normalizer['action']['poly_coef'].normalize(poly_coef)
            # start_conf = self.normalizer['obs']['jnt_pos'].normalize(start_conf)
            # goal_conf = self.normalizer['obs']['jnt_pos'].normalize(goal_conf)
        
        data = {
            'start_conf': start_conf,
            'goal_conf': goal_conf,
            'poly_coef': poly_coef,
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data