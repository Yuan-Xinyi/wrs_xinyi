from typing import Dict
import torch
import numpy as np
import copy
from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
from cleandiffuser.dataset.dataset_utils import SequenceSampler, MinMaxNormalizer, ImageNormalizer, dict_apply

class MotionPlanningDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            obs_keys=['interp_confs', 'interp_spds'], 
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=False
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
        
        self.normalizer = self.get_normalizer()

    def get_normalizer(self):
        interp_confs_normalizer = MinMaxNormalizer(self.replay_buffer['interp_confs'][:])
        interp_spds_normalizer = MinMaxNormalizer(self.replay_buffer['interp_spds'][:])

        return {
            "obs": {
                "interp_confs": interp_confs_normalizer,
                "interp_spds": interp_spds_normalizer
            }
        }

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # image
        obs_jnt_cfg = sample['interp_confs'][:-1]
        nobs = self.normalizer['obs']['interp_confs'].normalize(obs_jnt_cfg)

        action_jnt_cfg = sample['interp_confs'][1:]
        action_jnt_spd = sample['interp_spds'][1:]
        naction_jnt_cfg = self.normalizer['obs']['interp_confs'].normalize(action_jnt_cfg)
        naction_jnt_spd = self.normalizer['obs']['interp_spds'].normalize(action_jnt_spd)
        naction = np.concatenate([naction_jnt_cfg, naction_jnt_spd], axis=-1)
        action = np.concatenate([action_jnt_cfg, action_jnt_spd], axis=-1)
        
        '''if normalize'''
        # data = {
        #     'obs': nobs, 
        #     'action': naction, 
        #     'trajectory': np.concatenate([naction, nobs], axis=-1)
        # }

        '''if not normalize'''
        data = {
            'obs': obs_jnt_cfg, 
            'action': action, 
            'trajectory': np.concatenate([action, obs_jnt_cfg], axis=-1)
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
            abs_action=False
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
        
        self.normalizer = self.get_normalizer()

    def get_normalizer(self):
        jnt_pos_normalizer = MinMaxNormalizer(self.replay_buffer['jnt_pos'][:])
        jnt_vel_normalizer = MinMaxNormalizer(self.replay_buffer['jnt_vel'][:])
        jnt_acc_normalizer = MinMaxNormalizer(self.replay_buffer['jnt_acc'][:])

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
        # image
        jnt_pos = sample['jnt_pos']
        njnt_pos = self.normalizer['obs']['jnt_pos'].normalize(jnt_pos)
        obstacles = sample['obstacles']

        jnt_vel = sample['jnt_vel'][:]
        jnt_acc = sample['jnt_acc'][:]
        njnt_vel = self.normalizer['obs']['jnt_vel'].normalize(jnt_vel)
        njnt_acc = self.normalizer['obs']['jnt_acc'].normalize(jnt_acc)
        naction = np.concatenate([njnt_vel, njnt_acc], axis=-1)
        action = np.concatenate([jnt_vel, jnt_acc], axis=-1)
        
        '''if normalize'''
        # data = {
        #     'obs': nobs, 
        #     'action': naction, 
        #     'trajectory': np.concatenate([naction, nobs], axis=-1)
        # }

        '''if not normalize'''
        data = {
            'obs': {
                jnt_pos: jnt_pos,
                obstacles: obstacles
            }, 
            'action': action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data