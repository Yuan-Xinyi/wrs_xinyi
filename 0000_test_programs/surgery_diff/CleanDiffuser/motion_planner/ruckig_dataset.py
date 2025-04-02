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
            obs_keys=['jnt_pos', 'jnt_vel', 'jnt_acc'], 
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
        jnt_p_normalizer = MinMaxNormalizer(self.replay_buffer['jnt_pos'])
        jnt_v_normalizer = MinMaxNormalizer(self.replay_buffer['jnt_vel'])
        jnt_a_normalizer = MinMaxNormalizer(self.replay_buffer['jnt_acc'])

        return {
            "obs": jnt_p_normalizer,
            "action": {
                "jnt_vel": jnt_v_normalizer,
                "jnt_acc": jnt_a_normalizer
            }
        }

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # image
        jnt_pos = sample['jnt_pos']
        njnt_pos = self.normalizer['obs'].normalize(jnt_pos)

        jnt_vel = sample['jnt_vel']
        jnt_acc = sample['jnt_acc']
        njnt_vel = self.normalizer['action']['jnt_vel'].normalize(jnt_vel)
        njnt_acc = self.normalizer['action']['jnt_acc'].normalize(jnt_acc)
        naction = np.concatenate([njnt_vel, njnt_acc], axis=-1)
        action = np.concatenate([jnt_vel, jnt_acc], axis=-1)

        '''if not normalize'''
        data = {
            'obs': jnt_pos, 
            'action': action,
            'jnt_vel': jnt_vel,
            'jnt_acc': jnt_acc,
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
        
        # self.normalizer = self.get_normalizer()

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
        jnt_vel = sample['jnt_vel'][:]
        jnt_acc = sample['jnt_acc'][:]
        action = np.concatenate([jnt_vel, jnt_acc], axis=-1)

        '''condition'''
        condition = np.concatenate([sample['jnt_pos'][0], sample['jnt_pos'][-1], 
                                    sample['obstacles'][0]], axis=-1)
        

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