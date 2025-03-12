from typing import Dict
import torch
import numpy as np
import copy
from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.replay_buffer import ReplayBuffer
from cleandiffuser.dataset.dataset_utils import SequenceSampler, MinMaxNormalizer, ImageNormalizer, dict_apply

class DrillHoleImageDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            obs_keys=['img', 'state', 'action'], 
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
        agent_pos_normalizer = MinMaxNormalizer(self.replay_buffer['state'][...,:2])
        image_normalizer = ImageNormalizer()
        action_normalizer = MinMaxNormalizer(self.replay_buffer['action'][:])
        
        return {
            "obs": {
                "image": image_normalizer,
                "agent_pos": agent_pos_normalizer
            },
            "action": action_normalizer
        }

    def __str__(self) -> str:
        return f"Keys: {self.replay_buffer.keys()} Steps: {self.replay_buffer.n_steps} Episodes: {self.replay_buffer.n_episodes}"
    
    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # image
        image = np.moveaxis(sample['img'], -1, 1) / 255
        image = self.normalizer['obs']['image'].normalize(image)
        
        # agent_pos
        agent_pos = sample['state'][:,:2].astype(np.float32)  # (T, 2)
        agent_pos = self.normalizer['obs']['agent_pos'].normalize(agent_pos)
        
        # action
        action = sample['action'].astype(np.float32)  # (T, 2)
        action = self.normalizer['action'].normalize(action)
        
        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': action,  # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.tensor)
        return torch_data