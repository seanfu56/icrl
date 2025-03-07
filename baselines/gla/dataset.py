import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, log_dir):
        
        total_obs_history = np.load(os.path.join(log_dir, 'obs.npy'))
        total_actions = np.load(os.path.join(log_dir, 'actions.npy'))
        total_rewards = np.load(os.path.join(log_dir, 'reward.npy'))

        total_obs_history = total_obs_history[:100000]
        total_actions = total_actions[:100000]
        total_rewards = total_rewards[:100000]

        num_traj = total_obs_history.shape[0] // 1000


        total_obs_history = total_obs_history.reshape(num_traj, -1, total_obs_history.shape[-1])
        total_actions = total_actions.reshape(num_traj, -1, total_actions.shape[-1])
        total_rewards = total_rewards.reshape(num_traj, -1)

        num_gap = 10

        gap_obs_history = total_obs_history[::num_gap]
        gap_actions = total_actions[::num_gap]
        gap_rewards = total_rewards[::num_gap]

        gap_obs_history = gap_obs_history.reshape(-1, gap_obs_history.shape[-1])
        gap_actions = gap_actions.reshape(-1, gap_actions.shape[-1])
        gap_rewards = gap_rewards.reshape(-1, 1)


        self.obs_history = gap_obs_history
        self.actions = gap_actions
        self.rewards = gap_rewards

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        s_t = self.obs_history[:-1]
        a_t = self.actions[:-1]
        r_t = self.rewards[:-1]
        s_t1 = self.obs_history[1:]
        
        a_t1 = self.actions[1:]

        state_projector = np.random.randn()
        action_projector = np.random.randn()

        # print(s_t.shape, a_t.shape, r_t.shape, s_t1.shape, a_t1.shape)

        data = (
            torch.tensor(s_t, dtype=torch.float32),
            torch.tensor(a_t, dtype=torch.float32),
            torch.tensor(r_t, dtype=torch.float32),
            torch.tensor(s_t1, dtype=torch.float32),
        )
        label = torch.tensor(a_t1, dtype=torch.float32)

        return data, label