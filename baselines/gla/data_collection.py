import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

class TrainingLoggerCallback(BaseCallback):
    def __init__(self, log_dir="data/gla/default", verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.obs = []
        self.actions = []
        self.rewards = []

    def _on_step(self) -> bool:
        # print(self.locals)
        self.obs.append(self.locals["obs_tensor"].squeeze().cpu().numpy())
        self.actions.append(np.squeeze(self.locals['actions']))
        self.rewards.append(self.locals['rewards'])

        return True

    def _on_training_end(self) -> None:
        rewards_array = np.array(self.rewards, dtype=np.float32)
        obs_array = np.array(self.obs, dtype=np.float32)
        actions_array = np.array(self.actions, dtype=np.float32)
        np.save(os.path.join(self.log_dir, 'reward.npy'), rewards_array)
        np.save(os.path.join(self.log_dir, 'obs.npy'), obs_array)
        np.save(os.path.join(self.log_dir, 'actions.npy'), actions_array)
        print(f"Training history saved to {self.log_dir}")

def data_collection(env_name):
    env = gym.make(env_name)
    model = PPO("MlpPolicy", env, verbose=1)
    
    callback = TrainingLoggerCallback(log_dir=f"data/gla/{env_name}")
    
    model.learn(total_timesteps=100000, callback=callback)
    model.save("ppo_cartpole")

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == '__main__':
    data_collection("HalfCheetah-v5")