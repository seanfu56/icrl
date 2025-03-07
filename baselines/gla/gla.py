import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import numpy as np

from baselines.gla.transformer import GLATransformer
from baselines.gla.data_collection import data_collection
from baselines.gla.dataset import TrajectoryDataset

def train_model(model, dataset, epochs=10, batch_size=16, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # 預設離散動作預測

    print(len(dataloader))

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data, label in dataloader:
            optimizer.zero_grad()
            # print(data)
            states, actions, rewards, next_states = data
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            label = label.to(device)
            logits = model(states, actions, rewards, next_states)  # logits 形狀: (B, action_dim)
            # 目標取自序列中最後一個動作
            target = label
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(dataloader):.4f}")

# -----------------------------
# 主程式：收集資料、訓練與測試
# -----------------------------
def gla_main():
    # 建立 CartPole 環境

    env_name = 'HalfCheetah-v5'

    env = gym.make(env_name)

    # 收集真實軌跡資料 (例如 50 個 episode)
    num_episodes = 50
    # trajectories = data_collection(env, num_episodes)
    # print(f"Collected {len(trajectories)} trajectories.")

    # 設定每個序列長度 (例如 10)
    log_dir = f'data/gla/{env_name}'
    dataset = TrajectoryDataset(log_dir=log_dir)

    # 模型參數 (依據 CartPole，state_dim=4，action_dim=env.action_space.n)
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.shape[0]             # 2
    hidden_dim = 32
    n_layers = 2
    n_heads = 2
    seq_len = 110000
    proj_dims = {'obs': 16, 'act': 16}

    model = GLATransformer(state_dim, action_dim, hidden_dim, n_layers, n_heads, seq_len, proj_dims)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    # 執行訓練 (Meta-Training)
    train_model(model, dataset, epochs=20, batch_size=16, lr=1e-4)

    # 執行 meta-testing：模型利用 in-context RL 在環境中互動
    print("Meta-testing:")
    # meta_test(model, env, seq_len=seq_len, max_steps=50)

