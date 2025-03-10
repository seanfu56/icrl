import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from baselines.gla.transformer import GLATransformer
from baselines.gla.data_collection import data_collection
from baselines.gla.dataset import TrajectoryDataset

def train_model(model, dataset, epochs=10, batch_size=4, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # 預設離散動作預測

    print(len(dataloader))

    model.train()
    for epoch in tqdm(range(epochs)):
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

    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.shape[0]             # 2
    hidden_dim = 32
    n_layers = 12
    n_heads = 8
    seq_len = 110000
    proj_dims = {'obs': 16, 'act': 16}
    # 設定每個序列長度 (例如 10)
    log_dir = f'data/gla/{env_name}'
    dataset = TrajectoryDataset(log_dir=log_dir,
                                obs_dim=state_dim,
                                act_dim=action_dim,
                                obs_emb_dim=proj_dims['obs'],
                                act_emb_dim=proj_dims['act'])

    # 模型參數 (依據 CartPole，state_dim=4，action_dim=env.action_space.n)


    model = GLATransformer(proj_dims['obs'], proj_dims['act'], hidden_dim, n_layers, n_heads, seq_len)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    # 執行訓練 (Meta-Training)
    train_model(model, dataset, epochs=2000, batch_size=4, lr=1e-4)

    # 儲存模型
    torch.save(model.state_dict(), f'data/gla/{env_name}/transformer.pt')

