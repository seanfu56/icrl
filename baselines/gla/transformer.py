import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class GLATransformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, n_layers, n_heads, max_seq_len, proj_dims):
        """
        state_dim: 原始 state 維度
        action_dim: 連續動作的維度
        hidden_dim: Transformer 隱藏層維度
        n_layers: Transformer 層數
        n_heads: 注意力頭數
        max_seq_len: 位置編碼的最大序列長度
        proj_dims: 字典，包含 'obs' 與 'act' 的投影後維度
        """
        super(GLATransformer, self).__init__()
        self.action_dim = action_dim
        
        # 隨機投影層 (data augmentation)；參數固定，不進行訓練
        self.obs_proj = nn.Linear(state_dim, proj_dims['obs'], bias=False)
        self.act_proj = nn.Linear(action_dim, proj_dims['act'], bias=False)
        for param in self.obs_proj.parameters():
            param.requires_grad = False
        for param in self.act_proj.parameters():
            param.requires_grad = False

        # 嵌入層
        self.state_embed = nn.Linear(proj_dims['obs'], hidden_dim)
        self.action_embed = nn.Linear(proj_dims['act'], hidden_dim)
        self.reward_embed = nn.Linear(1, hidden_dim)
        # self.done_embed   = nn.Linear(1, hidden_dim)

        # 可訓練的位置編碼
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

        # Transformer 編碼器
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 預測下一個時間步的動作
        self.out_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, states, actions, rewards, next_states):
        """
        states: Tensor, shape (B, L, state_dim)
        actions: Tensor, shape (B, L, action_dim)  # 連續動作
        rewards: Tensor, shape (B, L)
        dones: Tensor, shape (B, L)
        """
        batch_size, seq_len, _ = states.shape
        states_proj = self.obs_proj(states)  # (B, L, proj_dims['obs'])
        actions_proj = self.act_proj(actions)  # (B, L, proj_dims['act'])
        next_states_proj = self.obs_proj(next_states)  # (B, L, proj_dims['obs'])
        # 嵌入層
        state_tokens  = self.state_embed(states_proj)
        action_tokens = self.action_embed(actions_proj)
        reward_tokens = self.reward_embed(rewards)
        next_states_tokens = self.state_embed(next_states_proj)
        # done_tokens   = self.done_embed(dones.unsqueeze(-1))

        # 位置編碼
        pos_encoding = self.pos_embedding[:, :seq_len, :]  # (1, L, hidden_dim)
        # tokens = state_tokens + action_tokens + reward_tokens + done_tokens + pos_encoding
        
        tokens = state_tokens + action_tokens + reward_tokens + next_states_tokens + pos_encoding

        # Transformer
        transformer_out = self.transformer(tokens)  # (B, L, hidden_dim)

        # **預測所有時間步的下一個動作**
        logits = self.out_head(transformer_out)  # (B, L, action_dim)

        return logits
    

def test_glatransformer_continuous():
    # 超參數
    state_dim = 10
    action_dim = 3  # 連續動作空間
    hidden_dim = 64
    n_layers = 8
    n_heads = 4
    seq_len = 100000
    proj_dims = {'obs': 32, 'act': 16}
    
    # 初始化模型
    model = GLATransformer(state_dim, action_dim, hidden_dim, n_layers, n_heads, seq_len, proj_dims)
    model.eval()  # 設定為推理模式

    # 測試不同的 seq_len
    batch_size = 4
    test_seq_len = 16  # 可更改為 32, 64 測試不同長度

    # 生成隨機輸入
    states = torch.randn(batch_size, test_seq_len, state_dim)  # (B, L, state_dim)
    actions = torch.randn(batch_size, test_seq_len, action_dim)  # (B, L, action_dim) (連續動作)
    rewards = torch.randn(batch_size, test_seq_len)  # (B, L) 獎勵
    dones = torch.randint(0, 2, (batch_size, test_seq_len)).float()  # (B, L) 0 或 1

    # 前向傳播
    with torch.no_grad():
        logits = model(states, actions, rewards, dones)

    print(states.shape, actions.shape, rewards.shape, dones.shape)
    print(logits.shape)

    # 檢查輸出形狀
    assert logits.shape == actions.shape, f"錯誤: 預測輸出形狀錯誤 {logits.shape}, 預期為 {(batch_size, action_dim)}"

    print("✅ 測試通過：模型可以處理連續動作並適應不同 seq_len！")

if __name__ == "__main__":
    test_glatransformer_continuous()