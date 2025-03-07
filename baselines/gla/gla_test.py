import torch
import gymnasium as gym
import numpy as np
from baselines.gla.transformer import GLATransformer

def meta_test(env_name, test_env_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = 'HalfCheetah-v5'

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.shape[0]             # 2
    hidden_dim = 32
    n_layers = 2
    n_heads = 2
    seq_len = 110000
    proj_dims = {'obs': 16, 'act': 16}

    model = GLATransformer(proj_dims['obs'], proj_dims['act'], hidden_dim, n_layers, n_heads, seq_len)

    state_projector = np.random.randn(state_dim, proj_dims['obs'])
    action_projector = np.random.randn(action_dim, proj_dims['act'])

    inv_action_projector = np.linalg.pinv(action_projector)

    model.load_state_dict(torch.load(f"data/gla/{env_name}/transformer.pt", weights_only=True))
    model.to(device)

    with torch.no_grad():
        for i in range(10):
            total_reward = 0
            state_list = []
            action_list = []
            reward_list = []
            nxt_state_list = []

            state, _ = env.reset()
            action = env.action_space.sample()
            nxt_state, reward, _, _, _ = env.step(action)

            state_list.append(torch.tensor(np.tanh(np.dot(state, state_projector)), dtype=torch.float32))
            action_list.append(torch.tensor(np.tanh(np.dot(action, action_projector)), dtype=torch.float32))
            reward_list.append(torch.tensor(np.array([reward]), dtype=torch.float32))
            nxt_state_list.append(torch.tensor(np.tanh(np.dot(nxt_state, state_projector)), dtype=torch.float32))


            for j in range(1000):

                # print(
                #     torch.stack(state_list).unsqueeze(0).to(device).shape,
                #     torch.stack(action_list).unsqueeze(0).to(device).shape,
                #     torch.stack(reward_list).unsqueeze(0).to(device).shape,
                #     torch.stack(nxt_state_list).unsqueeze(0).to(device).shape           
                # )

                action = model(
                    torch.stack(state_list).unsqueeze(0).to(device),
                    torch.stack(action_list).unsqueeze(0).to(device),
                    torch.stack(reward_list).unsqueeze(0).to(device),
                    torch.stack(nxt_state_list).unsqueeze(0).to(device)
                )
                action = action.squeeze().cpu().numpy()
                real_action = np.dot(np.arctanh(action), inv_action_projector)
                if len(real_action.shape) == 2:
                    real_action = real_action[-1]
                print(real_action)
                state = nxt_state
                nxt_state, reward, done, _, _ = env.step(real_action)
                total_reward += reward

                state_list.append(torch.tensor(np.tanh(np.dot(state, state_projector)), dtype=torch.float32))
                action_list.append(torch.tensor(np.tanh(np.dot(real_action, action_projector)), dtype=torch.float32))
                reward_list.append(torch.tensor(np.array([reward]), dtype=torch.float32))
                nxt_state_list.append(torch.tensor(np.tanh(np.dot(nxt_state, state_projector)), dtype=torch.float32))

            print(f"Episode {i+1} Reward: {total_reward}")

if __name__ == '__main__':
    meta_test("HalfCheetah-v5", "HalfCheetah-v5")