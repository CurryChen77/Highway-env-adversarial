import gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Define the Dueling DQN model
class DuelingDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DuelingDQN, self).__init__()
        self.fc_adv = nn.Linear(input_size, 64)
        self.fc_val = nn.Linear(input_size, 64)
        self.fc_out_adv = nn.Linear(64, output_size)
        self.fc_out_val = nn.Linear(64, 1)

    def forward(self, x):
        adv = self.fc_out_adv(self.fc_adv(x))
        val = self.fc_out_val(self.fc_val(x))
        return val + adv - adv.mean()

# Hyperparameters
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.95
batch_size = 64

# Create highway-env environment
env = gym.make('highway-v0')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# Create Dueling DQN model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DuelingDQN(input_size, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Experience replay buffer
replay_buffer = deque(maxlen=2000)

# Training loop
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            state_batch = torch.tensor(states, dtype=torch.float32).to(device)
            action_batch = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
            reward_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
            next_state_batch = torch.tensor(next_states, dtype=torch.float32).to(device)
            done_batch = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

            q_values = model(state_batch)
            next_q_values = model(next_state_batch)
            next_q_target = reward_batch + gamma * next_q_values.max(1)[0].unsqueeze(1) * (1 - done_batch)

            q_values = q_values.gather(1, action_batch)
            loss = nn.MSELoss()(q_values, next_q_target.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    print(f"Episode: {episode+1}/{num_episodes}, Total Reward: {total_reward}")

# Testing loop
num_test_episodes = 10

for episode in range(num_test_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Test Episode: {episode+1}/{num_test_episodes}, Total Reward: {total_reward}")
