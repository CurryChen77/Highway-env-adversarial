import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def add(self, state, action, next_state, reward, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones)


# 定义SAC的神经网络架构
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# SAC算法的主要代码
class SACAgent:
    def __init__(self, state_dim, action_dim, discount=0.99, tau=0.005, alpha=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.alpha = alpha

        # 创建Actor、两个Critic和Target网络
        self.actor = QNetwork(state_dim, action_dim)
        self.critic1 = QNetwork(state_dim + action_dim, 1)
        self.critic2 = QNetwork(state_dim + action_dim, 1)
        self.target_critic1 = QNetwork(state_dim + action_dim, 1)
        self.target_critic2 = QNetwork(state_dim + action_dim, 1)

        # 使用均方误差作为Critic的损失函数
        self.critic_criterion = nn.MSELoss()

        # 创建优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.0003)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.0003)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).detach().numpy().flatten()

    def train(self, replay_buffer, batch_size=64):
        # 从经验回放缓冲区中随机采样一批数据
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = replay_buffer.sample(batch_size)


        # 转换为张量
        state_batch = torch.FloatTensor(state_batch).reshape((batch_size, -1))  # [64, 25]
        action_batch = torch.FloatTensor(action_batch)  # [64, 1]
        next_state_batch = torch.FloatTensor(next_state_batch).reshape((batch_size, -1))  # [64, 25]
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(-1)  # [64, 1]
        done_batch = torch.FloatTensor(np.float32(done_batch)).unsqueeze(-1)  # [64, 1]

        # 计算Target Q值
        with torch.no_grad():
            next_action = self.actor(next_state_batch)
            # temp = torch.cat((next_state_batch, next_action), 1)
            target_q1 = self.target_critic1(torch.cat((next_state_batch, next_action), 1))
            target_q2 = self.target_critic2(torch.cat((next_state_batch, next_action), 1))
            target_q = torch.min(target_q1, target_q2)
            target_q = reward_batch + (1 - done_batch) * self.discount * target_q

        # 更新Critic网络
        current_q1 = self.critic1(torch.cat((state_batch, action_batch), 1))
        current_q2 = self.critic2(torch.cat((state_batch, action_batch), 1))
        critic1_loss = self.critic_criterion(current_q1, target_q)
        critic2_loss = self.critic_criterion(current_q2, target_q)
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # 更新Actor网络
        sampled_actions = self.actor(state_batch)
        actor_loss = -self.critic1(torch.cat((state_batch, sampled_actions), 1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新Target网络
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, model_name):
        os.makedirs(f"./BV_model/{model_name}", exist_ok=True)
        # 保存策略网络和Q网络的模型参数
        torch.save(self.actor.state_dict(), f"./BV_model/{model_name}/SAC-actor.pth")
        torch.save(self.critic1.state_dict(), f"./BV_model/{model_name}/SAC-critic1.pth")
        torch.save(self.critic2.state_dict(), f"./BV_model/{model_name}/SAC-critic2.pth")
        print("Successfully save the model")

    def load(self, model_name):
        # 创建SAC代理并加载策略网络和Q网络的模型参数
        self.actor.load_state_dict(torch.load(f"./BV_model/{model_name}/SAC-actor.pth"))
        self.critic1.load_state_dict(torch.load(f"./BV_model/{model_name}/SAC-critic1.pth"))
        self.critic2.load_state_dict(torch.load(f"./BV_model/{model_name}/SAC-critic2.pth"))
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        print("--------Successfully load the model--------")