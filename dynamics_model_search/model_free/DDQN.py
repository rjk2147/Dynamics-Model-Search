from collections import deque
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, next_state, reward, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).to(device)
class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        state = Variable(torch.FloatTensor(np.float32(state)), volatile=True)
        # print(state.shape)
        q_value = self.forward(state)
        max_act = q_value.max(1)[1]
        if random.random() > epsilon:
            return max_act
        else:
            return torch.randint_like(max_act, self.num_actions)
        # if random.random() > epsilon:
        #     state = Variable(torch.FloatTensor(np.float32(state)), volatile=True)
        #     q_value = self.forward(state)
        #     action = q_value.max(1)[1].data[0]
        # else:
        #     action = random.randrange(self.num_actions)
        # return action

class DDQN:
    def __init__(self, env, buff_size=100000, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=30000, lr=0.00001,
                 batch_size=32, gamma=0.99):
        self.current_model = CnnDQN(env.observation_space.shape, env.action_space.n).to(device)
        self.target_model = CnnDQN(env.observation_space.shape, env.action_space.n).to(device)
        self.optimizer = optim.Adam(self.current_model.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayBuffer(buff_size)

        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        self.steps = 0
        self.update_freq = 4
        self.target_freq = 10000

        self.update_target()

    def act(self, state):
        epsilon = self.epsilon_by_frame(self.steps)
        action = self.current_model.act(state, epsilon)
        # print(action)
        return action

    def value(self, state, act, next_state):
        v = self.current_model(state)
        return v

    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def update(self, batch_size, updates):
        batch_size = self.batch_size
        if len(self.replay) > batch_size and self.steps % self.update_freq == 0:
            state, action, reward, next_state, done = self.replay.sample(batch_size)

            state = Variable(torch.FloatTensor(np.float32(state)))
            next_state = Variable(torch.FloatTensor(np.float32(next_state)))
            action = Variable(torch.LongTensor(action))
            reward = Variable(torch.FloatTensor(reward))
            done = Variable(torch.FloatTensor(done))

            q_values = self.current_model(state)
            next_q_values = self.current_model(next_state)
            next_q_state_values = self.target_model(next_state)

            # print(q_values.shape)
            # print(action.shape)
            q_value = q_values.gather(1, action).squeeze(1)
            next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)

            loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # if updates % 100 == 0: # works for pong
        if self.steps % self.target_freq  == 0:
            self.update_target()

    def save(self, filename):
        torch.save(self.current_model.state_dict(), filename+'_DDQN_current')
        torch.save(self.target_model.state_dict(), filename+'_DDQN_target')

    def load(self, filename):
        self.current_model.load_state_dict(torch.load(filename + '_DDQN_current'))
        self.target_model.load_state_dict(torch.load(filename + '_DDQN_target'))
