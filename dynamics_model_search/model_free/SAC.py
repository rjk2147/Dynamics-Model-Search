# From https://github.com/pranz24/pytorch-soft-actor-critic

import os
import torch
import random
import numpy as np
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Normal

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        self.action_scale = 1.
        self.action_bias = 0.

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        # self.action_scale = self.action_scale.to(device)
        # self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

class ReplayMemory:
    def __init__(self, state_dim=None, action_dim=None, max_size=int(1e6)):
        self.capacity = max_size
        self.buffer = []
        self.position = 0

    def add(self, state, action, next_state, reward, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, 1.0)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class SAC(object):
    def __init__(self, env, gamma=0.99, tau=0.005, alpha=0.2,
                 target_update_interval=1, automatic_entropy_tuning=False, hidden_size=256, lr=0.0003):

        act_dim = env.action_space.shape[0]
        num_inputs = env.observation_space.shape[0]

        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.replay = ReplayMemory(num_inputs, act_dim)

        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = QNetwork(num_inputs, act_dim, hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(num_inputs, act_dim, hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.alpha = 0
        self.automatic_entropy_tuning = False
        self.policy = DeterministicPolicy(num_inputs, act_dim, hidden_size).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.steps = 0


    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def act(self, obs):
        # if self.steps > 10000:
        #     action = self.select_action(obs, eval=False)
        # else:
        #     action = np.random.uniform(-1, 1, self.act_dim)
        # return action

        # if self.steps > 10000:
        if torch.is_tensor(obs):
            obs = obs.to(self.device)
        else:
            obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        action, _, _ = self.policy.sample(obs)
        # action = self.actor(obs).detach()
        # action += torch.randn_like(action)*self.expl_noise
        # action.clamp(-self.max_action, self.max_action)
        # else:
        #     action = torch.from_numpy(np.random.uniform(-1, 1, (len(obs), self.act_dim))).to(self.device).float()
        return action.detach()

    def value(self, obs, act, new_obs):
        if not torch.is_tensor(obs):
           obs = torch.Tensor(obs).to(self.device)
        if not torch.is_tensor(act):
            act = torch.Tensor(act).to(self.device)
        if len(obs.shape) > 2:
            obs = obs.unsqueeze(1)
        if len(act.shape) > 2:
            act = act.squeeze(1)
        qf1, qf2 = self.critic(obs, act)  # Two Q-functions to mitigate positive bias in the policy improvement step
        min_qf_pi = torch.min(qf1, qf2)
        return min_qf_pi.cpu().detach().numpy()

    def update(self, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.replay.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save(self, filename):
        # print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), filename+'_SAC_actor')
        torch.save(self.critic.state_dict(), filename+'_SAC_critic')

    # Load model parameters
    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename+'_SAC_actor'))
        self.critic.load_state_dict(torch.load(filename+'_SAC_critic'))

