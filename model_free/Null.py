import torch
import numpy as np
class NullReplay(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        pass

    def add(self, state, action, next_state, reward, done):
        pass

    def __len__(self):
        return 0

class NullAgent:
    def __init__(self, state_dim, act_dim):
        print('Null agent chosen')
        self.act_dim = act_dim
        self.steps = 0

    def act(self, obs):
        action = torch.randn((obs.shape[0], self.act_dim))
        return action.to(obs.device)

    def value(self, obs, act, new_obs):
        r = new_obs[:,0].cpu().detach().numpy()
        return np.zeros_like(r)

    def save(self, filename):
        pass
