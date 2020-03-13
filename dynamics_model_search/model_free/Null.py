import torch
import numpy as np
class NullReplay(object):
    def __init__(self, state_dim=None, action_dim=None, max_size=int(1e6)):
        pass

    def add(self, state, action, next_state, reward, done):
        pass

    def __len__(self):
        return 0

class NullAgent:
    def __init__(self, env):
        print('Null agent chosen')
        # Continuous Action Space
        try:
            self.act_mul_const = env.action_space.high
            self.act_dim = env.action_space.shape[0]
            self.action_space = env.action_space

        # Discrete Action Space
        except:
            self.act_dim = env.action_space.n
        self.steps = 0
        self.replay = NullReplay()

    def act(self, obs):
        if torch.is_tensor(obs):
            action = torch.randn((obs.shape[0], self.act_dim))
            return action.to(obs.device)
        else:
            action = torch.randn((self.act_dim, ))
            return action

        # return np.random.normal(0, 1, self.act_dim).clip(-1, 1)

    def value(self, obs, act, new_obs):
        r = new_obs[:,0].cpu().detach().numpy()
        return np.zeros_like(r)

    def save(self, filename):
        pass
