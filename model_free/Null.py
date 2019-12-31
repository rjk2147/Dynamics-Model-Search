import torch
class NullReplay(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        pass

    def add(self, state, action, next_state, reward, done):
        pass

class NullAgent:
    def __init__(self, state_dim, act_dim):
        print('Null agent chosen')
        self.act_dim = act_dim

    def act(self, obs):
        action = torch.randn((obs.shape[0], self.act_dim))
        return action

    def value(self, obs, act, new_obs):
        r = new_obs[:,0].cpu().detach().numpy()
        return torch.zeros_like(r)
