import gym
import numpy as np
from gym import spaces
import copy

class NullWrapper(gym.Env):
    def __init__(self, env, env_len=None, env_learner=None, loop=None):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

class StateWrapper(gym.Env):
    def __init__(self, env, env_len=None, env_learner=None, loop=None):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        obs = self.env.reset()
        state = copy.deepcopy(self.env)
        return (obs, state)

    def step(self, action):
        obs, r, done, info =  self.env.step(action)
        state = copy.deepcopy(self.env)
        return (obs, state), r, done, info

class RealerWalkerWrapper(gym.Env):
    def __init__(self, ant_env, ep_len=100):
        self.env = ant_env
        # obs_scale = 5
        obs_scale = 1
        act_scale = 1
        self.action_space = self.env.action_space
        # self.action_space.high *= act_scale
        # self.action_space.low *= act_scale

        self.front = 3 # first 3 need to be skipped as they are angle to target, 2nd 3 (totaling 6) account for velocity
        self.back = 4 # cutting out the feet contacts
        obs_ones = obs_scale*np.ones(shape=(self.env.observation_space.shape[0]-self.front-self.back,))
        self.observation_space = spaces.Box(high=self.env.observation_space.high[self.front:-self.back],
                                            low=self.env.observation_space.low[self.front:-self.back])
        # self.observation_space = self.env.observation_space.high
        print('State Dim: '+str(obs_ones.shape[0]))
        # State Summary (dim=25):
        # state[0] = vx
        # state[1] = vy
        # state[2] = vz
        # state[3] = roll
        # state[4] = pitch
        # state[5 to -4] = Joint relative positions
        #    even elements [0::2] position, scaled to -1..+1 between limits
        #    odd elements  [1::2] angular speed, scaled to show -1..+1
        # state[-4 to -1] = feet contacts
        self.timestep = 0
        self.max_time = ep_len

    def reset(self):
        obs = self.env.reset()
        # Could clip to +/- 5 since thats what they do in pybullet_envs robot_locomotors.py
        # obs =  np.clip(obs[self.front:-self.back], -5, +5)
        obs = obs[self.front:-self.back]
        self.timestep = 0
        self.ep_rew = 0
        return obs

    def step(self, action):
        new_obs, _, done, info = self.env.step(action)
        # r = (new_obs[3]/0.3)/60 # x velocity
        # r = new_obs[4] # y velocity
        r = new_obs[3]

        # Could clip to +/- 5 since thats what they do in pybullet_envs robot_locomotors.py
        # obs =  np.clip(obs[self.front:-self.back], -5, +5)
        new_obs = new_obs[self.front:-self.back]
        self.timestep += 1
        done = self.timestep >= self.max_time
        self.ep_rew += r
        info = {}
        if done:
            info['episode'] = {}
            info['episode']['r'] = self.ep_rew
            info['episode']['l'] = self.timestep
        return new_obs, r, done, info

    def reset_raw(self):
        obs = self.env.reset()
        return obs
    def step_raw(self, action):
        new_obs, r, done, info = self.env.step(action)
        return new_obs, r, done, info
    def render(self, mode='human'):
        return self.env.render(mode)
