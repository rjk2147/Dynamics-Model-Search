from model_based.mpc import MPC, NullAgent
import numpy as np
# from queue import Queue
from collections import deque
from threading import Thread, Lock
import multiprocessing
# from multiprocessing import Process, Lock
import time
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
counter = 0

class NullDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.size = data.size
        self.dim = data.dim
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

class Node:
    def __init__(self, obs, past_obs=None, act=None, r=None, avg_future_r=0):
        self.past = []
        self.obs = obs
        if past_obs is not None:
            self.past.append((past_obs, act, r))
        self.future = dict() # self.future[act] = (new_obs, r+future_r)
        self.avg_future_r = avg_future_r
        self.best_act = None
        self.depth = 0

class State:
    def __init__(self, obs):
        self.obs = obs
        self.Q = 0
        self.acts = []
        self.rs = []
        self.future = []
        self.Qs = []

    def connect(self, act, r, new_state):
        self.rs.append(r)
        self.acts.append(act)
        self.future.append(new_state)

    def get_avg_r(self):
        return np.mean(self.rs)

    def update_Q(self, discount=0.99):
        self.Q = 0
        self.Qs = []
        for i in range(len(self.rs)):
            Q = self.rs[i] + self.future[i].Q * discount
            self.Q += Q
            self.Qs.append(Q)
        self.Q = self.Q / max(len(self.future), 1)
        return self.Q

if torch.cuda.is_available():
    devices = [torch.device("cuda:"+str(i)) for i in range(0, torch.cuda.device_count())]
else:
    devices = [torch.device('cpu')]

class MCTS(MPC):
    def __init__(self, lookahead, env_learner, agent=None, initial_width=2):
        MPC.__init__(self, lookahead, env_learner, agent)
        self.width = initial_width
        self.populate_queue = deque()
        self.start = time.time()
        self.batch_size = 262144
        self.clear()

    def clear(self):
        self.state_list = []

    def add(self, new_obs, state=None, act=None, r=None, depth=0):
        new_state = State(new_obs)
        if state is not None:
            state.connect(act, r, new_state)
        self.state_list.append((new_state, depth))
        return new_state

    def serve_queue(self, q):
        n_skips = 0
        while len(q) > 0:
            obs = []
            states = []
            depths = []
            acts = []
            while len(acts)+self.width <= self.batch_size and len(q) > 0:
                item = q.pop()
                if item is None:
                    return
                state, depth = item
                if depth < self.lookahead:
                    for _ in range(self.width):
                        act = self.agent.act(state.obs[0])
                        acts.append(act)
                        obs.append(state.obs)
                        states.append(state)
                        depths.append(depth)
            if len(acts) == 0:
                continue

            acts = np.array(acts)
            acts_in = torch.from_numpy(acts).float().to(devices[0])
            while len(acts_in.shape) < 3:
                acts_in = acts_in.unsqueeze(1)
            obs_in = (torch.cat([obs[i][0] for i in range(len(obs))]),
                   torch.cat([obs[i][1] for i in range(len(obs))]))

            new_obs = self.env_learner.step_parallel(obs_in=obs_in, action_in=acts_in, state=True, state_in=True)
            rs = self.agent.value(obs[0], acts, new_obs[0].cpu())

            these_new_obs = [(new_obs[0][i], new_obs[1][i].unsqueeze(0)) for i in range(len(acts))]

            for i in range(len(acts)):
                new_state = self.add(these_new_obs[i], states[i], acts[i], rs[i].item(), depth=depths[i]+1)
                q.appendleft((new_state, depths[i]+1))

    def populate(self, obs, depth=0):
        self.populate_queue.appendleft((obs, depth))
        self.serve_queue(self.populate_queue)

    def best_move(self, obs):
        global A, B, C, D, E, Q
        self.clear()
        obs = (torch.from_numpy(obs[0]), obs[1])
        root = self.add(obs)
        self.populate(root)
        self.state_list.reverse()
        for state, depth in self.state_list:
            state.update_Q()
        i = np.argmax(root.Qs)
        best_act = root.acts[i]
        root.best_act = best_act
        root.best_r = root.rs[i]
        return best_act, root
