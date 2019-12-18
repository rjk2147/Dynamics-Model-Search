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
import torch.multiprocessing as mp
# import queue as ctx
# counter = 0
ctx = mp.get_context("spawn")

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
    # print(devices)
else:
    devices = [torch.device('cpu')]

class MCTS(MPC):
    def __init__(self, lookahead, env_learner, agent=None, initial_width=2, with_hidden=False):
        MPC.__init__(self, lookahead-1, env_learner, agent)
        self.width = initial_width
        self.populate_queue = deque()
        self.start = time.time()
        self.batch_size = 262144
        self.clear()
        self.n_proc = initial_width
        self.with_hidden = with_hidden
        self.spawn_processes()

    def spawn_processes(self):
        self.processes = []
        self.q_in = [ctx.Queue() for _ in range(self.n_proc)]
        self.q_out = [ctx.Queue() for _ in range(self.n_proc)]
        for i in range(self.n_proc):
            dev_id = i%len(devices)
            p = ctx.Process(target=self.multi_best_act, args=(dev_id, self.q_in[i], self.q_out[i]))
            self.processes.append(p)
            p.start()

    def clear(self):
        self.state_list = []

    def add(self, new_obs, state=None, act=None, r=None, depth=0):
        new_state = State(new_obs)
        if state is not None:
            state.connect(act, r, new_state)
        self.state_list.append((new_state, depth))
        return new_state

    def serve_queue(self, q):
        while len(q) > 0:
            obs = []
            states = []
            depths = []
            acts = []
            while len(acts)+self.width <= self.batch_size and len(q) > 0:
                item = q.pop()
                state, depth = item
                if depth < self.lookahead:
                    for _ in range(self.width):
                        obs.append(state.obs)
                        states.append(state)
                        depths.append(depth)
            if len(states) == 0:
                continue
            obs_in = (torch.cat([obs[i][0].unsqueeze(0) for i in range(len(obs))]),
                   torch.cat([obs[i][1] for i in range(len(obs))]))
            if self.with_hidden:
                tmp_obs = torch.cat([obs_in[0], obs_in[1].squeeze(1)], -1)
            else:
                tmp_obs = obs_in[0]
            acts_in = self.agent.act(tmp_obs)
            while len(acts_in.shape) < 3:
                acts_in = acts_in.unsqueeze(1)
            new_obs = self.env_learner.step_parallel(obs_in=obs_in, action_in=acts_in, state=True, state_in=True)
            rs = self.agent.value(tmp_obs, acts_in, new_obs[0])
            these_new_obs = [(new_obs[0][i], new_obs[1][i].unsqueeze(0)) for i in range(len(states))]
            for i in range(len(states)):
                new_state = self.add(these_new_obs[i], states[i], acts_in[i], rs[i].item(), depth=depths[i]+1)
                q.appendleft((new_state, depths[i]+1))

    def populate(self, obs, depth=0):
        self.populate_queue.appendleft((obs, depth))
        self.serve_queue(self.populate_queue)

    def multi_best_act(self, dev_id, q_in, q_out):
        global A, B, C, D, E, Q
        self.device = devices[dev_id]
        print(self.device)
        self.env_learner.to(self.device)
        self.agent.to(self.device)
        while True:
            item = q_in.get()
            if item is None: return
            obs, agent_state, model_state = item
            self.agent.load_dict(agent_state)
            self.env_learner.load_dict(model_state)
            # new_agent_state = self.agent.save_dict()
            # print(new_agent_state[0]['l1.weight'])
            obs = (obs[0].to(self.device), obs[1].to(self.device))
            self.clear()
            root = self.add(obs)
            self.populate(root)
            self.state_list.reverse()
            for state, depth in self.state_list:
                state.update_Q()
            i = np.argmax(root.Qs)
            best_act = root.acts[i]
            root.best_act = best_act
            root.best_r = root.rs[i]
            q_out.put((root.acts[i], root.rs[i], root.Qs[i]))

    def exit(self):
        for q_in in self.q_in:
            q_in.put(None)

    def best_move(self, obs):
        obs = (torch.from_numpy(obs[0]).cpu(), obs[1].cpu())
        agent_state = self.agent.save_dict()
        model_state = self.env_learner.save_dict()
        # print('----------------------------------------------------------------------')
        # print(agent_state[0]['l1.weight'])
        root = State(obs)
        root.best_act = None
        root.best_r = None
        best_q = None
        for q_in in self.q_in:
            q_in.put((obs, agent_state, model_state))
        for q_out in self.q_out:
            new_act, new_r, new_q = q_out.get()
            if best_q is None or new_q > best_q:
                best_q = new_q
                root.best_act = new_act
                root.best_r = new_r
        return root.best_act.cpu().data.numpy().flatten(), root
