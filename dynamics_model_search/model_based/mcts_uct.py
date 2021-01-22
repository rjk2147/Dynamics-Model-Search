from dynamics_model_search.model_based.mpc import MPC, NullAgent
from dynamics_model_search.model_based.cem import CEM

import numpy as np
from collections import deque
import heapq
import time
import torch

class State:
    def __init__(self, obs, p=1):
        self.obs = obs
        self.Q = 0
        self.acts = []
        self.rs = []
        self.future = []
        self.Qs = []
        self.p = p
        self.updated = False

    # # for heapsort
    def __lt__(self,other):
        # using greater than instead of less than because heap sort returns lowest value and we want highest Q
        return (self.Q > other.Q)

    def connect(self, act, r, new_state):
        self.rs.append(r)
        self.acts.append(act)
        self.future.append(new_state)

    def exit(self):
        return

    def update_Q(self, discount=0.99):
        self.updated = True
        if len(self.future) == 0:
            return self.Q
        self.Qs = []
        for i in range(len(self.future)):
            assert self.future[i].updated == True
            Q = self.rs[i] + self.future[i].Q * discount
            self.Qs.append(Q)

        # self.Q = sum(self.Qs) / len(self.Qs)
        self.Q = self.p * max(self.Qs)
        return self.Q

if torch.cuda.is_available():
    devices = [torch.device("cuda:"+str(i)) for i in range(0, torch.cuda.device_count())]
else:
    devices = [torch.device('cpu')]

class MCTS(MPC):
    def __init__(self, lookahead, dynamics_model, agent=None, initial_width=2, nodes=2048, cross_entropy=False):
        MPC.__init__(self, lookahead, dynamics_model, agent)
        self.width = initial_width
        # self.populate_queue = deque()
        self.populate_queue = []
        # self.populate_queue = heapq()
        self.start = time.time()
        # self.batch_size = 262144
        self.batch_size = 256
        self.max_tree = nodes
        self.clear()

    def clear(self):
        self.state_list = []
        self.populate_queue = []

    def add(self, new_obs, state=None, act=None, V=0, r=0, p=1, depth=0):
        new_state = State(new_obs, p)
        new_state.Q = V * p
        if state is not None:
            # reward is a function s and a which is the first element of s'
            state.connect(act, r, new_state)
        self.state_list.append((new_state, depth))
        return new_state

    def serve_queue(self, q):
        n_samples = 100

        i_d = 0
        while len(q) > 0 and len(self.state_list) < self.max_tree:
            obs = []
            states = []
            depths = []

            while len(obs)+self.width <= self.batch_size and len(q) > 0:
                # item = q.pop()
                item = heapq.heappop(q)
                state, depth = item
                if depth < self.lookahead:
                    for _ in range(self.width):
                        obs.append(state.obs)
                        states.append(state)
                        depths.append(depth)
            i_d += 1

            if len(states) == 0:
                continue
            obs_in = (torch.cat([obs[i][0].unsqueeze(0) for i in range(len(obs))]).unsqueeze(1),
                   [obs[i][1] for i in range(len(obs))])

            acts_in = self.agent.act(obs_in[0].squeeze(1)).unsqueeze(1)
            new_obs_mean, new_h, new_obs_sd = self.dynamics_model.step(obs_in=obs_in, action_in=acts_in, state=True,
                                                        state_in=True,certainty=True)

            p = torch.mean(torch.exp(-new_obs_sd), -1).detach().cpu().numpy()
            N = torch.distributions.normal.Normal(new_obs_mean, new_obs_sd)
            new_s = N.sample((n_samples,))
            probs = torch.exp(torch.sum(N.log_prob(new_s), -1))

            new_s = new_s.view(-1, new_obs_mean.shape[-1]) # Merging components
            V = self.agent.value(new_s).view(n_samples, new_obs_mean.shape[0]) # V(s')
            EV = (torch.sum(probs*V, 0)/torch.sum(probs, 0)).cpu().detach().numpy()
            r = new_obs_mean[:,0].cpu().numpy()

            these_new_obs = [(new_obs_mean[i], new_h[i].unsqueeze(0)) for i in range(len(states))]
            for i in range(len(states)):
                new_state = self.add(these_new_obs[i], states[i], acts_in[i], EV[i], r[i], p[i], depths[i]+1)
                heapq.heappush(q, (new_state, depths[i]+1))

    def populate(self, obs, depth=0):
        heapq.heappush(self.populate_queue, (obs, depth))
        self.serve_queue(self.populate_queue)

    def best_move(self, obs):
        self.clear()
        obs = (torch.from_numpy(obs[0]).to(devices[0]).float(), obs[1].to(devices[0]))
        root = self.add(obs)

        self.populate(root)
        self.state_list.reverse()
        for state, depth in self.state_list:
            state.update_Q()
        i = np.argmax(root.Qs)
        best_act = root.acts[i]
        root.best_act = best_act
        root.best_r = root.rs[i]
        return best_act, root.best_r, self.state_list, root.future[i].obs[1]

    def exit(self):
        self.clear()
