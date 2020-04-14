from dynamics_model_search.model_based.mpc import MPC, NullAgent
from dynamics_model_search.model_based.cem import CEM

import numpy as np
from collections import deque
import heapq
import time
import torch

class State:
    def __init__(self, obs):
        self.obs = obs
        self.Q = 0
        self.acts = []
        self.rs = []
        self.future = []
        self.Qs = []

    # for heapsort
    def __lt__(self,other):
        # using greater than instead of less than because heap sort returns lowest value and we want highest Q
        return (self.Q > other.Q)

    def connect(self, act, r, new_state):
        self.rs.append(r)
        self.acts.append(act)
        self.future.append(new_state)

    def exit(self):
        return

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
    def __init__(self, lookahead, dynamics_model, agent=None, initial_width=2, cross_entropy=False):
        MPC.__init__(self, lookahead, dynamics_model, agent)
        self.width = initial_width
        # self.populate_queue = deque()
        self.populate_queue = []
        # self.populate_queue = heapq()
        self.start = time.time()
        # self.batch_size = 262144
        self.batch_size = 256
        self.max_tree = 2048
        self.memory_buffer = torch.zeros([1])
        self.epsilon = 0.1
        self.ep_min_val = float('inf')
        self.clear()

    def clear(self):
        self.state_list = []
        self.populate_queue = []

    def add(self, new_obs, state=None, act=None, r=0, depth=0):
        new_state = State(new_obs)
        new_state.Q = r
        if state is not None:
            state.connect(act, r, new_state)
        self.state_list.append((new_state, depth))
        return new_state

    def serve_queue(self, q):
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
            # print(i_d)
            # print(len(self.state_list))
            # print(len(obs))
            # print('')
            if len(states) == 0:
                continue
            obs_in = (torch.cat([obs[i][0].unsqueeze(0) for i in range(len(obs))]).unsqueeze(1),
                   [obs[i][1] for i in range(len(obs))])
            acts_in = self.agent.act(obs_in[0].squeeze(1)).unsqueeze(1)
            tmp_obs, tmp_h, uncertainty = self.dynamics_model.step_parallel(obs_in=obs_in, action_in=acts_in, state=True,
                                                        state_in=True,certainty=True)
            tmp_obs = self.replace_obs(tmp_obs, uncertainty)

            new_obs = (tmp_obs, tmp_h)
            rs = self.agent.value(obs_in[0].squeeze(1), acts_in.squeeze(1), new_obs[0])

            # Discounting the reward by the uncertainty
            rs *= torch.mean(uncertainty, -1).unsqueeze(-1).detach().cpu().numpy()

            these_new_obs = [(new_obs[0][i], new_obs[1][i].unsqueeze(0)) for i in range(len(states))]
            for i in range(len(states)):
                new_state = self.add(these_new_obs[i], states[i], acts_in[i], rs[i].item(), depth=depths[i]+1)
                # q.appendleft((new_state, depths[i]+1))
                heapq.heappush(q, (new_state, depths[i]+1))


    def replace_obs(self, obs, uncertainty):
        for i in range(obs.shape[0]):
            temp_obs_norm = torch.norm(obs[i]).expand(self.memory_buffer.shape[0], 1)
            mem_norm = torch.unsqueeze(torch.norm(self.memory_buffer, dim = 1), dim = 1)
            to_div = torch.cat([temp_obs_norm, mem_norm], dim = 1).max(dim = 1)[0]
            
            diffs = torch.norm((self.memory_buffer - obs[i]) / uncertainty[i], dim = 1)/to_div
            #normalize
            min_val, min_ind = diffs.min(dim=0)
            if min_val < self.epsilon:
                obs[i] = self.memory_buffer[min_ind]
            if min_val < self.ep_min_val:
                self.ep_min_val = min_val
        return obs

    def populate(self, obs, depth=0):
        # self.populate_queue.appendleft((obs, depth))
        heapq.heappush(self.populate_queue, (obs, depth))
        self.serve_queue(self.populate_queue)

    def best_move(self, obs):
        self.clear()
        obs = (torch.from_numpy(obs[0]).to(devices[0]).float(), obs[1].to(devices[0]))
        root = self.add(obs)
        self.populate(root)
        start = time.time()
        self.state_list.reverse()
        # print(len(self.state_list))
        for state, depth in self.state_list:
            state.update_Q()
        i = np.argmax(root.Qs)
        best_act = root.acts[i]
        root.best_act = best_act
        root.best_r = root.rs[i]
        return best_act, root.best_r

    def exit(self):
        self.clear()
