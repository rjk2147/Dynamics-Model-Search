from dynamics_model_search.model_based.mpc import MPC, NullAgent
from dynamics_model_search.model_based.cem import CEM

import numpy as np
from collections import deque
import heapq
import time
import torch

class State:
    def __init__(self, obs, past_rew, q, r, ori_act, gamma = 0.99):
        self.obs = obs
        self.total_rew = past_rew + gamma * q
        self.ori_act = ori_act
        self.past_rew = past_rew + gamma * r

    # for heapsort
    def __lt__(self,other):
        # using greater than instead of less than because heap sort returns lowest value and we want highest Q
        return (self.total_rew > other.total_rew)

    # def connect(self, act, r, new_state):
    #     self.rs.append(r)
    #     self.acts.append(act)
    #     self.future.append(new_state)

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
        self.env_obs_len = self.dynamics_model.state_mul_const_tensor.shape[0]
        # self.leafs = set()
        self.clear()

    def clear(self):
        self.state_list = []
        self.populate_queue = []

    def add(self, new_obs, q, r, act = None, state=None, depth=0):
        if state:
            new_state = State(new_obs, state.past_rew, q, r, act)
        else:
            new_state = State(new_obs, 0, q, r, act)




        # new_state.Q = r
        # if state is not None:
        #     state.connect(act, r, new_state)
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
            tmp_obs = tmp_obs[:,:self.env_obs_len]
            states_rewards = tmp_obs[:,-1].unsqueeze(axis = 1).cpu().numpy()
            new_obs = (tmp_obs, tmp_h)

            qs = self.agent.value(obs_in[0].squeeze(1), acts_in.squeeze(1), new_obs[0])
            # Discounting the reward by the uncertainty
            uncertainty_factor = torch.mean(uncertainty, -1).unsqueeze(-1).detach().cpu().numpy()
            qs *= uncertainty_factor
            states_rewards *= uncertainty_factor

            these_new_obs = [(new_obs[0][i], new_obs[1][i].unsqueeze(0)) for i in range(len(states))]
            for i in range(len(states)):
                if states[i].ori_act is not None:
                    new_state = self.add(these_new_obs[i], qs[i].item(), states_rewards[i].item(),  states[i].ori_act, states[i], depth=depths[i]+1)
                else:
                    new_state = self.add(these_new_obs[i], qs[i].item(), states_rewards[i].item(),  acts_in[i], states[i], depth=depths[i]+1)

                # q.appendleft((new_state, depths[i]+1))
                heapq.heappush(q, (new_state, depths[i]+1))

    def populate(self, obs, depth=0):
        # self.populate_queue.appendleft((obs, depth))
        heapq.heappush(self.populate_queue, (obs, depth))
        self.serve_queue(self.populate_queue)

    def best_move(self, obs):
        self.clear()
        obs = (torch.from_numpy(obs[0]).to(devices[0]).float(), obs[1].to(devices[0]))
        root = self.add(obs, 0, 0)
        self.populate(root)
        start = time.time()
        # self.state_list.reverse()
        # print(len(self.state_list))
        # for state, depth in self.state_list:
            # state.update_Q()        
        # i = np.argmax(root.Qs)
        # best_act = root.acts[i]
        # root.best_act = best_act
        # root.best_r = root.rs[i]
        # return best_act, root.best_r
        best_leaf, _ = heapq.heappop(self.populate_queue)
        best_act = best_leaf.ori_act
        best_r = best_leaf.total_rew
        return best_act, best_r

    def exit(self):
        self.clear()
