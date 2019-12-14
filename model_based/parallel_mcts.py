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

# 2.7s to run no CE with width 4 on standard
# 2.2s to run no CE with width 4 on parallel
# 37s to run Cross Entropy with width 4 on standard
# 8s to run Cross Entropy with width 4 on parallel
# 78s to run no CE with width 8 on standard
# 62s to run no CE with width 8 on parallel
if torch.cuda.is_available():
    devices = [torch.device("cuda:"+str(i)) for i in range(0, torch.cuda.device_count())]
else:
    devices = [torch.device('cpu')]
class MCTS(MPC):
    def __init__(self, lookahead, env_learner, agent=None, initial_width=2):
        MPC.__init__(self, lookahead, env_learner, agent)
        self.discount = 0.99
        # self.states = pickle.load(open('mcts_ant.pkl', 'rb'))
        # manager = multiprocessing.Manager()
        self.states = dict()
        # self.discritize = lambda x: np.round(x, 2)
        self.discritize = lambda x: x[0]
        self.width = initial_width
        # Error where more than 1 thread causes 2-4x slowdown
        self.tree_processes = 1
        self.env_processes = 1
        self.tree_lock = Lock()
        self.populate_queue = deque()
        # self.spawn_processes()
        self.start = time.time()
        # self.env_learner.model = nn.DataParallel(self.env_learner.model, device_ids=[1,2])
        # self.env_learner.model.to(devices[0])

        self.batch_size = 32768
        self.CE_N = 128
        self.n_acts = self.batch_size/self.CE_N

    def cross_entropy(self, obs, act):
        # Initialize parameters
        epsilon = 1e-3
        mu = np.zeros(act.shape)
        sigma2 = np.ones(act.shape)
        t = 0
        maxits = 50
        N = self.CE_N
        Ne = 16
        b = act.shape[0]
        # While maxits not exceeded and not
        # Expands the states and actions batch size to be b*N
        X = (np.tile(obs[0], (N,1)), obs[1].repeat((N, 1, 1)).transpose(0,1))
        # print(X[1].shape)
        A_raw = np.tile(act, (N,1))
        # print(A_raw.shape)
        while t < maxits and (sigma2 > epsilon).any():
            # Obtain N samples from current sampling distribution
            # Expanding mu and sigma to b*N and adding to expanded acts
            noise = np.random.normal(np.tile(mu, (N,1)), np.tile(sigma2, (N,1)))
            if t == 0:
                A = (A_raw+noise).clip(-1, 1)
            else:
                A = noise.clip(-1, 1)
            # A = np.array([(act+np.random.normal(mu, sigma2)).clip(-1, 1) for _ in range(N)])
            # Evaluate objective function at sampled points
            S = self.env_learner.step(obs_in=X, action_in=A, state=True, state_in=True)
            R = self.agent.value(X[0], A, S[0])
            # Splitting Rs and As into their initial groups
            R = np.split(R, b)
            A = np.split(A, b)
            # Sort X by objective function values in descending order
            A = [A[i][R[i].argsort().flatten()][::-1] for i in range(b)]
            # A = A[R.argsort().flatten()][::-1]
            # Update parameters of sampling distribution
            mu = np.array([np.mean(A[i][:Ne], axis=0) for i in range(b)])
            sigma2 = np.array([np.std(A[i][:Ne], axis=0) for i in range(b)])
            t = t + 1
        # Return mean of final sampling distribution as solution
        return mu

    def clear(self):
        self.states = dict()

    def spawn_processes(self):
        self.workers = []
        for i in range(self.tree_processes):
            worker = Thread(target=self.populate_threaded, args=(self.populate_queue,))
            worker.start()
            self.workers.append(worker)
            # time.sleep(0.01)

    def to_key(self, x):
        if x is None:
            return None
        key = self.discritize(x).tostring()
        # print(key)
        return key

    def update_node(self, key, obs, avg_future_r, states):
        for i in range(len(states[key].past)):
            past_obs = states[key].past[i][0]
            past_act = states[key].past[i][1]
            past_r = states[key].past[i][2]
            past_key = self.to_key(past_obs)
            if past_key is not None:
                if str(past_act) not in states[past_key].future:
                    discounted_past_r = past_r*(self.discount**states[past_key].depth)
                    states[past_key].future[str(past_act)] = [key, avg_future_r+discounted_past_r, past_act, obs, discounted_past_r]
                else:
                    states[past_key].future[str(past_act)][1] = avg_future_r
                states[past_key].avg_future_r = 0
                states[past_key].best_act = None
                max_future = 0
                for past_future_key in states[past_key].future:
                    discounted_q = states[past_key].future[past_future_key][1]
                    states[past_key].avg_future_r += discounted_q
                    if discounted_q > max_future or states[past_key].best_act is None:
                        max_future = discounted_q
                        states[past_key].best_act = states[past_key].future[past_future_key][2]
                states[past_key].avg_future_r /= len(states[past_key].future)
                self.update_node(past_key, past_obs, states[past_key].avg_future_r, states)

    def tree_add(self, key, obs, states, past_obs=None, act=None, r=None, avg_future_r=0, depth=0):
        if key not in states:
            states[key] = Node(obs, past_obs, act, r, avg_future_r)
            states[key].depth = depth
        elif past_obs is not None:
            states[key].past.append((past_obs, act, r))
            return
        self.update_node(key, obs, avg_future_r, states)

    def serve_queue(self, q):
        n_skips = 0
        while len(q) > 0:
            obs = []
            depth = []
            acts = []
            while len(acts)+self.width <= self.batch_size and len(q) > 0:
                item = q.pop()
                if item is None:
                    return
                depth = item[1]
                if depth < self.lookahead:
                    for _ in range(self.width):
                        # print(depth)
                        state = item[0][0]
                        act = self.agent.act(state)
                        acts.append(act)
                        obs.append(item[0])
            if len(acts) == 0:
                continue
            # else:
                # print(str(len(acts))+' '+str(time.time()))
            acts = np.array(acts)
            acts_in = torch.from_numpy(acts).float().to(devices[0])
            while len(acts_in.shape) < 3:
                acts_in = acts_in.unsqueeze(1)
            obs = (np.array([obs[i][0] for i in range(len(obs))]),
                   torch.cat([obs[i][1] for i in range(len(obs))]).to(devices[0]))
            obs_in = (torch.from_numpy(obs[0]).float().to(devices[0]), obs[1])

            # acts = self.cross_entropy(obs, acts)
            new_obs = self.env_learner.step_parallel(obs_in=obs_in, action_in=acts_in, state=True, state_in=True)
            rs = self.agent.value(obs[0], acts, new_obs[0].cpu().numpy())
            for i in range(len(acts)):
                this_act = acts[i]
                this_obs = (obs[0][i], obs[1][i].unsqueeze(0))
                this_new_obs = (new_obs[0][i].cpu().numpy(), new_obs[1][i].unsqueeze(0))
                this_r = rs[i]
                state_key = self.to_key(this_new_obs)
                if state_key not in self.states:
                    self.tree_add(state_key, this_new_obs, self.states, this_obs, this_act, this_r, 0, depth=depth+1)
                    q.appendleft((this_new_obs, depth+1))
                else:
                    n_skips += 1
                #     print('Skip')
        if n_skips > 0:
            print('n_skips: '+str(n_skips)+'/'+str(len(self.states)+n_skips))

    def populate(self, obs, depth=0):
        self.populate_queue.appendleft((obs, depth))
        self.serve_queue(self.populate_queue)

    def best_move(self, obs):

        self.clear()
        self.tree_add(self.to_key(obs), obs, self.states)
        self.populate(obs)
        best_act = self.states[self.to_key(obs)].best_act
        # print(len(self.states))
        return best_act, self.states[self.to_key(obs)]

