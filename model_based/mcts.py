from model_based.mpc import MPC, NullAgent
import numpy as np
from collections import deque
import time
import torch

class State:
    def __init__(self, obs, cur_V):
        self.obs = obs
        self.cur_V = cur_V
        self.Q = 0
        self.acts = []
        self.rs = []
        self.future = []
        self.Qs = []

    def connect(self, act, r, new_state):
        self.rs.append(r)
        self.acts.append(act)
        self.future.append(new_state)

    def exit(self):
        return

    def get_avg_r(self):
        return np.mean(self.rs)

    def update_Q(self, discount=0.99):
        # self.Q = 0
        self.Qs = []
        for i in range(len(self.rs)):
            Q = self.cur_V + (self.future[i].Q - self.rs[i]) * discount
            self.Q += Q
            self.Qs.append(Q)
        self.Q = self.Q / max(len(self.future), 1)
        return self.Q

if torch.cuda.is_available():
    devices = [torch.device("cuda:"+str(i)) for i in range(0, torch.cuda.device_count())]
else:
    devices = [torch.device('cpu')]

class MCTS(MPC):
    def __init__(self, lookahead, env_learner, agent=None, initial_width=2, with_hidden=False, cross_entropy=False):
        MPC.__init__(self, lookahead, env_learner, agent)
        self.width = initial_width
        self.populate_queue = deque()
        self.start = time.time()
        self.batch_size = 262144
        self.CE_N = 1
        self.clear()
        self.with_hidden = with_hidden
        self.with_CE = cross_entropy

        self.estimated_Q = 0

        if self.with_CE:
            self.CE_N = 64

    def clear(self):
        self.state_list = []

    def add(self, new_obs, state=None, act=None, r=0, depth=0):
        new_state = State(new_obs, r)
        if state is not None:
            state.connect(act, r, new_state)
        self.state_list.append((new_state, depth))
        return new_state

    def cross_entropy(self, obs, act):
        # Initialize parameters
        epsilon = 1e-3
        t = 0
        maxits = 32
        N = self.CE_N
        Ne = 16
        b = act.shape[0]
        # While maxits not exceeded and not
        # Expands the states and actions batch size to be b*N
        X = (obs[0].repeat((N, 1)), obs[1].repeat((N, 1, 1)))
        # A_raw = act.repeat((N, 1, 1))
        # mu = torch.zeros(act.shape)
        mu = act
        sigma = torch.ones(mu.shape).to(mu.device)*0.1
        A_time = 0
        B_time = 0
        C_time = 0
        D_time = 0
        E_time = 0
        while t < maxits and (sigma > epsilon).any():
            start = time.time()
            # Obtain N samples from current sampling distribution
            mu = mu.repeat((N, 1))
            sigma = sigma.repeat((N, 1))
            A = (torch.randn_like(mu)*sigma + mu).clamp(-1, 1)
            A_time += time.time()-start
            start = time.time()
            # Evaluate objective function at sampled points
            S = self.env_learner.step_parallel(obs_in=X, action_in=A, state=True, state_in=True)
            R = self.agent.value(X[0], A, S[0]).flatten()
            B_time += time.time()-start
            start = time.time()
            # Splitting Rs and As into their initial groups
            R = np.split(R, b, 0)
            A = torch.chunk(A, b, 0)
            C_time += time.time()-start
            start = time.time()
            # Select top Ne actions based on their R score
            A = [A[i][np.argsort(-R[i])[:Ne]] for i in range(b)]
            # A = [A[i][np.argpartition(-R[i], Ne)[:Ne]] for i in range(b)]
            D_time += time.time()-start
            start = time.time()
            # Update parameters of sampling distribution
            mu = torch.cat([torch.mean(A[i], 0).unsqueeze(0) for i in range(b)])
            sigma = torch.cat([torch.std(A[i], 0).unsqueeze(0) for i in range(b)])
            t = t + 1
            E_time += time.time()-start
        # Return mean of final sampling distribution as solution
        # print(mu.shape[0])
        # print('A: '+str(A_time))
        # print('B: '+str(B_time))
        # print('C: '+str(C_time))
        # print('D: '+str(D_time))
        # print('E: '+str(E_time))
        # print('=================')
        return mu

    def serve_queue(self, q):
        while len(q) > 0:
            obs = []
            states = []
            depths = []
            acts = []
            while len(acts)+self.width <= self.batch_size/self.CE_N and len(q) > 0:
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
            if self.with_CE:
                acts_in = self.cross_entropy(obs_in, acts_in)
            new_obs = self.env_learner.step_parallel(obs_in=obs_in, action_in=acts_in, state=True, state_in=True)
            rs = self.agent.value(tmp_obs, acts_in, new_obs[0])
            these_new_obs = [(new_obs[0][i], new_obs[1][i].unsqueeze(0)) for i in range(len(states))]
            for i in range(len(states)):
                new_state = self.add(these_new_obs[i], states[i], acts_in[i], rs[i].item(), depth=depths[i]+1)
                if depth + 1 == self.lookahead:
                    new_state.Q = rs[i].item()
                q.appendleft((new_state, depths[i]+1))

    def populate(self, obs, depth=0):
        self.populate_queue.appendleft((obs, depth))
        self.serve_queue(self.populate_queue)

    def best_move(self, obs):
        self.clear()
        obs = (torch.from_numpy(obs[0]).to(devices[0]), obs[1].to(devices[0]))
        root = self.add(obs)
        self.populate(root)
        start = time.time()
        self.state_list.reverse()

        discount = 0.99

        for state, depth in self.state_list:
            state.update_Q(discount)

        for i in range(len(root.Qs)):
            root.Qs[i] += root.rs[i] * 1/discount

        i = np.argmax(root.Qs)
        root.exp_V = root.Qs[i]
        best_act = root.acts[i]
        root.best_act = best_act
        root.best_r = root.rs[i]
        return best_act.cpu().data.numpy().flatten(), root

    def exit(self):
        self.clear()
