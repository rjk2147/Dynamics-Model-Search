from model_based.mpc import MPC, NullAgent
import numpy as np
from queue import Queue
from threading import Thread, Lock
import multiprocessing
# from multiprocessing import Process, Lock
import time
import pickle
import torch

counter = 0
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
        self.populate_queue = Queue()
        # self.spawn_processes()
        self.start = time.time()

    def cross_entropy(self, obs, act):
        # Initialize parameters
        epsilon = 1e-3
        mu = act
        sigma2 = np.ones(self.act_dim)
        t = 0
        maxits = 50
        N = 100
        Ne = 10
        # While maxits not exceeded and not converged
        X = (np.array([obs[0] for i in range(N)]), torch.cat([obs[1] for i in range(N)]).transpose(0,1))
        while t < maxits and (sigma2 > epsilon).any():
            # Obtain N samples from current sampling distribution
            A = np.array([(act+np.random.normal(mu, sigma2)).clip(-1, 1) for _ in range(N)])
            # Evaluate objective function at sampled points
            S = self.env_learner.step(obs_in=X, action_in=A, state=True, state_in=True)
            R = self.agent.value(X[0], A, S[0])
            # Sort X by objective function values in descending order
            A = A[R.argsort()[::-1]]
            # A = A[R.argsort()[:,0]][::-1]
            # Update parameters of sampling distribution
            mu = np.mean(A[:Ne], axis=0)
            sigma2 = np.std(A[:Ne], axis=0)
            t = t + 1
        # Return mean of final sampling distribution as solution
        # print(t)
        # print(time.time())
        # print(mu)
        # print(act)
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
        return str(self.discritize(x))

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

    def populate_threaded(self, q):
        while not q.empty():
        # while True:
            item = q.get()
            if item is None:
                return
            obs, depth = item
            if depth < self.lookahead:
                # for i in range(max(int(self.width/(2**depth)), 2)):
                for i in range(self.width):
                    # state = obs[0]
                    # state = obs[1].cpu().detach().numpy()
                    # state = np.concatenate([obs[0], obs[1].cpu().flatten().detach().numpy()])
                    state = obs[0]
                    act = self.agent.act(state)
                    # act = self.cross_entropy(obs, act)
                    if torch.is_tensor(act):
                        act_mul_const = torch.from_numpy(self.act_mul_const).to(act.device)
                        act = act*act_mul_const
                    else:
                        act = act*self.act_mul_const
                    new_obs = self.env_learner.step(obs_in=obs, action_in=act, state=True, state_in=True)
                    state_key = self.to_key(new_obs)
                    r = self.agent.value(obs[0], act, new_obs[0])
                    if state_key not in self.states:
                        self.tree_add(state_key, new_obs, self.states, obs, act, r, 0, depth=depth+1)
                        q.put((new_obs, depth+1))
                    else:
                        print('Skip')
            q.task_done()

    def populate(self, obs, depth=0):
        self.populate_queue.put((obs, depth))
        self.populate_threaded(self.populate_queue)
        # self.spawn_processes()
        # self.populate_queue.join()

    def best_move(self, obs):
        self.clear()
        i = 0
        self.tree_add(self.to_key(obs), obs, self.states)
        self.populate(obs)
        best_act = self.states[self.to_key(obs)].best_act
        # while self.states[self.to_key(obs)].future[str(best_act)][1] < 0 and i < 10:
        #     self.populate(obs)
        #     best_act = self.states[self.to_key(obs)].best_act
        #     i += 1
        # pred_r = self.states[self.to_key(obs)].future[str(best_act)][-1]
        # print(len(self.states))
        return best_act, self.states[self.to_key(obs)]
