from model_based.mcts import MCTS, State
import numpy as np
import torch
import torch.multiprocessing as mp
ctx = mp.get_context("spawn")

if torch.cuda.is_available():
    devices = [torch.device("cuda:"+str(i)) for i in range(0, torch.cuda.device_count())]
else:
    devices = [torch.device('cpu')]

class ParallelMCTS(MCTS):
    def __init__(self, lookahead, env_learner, agent=None, initial_width=2, with_hidden=False, cross_entropy=False):
        MCTS.__init__(self, lookahead-1, env_learner, agent, initial_width, with_hidden, cross_entropy)
        self.n_proc = initial_width
        self.spawn_processes()

    def spawn_processes(self):
        from multiprocessing import Pipe
        self.processes = []
        self.q_in = [Pipe() for _ in range(self.n_proc)]
        self.q_out = [Pipe() for _ in range(self.n_proc)]
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
            item = q_in[0].recv()
            # print('Got')
            if item is None: return
            obs, agent_state, model_state = item
            self.agent.load_dict(agent_state)
            self.env_learner.load_dict(model_state)
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
            q_out[0].send((root.acts[i], root.rs[i], root.Qs[i]))

    def best_move(self, obs):
        obs = (torch.from_numpy(obs[0]).cpu(), obs[1].cpu())
        agent_state = self.agent.save_dict()
        model_state = self.env_learner.save_dict()
        root = State(obs)
        best_q = None
        for q_in in self.q_in:
            q_in[1].send((obs, agent_state, model_state))
        for q_out in self.q_out:
            new_act, new_r, new_q = q_out[1].recv()
            if best_q is None or new_q > best_q:
                best_q = new_q
                root.best_act = new_act
                root.best_r = new_r
        return root.best_act.cpu().data.numpy().flatten(), root.best_r

    def exit(self):
        for q_in in self.q_in:
            q_in.put(None)
