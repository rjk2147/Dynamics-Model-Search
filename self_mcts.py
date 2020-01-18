import torch
import time
import dill
import numpy as np
from torch import nn, optim
from collections import deque
import random
import datetime
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Agent:
    def __init__(self, env_learner, width=64, depth=1, agent='TD3', with_tree=True, with_hidden=False,
                 model_rew=False, parallel=False, cross_entropy=False):
        self.act_dim = env_learner.act_dim
        self.state_dim = env_learner.state_dim
        self.act_mul_const = env_learner.act_mul_const
        self.lookahead = depth
        self.from_update = 0
        self.sm_batch = 512
        self.null_agent = False
        if agent == 'TD3':
            from model_free.TD3 import TD3 as Agent
            # from model_free.TD3 import ReplayBuffer as Replay
        elif agent == 'SAC':
            from model_free.SAC import SAC as Agent
        elif agent == 'PPO':
            from model_free.PPO import PPO as Agent
        elif agent == 'TRPO':
            from model_free.TRPO import TRPO as Agent
        elif agent == 'None':
            from model_free.Null import NullAgent as Agent
            self.null_agent = True
        else:
            from model_free.TD3 import TD3 as Agent

        if parallel:
            from model_based.parallel_mcts import ParallelMCTS as MCTS
        else:
            from model_based.mcts import MCTS

        self.model = env_learner
        self.model.model.train()
        if with_hidden:
            self.rl_learner = Agent(self.state_dim+self.model.model.latent_size, self.act_dim)
        else:
            self.rl_learner = Agent(self.state_dim, self.act_dim)
        self.rl_learner.model_rew = model_rew
        self.planner = MCTS(self.lookahead, env_learner, self.rl_learner, initial_width=width,
                            with_hidden=with_hidden, cross_entropy=cross_entropy)
        self.model_replay = deque(maxlen=100000)
        
        if not os.path.exists('rl_models/'):
            os.mkdir('rl_models/')
        self.save_str = 'rl_models/'+datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.with_tree = with_tree
        self.with_hidden = with_hidden

    def print_stats(self):
        if self.ep_lens and len(self.ep_rs) > 0:
            print('Last Episode Reward: '+str(self.ep_rs[-1]))
        if self.ep_rs and len(self.ep_rs) > 0:
            print('Mean Episode Reward: '+str(np.mean(self.ep_rs)))
        if self.ep_rs and len(self.ep_rs) > 0:
            print('Stdev Episode Reward: '+str(np.std((self.ep_rs))))
        if self.ex_ep_rs and len(self.ex_ep_rs) > 0:
            print('Mean Expected Episode Reward: '+str(np.mean(self.ex_ep_rs)))
        if self.ex_ep_rs and len(self.ex_ep_rs) > 0:
            print('Stdev Expected Episode Reward: '+str(np.std(self.ex_ep_rs)))
        if self.avg_train_loss is not None and self.u > 0:
            print('Avg Train Loss: '+str(self.avg_train_loss/float(self.u)))
        if self.steps:
            print('Total Timesteps: '+str(self.steps))
        if self.start_time:
            print('Total Time: '+str(round(time.time()-self.start_time, 2)))
        print('--------------------------------------\n')

    def rl_update(self, batch_size=256):
        if len(self.rl_learner.replay) > batch_size:
            self.rl_learner.update(batch_size, self.n_updates)
            self.n_updates += 1

    def sm_update(self, obs, act, new_obs, done):
        # epsilon = 0
        epsilon = 1e-5

        self.x_seq.append(obs[0] / self.model.state_mul_const)
        self.a_seq.append(act / self.act_mul_const)
        self.y_seq.append(new_obs[0] / self.model.state_mul_const)
        if len(self.x_seq) == self.seq_len:
            self.model_replay.append((np.array(self.x_seq), np.array(self.a_seq), np.array(self.y_seq)))
        if len(self.model_replay) >= self.sm_batch:
            data = random.sample(self.model_replay, self.sm_batch)
            # data = list(self.model_replay)[:self.sm_batch]

            obs_dist = np.array([step[0][0] for step in data])
            obs_mean = np.mean(obs_dist, 0)
            obs_std = np.std(obs_dist, 0)

            # This line ensures that is there is no variance in an element of the states it is unchanged
            # Otherwise when the observations are divided those values will become NaN
            obs_std[obs_std <= epsilon] = 1

            self.model.model.norm_mean = obs_mean
            self.model.model.norm_std = obs_std
            train_loss = self.model.update(data)
            if self.avg_train_loss is None:
                self.avg_train_loss = np.zeros(len(train_loss))
            train_loss = np.array(train_loss)
            self.avg_train_loss += np.array(train_loss)
            self.u += 1
        if done:
            self.x_seq = deque(maxlen=self.seq_len)
            self.a_seq = deque(maxlen=self.seq_len)
            self.y_seq = deque(maxlen=self.seq_len)

    def learn(self, env, num_episodes=100):
        self.model.max_seq_len = 10
        self.seq_len = self.model.max_seq_len
        self.x_seq = deque(maxlen=self.seq_len)
        self.a_seq = deque(maxlen=self.seq_len)
        self.y_seq = deque(maxlen=self.seq_len)
        self.u = 0
        self.n_updates = 0
        self.avg_train_loss = None
        self.steps = 0

        self.ep_rs = deque(maxlen=100)
        self.ex_ep_rs = deque(maxlen=100)
        self.ep_lens = deque(maxlen=100)
        # self.steps = 0
        self.start_time = time.time()
        for i in range(num_episodes):
            obs = env.reset()
            obs = self.planner.env_learner.reset(obs, None)
            done = False
            ep_r = 0
            ep_exp_r = 0
            ep_len = 0
            while not done:
                if self.with_tree and not self.null_agent:
                    act, node = self.planner.best_move(obs)
                    act = act.flatten()
                    ex_r = node.best_r
                    self.planner.clear()
                else:
                    act = self.rl_learner.act(obs[0]).cpu().numpy().flatten()
                    ex_r = 0
                new_obs, r_raw, done, info = env.step(act*self.act_mul_const)

                # TODO: Efficiently pass this h value from the search since it is already calculated
                _, h = self.planner.env_learner.step_parallel(obs_in=(torch.from_numpy(obs[0]).unsqueeze(0).to(device), obs[1].to(device)),
                                                              action_in=torch.from_numpy(act).unsqueeze(0).to(device),
                                                              state=True, state_in=True)
                # _, h = self.planner.env_learner.step_parallel(obs, act)
                new_obs = self.planner.env_learner.reset(new_obs, h)

                # Statistics update
                self.steps += 1
                r = r_raw
                ep_r += r
                ep_exp_r += ex_r
                ep_len += 1

                # TODO could try to use state and rerun all obs through the self-model before RL-update
                ## RL Learner Update
                if self.with_hidden:
                    obs_cat = torch.cat([torch.from_numpy(obs[0]), obs[1].flatten().cpu()], -1).numpy()
                    new_obs_cat = torch.cat([torch.from_numpy(new_obs[0]), new_obs[1].flatten().cpu()], -1).numpy()
                    self.rl_learner.replay.add(obs_cat, act, new_obs_cat, r, done)
                else:
                    self.rl_learner.replay.add(obs[0], act, new_obs[0], r, done)
                self.from_update += 1
                self.rl_update()
                self.rl_learner.steps += 1

                ## Self-Model Update
                if self.with_tree:
                    self.sm_update(obs, act, new_obs, done)
                obs = new_obs

            self.ep_rs.append(ep_r)
            self.ep_lens.append(ep_len)
            self.ex_ep_rs.append(ep_exp_r)
            print('Models saved to '+str(self.save_str))
            self.print_stats()
            self.from_update = 0
            self.u = 0
            if self.avg_train_loss is not None:
                self.avg_train_loss = None
            self.rl_learner.save(self.save_str)
            self.model.save(self.save_str+'_self_model.pt')
        self.planner.exit()

    def play(self, env, num_episodes):
        for i in range(num_episodes):
            obs = env.reset()
            done = False
            ep_r = 0
            ep_len = 0
            while not done:
                obs = self.planner.env_learner.reset(obs)
                act, pred_obs = self.planner.best_move(obs)
                new_obs, r_raw, done, info = env.step(act)
                ep_r += r_raw
                ep_len += 1
            print('Episode '+str(i)+'/'+str(num_episodes)+'in '+str(ep_len)+' steps with reward '+str(ep_r))

