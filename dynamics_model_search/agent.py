import torch
import time
import numpy as np
from torch import nn, optim
from collections import deque
import random
import datetime
import os

from model_free.DQN import DQN
from model_free.DDQN import DDQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, dynamics_model, rl, planner, batch_size=512, replay_size=1e5, seq_len=10):
        self.from_update = 0
        self.batch_size = batch_size

        self.model = dynamics_model
        self.seq_len = seq_len
        self.rl_learner = rl
        self.planner = planner

        replay_size = max(replay_size, batch_size)
        if dynamics_model is not None:
            self.act_mul_const = dynamics_model.act_mul_const
            self.model.model.train()
            self.model.replay = deque(maxlen=int(replay_size))
        else:
            self.act_mul_const = 1.0
        
        if not os.path.exists('rl_models/'):
            os.mkdir('rl_models/')
        self.save_str = 'rl_models/'+datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

        # Modifed By Yu
        from torch.utils.tensorboard import SummaryWriter
        self.writer1 = SummaryWriter('runs/DQN')
        self.writer2 = SummaryWriter('runs/DDQN')
        if isinstance(self.rl_learner, DQN):
            self.writer = self.writer1
            print("Logged in runs/DQN.")
        elif isinstance(self.rl_learner, DDQN):
            self.writer = self.writer2
            print("Logged in runs/DDQN.")
        else:
            print('error')

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
        # Modifed By Yu
        # os.system("tensorboard --logdir runs/test")
        # print(self.rl_learner)
        self.writer.add_scalar('Reward',
                          self.ep_rs[-1],
                          self.steps)
        if self.ep_rs[-1] > 20:
            print('Target achived!')
        # self.writer.close()



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
            self.model.replay.append((np.array(self.x_seq), np.array(self.a_seq), np.array(self.y_seq)))
        if len(self.model.replay) >= self.batch_size:
            data = random.sample(self.model.replay, self.batch_size)

            # obs_dist = np.array([step[0][0] for step in data], dtype=np.float32)
            obs_dist = np.array([np.mean(step[0], 0) for step in data], dtype=np.float32) # also normalizes across time
            obs_mean = np.mean(obs_dist, 0)
            obs_std = np.std(obs_dist, 0)

            # This line ensures that is there is no variance in an element of the states it is unchanged
            # Otherwise when the observations are divided those values will become NaN
            obs_std[obs_std <= epsilon] = 1

            self.model.model.norm_mean = obs_mean
            self.model.model.norm_std = obs_std
            train_loss = self.model.update(data)
            train_loss = np.array(train_loss)
            if self.avg_train_loss is None:
                self.avg_train_loss = np.array(train_loss)
            else:
                self.avg_train_loss += np.array(train_loss)
            self.u += 1
        if done:
            self.x_seq = deque(maxlen=self.seq_len)
            self.a_seq = deque(maxlen=self.seq_len)
            self.y_seq = deque(maxlen=self.seq_len)

    def learn(self, env, max_timesteps=1e6):
        return self.run(training=True, env=env, max_timesteps=max_timesteps)

    def play(self, env, max_timesteps=1000):
        return self.run(training=False, env=env, max_timesteps=max_timesteps)

    def run(self, training, env, max_timesteps=1e6):
        if self.model is not None:
            self.model.max_seq_len = self.seq_len
            # self.seq_len = self.model.max_seq_len
        self.x_seq = deque(maxlen=self.seq_len)
        self.a_seq = deque(maxlen=self.seq_len)
        self.y_seq = deque(maxlen=self.seq_len)
        self.u = 0
        self.n_updates = 0
        self.avg_train_loss = None
        self.steps = 0
        obs_lists = []

        self.ep_rs = deque(maxlen=100)
        self.ex_ep_rs = deque(maxlen=100)
        self.ep_lens = deque(maxlen=100)
        # self.steps = 0
        self.start_time = time.time()
        while self.steps < max_timesteps:
            obs_list = []
            obs = env.reset()
            if self.planner is not None:
                obs = self.planner.dynamics_model.reset(obs, None)
            else:
                obs = (obs, None)
            done = False
            ep_r = 0
            ep_exp_r = 0
            ep_len = 0
            while not done:
                if self.planner is not None:
                    act, best_r, explored = self.planner.best_move(obs)
                    act = act.cpu().data.numpy().flatten()
                    ex_r = best_r
                    self.planner.clear()
                else:
                    act = self.rl_learner.act(np.expand_dims(obs[0], 0)).cpu().numpy().flatten()
                    ex_r = 0
                # modified by yu
                # action.int()
                action = act*self.act_mul_const
                if type(action) == int:
                    pass
                else:
                    action = action.astype(int)
                new_obs, r, done, info = env.step(action)

                if self.planner is not None and self.model is not None:
                    # TODO: Efficiently pass this h value from the search since it is already calculated
                    _, h = self.planner.dynamics_model.step(obs_in=(torch.from_numpy(obs[0]).unsqueeze(0).unsqueeze(1).to(device), [obs[1].to(device)]),
                                                                  action_in=torch.from_numpy(act).unsqueeze(0).unsqueeze(1).to(device),
                                                                  state=True, state_in=True)
                    # _, h = self.planner.dynamics_model.step_parallel(obs, act)
                    new_obs = self.planner.dynamics_model.reset(new_obs, h)
                else:
                    new_obs = (new_obs, None)

                # Statistics update
                self.steps += 1
                ep_r += r
                ep_exp_r += ex_r
                ep_len += 1

                if training:
                    ## RL Learner Update
                    self.rl_learner.replay.add(obs[0], act, new_obs[0], r, done)
                    self.from_update += 1
                    self.rl_update()
                    self.rl_learner.steps += 1

                    ## Self-Model Update
                    if self.model is not None:
                        self.sm_update(obs, act, new_obs, done)
                else:
                    obs_list.append(obs[0])
                obs = new_obs

            if training:
                print('Models saved to '+str(self.save_str))
                self.rl_learner.save(self.save_str)
                if self.model is not None:
                    self.model.save(self.save_str+'_self_model.pt')
            else:
                obs_lists.append(obs_list)
            self.ep_rs.append(ep_r)
            self.ep_lens.append(ep_len)
            self.ex_ep_rs.append(ep_exp_r)
            self.print_stats()
            self.from_update = 0
            self.u = 0
            if self.avg_train_loss is not None:
                self.avg_train_loss = None
        # self.planner.exit()
        if not training:
            return obs_lists