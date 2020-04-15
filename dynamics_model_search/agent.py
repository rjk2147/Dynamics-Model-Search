import torch
import time
import numpy as np
from torch import nn, optim
from collections import deque
import random
import datetime
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_rew_value(obs, act, new_obs):
    return new_obs[...,0].detach().cpu().numpy()

class Agent:
    def __init__(self, dynamics_model, rl, planner, with_tree=True, model_rew=False, batch_size=512, replay_size=1e5, with_memory = False):
        self.model_rew = model_rew
        self.act_mul_const = dynamics_model.act_mul_const
        self.from_update = 0
        self.batch_size = batch_size
        self.with_tree = with_tree

        self.model = dynamics_model
        self.rl_learner = rl
        self.planner = planner

        self.with_memory = with_memory
        self.replay_size = replay_size
        
        if self.model_rew: # nullifying the value function in favor or the model based approach
            self.rl_learner.value = model_rew_value
        self.model.model.train()

        replay_size = max(replay_size, batch_size)
        self.model_replay = deque(maxlen=int(replay_size))
        if not os.path.exists('rl_models/'):
            os.mkdir('rl_models/')
        self.save_str = 'rl_models/'+datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

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
        if self.with_memory:
            print('ep_min_val: ' + str(self.ep_min_val))
            print('repl_count: ' + str(self.planner.replace_count))
        print('--------------------------------------\n')

    def logging(self, ep):
            st_dir = './log/2_temp_log_rl.txt'
            # if os.path.exists(st_dir):
            with open(st_dir, 'a+') as f:
                f.write("episode : {} \n".format(ep))
                if self.ep_lens and len(self.ep_rs) > 0:
                    f.write('Last Episode Reward: ' + str(self.ep_rs[-1]) + '\n')
                if self.ep_rs and len(self.ep_rs) > 0:
                    f.write('Mean Episode Reward: ' + str(np.mean(self.ep_rs)) + '\n')
                if self.ep_rs and len(self.ep_rs) > 0:
                    f.write('Stdev Episode Reward: ' + str(np.std((self.ep_rs))) + '\n')
                if self.ex_ep_rs and len(self.ex_ep_rs) > 0:
                    f.write('Mean Expected Episode Reward: ' + str(np.mean(self.ex_ep_rs))+ '\n')
                # if self.ex_ep_rs and len(self.ex_ep_rs) > 0:
                #     f.write('Expected Episode Reward: ' + str(self.expected_reward) + '\n')
                if self.ex_ep_rs and len(self.ex_ep_rs) > 0:
                    f.write('Stdev Expected Episode Reward: ' + str(np.std(self.ex_ep_rs)) + '\n')
                if self.avg_train_loss is not None and self.u > 0:
                    f.write('Avg Train Loss: ' + str(self.avg_train_loss / float(self.u)) + '\n')
                if self.steps:
                    f.write('Total Timesteps: ' + str(self.steps) + '\n')
                if self.start_time:
                    f.write('Total Time: ' + str(round(time.time() - self.start_time, 2)) + '\n')
                if self.with_memory:
                    f.write('ep_min_val: ' + str(self.ep_min_val) + '\n')
                    f.write('repl_count: ' + str(self.planner.replace_count))

                # f.write('num_iter: ' + str(self.num_iter_history) + '\n')

                f.write('\n\n\n')

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
        if len(self.model_replay) >= self.batch_size:
            data = random.sample(self.model_replay, self.batch_size)

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

    def learn(self, env, max_timesteps=1e6):
        return self.run(training=True, env=env, max_timesteps=max_timesteps)

    def play(self, env, max_timesteps=1000):
        return self.run(training=False, env=env, max_timesteps=max_timesteps)

    def run(self, training, env, max_timesteps=1e6):
        self.model.max_seq_len = 10
        self.seq_len = self.model.max_seq_len
        self.x_seq = deque(maxlen=self.seq_len)
        self.a_seq = deque(maxlen=self.seq_len)
        self.y_seq = deque(maxlen=self.seq_len)
        self.u = 0
        self.n_updates = 0
        self.avg_train_loss = None
        self.steps = 0
        obs_lists = []
        ep = 0

        self.ep_rs = deque(maxlen=100)
        self.ex_ep_rs = deque(maxlen=100)
        self.ep_lens = deque(maxlen=100)
        # self.steps = 0
        self.start_time = time.time()

        if self.with_memory:
            self.ep_min_val = float('inf')

        while self.steps < max_timesteps:
            obs_list = []
            ep += 1
            obs = env.reset()


            if self.with_memory:
                to_cat = torch.unsqueeze(torch.from_numpy(obs), dim = 0).cuda()
                if len(self.planner.memory_buffer.shape) == 1:
                    self.planner.memory_buffer = to_cat
                    self.planner.memory_buffer_usage = np.array([0])
                else:
                    if len(self.planner.memory_buffer_usage) < self.replay_size:
                        self.planner.memory_buffer = torch.cat([self.planner.memory_buffer, to_cat],dim = 0)
                        self.planner.memory_buffer_usage = np.concatenate([self.planner.memory_buffer_usage, np.array([0])], axis = 0)
                    else:
                        self.planner.clean_and_input(to_cat)
                
            if self.model_rew: # appending initial reward of 0 to obs
                obs = np.concatenate([np.zeros(1), obs]).astype(obs.dtype)
            if self.with_tree:
                obs = self.planner.dynamics_model.reset(obs, None)
            else:
                obs = (obs, None)
            done = False
            ep_r = 0
            ep_exp_r = 0
            ep_len = 0
            self.planner.replace_count = 0
            
            while not done:
                if self.with_tree:
                    act, best_r = self.planner.best_move(obs)
                    act = act.cpu().data.numpy().flatten()
                    ex_r = best_r
                    self.planner.clear()
                else:
                    act = self.rl_learner.act(np.expand_dims(obs[0], 0)).cpu().numpy().flatten()
                    ex_r = 0
                new_obs, r, done, info = env.step(act*self.act_mul_const)

                if self.with_memory:
                    to_cat = torch.unsqueeze(torch.from_numpy(new_obs),dim=0).cuda()
                    if len(self.planner.memory_buffer_usage) < self.replay_size:
                        self.planner.memory_buffer = torch.cat([self.planner.memory_buffer, to_cat],dim=0)
                        self.planner.memory_buffer_usage = np.concatenate([self.planner.memory_buffer_usage, np.array([0])], axis = 0)
                    else:
                        self.planner.clean_and_input(to_cat)

                if self.model_rew: # appending reward to obs
                    new_obs = np.concatenate([np.ones(1)*r, new_obs]).astype(new_obs.dtype)
                if self.with_tree:
                    # TODO: Efficiently pass this h value from the search since it is already calculated
                    _, h = self.planner.dynamics_model.step_parallel(obs_in=(torch.from_numpy(obs[0]).unsqueeze(0).unsqueeze(1).to(device), [obs[1].to(device)]),
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
        
                if self.with_memory:
                    if self.planner.ep_min_val < self.ep_min_val:
                        self.ep_min_val = self.planner.ep_min_val

                if training:
                    # TODO could try to use state and rerun all obs through the self-model before RL-update
                    ## RL Learner Update
                    self.rl_learner.replay.add(obs[0], act, new_obs[0], r, done)
                    self.from_update += 1
                    self.rl_update()
                    self.rl_learner.steps += 1

                    ## Self-Model Update
                    if self.with_tree:
                        self.sm_update(obs, act, new_obs, done)
                else:
                    obs_list.append(obs[0])
                obs = new_obs

            if training:
                print('Models saved to '+str(self.save_str))
                self.rl_learner.save(self.save_str)
                self.model.save(self.save_str+'_self_model.pt')
            else:
                obs_lists.append(obs_list)
            self.ep_rs.append(ep_r)
            self.ep_lens.append(ep_len)
            self.ex_ep_rs.append(ep_exp_r)
            self.print_stats()
            self.logging(ep)
            self.from_update = 0
            self.u = 0
            if self.avg_train_loss is not None:
                self.avg_train_loss = None
        self.planner.exit()
        if not training:
            return obs_lists

    def play_old(self, env, num_episodes):
        self.model.max_seq_len = 10
        self.seq_len = self.model.max_seq_len
        self.steps = 0
        self.start_time = time.time()

        obs_lists = []
        ep_rs = []

        for i in range(num_episodes):
            obs_list = []
            obs = env.reset()
            obs = self.planner.dynamics_model.reset(obs, None)
            done = False
            ep_r = 0
            ep_exp_r = 0
            ep_len = 0
            while not done:
                obs_list.append(obs[0])
                if self.with_tree:
                    act, best_r = self.planner.best_move(obs)
                    act = act.cpu().data.numpy().flatten()
                    ex_r = best_r
                    self.planner.clear()
                else:
                    act = self.rl_learner.act(np.expand_dims(obs[0], 0)).cpu().numpy().flatten()
                    ex_r = 0
                new_obs, r, done, info = env.step(act*self.act_mul_const)

                # TODO: Efficiently pass this h value from the search since it is already calculated
                _, h = self.planner.dynamics_model.step_parallel(obs_in=(torch.from_numpy(obs[0]).unsqueeze(0).to(device), obs[1].to(device)),
                                                              action_in=torch.from_numpy(act).unsqueeze(0).to(device),
                                                              state=True, state_in=True)
                new_obs = self.planner.dynamics_model.reset(new_obs, h)

                # Statistics update
                self.steps += 1
                ep_r += r
                ep_exp_r += ex_r
                ep_len += 1

                obs = new_obs
            obs_lists.append(obs_list)
            ep_rs.append(ep_r)
            print('Episode '+str(len(obs_lists))+' Reward: '+str(ep_r))
        self.planner.exit()
        print('---------------------------')
        print('Mean Episode Reward: '+str(np.mean(ep_rs)))
        print('Stdev Episode Reward: '+str(np.std(ep_rs)))
        return obs_lists
