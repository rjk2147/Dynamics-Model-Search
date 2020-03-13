import time
import numpy as np
import torch
# from stable_baselines import SAC, PPO1, A2C, DDPG, TD3

# Rand MPC Data on 1000 episodes for Ant:
# Reward Mean: -0.003302614366036062
# Reward Median: -0.0021391629610055347
# Reward Stdev: 0.09327849355344207


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class NullAgent:
    def __init__(self, act_dim):
        self.act_dim = act_dim
    def act(self, obs):
        a = np.random.uniform(-1, 1, self.act_dim)
        # a = np.random.normal(0, 0.1, self.act_dim).clip(-1, 1)
        # a = np.zeros(self.act_dim)
        # if torch.is_tensor(obs):
        #     return torch.from_numpy(a).float().to(obs.device).unsqueeze(0)
        return a

    def value(self, obs, act, new_obs):
        # print(obs)
        if new_obs.shape[0] > 1 and len(new_obs.shape) > 1:
            return new_obs[:,0]
        else:
            return new_obs[0]

class MPC:
    def __init__(self, lookahead, dynamics_model, agent=None):
        self.lookahead = lookahead
        self.dynamics_model = dynamics_model
        self.act_dim = self.dynamics_model.act_dim
        # self.act_mul_const = self.dynamics_model.act_mul_const

        self.agent = agent
        if self.agent is None:
            self.agent = NullAgent(self.act_dim)

    def plan(self, env, num_eps):
        final_rs = []
        for i in range(num_eps):
            # start = time.time()
            final_r = self.plan_episode(env)
            # t = round(time.time()-start, 2)
            # print('Trial '+str(i+1)+'/'+str(num_eps)+' completed in '+str(t)+'s with Final Reward: '+str(final_r))
            final_rs.append(final_r)
        final_rs = np.array(final_rs)
        return final_rs

    def plan_episode(self, env):
        done = False
        ep_r = 0
        start = time.time()
        obs = env.reset()
        i = 0
        expected_r = 0
        while not done:
            obs = self.dynamics_model.reset(obs)
            act, node = self.best_move(obs)
            pred_r = node.future[str(act)][-1]
            # pred_r = 0
            # act = self.agent.act(obs[0])
            new_obs, r_raw, done, info = env.step(act)
            expected_r += pred_r
            ep_r += r_raw
            if i%10 == 0:
                print('Timestep '+str(i)+' in '+str(round(time.time()-start, 2))+'s with reward so far: '+str(ep_r))
                print('Expected Reward: '+str(expected_r))
            i += 1
            obs = new_obs
        return ep_r

    def best_move(self, obs):
        raise NotImplementedError

    def clear(self):
        pass