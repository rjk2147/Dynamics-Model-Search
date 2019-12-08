import pybullet, pybullet_envs
from pybullet_wrappers import NullWrapper, RealerWalkerWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import gym
import numpy as np
import datetime
import argparse
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.td3.policies import MlpPolicy as TD3MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPGMlpPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import SAC, PPO1, A2C, DDPG, TD3, TRPO
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    def reward(self, r):
        return r * self.scale

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', default="AntBulletEnv-v0")
parser.add_argument('--algorithm', default="td3")
parser.add_argument('--play', default=10)
parser.add_argument('--num_timesteps', default=1000000)
parser.add_argument('--load', default=None)
args = parser.parse_args()
if args.env == 'ant':
    env_name = 'AntBulletEnv-v0'
    env = RealerWalkerWrapper(gym.make('AntBulletEnv-v0'))
else:
    env_name = args.env
    env = RealerWalkerWrapper(gym.make(args.env))
if args.algorithm == 'sac':
    print('SAC selected')
    model = SAC(SACMlpPolicy, env, verbose=1)
    model_class = SAC
elif args.algorithm == 'ppo':
    print('PPO selected')
    model = PPO1(MlpPolicy, env, verbose=1)
    model_class = PPO1
elif args.algorithm == 'trpo':
    print('TRPO selected')
    model = TRPO(MlpPolicy, env, verbose=1)
    model_class = TRPO
elif args.algorithm == 'a2c':
    print('A2C selected')
    model = A2C(MlpPolicy, DummyVecEnv([env]), verbose=1)
    model_class = A2C
elif args.algorithm == 'ddpg':
    print('DDPG selected')
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
    model = DDPG(DDPGMlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
    model_class = DDPG
elif args.algorithm == 'td3':
    print('TD3 selected')
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = TD3(TD3MlpPolicy, env, action_noise=action_noise, verbose=1)
    model_class = TD3
else:
    print('Unknown algorithm \"'+args.algorithm+'\" chosen')
    print('Defaulting to SAC')
    model = SAC(SACMlpPolicy, env, verbose=1)
    model_class = SAC

timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
if int(args.num_timesteps) > 0:
    model.learn(total_timesteps=int(args.num_timesteps), log_interval=10)
    # model.save(env_name+"_"+args.algorithm+'_model')
if args.load is not None:
    model = model_class.load(args.load)
num_episodes = int(args.play)
if num_episodes > 0:
    poses = []
    for i in range(num_episodes):
        n = 0
        obs = env.reset()
        # print(obs)
        done = False
        pos = 0
        while not done:
            action, _states = model.predict(obs)
            action = action+np.random.normal(0, 0.1, env.action_space.shape[-1])
            obs, rewards, done, info = env.step(action)
            pos += obs[0]
            # env.render()
            # if n%10 == 0:
            #     print('Timestep '+str(n)+' with reward so far: '+str(pos))
            n += 1
            if done:
                print(str(i)+'/'+str(num_episodes)+' in '+str(n)+' steps: '+str(pos))
                print(env.ep_rew)
                poses.append(pos)
    poses = np.array(poses)
    print('Mean: '+str(np.mean(poses)))
    print('Median: '+str(np.median(poses)))
    print('Stdev: '+str(np.std(poses)))
