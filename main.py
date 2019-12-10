import pybullet_envs
import gym
from self_mcts import Agent
from models.preco_gen_env_learner import PreCoGenEnvLearner
from pybullet_wrappers import RealerWalkerWrapper
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', type=str, default='AntBulletEnv-v0') # pybullet environment
    parser.add_argument('--agent', type=str, default='TD3') # model free agent algorithm
    parser.add_argument('--width', type=str, default=64) # width of the search tree at every level
    parser.add_argument('--depth', type=int, default=1) # depth of the search tree
    parser.add_argument('--episodes', type=int, default=10000) # training episodes
    args = parser.parse_args()

    # print(args.env)
    # print(args.agent)
    # print(args.width)
    # print(args.depth)
    # print(args.episodes)

    env = RealerWalkerWrapper(gym.make(args.env))
    env_learner = PreCoGenEnvLearner(env)
    agent = Agent(env_learner, width=int(args.width), depth=int(args.depth), agent=args.agent)
    agent.learn(env, int(args.episodes))
    # agent.play(env, 10)
