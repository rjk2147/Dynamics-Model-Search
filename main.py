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
    parser.add_argument('--width', type=str, default=4) # width of the search tree at every level
    parser.add_argument('--depth', type=int, default=5) # depth of the search tree
    parser.add_argument('--episodes', type=int, default=10000) # training episodes
    parser.add_argument('--load-all', type=str, default=None) # path to general model
    parser.add_argument('--load-model', type=str, default=None) # path to self-model
    parser.add_argument('--load-agent', type=str, default=None) # path to agent model
    parser.add_argument('--use-state', action='store_true', default=False)
    parser.add_argument('--model-reward', action='store_true', default=False)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--cross-entropy', action='store_true', default=True)
    args = parser.parse_args()

    print(args.use_state)
    env = RealerWalkerWrapper(gym.make(args.env))
    env_learner = PreCoGenEnvLearner(env)
    agent = Agent(env_learner, width=int(args.width), depth=int(args.depth), agent=args.agent,
                  with_hidden=args.use_state, model_rew=args.model_reward, parallel=args.parallel,
                  cross_entropy=args.cross_entropy)
    if args.load_all is not None:
        args.load_model = args.load_all
        args.load_agent = args.load_all
    if args.load_model is not None:
        print('Loading Model...')
        env_learner.load(args.load_model+'_self_model.pt')
    if args.load_agent is not None:
        print('Loading Agent...')
        agent.rl_learner.load(args.load_agent)
    agent.learn(env, int(args.episodes))
