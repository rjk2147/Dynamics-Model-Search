import pybullet_envs
import gym
import torch
from self_mcts import Agent
from models.preco_gen_env_learner import PreCoGenEnvLearner
from models.seq_env_learner import SeqEnvLearner
from pybullet_wrappers import RealerWalkerWrapper
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', type=str, default='JumpAntBulletEnv-v0') # pybullet environment
    parser.add_argument('--agent', type=str, default='TD3') # model free agent algorithm
    parser.add_argument('--width', type=str, default=4) # width of the search tree at every level
    parser.add_argument('--depth', type=int, default=5) # depth of the search tree
    parser.add_argument('--episodes', type=int, default=10000) # training episodes
    parser.add_argument('--batch-size', type=int, default=512) # SM batch size
    parser.add_argument('--replay-size', type=int, default=100000) # SM replay memory size
    parser.add_argument('--seed', type=int, default=None) # Initial seed
    parser.add_argument('--load-all', type=str, default=None) # path to general model
    parser.add_argument('--load-model', type=str, default=None) # path to self-model
    parser.add_argument('--load-agent', type=str, default=None) # path to agent model
    parser.add_argument('--model-arch', type=str, default='seq') # type of self-model
    parser.add_argument('--use-state', action='store_true', default=False)
    parser.add_argument('--model-reward', action='store_true', default=False)
    parser.add_argument('--parallel', action='store_true', default=False)
    parser.add_argument('--cross-entropy', action='store_true', default=False)
    parser.add_argument('--no-search', action='store_true', default=False)
    args = parser.parse_args()
    cmd = 'python main.py --env '+str(args.env)+' --agent '+str(args.agent)+' --width '+str(args.width)+' --depth '+str(args.depth)+\
          ' --episodes '+str(args.episodes)+' --batch-size '+str(args.batch_size)+' --replay-size '+str(args.replay_size)+\
          ' --model-arch '+str(args.model_arch)
    if args.seed is not None: cmd += ' --seed '+str(args.seed)
    if args.use_state:      cmd += ' --use-state'
    if args.model_reward:   cmd += ' --model-reward'
    if args.parallel:       cmd += ' --parallel'
    if args.cross_entropy:  cmd += ' --cross-entropy'
    if args.no_search:      cmd += ' --no-search'
    print(cmd)

    if args.env[:4].lower() == 'jump' and 'Bullet' in args.env:
        print('Jumping task chosen')
        env = RealerWalkerWrapper(gym.make(args.env[4:]), rew='jump')
    elif 'Bullet' in args.env:
        print('Bullet env chosen')
        env = RealerWalkerWrapper(gym.make(args.env))
    else:
        env = gym.make(args.env)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        env.seed(args.seed)

    if args.model_arch == 'precogen':
        env_learner = PreCoGenEnvLearner(env)
    elif args.model_arch == 'seq':
        env_learner = SeqEnvLearner(env)
    agent = Agent(env_learner, width=int(args.width), depth=int(args.depth), agent=args.agent,
                  with_hidden=args.use_state, model_rew=args.model_reward, parallel=args.parallel,
                  cross_entropy=args.cross_entropy, with_tree=not args.no_search, batch_size=args.batch_size,
                  replay_size=args.replay_size)
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
