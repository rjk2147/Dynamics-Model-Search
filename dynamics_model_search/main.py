import pybullet_envs
import gym
import torch
import numpy as np
from agent import Agent
from pybullet_wrappers import RealerWalkerWrapper
import argparse
import torch
torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Algorithms
    parser.add_argument('--env', type=str, default='AntBulletEnv-v0') # pybullet environment
    # parser.add_argument('--env', type=str, default='Pong-v0') # pybullet environment
    parser.add_argument('--rl', type=str, default='SAC') # model free agent algorithm
    parser.add_argument('--planner', type=str, default='MCTS-UCT') # model based algorithm
    parser.add_argument('--model-arch', type=str, default='2dcnn') # type of self-model
    parser.add_argument('--atari', action='store_true', default=False)

    # Training Parameters
    parser.add_argument('--steps', type=int, default=1e6) # training steps
    parser.add_argument('--batch-size', type=int, default=512) # SM batch size
    parser.add_argument('--seq-len', type=int, default=20) # SM sequence modeling window size
    parser.add_argument('--replay-size', type=int, default=100000) # SM replay memory size

    parser.add_argument('--width', type=str, default=8) # width of the search tree at every level
    parser.add_argument('--depth', type=int, default=5) # depth of the search tree
    parser.add_argument('--nodes', type=int, default=2048) # depth of the search tree

    # Repeatability
    parser.add_argument('--seed', type=int, default=None) # Initial seed
    parser.add_argument('--load-all', type=str, default=None) # path to general model
    parser.add_argument('--load-model', type=str, default=None) # path to self-model
    parser.add_argument('--load-agent', type=str, default=None) # path to agent model

    args = parser.parse_args()
    cmd = 'python main.py --env '+str(args.env)+' --agent '+str(args.rl)+' --planner '+str(args.planner)+' --width '+str(args.width)+\
          ' --depth '+str(args.depth)+' --steps '+str(args.steps)+' --batch-size '+str(args.batch_size)+\
          ' --replay-size '+str(args.replay_size)+' --model-arch '+str(args.model_arch)
    if args.seed is not None: cmd += ' --seed '+str(args.seed)
    print(cmd)

    if args.env[:4].lower() == 'jump' and 'Bullet' in args.env:
        print('Jumping task chosen')
        env = RealerWalkerWrapper(gym.make(args.env[4:]), rew='jump')
    elif 'Bullet' in args.env:
        print('Bullet env chosen')
        env = RealerWalkerWrapper(gym.make(args.env))
    elif args.atari:
        from atari_wrapper import make_atari, wrap_deepmind, wrap_pytorch
        print('Atari env chosen')

        env = make_atari(args.env)
        env = wrap_deepmind(env)
        env = wrap_pytorch(env)
    else:
        env = gym.make(args.env)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        env.seed(args.seed)

    ensemble = False
    if args.model_arch[:len('ensemble-')] == 'ensemble-':
        ensemble = True
        args.model_arch = args.model_arch[len('ensemble-'):]

    if args.model_arch == 'precogen':
        from models.preco_gen_dynamics_model import PreCoGenDynamicsModel as DyanmicsModel
    elif args.model_arch == 'rnn':
        from models.rnn_dynamics_model import RNNDynamicsModel as DyanmicsModel
    elif args.model_arch == 'rnn-vae':
        from models.rnn_vae import RNNVAE as DyanmicsModel
    elif args.model_arch == 'mdrnn':
        from models.mdrnn_dynamics_model import MDRNNDynamicsModel as DyanmicsModel
    elif args.model_arch == 'mdn-seq':
        from models.mdn_seq_dynamics_model import MDNSeqDynamicsModel as DyanmicsModel
    elif args.model_arch == 'latent-seq':
        from models.latent_seq_dynamics_model import LatentSeqDynamicsModel as DyanmicsModel
    elif args.model_arch == 'vrnn':
        from models.vrnn_dynamics_model import VRNNDynamicsModel as DyanmicsModel
    elif args.model_arch == 'bseq':
        from models.bayesian_dynamics_model import BayesianSequenceDynamicsModel as DyanmicsModel
    elif args.model_arch == 'biased-bseq':
        from models.biased_bayesian_dynamics_model import BayesianSequenceDynamicsModel as DyanmicsModel
    elif args.model_arch == 'seq-cnn':
        from models.cnn1D_dynamics_model import SeqCNNDynamicsModel as DyanmicsModel
    elif args.model_arch == '1dcnn':
        from models.cnn1D_dynamics_model import SeqCNNDynamicsModel as DyanmicsModel
    elif args.model_arch == '2dcnn':
        from models.cnn2D_dynamics_model import SeqCNNDynamicsModel as DyanmicsModel
    elif args.model_arch.lower() == 'none': # TODO Test to ensures this doesn't break anything
        DyanmicsModel = None
        dynamics_model = None
    else:
        print('No Valid Dynamics Model chosen exiting...')
        exit(1)

    if DyanmicsModel is not None:
        if ensemble:
            from models.ensemble import Ensemble as Ensemble
            # from models.ensemble_parallel import ParallelEnsemble as Ensemble
            dynamics_model = Ensemble(DyanmicsModel, env)
            args.batch_size = int(args.batch_size)*dynamics_model.ensemble_size
        else:
            dynamics_model = DyanmicsModel(env, seq_len=int(args.seq_len))

    if args.rl.upper() == 'TD3':
        from model_free.TD3 import TD3
        rl_learner = TD3(env)
    elif args.rl.upper() == 'SAC':
        from model_free.SAC import SAC
        rl_learner = SAC(env)
    elif args.rl.upper() == 'DQN':
        from model_free.DQN import DQN
        rl_learner = DQN(env)
    elif args.rl.upper() == 'DDQN':
        from model_free.DDQN import DDQN
        rl_learner = DDQN(env)
    elif args.rl.lower() == 'none' or args.rl.lower() == 'null':
        from model_free.Null import NullAgent
        rl_learner = NullAgent(env)
    else:
        from model_free.Null import NullAgent
        rl_learner = NullAgent(env)

    if args.planner == 'MCTS':
        from model_based.mcts import MCTS
        planner = MCTS(int(args.depth), dynamics_model, rl_learner, int(args.width))
    if args.planner == 'MCTS-UCT':
        from model_based.mcts_uct import MCTS
        planner = MCTS(int(args.depth), dynamics_model, rl_learner, int(args.width), nodes=int(args.nodes))
    elif args.planner == 'CEM':
        from model_based.cem import CEM
        planner = CEM(int(args.depth), dynamics_model, rl_learner, int(args.width))
    elif args.planner.lower() == 'null' or args.planner.lower() == 'none':
        planner = None
    else:
        from model_based.mcts import MCTS
        planner = MCTS(int(args.depth), dynamics_model, rl_learner, int(args.width))

    if dynamics_model is None and planner is not None:
        print('Error: Cannot have a null model with a planner')
        exit(1)

    agent = Agent(dynamics_model, rl_learner, planner,
                  batch_size=int(args.batch_size), replay_size=int(args.replay_size), seq_len=int(args.seq_len))
    if args.load_all is not None:
        args.load_model = args.load_all
        args.load_agent = args.load_all
    if args.load_model is not None:
        print('Loading Model...')
        dynamics_model.load(args.load_model+'_self_model.pt')
    if args.load_agent is not None:
        print('Loading Agent...')
        agent.rl_learner.load(args.load_agent)
    agent.learn(env, int(args.steps))
