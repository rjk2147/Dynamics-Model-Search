import pybullet_envs
import gym
from self_mcts import Agent
from models.preco_gen_env_learner import PreCoGenEnvLearner
from pybullet_wrappers import RealerWalkerWrapper

if __name__ == '__main__':
    env = RealerWalkerWrapper(gym.make("AntBulletEnv-v0"))
    env_learner = PreCoGenEnvLearner(env)
    agent = Agent(env_learner)
    agent.learn(env, 10000)
    # agent.play(env, 10)
