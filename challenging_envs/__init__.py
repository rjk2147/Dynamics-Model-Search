

import gym
from gym.envs.registration import registry, make, spec


def register(id, *args, **kvargs):
  if id in registry.env_specs:
    return
  else:
    return gym.envs.registration.register(id, *args, **kvargs)


register(id='AntBulletEnv-v1',
         entry_point='challenging_envs.envs.gym_locomotion_envs:AntBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)