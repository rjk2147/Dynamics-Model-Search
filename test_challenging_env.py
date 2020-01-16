

import gym
import challenging_envs

env = gym.make('AntBulletEnv-v1')
env.render(mode="human")

obs = env.reset()
height_lst = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

for height in height_lst:
    print(height)
    obs = env.reset()
    # obs_idx = env.unwrapped.load_obstacle((0, 1.2, 0), (0, 0, 0), 'cube', height)

    for k in range(20000):
        a = env.action_space.sample()
        obs, r, done, _ = env.step(a)

    # env.unwrapped.remove_obstacle(obs_idx)