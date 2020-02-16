import gym
import pybullet_envs
import numpy as np
from data_loader import DataLoader
import os

if __name__ == "__main__":
    env = gym.make('AntBulletEnv-v0')
    episodes = 2000
    timesteps = 100
    experiences = []

    #----------------- COLLECT TRAJECTORIES ---------------------
    # env.render()
    for episode in range(episodes):
        observation = env.reset()
        for t in range(timesteps):
            # env.render()
            action = env.action_space.sample()
            next_observation, reward, done, info = env.step(action)
            experiences.append((observation[3:-4], action, next_observation[3:-4], done))   # first 3 elements in state are target quantities, last 4 elements are feet contact quantities
            if done:
                print("Episode %d finished after %d timesteps" % (episode, t+1))
                break
            observation = np.copy(next_observation)
        print("Episode %d finished after %d timesteps" % (episode, 100))

    env.close()
    #----------------- COLLECT TRAJECTORIES ---------------------

    dataloader = DataLoader(env_in=env)
    X, A, Y = dataloader.process_data(data=experiences, max_seq_len=100) 

    dataloader.save_data(X, path=os.getcwd()+"/../datasets/states.npy")    
    dataloader.save_data(A, path=os.getcwd()+"/../datasets/actions.npy")    
    dataloader.save_data(Y, path=os.getcwd()+"/../datasets/next_states.npy")    

    print("Trajectories saved with the shapes: ", X.shape, A.shape, Y.shape)