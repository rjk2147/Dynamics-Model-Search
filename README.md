# Dynamics Model Search

## Installation
Clone the repo and cd into it
~~~
git clone https://github.com/rjk2147/Dynamics-Model-Search
cd Dynamics-Model-Search
~~~
Install the package
~~~
pip3 install -e .
~~~

## Usage

~~~~
usage: main.py [-h] [--env ENV] [--rl RL] [--planner PLANNER]
               [--model-arch MODEL_ARCH] [--atari] [--steps STEPS]
               [--batch-size BATCH_SIZE] [--seq-len SEQ_LEN]
               [--replay-size REPLAY_SIZE] [--width WIDTH] [--depth DEPTH]
               [--nodes NODES] [--seed SEED] [--load-all LOAD_ALL]
               [--load-model LOAD_MODEL] [--load-agent LOAD_AGENT]
~~~~

Example Usage:

To train TD3 using UAMS with an MDRNN and width 8, depth 5 and exploring 2048 tree branches on the Ant environment: 
~~~~
python3 main.py --env AntBulletEnv-v0 --rl TD3 --width 8 --depth 5 --nodes 2048 --planner MCTS-UCT --model-arch mdrnn
~~~~
To train TD3 using DMS with DQN and a CNN architecture on the Pong Atari environment: 
~~~~
python3 main.py --env PongNoFrameskip-v4 --rl DQN --model-arch seq-cnn --atari
~~~~
To train SAC without DMS just using the pure RL baseline on HalfCheetah:
~~~~
python3 main.py --env HalfCheetahBulletEnv-v0 --rl SAC --planner none--planner none --model-arch none
~~~~
To train just the dynamics model using the "rnn" model architecture on Humanoid:
~~~~
python3 main.py --model-arch rnn --env HumanoidBulletEnv-v0 --rl none --planner none 
~~~~
