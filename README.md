# Dynamics Model Search
~~~~
usage: main.py [-h] [--env ENV] [--rl RL] [--planner PLANNER] [--width WIDTH]
               [--depth DEPTH] [--episodes EPISODES] [--batch-size BATCH_SIZE]
               [--replay-size REPLAY_SIZE] [--seed SEED] [--load-all LOAD_ALL]
               [--load-model LOAD_MODEL] [--load-agent LOAD_AGENT]
               [--model-arch MODEL_ARCH] [--use-state] [--model-reward]
               [--cross-entropy] [--no-search]
~~~~

Example Usage:

To train TD3 using DMS with width 4, depth 5 on the Ant environment: 
~~~~
python3 main.py --env AntBulletEnv-v0 --rl TD3 --width 4 --depth 5
~~~~
To train SAC without DMS just using the pure RL baseline on HalfCheetah:
~~~~
python3 main.py --env HalfCheetahBulletEnv-v0 --rl SAC --no-search
~~~~
To train TD3 with DMS with a search space of 64 using the Cross Entropy Method with a lookahead of 4:
~~~~
python3 main.py --env HalfCheetahBulletEnv-v0 --rl SAC --planner CEM --width 64 --depth 4
~~~~
To train just the dynamics model using the "seq" model architecture on Humanoid:
~~~~
python3 main.py --model-arch seq --env HumanoidBulletEnv-v0 --rl Null --no-search
~~~~
