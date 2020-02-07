# Dynamics Model Search
~~~~
usage: main.py [-h] [--env ENV] [--agent AGENT] [--width WIDTH]
               [--depth DEPTH] [--episodes EPISODES] [--batch-size BATCH_SIZE]
               [--replay-size REPLAY_SIZE] [--seed SEED] [--load-all LOAD_ALL]
               [--load-model LOAD_MODEL] [--load-agent LOAD_AGENT]
               [--model-arch MODEL_ARCH] [--use-state] [--model-reward]
               [--parallel] [--cross-entropy] [--no-search]
~~~~

Example Usage:

To train TD3 using DMS with width 4, depth 5 on the Ant environment: 
~~~~
python3 main.py --env AntBulletEnv-v0 --agent TD3 --width 4 --depth 5
~~~~
To train SAC without DMS just using the pure RL baseline on HalfCheetah:
~~~~
python3 main.py --env HalfCheetahBulletEnv-v0 --agent SAC --no-search
~~~~
To train just the dynamics model on HalfCheetah:
~~~~
python3 main.py --env HalfCheetahBulletEnv-v0 --agent Null
~~~~
