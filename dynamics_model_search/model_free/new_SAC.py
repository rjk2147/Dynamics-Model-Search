from copy import deepcopy
import itertools
import numpy as np
from torch.optim import Adam
import gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            # return a.cpu().numpy()
            return a

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size


    def add(self, state, action, next_state, reward, done):
        return self.store(state, action, reward, next_state, done)

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def __len__(self):
        return self.size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC(object):
    def __init__(self, env, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1):
        """
        Soft Actor-Critic (SAC)

        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with an ``act``
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of
                observations as inputs, and ``q1`` and ``q2`` should accept a batch
                of observations and a batch of actions as inputs. When called,
                ``act``, ``q1``, and ``q2`` should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each
                                               | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                               | of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current
                                               | estimate of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ===========  ================  ======================================

                Calling ``pi`` should return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                               | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                               | actions in ``a``. Importantly: gradients
                                               | should be able to flow back into ``a``.
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to SAC.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target
                networks. Target networks are updated towards main networks
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually
                close to 1.)

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to
                inverse of reward scale in the original SAC paper.)

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.

            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long
                you wait between updates, the ratio of env steps to gradient steps
                is locked to 1.

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """

        self.gamma = gamma
        self.alpha = alpha
        self.polyak = polyak
        self.batch_size = batch_size
        self.update_after = update_after
        self.update_every = update_every
        self.start_steps = start_steps
        self.steps = 0

        torch.manual_seed(seed)
        np.random.seed(seed)

        # env, test_env = env_fn(), env_fn()
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape[0]
        self.action_space = env.action_space

        # Create actor-critic module and target networks
        self.ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
        self.ac = self.ac.to(device)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o = o.to(device)
        a = a.to(device)
        r = r.to(device)
        o2 = o2.to(device)
        d = d.to(device)

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        o = o.to(device)
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        return loss_pi, pi_info

    def update(self, batch_size=None, n_updates=None):
        if self.steps >= self.update_after and self.steps % self.update_every == 0:
            for j in range(self.update_every):
                batch = self.replay.sample_batch(self.batch_size)
                self.__update__(data=batch)

    def __update__(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32).to(device),
                      deterministic)

    def act(self, obs):
        if self.steps < self.start_steps:
            sample = np.array([self.action_space.sample() for i in range(len(obs))])
            return torch.from_numpy(sample).to(device)
        if torch.is_tensor(obs):
            obs = obs.to(device)
        else:
            obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        action = self.get_action(obs).detach()
        return action

    def value(self, obs):
        if not torch.is_tensor(obs):
           obs = torch.Tensor(obs).to(device)
        if len(obs.shape) > 2:
            obs = obs.unsqueeze(1)
        pi, logp_pi = self.ac.pi(obs)
        q1_pi = self.ac.q1(obs, pi)
        q2_pi = self.ac.q2(obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        return q_pi.unsqueeze(-1).cpu().detach().numpy()

    def save(self, filename):
        pass

    def load(self, filename):
        pass

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    torch.set_num_threads(torch.get_num_threads())
    env = gym.make(args.env)
    env = env.env
    sac = SAC(env, actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs)
    # Prepare for interaction with environment
    total_steps = 1000000
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    from collections import deque
    rews = deque(maxlen = 4)
    ep_n = 0
    max_ep_len = 1000

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        sac.steps = t
        # a = sac.act(o)

        a = sac.act(np.expand_dims(o, 0))
        a = a.cpu().detach().numpy().flatten()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        sac.replay.add(o, a, o2, r, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            rews.append(ep_ret)
            ep_n += 1

            print('Timesteps: '+str(t+1))
            print('Time: '+str(round(time.time()-start_time, 3)))
            print('Avg Reward: '+str(np.mean(rews)))
            print('Episode '+str(ep_n))
            print('Length: '+str(ep_len))
            print('Return: '+str(ep_ret))
            print('')
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        sac.update()

        # End of epoch handling
        # if (t + 1) % steps_per_epoch == 0:
        #     epoch = (t + 1) // steps_per_epoch
