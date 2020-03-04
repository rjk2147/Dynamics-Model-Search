import numpy as np
from collections import deque

class EnvLearner:
    def __init__(self, env_in):
        self.state_mul_const = env_in.observation_space.high
        # print(self.state_mul_const)
        self.state_mul_const[self.state_mul_const == np.inf] = 1
        #add 1 dimension for reward
        self.state_mul_const = np.append(self.state_mul_const, 1)
        print(self.state_mul_const)
        self.act_mul_const = env_in.action_space.high
        self.act_dim = env_in.action_space.shape[0]
        self.state_dim = env_in.observation_space.shape[0]

        self.action_space = env_in.action_space
        self.observation_space = env_in.observation_space

        self.buff_init = [np.zeros(self.state_dim+self.act_dim)]
        self.seq_init = [np.zeros(self.act_dim)]

        self.norm_mean = 0
        self.norm_std = 1

        # To be changed by child classes
        self.buff_len = 1
        self.seq_len = 1
        self.max_seq_len = 1
        self.batch_size = 64

    def initialize(self, session, load=False):
        return NotImplementedError

    def __batch__(self, data, batch_size):
        batches = []
        if batch_size > len(data) or batch_size < 0:
            return [data]
        while len(data) >= batch_size:
            batches.append(data[:batch_size])
            data = data[batch_size:]
        return batches

    def __prep_data__(self, data, batch_size=None, big=False):
        if batch_size is None: batch_size = self.batch_size

        Xs = []
        Ys = []
        As = []

        # x = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
        # a = deque([np.zeros(self.act_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
        # y = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)

        x = deque(maxlen=self.max_seq_len)
        a = deque(maxlen=self.max_seq_len)
        y = deque(maxlen=self.max_seq_len)

        for i in range(len(data)):
            obs = data[i][0]/self.state_mul_const
            act = data[i][1]/self.act_mul_const
            new_obs = data[i][3]/self.state_mul_const

            x.append(obs)
            a.append(act)
            y.append(new_obs)

            x_tmp = np.array(x)
            y_tmp = np.array(y)
            a_tmp = np.array(a)
            # x_tmp = np.concatenate([x_tmp, np.zeros((self.max_seq_len-x_tmp.shape[0], x_tmp.shape[1]))])
            # y_tmp = np.concatenate([y_tmp, np.zeros((self.max_seq_len-y_tmp.shape[0], y_tmp.shape[1]))])
            # a_tmp = np.concatenate([a_tmp, np.zeros((self.max_seq_len-a_tmp.shape[0], a_tmp.shape[1]))])
            if len(x) == self.max_seq_len:
                Xs.append(x_tmp)
                As.append(a_tmp)
                Ys.append(y_tmp)

            reset = data[i][4]

            if reset:
                # x = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
                # a = deque([np.zeros(self.act_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
                # y = deque([np.zeros(self.state_dim)]*self.max_seq_len, maxlen=self.max_seq_len)
                x = deque(maxlen=self.max_seq_len)
                a = deque(maxlen=self.max_seq_len)
                y = deque(maxlen=self.max_seq_len)

        assert len(Ys) == len(As) == len(Xs)
        # p = np.random.permutation(len(Xs))
        p = np.arange(len(Xs))
        if big is False:
            Xs = np.array(Xs)[p]
            As = np.array(As)[p]
            Ys = np.array(Ys)[p]
            Xs = self.__batch__(Xs, self.batch_size)
            As = self.__batch__(As, self.batch_size)
            Ys = self.__batch__(Ys, self.batch_size)
            return Xs, As, Ys
        else:
            return Xs, As, Ys, p

    def train(self, train, epochs, valid=None, log_interval=10, early_stopping=-1, save_str=None):
        return NotImplementedError

    def get_loss(self, data):
        return NotImplementedError

    def reset(self, obs_in, h=None):
        return NotImplementedError

    def step(self, action_in, obs_in=None, save=True, state=False, state_in=None):
        return NotImplementedError
    def get_uncertainty(self, obs_in, action_in, episode_step, save=True, buff=None):
        return 0.01

    def next_move(self, obs_in, episode_step):
        return np.random.uniform(-1, 1, self.act_dim)
