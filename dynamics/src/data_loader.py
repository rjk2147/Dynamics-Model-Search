import numpy as np
from collections import deque
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, env_in): 
        self.action_space = env_in.action_space
        self.observation_space = env_in.observation_space
        self.act_dim = env_in.action_space.shape[0]
        self.state_dim = env_in.observation_space.shape[0]
        # normalizing constants for the state and action vectors
        self.state_norm_const = env_in.observation_space.high[3:-4]     # ignore the first three and last four state space values
        self.state_norm_const[self.state_norm_const == np.inf] = 1
        self.action_norm_const = env_in.action_space.high

    # returns data as batches
    @staticmethod
    def batch(data, batch_size):
        batches = []
        if batch_size > len(data) or batch_size < 0:
            return np.array([data])
        while len(data) >= batch_size:
            batches.append(data[:batch_size])
            data = data[batch_size:]
        return np.array(batches)

    @staticmethod
    def split_train_val_test(X_in, Y_in, train=0.7, val=0.2, test=0.1):
        X, X_test, Y, Y_test = train_test_split(X_in, Y_in, test_size=test)
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val/(1-test))
        X_train, Y_train = X_train, Y_train
        X_val, Y_val = X_val, Y_val
        X_test, Y_test = X_test, Y_test
        return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

    @staticmethod
    def save_data(X, path):
        np.save(path, X)

    # data is an array of tuples of the form (s, a, s', done)
    def process_data(self, data, max_seq_len=100):
        # if batch_size is None: batch_size = self.batch_size
        # X = [s...], A = [a...], Y = [s'...]
        # for each (s, a, s', done) tuple
        X, A, Y = [], [], []

        # after max_seq_len elements from the beginning of array are removed to maintain max length
        x = deque(maxlen=max_seq_len)
        a = deque(maxlen=max_seq_len)
        y = deque(maxlen=max_seq_len)

        for i in range(len(data)):
            curr_s, action, next_s, done = data[i][0], data[i][1], data[i][2], data[i][3]
            
            # normalize states and action
            curr_s /= self.state_norm_const
            next_s /= self.state_norm_const
            action /= self.action_norm_const
            x.append(curr_s)
            a.append(action)
            y.append(next_s)

            if len(x) == max_seq_len:
                X.append(np.array(x))
                A.append(np.array(a))
                Y.append(np.array(y))
            if done:
                x = deque(maxlen=max_seq_len)
                a = deque(maxlen=max_seq_len)
                y = deque(maxlen=max_seq_len)

        assert len(X) == len(A) == len(Y)

        # shuffle the data
        p = np.random.permutation(len(X))
        # p = np.arange(len(X))
        X = np.array(X)[p]
        A = np.array(A)[p]
        Y = np.array(Y)[p]
        return X, A, Y

        
    