"""
    From https://github.com/transedward/pytorch-dqn
"""
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from stable_baselines.common.policies import nature_cnn
import tensorflow as tf
# print(tf.__version__) # 1.15.0
import tfpyth
from stable_baselines.common.tf_layers import linear
import tensorflow.contrib.layers as tf_layers


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def linear_interpolation(left, right, alpha):
    """
    Linear interpolation between `left` and `right`.
    :param left: (float) left boundary
    :param right: (float) right boundary
    :param alpha: (float) coeff in [0, 1]
    :return: (float)
    """

    return left + alpha * (right - left)


class PiecewiseSchedule(object):
    """
    Piecewise schedule.
    :param endpoints: ([(int, int)])
        list of pairs `(time, value)` meaning that schedule should output
        `value` when `t==time`. All the values for time must be sorted in
        an increasing order. When t is between two times, e.g. `(time_a, value_a)`
        and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
        `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
        time passed between `time_a` and `time_b` for time `t`.
    :param interpolation: (lambda (float, float, float): float)
        a function that takes value to the left and to the right of t according
        to the `endpoints`. Alpha is the fraction of distance from left endpoint to
        right endpoint that t has covered. See linear_interpolation for example.
    :param outside_value: (float)
        if the value is requested outside of all the intervals specified in
        `endpoints` this value is returned. If None then AssertionError is
        raised when outside value is requested.
    """

    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, step):
        for (left_t, left), (right_t, right) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if left_t <= step < right_t:
                alpha = float(step - left_t) / (right_t - left_t)
                return self._interpolation(left, right, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the typical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx = 0
        self.num_in_buffer = 0

        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        # print(self.num_in_buffer)
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch = np.concatenate([self._encode_observation(idx)[np.newaxis, :] for idx in idxes], 0)
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[np.newaxis, :] for idx in idxes], 0)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def add(self, state, action, next_state, reward, done):
        ### Step the env and store the transition
        # Store lastest observation in replay memory and last_idx can be used to store action, reward, done
        # last_idx = self.replay.store_frame(last_obs)
        last_idx = self.store_frame(state)
        # Store other info in replay memory
        self.store_effect(last_idx, action, reward, done)

    def __len__(self):
        return self.num_in_buffer

    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_c * frame_history_len, img_h, img_w)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """

        def sample_n_unique(sampling_f, n):
            """Helper function. Given a function `sampling_f` that returns
            comparable objects, sample n such unique objects.
            """
            res = []
            while len(res) < n:
                candidate = sampling_f()
                if candidate not in res:
                    res.append(candidate)
            return res

        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        return self.obs[idx]
        # end_idx   = idx + 1 # make noninclusive
        # start_idx = end_idx - self.frame_history_len
        # # this checks if we are using low-dimensional observations, such as RAM
        # # state, in which case we just directly return the latest RAM.
        # if len(self.obs.shape) == 2:
        #     return self.obs[end_idx-1]
        # # if there weren't enough frames ever in the buffer for context
        # if start_idx < 0 and self.num_in_buffer != self.size:
        #     start_idx = 0
        # for idx in range(start_idx, end_idx - 1):
        #     if self.done[idx % self.size]:
        #         start_idx = idx + 1
        # missing_context = self.frame_history_len - (end_idx - start_idx)
        # # if zero padding is needed for missing context
        # # or we are on the boundry of the buffer
        # if start_idx < 0 or missing_context > 0:
        #     frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
        #     for idx in range(start_idx, end_idx):
        #         frames.append(self.obs[idx % self.size])
        #     return np.concatenate(frames, 0)
        # else:
        #     # this optimization has potential to saves about 30% compute time \o/
        #     img_h, img_w = self.obs.shape[2], self.obs.shape[3]
        #     return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            and the frame will transpose to shape (img_h, img_w, img_c) to be stored
        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        # make sure we are not using low-dimensional observations, such as RAM
        # if len(frame.shape) > 1:
        #     transpose image frame into (img_c, img_h, img_w)
        # frame = frame.transpose(2, 0, 1)

        if self.obs is None:
            self.obs = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action = np.empty([self.size], dtype=np.int32)
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.bool)

        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that one can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        data = data.to(self.device)
        super(Variable, self).__init__(data, *args, **kwargs)


# class DQNCNNModel(nn.Module):
#     def __init__(self, in_channels=4, num_actions=18):
#         """
#         Initialize a deep Q-learning network as described in
#         https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
#         Arguments:
#             in_channels: number of channel of input.
#                 i.e The number of most recent frames stacked together as describe in the paper
#             num_actions: number of action-value to output, one-to-one correspondence to action in game.
#         """
#         super(DQNCNNModel, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(8,1), stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,1), stride=2)  # 4 default refers to (4*4), different to 4*1
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,1), stride=1)
#         self.fc4 = nn.Linear(1 * 7 * 64, 512)
#         self.fc5 = nn.Linear(512, num_actions)
#
#     def forward(self, x):
#         x = x / 255.0
#         # From N, W, H, C to N, C, H, W
#         # print(x.size()) # torch.Size([1, 1, 84, 84])
#         x = x.permute(0, 3, 2, 1)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.fc4(x.view(x.size(0), -1)))
#         return self.fc5(x)
#
# class DQNLinearModel(nn.Module):
#     def __init__(self, state_dim=4, act_dim=1):
#         super(DQNLinearModel, self).__init__()
#         self.fc4 = nn.Linear(state_dim, 512)
#         self.fc5 = nn.Linear(512, act_dim)
#
#     def forward(self, x):
#         x = F.relu(self.fc4(x))
#         return self.fc5(x)

def cnn_model(scaled_images, **kwargs):
    """
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    with tf.compat.v1.variable_scope(kwargs["name"]):
        scaled_images = tf.cast(scaled_images, tf.float32) / 255.0
        scaled_images = tf.transpose(scaled_images, [0, 3, 2, 1])
        layer_out = nature_cnn(scaled_images)
        action_scores = tf_layers.fully_connected(layer_out, num_outputs=kwargs["num_actions"], activation_fn=None)
    return action_scores



def linear_model(scaled_images, **kwargs):
    with tf.compat.v1.variable_scope(kwargs["name"]):
        extracted_features = tf.layers.flatten(scaled_images)
        action_out = extracted_features
        action_out = tf_layers.fully_connected(action_out, num_outputs=512, activation_fn=None)
        action_out = tf.nn.relu(action_out)
        # action_scores = tf_layers.fully_connected(action_out, num_outputs=self.n_actions, activation_fn=None)
        action_scores = tf_layers.fully_connected(action_out, num_outputs=kwargs["num_actions"], activation_fn=None)
    return action_scores


# def build_calculation(self) -> None:
#     self.o_b = tf.compat.v1.placeholder(tf.uint8, [None, 1, 84, 84], name="obs_batch")
#     self.a_b = tf.compat.v1.placeholder(tf.int32, [None], name="act_batch")
#     self.r_b = tf.compat.v1.placeholder(tf.float32, [None], name="rew_batch")
#     self.next_o_b = tf.compat.v1.placeholder(tf.uint8, [None, 1, 84, 84], name="next_obs_batch")
#     self.d_m = tf.compat.v1.placeholder(tf.float32, [None], name="done_mask")
#     not_done_mask = tf.convert_to_tensor(1 - self.d_m)
#     current_Q_values = tf.gather(self.model, self.a_b, axis=1)  # act is the index of sec dim.
#     # .squeeze() not work for multi dim.#tf also have gather
#     # Compute next Q value based on which action gives max Q values
#     # Detach variable from the current graph since we don't want gradients for next Q to propagated
#     next_max_q = tf.reduce_max(self.target_model, axis=1)  # .detach().max(1)[0]
#     next_Q_values = not_done_mask * next_max_q
#
#     # Compute the target of the current Q values
#     target_Q_values = self.r_b + (self.gamma * next_Q_values)
#     # Compute Bellman error
#     bellman_error = tf.stop_gradient(target_Q_values) - current_Q_values
#     # clip the bellman error between [-1 , 1]
#     clipped_bellman_error = tf.clip_by_value(bellman_error, -1, 1)  # tf.clip_by_value vs pytorch .clamp vs np.clip
#     # Note: clipped_bellman_delta * -1 will be right gradient
#     self.d_error = clipped_bellman_error * -1.0

# exploration = PiecewiseSchedule([
#     (0, 1.0),  # (0, 1.0),
#     (1e6, 0.1),  # (1e6, 0.1)
#     (2e6, 0.01),  # (2e6, 0.1)  # 0.01 (5e6, 0.01)
#     (8e6, 0.001)  # (8e6, 0.005)
# ], outside_value=0.001),  # 0.005

class DQN():
    def __init__(self,
                 env,
                 exploration=PiecewiseSchedule([
                    (0, 1.0),  # (0, 1.0), # (1e6, 0.1)
                    (4e6, 0.01),  # (2e6, 0.1)  # 0.01 (5e6, 0.01)
                    (8e6, 0.001)  # (8e6, 0.005)
                ], outside_value=0.001),
                 replay_buffer_size=90000,
                 gamma=0.99,
                 lr=0.001, # 0.00008 even decrease# 0.0005 do not learn
                 alpha=0.90,
                 eps=0.01,
                 learning_starts=50000,  # 50000
                 learning_freq=4,
                 frame_history_len=4,
                 target_update_freq=10000 # 10000
                 ):
        # LinearSchedule(1e6, 0.1)
        """Run Deep Q-learning algorithm.

        You can specify your own convnet using q_func.

        All schedules are w.r.t. total number of steps taken in the environment.

        Parameters
        ----------
        env: gym.Env
            gym environment to train on.
        q_func: function
            Model to use for computing the q function. It should accept the
            following named arguments:
                input_channel: int
                    number of channel of input.
                num_actions: int
                    number of actions
        optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate schedule
            for the optimizer
        exploration: Schedule (defined in utils.schedule)
            schedule for probability of chosing random action.
        stopping_criterion: (env) -> bool
            should return true when it's ok for the RL algorithm to stop.
            takes in env and the number of steps executed so far.
        replay_buffer_size: int
            How many memories to store in the replay buffer.
        batch_size: int
            How many transitions to sample each time experience is replayed.
        gamma: float
            Discount Factor
        learning_starts: int
            After how many environment steps to start replaying experiences
        learning_freq: int
            How many steps of environment to take between every experience replay
        frame_history_len: int
            How many past frames to include as input to the model.
        target_update_freq: int
            How many experience replay rounds (not steps!) to perform between
            each update to the target Q network
        """
        assert type(env.observation_space) == gym.spaces.Box
        assert type(env.action_space) == gym.spaces.Discrete

        ###############
        # BUILD MODEL #
        ###############

        # if len(env.observation_space.shape) == 1:
        #     # This means we are running on low-dimensional observations (e.g. RAM)
        #     DQNModel = DQNLinearModel
        # else:
        #     DQNModel = DQNCNNModel
        # input_arg = env.observation_space.shape[-1] # 84
        self.num_actions = env.action_space.n
        self.learning_starts = learning_starts
        self.gamma = gamma
        self.learning_freq = learning_freq
        self.target_update_freq = target_update_freq
        self.exploration = exploration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # Initialize target q function and q function
        # # input_arg = frame_history_len
        # self.Q = DQNModel(input_arg, self.num_actions).type(dtype).to(self.device)
        # self.target_Q = DQNModel(input_arg, self.num_actions).type(dtype).to(self.device)
        #
        # # Construct Q network optimizer function
        # self.optimizer = optim.RMSprop(self.Q.parameters(), lr, alpha, eps)

        # Construct the replay buffer
        self.replay = ReplayBuffer(replay_buffer_size, frame_history_len)

        self.steps = 0
        self.num_param_updates = 0

        # Modified by Yu
        self.optimizer_tf = tf.compat.v1.train.RMSPropOptimizer(lr, alpha, eps)
        self.x = tf.compat.v1.placeholder(tf.float32, [None, 1, 84, 84], name="state")
        self.target_x = tf.compat.v1.placeholder(tf.float32, [None, 1, 84, 84], name="target_state")
        self.sess = tf.compat.v1.Session()

        with tf.variable_scope("model", reuse=tf.compat.v1.AUTO_REUSE):
            if len(env.observation_space.shape) == 1:
                self.model = linear_model(self.x, num_actions=self.num_actions, name="q_model")
                self.target_model = linear_model(self.target_x, num_actions=self.num_actions, name="target_q_model")
            else:
                self.model = cnn_model(self.x, num_actions=self.num_actions, name="q_model")
                self.target_model = cnn_model(self.target_x, num_actions=self.num_actions, name="target_q_model")
        # Get all the variables in the Q primary network.
        self.q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model/q_model")
        # Get all the variables in the Q target network.
        self.q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="model/target_q_model")
        self.predict = tf.argmax(self.model, 1)
        self.target_predict = tf.argmax(self.target_model, 1)

        self.build_calculation()

        # self.error = tf.placeholder(tf.float32, shape=[None, self.num_actions])
        self.error = tf.placeholder(tf.float32, shape=[None])
        self.loss = tf.losses.mean_squared_error(self.error, tf.reduce_max(self.model, axis=1))
        # self.loss = tf.reduce_mean(tf.square(self.error), name="loss")
        # self.train_op = self.optimizer_tf.minimize(self.loss, var_list=self.model, name="rms_optimizer")
        self.train_op = self.optimizer_tf.minimize(self.loss, name="rms_optimizer") #####
        # self.train_op = self.optimizer_tf.minimize(tf.reduce_mean(tf.square(self.error), name="rms_optimizer"))

        # self.grads = tf.gradients(self.loss, self.q_vars)
        # self.train_op = self.optimizer_tf.apply_gradient(zip(self.grads, self.q_vars), global_step=global_step)

        """update target"""

        # self.update_ops = self._update_target_vars()
        # self.sync = tf.group(
        #     *(
        #         [v1.assign(v2) for v1, v2 in zip(self.q_target_vars, self.q_vars)]
        #     ))

        online_vars = {var.name[len("model/q_model"):]: var
                       for var in self.q_vars}
        target_vars = {var.name[len("model/target_q_model"):]: var
                       for var in self.q_target_vars}
        # print(online_vars['q_model/c1/w:0'])
        # print("target:", target_vars['target_q_model/c1/w:0'])
        # We need an operation to copy the online DQN to the target DQN
        copy_ops = [target_var.assign(online_vars[var_name])
                    for var_name, target_var in target_vars.items()]
        self.copy_online_to_target = tf.group(*copy_ops)




        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        # self.sess.graph.finalize()
        writer = tf.summary.FileWriter('./name_scope', graph=tf.get_default_graph())
        writer.close()


    def build_calculation(self) -> None:
        # self.o_b = tf.compat.v1.placeholder(tf.uint8, [None, 1, 84, 84], name="obs_batch")
        self.a_b = tf.compat.v1.placeholder(tf.int32, [None], name="act_batch")
        self.r_b = tf.compat.v1.placeholder(tf.float32, [None], name="rew_batch")
        # self.next_o_b = tf.compat.v1.placeholder(tf.uint8, [None, 1, 84, 84], name="next_obs_batch")
        self.d_m = tf.compat.v1.placeholder(tf.float32, [None], name="done_mask")
        # not_done_mask = tf.convert_to_tensor(tf.ones_like(self.d_m) - self.d_m)
        not_done_mask = tf.convert_to_tensor(1 - self.d_m)
        # self.a_b = tf.expand_dims(self.a_b, -1)
        # dims = tf.range(self.a_b.shape()[0])  # tensor does not have len.
        self.dims = tf.compat.v1.placeholder(tf.int32, [None]) # tensor does not have len.
        new_slice = tf.stack([self.dims, self.a_b], 1)
        self.current_Q_values = tf.gather_nd(self.model, new_slice)  # act is the index of sec dim.
        # self.current_Q_values = self.model[index_dim_0, self.a_b]  # act is the index of sec dim.
        # self.current_Q_values = tf.gather(self.model, self.a_b, axis=1)  # act is the index of sec dim.
        # .squeeze() not work for multi dim.#tf also have gather
        # Compute next Q value based on which action gives max Q values
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        next_max_q = tf.reduce_max(self.target_model, axis=1)  # .detach().max(1)[0]
        next_Q_values = not_done_mask * next_max_q

        # Compute the target of the current Q values
        target_Q_values = self.r_b + (self.gamma * next_Q_values)
        # Compute Bellman error
        bellman_error = tf.stop_gradient(target_Q_values) - self.current_Q_values
        # clip the bellman error between [-1 , 1]
        clipped_bellman_error = tf.clip_by_value(bellman_error, -1, 1)  # tf.clip_by_value vs pytorch .clamp vs np.clip
        # Note: clipped_bellman_delta * -1 will be right gradient
        self.d_error = clipped_bellman_error * -1.0

    def _update_target_vars(self):
        update_ops = []
        for i, var in enumerate(self.q_vars):
            copy_op = self.q_target_vars[i].assign(var.value())
            # copy_op = self.q_target_vars[i].load(self.sess.run(var.value()), self.sess)
            update_ops.append(copy_op)
        return update_ops

    # def update_networks(self):
    #     """
    #     Args:
    #         sess: A Tensorflow session object
    #     Assigns the values of the parameters of the main network to the
    #     parameters of the target network
    #     """
    #     update_ops = self._update_target_vars()
    #     for copy_op in update_ops:
    #         self.sess.run(copy_op)

    def update_networks(self):
        """
        Args:
            sess: A Tensorflow session object
        Assigns the values of the parameters of the main network to the
        parameters of the target network
        """

        # for copy_op in update_ops:
        #         #     self.sess.run(copy_op)
        self.sess.run(self.update_ops)

    def act(self, state):
        # print(np.shape(state)) # 1
        sample = random.random()
        eps_threshold = self.exploration.value(self.steps)
        max_act = self.sess.run(self.predict, feed_dict={self.x: state})
        # max_act = self.sess.run(max_act) # max_act is still a tensor without assignment.
        if sample > eps_threshold and self.steps > self.learning_starts:
            return max_act
        else:
            max_act = torch.from_numpy(max_act)
            return torch.randint_like(max_act, self.num_actions)

    def value(self, state):
        if type(state) is not np.ndarray:
            state = state.numpy()
        q_values = self.sess.run(self.model, feed_dict={self.x: state})  # .max(1) ## array
        q_out = q_values.max(1)
        return q_out

    # def update(self, batch_size=32, num_param_updates=0):
    def update(self, batch_size=32, num_param_updates=0):

        ### Perform experience replay and train the network.
        # Note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (self.steps > self.learning_starts and
                self.steps % self.learning_freq == 0 and
                self.replay.can_sample(batch_size)):
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay.sample(batch_size)
            # print(next_obs_batch.dtype, np.shape(next_obs_batch))
            # print(next_obs_batch)
            act_batch.astype(np.int32)
            # print(act_batch)
            # print(self.sess.run(tf.expand_dims(act_batch, -1)))
            # print(np.shape(self.sess.run(tf.gather(self.model, tf.expand_dims(act_batch, -1), axis=1),
            #               feed_dict={self.x: obs_batch})))
            # print(self.sess.run(tf.gather(self.model, tf.expand_dims(act_batch, -1), axis=1),
            #                              feed_dict={self.x: obs_batch}))
            # act_batch = tf.convert_to_tensor(act_batch)
            # rew_batch = tf.convert_to_tensor(rew_batch)
            dims = np.arange(0, len(act_batch))  # tensor does not have len.
            # print(dims)
            # new_slice = tf.stack([dims, self.a_b], 1)
            # print(self.sess.run(new_slice, feed_dict={self.a_b: act_batch}))
            # self.current_Q_values = tf.gather_nd(self.model, new_slice)
            # print(self.sess.run(current_Q_values, feed_dict={self.a_b: act_batch}))
            # error = self.sess.run(self.d_error,
            #                       feed_dict={self.o_b: obs_batch, self.a_b: act_batch, self.r_b: rew_batch,
            #                                  self.next_o_b: next_obs_batch, self.d_m: done_mask,
            #                                  self.x: obs_batch, self.target_x: next_obs_batch})
            error = self.sess.run(self.d_error,
                                  feed_dict={self.a_b: act_batch, self.r_b: rew_batch,
                                             self.d_m: done_mask, self.dims: dims,
                                             self.x: obs_batch, self.target_x: next_obs_batch})
            # print(error)
            # # print(np.shape(self.current_Q_values))
            self.sess.run(self.train_op,
                          feed_dict={self.error: error, self.x: obs_batch, self.target_x: next_obs_batch})
            self.num_param_updates += 1
            # print(self.train_op)
            # print(self.q_vars)
            # print(tf.gradients(self.loss, self.q_vars))
            # Periodically update the target network by Q network to target Q network
            if self.num_param_updates % self.target_update_freq == 0:
                # self.optimizer_tf.apply_gradients([v_t.assign(v) for v_t, v in zip(self.q_target_vars, self.q_vars)])
                # self.update_networks()
                # self.optimizer_tf.apply_gradient(zip(grads, self.q_target_vars), global_step=global_step)
                # self.sess.run(self.sync)
                self.copy_online_to_target.run()
    # TODO Fill in later
    def save(self, *kwargs):
        pass

    def load(self, *kwargs):
        pass


def process_batch(obs, tftype):
    obs = tf.convert_to_tensor(obs)
    obs = tf.cast(obs, tftype)
    return obs
