import torch
import time
import numpy as np
from torch import nn, optim
from collections import deque
import random
import datetime
import os

from agent import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TBway:
    def __init__(self, agent):
        self.ep_lens = agent.ep_lens
        self.ep_rs = agent.ep_rs
    def print(self):
        if self.ep_lens and len(self.ep_rs) > 0:
            print('Hi! Last Episode Reward: ' + str(self.ep_rs[-1]))

        from visualization import TBway
        test = TBway(agent)
        test.print()