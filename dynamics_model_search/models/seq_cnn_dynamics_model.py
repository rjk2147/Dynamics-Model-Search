from models.dynamics_model import DynamicsModel
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch.nn.functional as F
import time
from collections import deque

def conv2d_bn_relu(inch,outch,kernel_size,stride=1,padding=0):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

def conv2d_bn_sigmoid(inch,outch,kernel_size,stride=1,padding=0):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_sigmoid(inch,outch,kernel_size,stride=1,padding=0):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_relu(inch,outch,kernel_size,stride=1,padding=0):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

def product(l):
    p = 1
    for i in l:
        p *= i
    return p

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SeqCNNModel(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(SeqCNNModel, self).__init__()
        self.state_dim = list(state_dim)
        state_size = product(state_dim)
        w, h, c = state_dim
        self.act_dim = act_dim
        self.h = None

        num_recurrent_layers = 1

        self.latent_size = 256
        self.norm_mean = np.array([0]).astype(np.float32)
        self.norm_std = np.array([1]).astype(np.float32)

        # Encode
        self.conv1 = conv2d_bn_relu(4, 16, kernel_size=5, stride=2)
        self.conv2 = conv2d_bn_relu(16, 32, kernel_size=5, stride=2)
        self.conv3 = conv2d_bn_relu(32, 32, kernel_size=5, stride=2)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32

        # RNN
        self.rnn = nn.GRU(linear_input_size+act_dim, self.latent_size, num_recurrent_layers)

        # Decode
        self.layer1 = nn.Linear(self.latent_size, self.latent_size)
        self.fc_out = nn.Linear(self.latent_size, state_size)

    def encode(self, x):
        out = []
        for i in range(x.shape[0]):
            # From N, W, H, C to N, C, H, W
            x_i = x[i].permute(0, 3, 2, 1).to(torch.float)
            x_i = self.conv1(x_i)
            x_i = self.conv2(x_i)
            x_i = self.conv3(x_i)
            out.append(x_i.view(x_i.size(0), -1).unsqueeze(0))
        return torch.cat(out)

    def decode_linear(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = F.sigmoid(self.fc_out(x))
        reshape_tuple = [x.shape[i] for i in range(len(x.shape)-1)]
        reshape_tuple.extend(self.state_dim)
        x = x.view(reshape_tuple)
        return x

    def decode(self, x):
        return self.decode_linear(x)

    def reset(self):
        return torch.ones((1,1,self.latent_size))

    def get_device(self):
        return self.fc_out._parameters['weight'].device

    # seq_len, batch_len, w, h, c
    def pred(self, x, a, h=None):
        x_enc = self.encode(x)
        tmp_in = torch.cat([x_enc, a.to(torch.float)], dim=-1)
        out_enc, enc_h = self.rnn(tmp_in, h)
        out_tmp = self.decode(out_enc)
        return out_tmp, enc_h

    def normalize(self, x):
        return (x-torch.from_numpy(self.norm_mean).to(x.device)) / torch.from_numpy(self.norm_std).to(x.device)

    def unnormalize(self, x):
        return x*torch.from_numpy(self.norm_std).to(x.device) + torch.from_numpy(self.norm_mean).to(x.device)

    def forward(self, x, a, h=None, y=None):
        if h is None:
            h = torch.ones((x.shape[0], 1, self.latent_size)).to(x.device)
        obs = x.transpose(0, 1)
        act = a.transpose(0, 1)
        h = h.transpose(0,1)

        seq_out, seq_h = self.pred(obs, act, h)
        # seq_out = self.normalize(seq_out)
        single_out = seq_out[0]
        final_out = seq_out[-1]
        if y is not None:
            new_obs = y.transpose(0, 1)
            # new_obs = self.normalize(new_obs)
            single = torch.abs(seq_out[0]-new_obs[0])
            final = torch.abs(seq_out[-1]-new_obs[-1])
            seq_errors = torch.abs(seq_out-new_obs)
            e = torch.mean(seq_errors, -1)
            e = torch.mean(e, -1)
            return torch.mean(single), torch.mean(seq_errors), torch.mean(final), single_out, seq_out, final_out
        else:
            return seq_out.to(torch.uint8), seq_h.transpose(0, 1)

class SeqCNNDynamicsModel(DynamicsModel):
    def __init__(self, env_in, dev=None, seq_len=100):
        DynamicsModel.__init__(self, env_in)
        self.lr = 1e-5
        self.is_reset = False
        self.val_seq_len = seq_len
        self.train_seq = 1
        self.look_ahead_per_epoch = 1
        self.batch_size = 64
        self.max_seq_len = seq_len
        
        self.model = SeqCNNModel(self.state_dim, self.act_dim)
        if dev is None:
            self.model.to(device)
            self.device = device
        else:
            self.model.to(dev)
            self.device = dev
        self.state_mul_const_tensor = torch.Tensor(self.state_mul_const).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.eval()

    def reinit(self, state_dim, state_mul_const, act_dim, act_mul_const):
        self.state_mul_const = state_mul_const
        self.state_mul_const[self.state_mul_const == np.inf] = 1
        self.act_mul_const = act_mul_const
        self.act_dim = act_dim
        self.state_dim = state_dim

        self.model = SeqCNNModel(self.state_dim, self.act_dim)
        self.model.to(self.device)
        self.state_mul_const_tensor = torch.Tensor(self.state_mul_const).to(self.device)
        self.act_mul_const_tensor = torch.Tensor(self.act_mul_const).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.eval()

    def save(self, save_str):
        torch.save(self.model.state_dict(), save_str)

    def load(self, save_str):
        self.model.load_state_dict(torch.load(save_str, map_location=self.device))
        self.model.eval()

    def save_dict(self):
        return self.model.state_dict()

    def load_dict(self, d):
        self.model.load_state_dict(d)

    def to(self, in_device):
        self.model.to(in_device)

    def get_loss(self, data):
        np.random.seed(0)
        Xs, As, Ys = self.__prep_data__(data)
        Corr = 0
        Single = 0
        Seq = 0
        Final = 0
        idx = 0
        self.model.eval()
        for Xsi, Asi, Ysi \
                in zip(enumerate(Xs), enumerate(As), enumerate(Ys)):
            Ysi = torch.from_numpy(Ysi[1].astype(np.float32)).to(self.device)
            Asi = torch.from_numpy(Asi[1].astype(np.float32)).to(self.device)
            Xsi = torch.from_numpy(Xsi[1].astype(np.float32)).to(self.device)

            single, seq, final, single_out, seq_out, final_out = self.model(Xsi, Asi, None, Ysi)
            Single += single.item()
            Seq += seq.item()
            Final += final.item()
            idx += 1
        return Corr/idx, Single/idx, Seq/idx, Final/idx

    def update(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        Xs = torch.from_numpy(np.array([step[0] for step in data]).astype(np.float32)).to(self.device)
        As = torch.from_numpy(np.array([step[1] for step in data]).astype(np.float32)).to(self.device)
        Ys = torch.from_numpy(np.array([step[2] for step in data]).astype(np.float32)).to(self.device)
        single, seq, final, single_out, seq_out, final_out = self.model(Xs, As, None, Ys)
        seq.backward()
        self.optimizer.step()
        return single.item(), seq.item(), final.item()

    def train_epoch(self, data):
        import sys, math
        big = True
        start = time.time()
        if not big:
            Xs, As, Ys = self.__prep_data__(data, big=big)
        else:
            Xs, As, Ys, p = self.__prep_data__(data, big=big)
        Corr = 0
        Single = 0
        Seq = 0
        Final = 0
        idx = 0
        self.model.train()
        self.optimizer.zero_grad()
        if big:
            i = 0
            while len(p) > 0:
                pi = p[:self.batch_size]
                p = p[self.batch_size:]

                Xsi = []
                Asi = []
                Ysi = []
                for i in range(len(pi)):
                    Xsi.append(Xs[pi[i]])
                    Ysi.append(Ys[pi[i]])
                    Asi.append(As[pi[i]])

                Xsi = np.array(Xsi)
                Asi = np.array(Asi)
                Ysi = np.array(Ysi)

                Xsi = torch.from_numpy(Xsi.astype(np.float32)).to(self.device)
                Asi = torch.from_numpy(Asi.astype(np.float32)).to(self.device)
                Ysi = torch.from_numpy(Ysi.astype(np.float32)).to(self.device)

                single, seq, final, single_out, seq_out, final_out = self.model(Xsi, Asi, None, Ysi)
                Single += single.item()
                Seq += seq.item()
                Final += final.item()

                idx += 1
                sys.stdout.write(str(round(float(100*idx)/float(len(data)/self.batch_size), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')

                seq.backward(retain_graph=True)
                i += self.batch_size
                self.optimizer.step()

        return Single/idx, Seq/idx, Final/idx

    def reset(self, obs_in, h=None):
        self.is_reset = True
        if h is None:
            return obs_in, self.model.reset()
        else:
            return obs_in, h

    def step_parallel(self, action_in, obs_in=None, save=True, state=False, state_in=None):
        self.model.eval()
        if obs_in is not None and state_in:
            state_in = torch.cat(obs_in[1])
            obs_in = obs_in[0]
            while len(obs_in.shape) < 3:
                obs_in = obs_in.unsqueeze(1)
        else:
            state_in = None
        tensor = True
        a = action_in.float()
        while len(a.shape) < 3:
            a = a.unsqueeze(1)
        if state_in is not None:
            new_obs, state_out = self.model(obs_in, a, state_in)
        elif obs_in is not None:
            x = torch.from_numpy(np.array([obs_in.astype(np.float32)/self.state_mul_const])).to(self.device).unsqueeze(0)
            if save:
                new_obs, self.h = self.model(x, a, self.h)
                self.h = self.h.detach()
                state_out = self.h
            else:
                new_obs, h = self.model(x, a, None)
                state_out = h
        else:
            new_obs, h = self.model(obs_in, a, self.h, None)
            self.h = h.detach()
            state_out = self.h
        self.is_reset = False

        new_obs = new_obs.squeeze(1).detach()*self.state_mul_const_tensor.to(new_obs.device).to(new_obs.dtype)
        if not tensor:
            new_obs = new_obs.detach().cpu().numpy()
        if new_obs.shape[0] == 1:
            new_obs = new_obs.squeeze(0)
        if state:
            return new_obs, state_out.detach()
        else:
            return new_obs

    def step(self, action_in, obs_in=None, save=True, state=False, state_in=None):
        return self.step_parallel(action_in, obs_in, save, state, state_in)
