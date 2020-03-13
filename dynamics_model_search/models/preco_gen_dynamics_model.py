from models.dynamics_model import DynamicsModel
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch.nn.functional as F
import time
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PreCoGenModel(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(PreCoGenModel, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h = None

        num_recurrent_layers = 1

        self.latent_size = 256
        self.norm_mean = np.array([0]).astype(np.float32)
        self.norm_std = np.array([1]).astype(np.float32)

        self.pred_rnn = nn.GRU(act_dim, self.latent_size, num_recurrent_layers)
        self.corr_rnn = nn.GRU(state_dim, self.latent_size, num_recurrent_layers)

        self.layer1 = nn.Linear(self.latent_size, self.latent_size)
        self.fc_out = nn.Linear(self.latent_size, state_dim)

    def decode(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x

    def get_device(self):
        return self.fc_out._parameters['weight'].device

    def pred_single(self, a, enc_h=None):
        out_enc, enc_h = self.pred_rnn(a, enc_h)
        out_tmp = self.decode(out_enc)
        return out_tmp, enc_h

    def autoencode(self, x):
        x_tmp = x.unsqueeze(0)
        out, h = self.corr_rnn(x_tmp, None)
        out = self.decode(out)
        return out, h

    # seq_len, batch_len, input_size
    def pred_seq(self, a, h=None):
        out_enc, enc_h = self.pred_rnn(a, h)
        out_tmp = self.decode(out_enc)
        return out_tmp, enc_h

    def normalize(self, x):
        return (x-torch.from_numpy(self.norm_mean).to(x.device)) / torch.from_numpy(self.norm_std).to(x.device)

    def unnormalize(self, x):
        return x*torch.from_numpy(self.norm_std).to(x.device) + torch.from_numpy(self.norm_mean).to(x.device)

    def forward(self, x, a, y=None, cmd='None'):
        x = x.transpose(0, 1)
        if a is not None:
            a = a.transpose(0, 1)
        if cmd == 'reset':
            return self.corr_rnn(x, a)
        if cmd == 'single':
            new_obs, h = self.pred_single(x, a)
            return new_obs.transpose(0, 1), h.transpose(0, 1)
        obs = x
        act = a
        new_obs = torch.transpose(y, 0, 1)
        new_obs = self.normalize(new_obs)

        corr_out, corr_h = self.autoencode(obs[0])
        seq_out, seq_h = self.pred_seq(act, corr_h)
        seq_out = self.normalize(seq_out)
        single_out = seq_out[0]
        final_out = seq_out[-1]
        if y is not None:
            single = torch.abs(seq_out[0]-new_obs[0])
            corr = torch.abs(corr_out[0]-obs[0])
            final = torch.abs(seq_out[-1]-new_obs[-1])
            seq_errors = torch.abs(seq_out-new_obs)
            return torch.mean(single), torch.mean(seq_errors), torch.mean(corr), torch.mean(final), \
                   single_out, seq_out, final_out
        else:
            return single_out, seq_out

class PreCoGenDynamicsModel(DynamicsModel):
    def __init__(self, env_in, dev=None):
        DynamicsModel.__init__(self, env_in)
        lr = 1e-5
        self.is_reset = False
        self.val_seq_len = 100
        self.train_seq = 1
        self.look_ahead_per_epoch = 1
        self.batch_size = 64
        self.max_seq_len = 100
        
        self.model = PreCoGenModel(self.state_dim, self.act_dim)
        if dev is None:
            self.model.to(device)
            self.device = device
        else:
            self.model.to(dev)
            self.device = dev
        self.state_mul_const_tensor = torch.Tensor(self.state_mul_const).to(self.device)
        self.act_mul_const_tensor = torch.Tensor(self.act_mul_const).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.eval()

    def reinit(self, state_dim, state_mul_const, act_dim, act_mul_const):
        self.state_mul_const = state_mul_const
        self.state_mul_const[self.state_mul_const == np.inf] = 1
        self.act_mul_const = act_mul_const
        self.act_dim = act_dim
        self.state_dim = state_dim

        self.model = PreCoGenModel(self.state_dim, self.act_dim)
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
        # print('Loading Model')
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

            single, seq, corr, final, single_out, seq_out, final_out = self.model(Xsi, Asi, Ysi)
            Corr += corr.item()
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
        single, seq, corr, final, single_out, seq_out, final_out = self.model(Xs, As, Ys)
        seq.backward()
        self.optimizer.step()
        return corr.item(), single.item(), seq.item(), final.item()

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

                single, seq, corr, final, single_out, seq_out, final_out = self.model(Xsi, Asi, Ysi)
                Corr += corr.item()
                Single += single.item()
                Seq += seq.item()
                Final += final.item()

                idx += 1
                sys.stdout.write(str(round(float(100*idx)/float(len(data)/self.batch_size), 2))+'% Done in '+str(round(time.time()-start, 2))+' s                                    \r')

                seq.backward(retain_graph=True)
                i += self.batch_size
                self.optimizer.step()

        return Corr/idx, Single/idx, Seq/idx, Final/idx

    def reset(self, obs_in, h=None):
        x = torch.from_numpy(np.array([obs_in.astype(np.float32)/self.state_mul_const])).to(self.device).unsqueeze(1)
        _, self.h = self.model(x, None, None, 'reset')
        self.h = self.h.detach()
        self.is_reset = True
        return obs_in, self.h

    def step_parallel(self, action_in, obs_in=None, save=True, state=False, state_in=None):
        self.model.eval()
        if obs_in is not None and state_in:
            state_in = obs_in[1]
            obs_in = obs_in[0]
        else:
            state_in = None
        tensor = True
        a = action_in
        if state_in is not None:
            new_obs, state_out = self.model(a, state_in, None, 'single')
        elif obs_in is not None:
            x = torch.from_numpy(np.array([obs_in.astype(np.float32)/self.state_mul_const])).to(self.device).unsqueeze(0)
            if save:
                _, self.h = self.model(x, None, None, cmd='reset')
                new_obs, h = self.model(a, self.h, None, cmd='single')
                self.h = self.h.detach()
                state_out = self.h
            else:
                _, h = self.model(x, None, None, cmd='reset')
                new_obs, h = self.model(a, h, None, cmd='single')
                state_out = h
        else:
            new_obs, h = self.model(a, self.h, None, cmd='single')
            self.h = self.h.detach()
            state_out = self.h
        self.is_reset = False

        new_obs = new_obs.squeeze(1).detach()*self.state_mul_const_tensor.to(new_obs.device)
        if not tensor:
            new_obs = new_obs.detach().cpu().numpy()
        if new_obs.shape[0] == 1:
            new_obs = new_obs.squeeze(0)
        if state:
            return new_obs, state_out.detach()
        else:
            return new_obs

    def step(self, action_in, obs_in=None, save=True, state=False, state_in=None):
        self.model.eval()
        if obs_in is not None and state_in:
            state_in = obs_in[1]
            obs_in = obs_in[0]
        else:
            state_in = None
        tensor = False
        if torch.is_tensor(action_in):
            tensor = True
            a = action_in/self.act_mul_const_tensor
            a = a.to(self.device)
        else:
            a = torch.from_numpy(np.array([action_in.astype(np.float32)/self.act_mul_const])).to(self.device)
        while len(a.shape) < 3:
            a = a.unsqueeze(0)
        if state_in is not None:
            new_obs, h = self.model.pred_single(a, state_in)
            state_out = h
        elif obs_in is not None:
            x = torch.from_numpy(np.array([obs_in.astype(np.float32)/self.state_mul_const])).to(self.device).unsqueeze(0)
            if save:
                _, self.h = self.model.corr_rnn(x, None)
                new_obs, self.h = self.model.pred_single(a, self.h)
                self.h = self.h.detach()
                state_out = self.h
            else:
                _, h = self.model.corr_rnn(x, None)
                new_obs, h = self.model.pred_single(a, h)
                state_out = h
        else:
            new_obs, self.h = self.model.pred_single(a, self.h)
            self.h = self.h.detach()
            state_out = self.h
        self.is_reset = False
        new_obs = new_obs.squeeze(0).detach()*self.state_mul_const_tensor
        if not tensor:
            new_obs = new_obs.detach().cpu().numpy()
        if new_obs.shape[0] == 1:
            new_obs = new_obs.squeeze(0)
        if state:
            return new_obs, state_out.detach()
        else:
            return new_obs
