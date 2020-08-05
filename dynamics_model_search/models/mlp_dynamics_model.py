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

class SeqMLPModel(nn.Module):
    def __init__(self, state_dim, act_dim, seq_len=10):
        super(SeqMLPModel, self).__init__()
        self.SeqMLPModel = list(state_dim)
        self.state_dim = state_dim
        self.state_size = sum(self.state_dim)
        self.act_dim = act_dim
        self.seq_len = seq_len
        self.in_dim = (self.state_size+self.act_dim)*seq_len
        self.with_vae = True
        if self.with_vae:
            from models.cnn1d_vae import VAEModel
            self.vae = VAEModel(state_dim, act_dim, seq_len)
            save_str = '/home/robert/Research/Dynamics-Model-Search/dynamics_model_search/rl_models/2020-07-08-11:27:45_self_model.pt'
            self.vae.load_state_dict(torch.load(save_str, map_location=device))
            self.vae.eval()
            self.in_dim = self.vae.latent_size

        self.mlp1 = nn.Linear(self.in_dim, 256)
        self.mlp2 = nn.Linear(256, 128)
        self.mlp3 = nn.Linear(128, 64)
        self.mlp4 = nn.Linear(64, self.state_size)

    def normalize(self, x):
        return (x-torch.from_numpy(self.norm_mean).to(x.device)) / torch.from_numpy(self.norm_std).to(x.device)

    def unnormalize(self, x):
        return x*torch.from_numpy(self.norm_std).to(x.device) + torch.from_numpy(self.norm_mean).to(x.device)

    def reset(self):
        h = torch.zeros(1, self.state_size+self.act_dim, self.seq_len).to(device)
        return h

    def forward(self, x, a, h=None, y=None):
        all_input = torch.cat([x,a], -1).transpose(1, 2)
        input_buffer = all_input.split(1, -1)

        if h is None:
            buffer = torch.zeros(1, self.state_size+self.act_dim, self.seq_len).to(x.device)
        else:
            buffer = h
        buffer = buffer.split(1, -1)
        buffer = deque(buffer, maxlen=len(buffer))
        for input in input_buffer:
            buffer.append(input)
        buffer = torch.cat(list(buffer), -1)
        if self.with_vae:
            mu, logvar = self.vae.encode(buffer)
            z = self.vae.reparameterize(mu, logvar)
            all_in = z.detach()
        else:
            all_in = buffer.flatten(1)

        pred_out = torch.relu(self.mlp1(all_in))
        pred_out = torch.relu(self.mlp2(pred_out))
        pred_out = torch.relu(self.mlp3(pred_out))
        pred_out = self.mlp4(pred_out)
        if y is not None:
            pred_out = self.normalize(pred_out)
            new_obs = y.transpose(0, 1)

            new_obs = self.normalize(new_obs[-1])
            seq_errors = torch.abs(pred_out-new_obs[-1])
            return torch.mean(seq_errors)
        else:
            return pred_out, buffer

class SeqMLPDynamicsModel(DynamicsModel):
    def __init__(self, env_in, dev=None, seq_len=10):
        DynamicsModel.__init__(self, env_in)
        self.lr = 1e-3
        self.is_reset = False
        self.val_seq_len = 100
        self.train_seq = 1
        self.look_ahead_per_epoch = 1
        self.batch_size = 64
        self.max_seq_len = seq_len
        
        self.model = SeqMLPModel(self.state_dim, self.act_dim, self.max_seq_len)
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

        self.model = SeqMLPModel(self.state_dim, self.act_dim)
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
        loss = self.model(Xs, As, None, Ys)
        loss.backward()
        self.optimizer.step()
        return loss.item()

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

    def step_parallel(self, action_in, obs_in=None, save=True, state=False, state_in=None, certainty=False):
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
            if certainty:
                return new_obs, state_out.detach(), torch.ones_like(new_obs)
            return new_obs, state_out.detach()
        else:
            if certainty:
                return new_obs, torch.ones_like(new_obs)
            return new_obs

    def step(self, action_in, obs_in=None, save=True, state=False, state_in=None, certainty=False):
        return self.step_parallel(action_in, obs_in, save, state, state_in, certainty)
