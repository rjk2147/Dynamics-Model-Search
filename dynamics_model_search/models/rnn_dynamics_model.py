from models.dynamics_model import DynamicsModel
import numpy as np
import torch
from torch import nn, optim
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class RNNModel(nn.Module):
    def __init__(self, state_dim, act_dim, drop_rate=0.5):
        super(RNNModel, self).__init__()
        self.state_dim = state_dim
        self.state_size = sum(state_dim)
        self.act_dim = act_dim
        self.h = None

        num_recurrent_layers = 1

        self.latent_size = 256
        self.norm_mean = np.array([0]).astype(np.float32)
        self.norm_std = np.array([1]).astype(np.float32)

        self.rnn = nn.GRU(self.state_size+act_dim, self.latent_size, num_recurrent_layers)

        self.layer1 = nn.Linear(self.latent_size, self.latent_size)
        self.fc_out = nn.Linear(self.latent_size, self.state_size)

    def decode(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.fc_out(x)
        return x

    def reset(self):
        return torch.ones((1,1,self.latent_size))

    def get_device(self):
        return self.fc_out._parameters['weight'].device

    # seq_len, batch_len, input_size
    def pred(self, x, a, h=None):
        tmp_in = torch.cat([x, a], dim=-1)
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
        if y is not None:
            seq_out = self.normalize(seq_out)
            single_out = seq_out[0]
            final_out = seq_out[-1]

            new_obs = y.transpose(0, 1)
            new_obs = self.normalize(new_obs)

            single = torch.abs(seq_out[0]-new_obs[0])
            final = torch.abs(seq_out[-1]-new_obs[-1])
            seq_errors = torch.abs(seq_out-new_obs)
            return torch.mean(single), torch.mean(seq_errors), torch.mean(final), single_out, seq_out, final_out
        else:
            return seq_out, seq_h.transpose(0, 1)

class RNNDynamicsModel(DynamicsModel):
    def __init__(self, env_in, dev=None):
        DynamicsModel.__init__(self, env_in)
        self.lr = 1e-5
        self.is_reset = False
        self.val_seq_len = 100
        self.train_seq = 1
        self.look_ahead_per_epoch = 1
        self.batch_size = 64
        self.max_seq_len = 100
        
        self.model = RNNModel(self.state_dim, self.act_dim)
        if dev is None:
            self.model.to(device)
            self.device = device
        else:
            self.model.to(dev)
            self.device = dev
        self.state_mul_const_tensor = torch.Tensor(self.state_mul_const).to(self.device)
        self.act_mul_const_tensor = torch.Tensor(self.act_mul_const).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.eval()

    def reinit(self, state_dim, state_mul_const, act_dim, act_mul_const):
        self.state_mul_const = state_mul_const
        self.state_mul_const[self.state_mul_const == np.inf] = 1

        self.act_mul_const = act_mul_const
        self.act_dim = act_dim
        self.state_dim = state_dim

        self.model = RNNModel(self.state_dim, self.act_dim)
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
        # x = torch.from_numpy(np.array([obs_in.astype(np.float32)/self.state_mul_const])).to(self.device).unsqueeze(1)
        # _, self.h = self.model(x, None, None, 'reset')
        # self.h = self.h.detach()
        self.is_reset = True
        if h is None:
            return obs_in, self.model.reset()
        else:
            return obs_in, h

    # Requires observation in shape l, b, d
    # l is length of sequence
    # b is batch size
    # d is dimension of state
    def step_parallel(self, action_in, obs_in=None, save=True, state=False, state_in=None, certainty=False):
        self.model.eval()
        if obs_in is not None and state_in:
            state_in = torch.cat(obs_in[1])
            obs_in = obs_in[0]
            # while len(obs_in.shape) < 3:
            # obs_in = obs_in.unsqueeze(1)
        else:
            state_in = None
        tensor = True
        a = action_in.float()
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

        new_obs = new_obs.squeeze(1).detach()*self.state_mul_const_tensor.to(new_obs.device)
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
