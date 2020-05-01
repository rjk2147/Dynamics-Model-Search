from models.dynamics_model import DynamicsModel
import numpy as np
import torch
from torch import nn, optim
import time, math
from torch.distributions import Normal, Uniform, OneHotCategorical

class Gaussian(nn.Module):
    def __init__(self, in_features, out_features):
        super(Gaussian, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.sigma = nn.Linear(in_features, out_features)
        self.mu = nn.Linear(in_features, out_features)

    def forward(self, x):
        epsilon = 1e-20
        mu = self.mu(x)
        sigma = torch.exp(1 + self.sigma(x))
        # sigma = torch.ones_like(mu)/100000.0

        # mu = torch.stack(mu.split(self.out_features, dim=-1), -2)
        # sigma = torch.stack(sigma.split(self.out_features, dim=-1), -2)
        if torch.isnan(mu).any() or torch.isnan(sigma).any():
            print('NaN in normal!')
            print('')
        return Normal(mu, torch.clamp(sigma, min=epsilon))

def gaussian_loss(normal, y):
    loglik = normal.log_prob(y)

    # Normalized logprob that scales to 0-1
    # loglik = 0.5*(((mu-x)/std)**2)

    loss = -torch.sum(loglik, dim=-1)

    # loss = -torch.logsumexp(loglik, dim=-1)
    if torch.isnan(loss).any():
        print('NaN in loss!')
        print('')
    return loss

class Encoder(nn.Module):
    def __init__(self, input_size=512, z_output_size=128, sigma_output_size=21):
        super().__init__()
        self.z_output_size = z_output_size
        self.sigma_output_size = sigma_output_size
        self.linear1 = nn.Linear(in_features=input_size, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=128)
        self.linear_z = nn.Linear(in_features=128, out_features=z_output_size * 2)
        self.linear_sigma = nn.Linear(in_features=128, out_features=sigma_output_size * 2)

    def forward(self, h):
        epsilon = 1e-20
        out = self.linear1(h)
        out = torch.relu(out)
        out = self.linear2(out)
        out = torch.relu(out)

        z = self.linear_z(out)
        z_loc = z[:, :self.z_output_size]
        z_scale = torch.exp(1+z[:, self.z_output_size:])

        return z_loc, z_scale, 0, 0

        # sigma = torch.exp(1+self.linear_sigma(out))
        # sigma = torch.clamp(sigma, min=epsilon)
        # sigma_loc = sigma[:, :self.sigma_output_size]
        # sigma_scale = sigma[:, self.sigma_output_size:]
        #
        # return z_loc, z_scale, sigma_loc, sigma_scale

class Decoder(nn.Module):
    def __init__(self, input_size=21 + 128, output_size=21, gaussian=True):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=32)
        self.linear4 = nn.Linear(in_features=32, out_features=16)
        if gaussian:
            self.linear5 = Gaussian(in_features=16, out_features=output_size)
        else:
            self.linear5 = nn.Linear(in_features=16, out_features=output_size)

    def forward(self, x, z):
        y = torch.cat([x, z], dim=-1).to(z.device)
        y = self.linear1(y)
        y = torch.relu(y)
        y = self.linear2(y)
        y = torch.relu(y)
        y = self.linear3(y)
        y = torch.relu(y)
        y = self.linear4(y)
        y = torch.relu(y)
        y = self.linear5(y)
        return y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MDNSeqModel(nn.Module):
    def __init__(self, state_dim=21, action_size=8, z_size=128, hidden_state_size=512):
        super(MDNSeqModel, self).__init__()
        self.h = None

        self.norm_mean = np.array([0]).astype(np.float32)
        self.norm_std = np.array([1]).astype(np.float32)
        state_size = sum(state_dim)
        self.z_size = z_size
        self.input_size = state_size + action_size
        self.output_size = state_size
        self.rnn_input_size = action_size + z_size
        self.rnn_hidden_size = hidden_state_size

        self.encoder = Encoder(input_size=self.rnn_hidden_size, z_output_size=self.z_size,
                           sigma_output_size=self.output_size).to(device)
        self.decoder = Decoder(input_size=self.z_size + self.output_size, output_size=self.output_size).to(device)
        self.rnn = nn.LSTMCell(self.rnn_input_size, self.rnn_hidden_size).to(device)

    def reset(self, obs):
        if torch.is_tensor(obs):
            self.x = obs.unsqueeze(0) # unsure about this
        else:
            self.x = torch.from_numpy(obs).unsqueeze(0)  # unsure about this
        return torch.zeros((1,self.z_size+2*self.rnn_hidden_size))

    # seq_len, batch_len, input_size
    def pred(self, a, z, h, c, x=None):
        means = []
        sds = []
        if x is None:
            x = self.x.repeat(a.shape[1], 1).to(a.device)

        for t in range(a.shape[0]):
            rnn_input = torch.cat([a[t], z], dim=-1)
            h, c = self.rnn(rnn_input, (h, c))
            z_loc, z_scale, sigma_loc, sigma_scale = self.encoder(h)
            z = Normal(z_loc, torch.clamp(z_scale, min=1e-20)).sample()
            normal = self.decoder(x, z) # unsure about this
            means.append(normal.mean.unsqueeze(0))
            sds.append(normal.stddev.unsqueeze(0))
        seq_h = torch.cat([z, h, c], -1)

        means = torch.cat(means)
        sds = torch.cat(sds)
        normals = Normal(means, sds)
        return normals, seq_h

    def normalize(self, x):
        return (x-torch.from_numpy(self.norm_mean).to(x.device)) / torch.from_numpy(self.norm_std).to(x.device)

    def forward(self, x, a, hidden_state=None, y=None):
        if hidden_state is None:
            hidden_state = torch.zeros((a.shape[0],self.z_size+2*self.rnn_hidden_size)).to(a.device)
            obs = x.transpose(0, 1)[0]
        else:
            obs = None
        act = a.transpose(0, 1)
        # hidden_state = hidden_state.transpose(0,1)
        z, h, c = hidden_state.split([self.z_size, self.rnn_hidden_size, self.rnn_hidden_size], -1)

        seq_normal, seq_h = self.pred(act, z, h, c, x=obs)
        if y is not None:
            new_obs = y.transpose(0, 1)
            seq_mean = self.normalize(seq_normal.mean)
            seq_std = seq_normal.stddev/torch.from_numpy(self.norm_std).to(x.device)
            new_obs_norm = self.normalize(new_obs)
            seq_norm = gaussian_loss(Normal(seq_mean, seq_std), new_obs_norm)
            mae_norm = torch.mean(torch.abs(seq_mean-new_obs_norm))

            seq = gaussian_loss(seq_normal, new_obs)
            mae = torch.abs(seq_normal.mean-new_obs)
            if torch.sum(torch.isnan(seq)):
                print('Seq NaN')
            # return torch.mean(seq), torch.mean(mae), torch.mean(seq_normal.stddev)
            return torch.mean(seq_norm), torch.mean(mae), torch.mean(seq_normal.stddev)
        else:
            seq_out = seq_normal.sample()
            sd = seq_normal.stddev

            return seq_out, sd, seq_h

class MDNSeqDynamicsModel(DynamicsModel):
    def __init__(self, env_in, dev=None):
        DynamicsModel.__init__(self, env_in)
        self.lr = 1e-5
        self.is_reset = False
        self.val_seq_len = 100
        self.train_seq = 1
        self.look_ahead_per_epoch = 1
        self.batch_size = 64
        self.max_seq_len = 100
        
        self.model = MDNSeqModel(self.state_dim, self.act_dim)
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

        self.model = MDNSeqModel(self.state_dim, self.act_dim)
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

    def update(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        Xs = torch.from_numpy(np.array([step[0] for step in data]).astype(np.float32)).to(self.device)
        As = torch.from_numpy(np.array([step[1] for step in data]).astype(np.float32)).to(self.device)
        Ys = torch.from_numpy(np.array([step[2] for step in data]).astype(np.float32)).to(self.device)
        seq, mae, sd = self.model(Xs, As, None, Ys)
        seq.backward()
        self.optimizer.step()
        return seq.item(), mae.item(), sd.item()

    def reset(self, obs_in, h=None):
        # x = torch.from_numpy(np.array([obs_in.astype(np.float32)/self.state_mul_const])).to(self.device).unsqueeze(1)
        # _, self.h = self.model(x, None, None, 'reset')
        # self.h = self.h.detach()
        self.is_reset = True
        if h is None:
            return obs_in, self.model.reset(obs_in)
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
            new_obs, sd, state_out = self.model(obs_in, a, state_in)
        elif obs_in is not None:
            x = torch.from_numpy(np.array([obs_in.astype(np.float32)/self.state_mul_const])).to(self.device).unsqueeze(0)
            if save:
                new_obs, sd, self.h = self.model(x, a, self.h)
                self.h = self.h.detach()
                state_out = self.h
            else:
                new_obs, sd, h = self.model(x, a, None)
                state_out = h
        else:
            new_obs, sd, h = self.model(obs_in, a, self.h, None)
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
                return new_obs, state_out.detach(), torch.exp(-sd).squeeze(0)
            return new_obs, state_out.detach()
        else:
            if certainty:
                return new_obs, torch.exp(-sd).squeeze(0)
            return new_obs

    def step(self, action_in, obs_in=None, save=True, state=False, state_in=None):
        return self.step_parallel(action_in, obs_in, save, state, state_in)
