from models.dynamics_model import DynamicsModel
import numpy as np
import torch
from torch import nn, optim
import time, math
from torch.distributions import Normal, Uniform, OneHotCategorical

class MDN(nn.Module):
    def __init__(self, in_features, out_features, num_gaussians, tanh=False):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.tanh = tanh
        # self.pi = CategoricalNetwork(in_features, num_gaussians)
        self.sigma = nn.Linear(in_features, out_features*num_gaussians)
        self.mu = nn.Linear(in_features, out_features*num_gaussians)

    def forward(self, x):
        epsilon = 1e-20
        pi = torch.clamp(self.pi(x), min=epsilon)
        mu = self.mu(x)
        if self.tanh:
            mu = torch.tanh(mu)
        sigma = torch.exp(1+self.sigma(x))
        # sigma = torch.ones_like(mu)/100000.0

        mu = torch.stack(mu.split(self.out_features, dim=-1),-2)
        sigma = torch.stack(sigma.split(self.out_features, dim=-1),-2)
        if torch.isnan(mu).any() or torch.isnan(sigma).any():
            print('NaN in normal!')
            print('')
        return OneHotCategorical(probs=pi), Normal(mu, torch.clamp(sigma, min=epsilon))

def mdn_loss(pi, normal, y):
    loglik = normal.log_prob(y.unsqueeze(-2).expand_as(normal.loc))
    loglik = torch.sum(loglik, dim=-1)
    loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=-1)
    if torch.isnan(loss).any():
        print('NaN in loss!')
        print('')
    return loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MDRNNModel(nn.Module):
    def __init__(self, state_dim, act_dim, state_mul_const=1, drop_rate=0.5):
        super(MDRNNModel, self).__init__()
        self.state_dim = state_dim
        self.state_size = sum(state_dim)
        self.act_dim = act_dim
        self.h = None

        num_recurrent_layers = 1

        self.latent_size = 256
        self.num_gaussians = 1
        self.norm_mean = np.array([0]).astype(np.float32)
        self.norm_std = np.array([1]).astype(np.float32)

        self.rnn = nn.GRU(self.state_size+act_dim, self.latent_size, num_recurrent_layers)

        self.layer1 = nn.Linear(self.latent_size, self.latent_size)
        self.fc_out = MDN(self.latent_size, self.state_size, self.num_gaussians)

    def reset(self):
        return torch.ones((1,1,self.latent_size))

    def decode(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        pi, normal = self.fc_out(x)
        return pi, normal

    # seq_len, batch_len, input_size
    def pred(self, x, a, h=None):
        tmp_in = torch.cat([x, a], dim=-1)
        out_enc, new_h = self.rnn(tmp_in, h)
        pi, normal = self.decode(out_enc)
        return pi, normal, new_h

    def forward(self, x, a, h=None, y=None):
        if h is None:
            h = torch.ones((x.shape[0], 1, self.latent_size)).to(x.device)
        obs = x.transpose(0, 1)
        act = a.transpose(0, 1)
        h = h.transpose(0,1)
        # obs = torch.transpose(x, 0, 1).to(self.layer1._parameters['weight'].device)
        # act = torch.transpose(a, 0, 1).to(self.layer1._parameters['weight'].device)
        # new_obs = torch.transpose(y, 0, 1).to(self.layer1._parameters['weight'].device)

        seq_pi, seq_normal, seq_h = self.pred(obs, act)

        # Added training to learn the change not the whole state
        # seq_normal = Normal(seq_normal.loc + obs.unsqueeze(-2).expand_as(seq_normal.loc), seq_normal.scale)

        if y is not None:
            new_obs = y.transpose(0, 1)
            seq = mdn_loss(seq_pi, seq_normal, new_obs)
            if torch.sum(torch.isnan(seq)):
                print('Seq NaN')
            mean = torch.sum(seq_pi.mean.unsqueeze(-1) * seq_normal.mean, dim=-2)
            mae = torch.mean(torch.abs(mean - new_obs))
            return torch.mean(seq), mae, torch.mean(seq_normal.stddev)
        else:
            seq_out = torch.sum(seq_pi.sample().unsqueeze(-1) * seq_normal.sample(), dim=-2)
            # sd = seq_normal.stddev
            mean = torch.sum(seq_pi.mean.unsqueeze(-1)*seq_normal.mean, dim=-2)
            sd = torch.sum(seq_pi.mean.unsqueeze(-1)*seq_normal.stddev, dim=-2)
            # Should not divide sd by seq_out so sd is a percentage [0-1] of the original value
            # if we did that we would be very certain about things with a large mean and very uncertain about small means
            # sd = sd / mean
            return seq_out, sd, seq_h.transpose(0, 1)

class MDRNNDynamicsModel(DynamicsModel):
    def __init__(self, env_in, dev=None, seq_len=100):
        DynamicsModel.__init__(self, env_in)
        self.lr = 1e-5
        self.is_reset = False
        self.val_seq_len = seq_len
        self.train_seq = 1
        self.look_ahead_per_epoch = 1
        self.batch_size = 64
        self.max_seq_len = seq_len
        
        self.model = MDRNNModel(self.state_dim, self.act_dim)
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

        self.model = MDRNNModel(self.state_dim, self.act_dim)
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

    def step(self, action_in, obs_in=None, save=True, state=False, state_in=None, certainty=False):
        return self.step_parallel(action_in, obs_in, save, state, state_in, certainty)
