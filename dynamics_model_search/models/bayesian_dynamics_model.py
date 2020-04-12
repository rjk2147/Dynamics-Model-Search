from collections import namedtuple
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.optim as optim
from models.dynamics_model import DynamicsModel

from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO


# from pyro.distributions import TransformedDistribution
# from pyro.distributions.transforms import affine_autoregressive

# from modules import MLP, Decoder, Encoder, Identity, Predict
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
        out = self.linear1(h)
        out = torch.relu(out)
        out = self.linear2(out)
        out = torch.relu(out)

        z = self.linear_z(out)
        z_loc = z[:, :self.z_output_size]
        z_scale = torch.nn.functional.softplus(z[:, self.z_output_size:])

        return z_loc, z_scale, 0, 0

        # sigma = self.linear_sigma(out)
        # sigma_loc = torch.sigmoid(sigma[:, :self.sigma_output_size])
        # sigma_scale = torch.nn.functional.softplus(sigma[:, self.sigma_output_size:])

        # return z_loc, z_scale, sigma_loc, sigma_scale

class Decoder(nn.Module):
    def __init__(self, input_size=21 + 128, output_size=21):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=32)
        self.linear4 = nn.Linear(in_features=32, out_features=16)
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

class BayesianSequenceModel(nn.Module):
    def __init__(self, state_size=21, action_size=8, z_size=128, hidden_state_size=512, likelihood_std=0.01,
                 device=None, path=None):
        super().__init__()
        self.device = device
        self.path = path

        self.z_size = z_size
        self.input_size = state_size + action_size
        self.output_size = state_size
        self.rnn_input_size = action_size + z_size
        self.rnn_hidden_size = hidden_state_size

        self.priors = {
            "z": {
                "loc": nn.Parameter(torch.zeros((1, self.z_size)).to(device), requires_grad=False),
                "scale": nn.Parameter(torch.ones((1, self.z_size)).to(device), requires_grad=False)
            },

            "sigma": {
                "loc": nn.Parameter(torch.zeros((1, self.output_size)).to(device), requires_grad=False),
                "scale": nn.Parameter(torch.ones((1, self.output_size)).to(device), requires_grad=False)
            }
        }
        # track priors by nn.Module
        self.prior_z_loc = self.priors["z"]["loc"]
        self.prior_z_scale = self.priors["z"]["scale"]
        self.prior_sigma_loc = self.priors["sigma"]["loc"]
        self.prior_sigma_scale = self.priors["sigma"]["scale"]

        self.init = {
            "z": nn.Parameter(torch.zeros(1, self.z_size).to(device)),
            "h": nn.Parameter(torch.zeros(1, self.rnn_hidden_size).to(device)),
            "c": nn.Parameter(torch.zeros(1, self.rnn_hidden_size).to(device))
        }
        # track learnable initial states by nn.Module
        self.init_z = self.init["z"]
        self.init_h = self.init["h"]
        self.init_c = self.init["c"]

        self.networks = {
            "encoder": Encoder(input_size=self.rnn_hidden_size, z_output_size=self.z_size,
                               sigma_output_size=self.output_size).to(device),
            "decoder": Decoder(input_size=self.z_size + self.output_size, output_size=self.output_size).to(device),
            "rnn": nn.LSTMCell(self.rnn_input_size, self.rnn_hidden_size).to(device)
        }
        # track learnable networks by nn.Module
        self.encoder = self.networks["encoder"]
        self.decoder = self.networks["decoder"]
        self.rnn = self.networks["rnn"]

        self.guide_cache = {
            "ix": None,
            "z_prev": None,
            "h_prev": None,
            "c_prev": None
        }

        self.likelihood_std = likelihood_std
        self.n_samples = 100
        self.uncertainty = False

    def guide(self, X, A, Y, batch_size, prev_z=None, prev_h=None, prev_c=None):
        pyro.module('rnn', self.networks["rnn"])
        pyro.module("encoder", self.networks["encoder"])
        pyro.param('h_init', self.init["h"])
        pyro.param('c_init', self.init["c"])
        pyro.param('z_init', self.init["z"])

        # X_ = torch.cat([X, A], dim=-1).to(self.device)

        with pyro.plate('data', X.shape[0], subsample_size=batch_size, device=self.device) as ix:
            batch_X = X[ix]
            batch_A = A[ix]

            z = self.init["z"].expand(batch_X.shape[0], self.z_size).to(self.device) if prev_z is None else prev_z
            h = self.init["h"].expand(batch_X.shape[0], self.rnn_hidden_size).to(
                self.device) if prev_h is None else prev_h
            c = self.init["c"].expand(batch_X.shape[0], self.rnn_hidden_size).to(
                self.device) if prev_c is None else prev_c

            Z = []
            for t in range(batch_X.shape[1]):
                z, h, c = self.guide_step(t=t, a=batch_A[:, t, :], prev_z=z, prev_h=h, prev_c=c)
                Z.append(z)
            Z = torch.stack(Z).transpose(0, 1)

            self.guide_cache = {
                "ix": ix.data.cpu().numpy(),
                "z_prev": z.data.cpu().numpy(),
                "h_prev": h.data.cpu().numpy(),
                "c_prev": c.data.cpu().numpy()
            }

            return Z.data.cpu().numpy()

    def guide_step(self, t, a, prev_z, prev_h, prev_c):
        rnn_input = torch.cat([a, prev_z], dim=-1)
        h, c = self.networks["rnn"](rnn_input, (prev_h, prev_c))
        z_loc, z_scale, sigma_loc, sigma_scale = self.networks["encoder"](h)  # could also encode(x, h)

        z = pyro.sample("z_{}".format(t),
                        dist.Normal(z_loc, z_scale)
                        .to_event(1)
                        )
        # sigma = pyro.sample("sigma_{}".format(t),
        #         dist.Normal(sigma_loc, sigma_scale)
        #             .to_event(1)
        #         )
        return z, h, c

    def model(self, X, A, Y, batch_size):
        pyro.module("decoder", self.networks["decoder"])

        # X_ = torch.cat([X, A], dim=-1).to(self.device)

        with pyro.plate('data', A.shape[0], device=self.device) as ix:
            batch_X = X[ix]
            batch_A = A[ix]
            batch_Y = Y[ix] if Y is not None else None

            Y_hat, Z = self.prior(X=batch_X, N=batch_X.shape[0], T=batch_X.shape[1])
            O = []

            for t in range(X.shape[1]):
                obs = pyro.sample('obs_{}'.format(t),
                                  dist.Normal(loc=Y_hat[:, t, :],
                                              scale=self.likelihood_std * torch.ones(Y_hat[:, t, :].shape[0],
                                                                                     Y_hat[:, t, :].shape[1]).to(
                                                  self.device))
                                  .to_event(1),
                                  obs=batch_Y[:, t, :] if batch_Y is not None else None
                                  )
                O.append(obs)

            O = torch.stack(O).transpose(0, 1)

        return Y_hat.data.cpu().numpy(), Z.data.cpu().numpy(), O.data.cpu().numpy()

    def prior(self, X, N, T):
        Y, Z = [], []
        for t in range(T):
            y, z = self.prior_step(x=X[:, 0, :], N=N, t=t)
            Y.append(y)
            Z.append(z)
        return torch.stack(Y).transpose(0, 1), torch.stack(Z).transpose(0, 1)

    def prior_step(self, x, N, t):
        z = pyro.sample("z_{}".format(t),
                        dist.Normal(self.priors["z"]["loc"].expand(N, self.z_size),
                                    self.priors["z"]["scale"].expand(N, self.z_size))
                        .to_event(1)
                        )
        y = self.networks["decoder"](x, z)
        return y, z

    def loss(self, X, A, Y, batch_size):
        trace = poutine.trace(self.guide).get_trace(X=X, A=A, Y=Y, batch_size=batch_size)
        batch_Y_hat, batch_Z = poutine.replay(self.prior, trace=trace)(X=X[self.guide_cache["ix"]], N=batch_size,
                                                                       T=X.shape[1])
        batch_Y = Y[self.guide_cache["ix"]]
        return torch.mean(torch.abs(
            batch_Y - batch_Y_hat)).item(), batch_Z.data.cpu().numpy(), batch_Y_hat.data.cpu().numpy(), batch_Y.data.cpu().numpy()

    def predict(self, x, A, Y=None, H=None):
        # X has shape (N, 1, D)
        # A has shape (N, T, D) for T actions
        N = x.shape[0]
        T = A.shape[1]
        Y_hat = []
        O = []
        if H is None:
            z, h, c = None, None, None
        else:
            z, h, c = H.split([self.z_size, self.rnn_hidden_size, self.rnn_hidden_size], -1)
        for t in range(0, T):
            a = A[:, t:t + 1, :]
            y = Y[:, t:t + 1, :] if Y is not None else None
            trace = None
            if z is None:
                trace = poutine.trace(self.guide).get_trace(X=x, A=a, Y=y, batch_size=N)
            else:
                trace = poutine.trace(self.guide).get_trace(X=x, A=a, Y=y, batch_size=N, prev_z=z, prev_h=h, prev_c=c)
            z, h, c = torch.from_numpy(self.guide_cache["z_prev"]).to(self.device), torch.from_numpy(
                self.guide_cache["h_prev"]).to(self.device), torch.from_numpy(self.guide_cache["c_prev"]).to(
                self.device)
            y_hat, _, o = poutine.replay(self.model, trace=trace)(X=x, A=a, Y=None, batch_size=N)
            O.append(o)
            Y_hat.append(y_hat)
        Y_hat = np.array(Y_hat)
        O = np.array(O)
        new_H = torch.cat([z, h, c], -1)
        return Y_hat.reshape((Y_hat.shape[1], Y_hat.shape[0], Y_hat.shape[-1])), O.reshape(
            (O.shape[1], O.shape[0], O.shape[-1])), new_H

    def predict_with_uncertainty(self, x, A, H=None, Y=None):
        new_x = x.repeat(self.n_samples, 1, 1)
        new_A = A.repeat(self.n_samples, 1, 1)
        new_H = torch.cat(H.split(1,0), 1).squeeze(0)
        Y_hat, O, tmp_H = self.predict(x=new_x, A=new_A, Y=Y, H=new_H)
        Y_hat = torch.from_numpy(Y_hat)
        split_y_hat = Y_hat.chunk(self.n_samples)
        v = torch.stack(split_y_hat)

        Hs = tmp_H.chunk(self.n_samples)
        Hs = torch.stack(Hs).transpose(0,1)
        return torch.mean(v, 0), torch.std(v, 0), Hs

    def forward(self, x, a, H=None, y=None):
        if self.uncertainty:
            new_obs, sd, new_H = self.predict_with_uncertainty(x, a, H=H)
            new_obs = new_obs.to(x.device)
        else:
            new_obs, O, new_H = self.predict(x, a, H=H)
            new_obs = torch.from_numpy(new_obs).to(x.device)
            sd = torch.zeros_like(new_obs)
        return new_obs, sd, new_H

    def reset(self):
        h = self.init_z, self.init_h, self.init_c
        h = torch.cat(h, -1)
        if self.uncertainty:
            h = h.repeat(self.n_samples, 1).unsqueeze(0)
        return h

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BayesianSequenceDynamicsModel(DynamicsModel):
    def __init__(self, env_in, dev=None):
        DynamicsModel.__init__(self, env_in)
        self.lr = 1e-3
        self.is_reset = False
        self.val_seq_len = 100
        self.train_seq = 1
        self.look_ahead_per_epoch = 1
        self.batch_size = 1600
        self.max_seq_len = 100

        self.adam = optim.Adam({"lr": self.lr})
        self.elbo = Trace_ELBO()
        self.model = BayesianSequenceModel(state_size=sum(self.state_dim), action_size=self.act_dim,
                                           z_size=32, hidden_state_size=256,likelihood_std=0.01)
        self.svi = SVI(self.model.model, self.model.guide, self.adam, loss=self.elbo)

        if dev is None:
            self.model.to(device)
            self.device = device
        else:
            self.model.to(dev)
            self.device = dev
        self.state_mul_const_tensor = torch.Tensor(self.state_mul_const).to(self.device)
        self.act_mul_const_tensor = torch.Tensor(self.act_mul_const).to(self.device)
        # self.model.eval()

    def reinit(self, state_dim, state_mul_const, act_dim, act_mul_const):
        self.state_mul_const = state_mul_const
        self.state_mul_const[self.state_mul_const == np.inf] = 1

        self.act_mul_const = act_mul_const
        self.act_dim = act_dim
        self.state_dim = state_dim

        self.model = BayesianSequenceModel(self.state_dim, self.act_dim)
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
        elbo_loss = self.svi.step(X=Xs, A=As, Y=Ys, batch_size=self.batch_size)
        # train_batch_mae_loss, train_batch_Z, train_batch_Y_hat, train_batch_Y = self.model.loss(X=Xs, A=As, Y=Ys,
        #                                                                                        batch_size=self.batch_size)
        return [elbo_loss.item()]

    def reset(self, obs_in, h=None):
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
            x = torch.from_numpy(np.array([obs_in.astype(np.float32) / self.state_mul_const])).to(
                self.device).unsqueeze(0)
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

        new_obs = new_obs.squeeze(1).detach() * self.state_mul_const_tensor.to(new_obs.device)
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
