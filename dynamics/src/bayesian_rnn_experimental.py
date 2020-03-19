from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
# from modules import MLP, Decoder, Encoder, Identity, Predict

class Encoder(nn.Module):
    def __init__(self, input_size=256, z_output_size=32, sigma_output_size=21):
        super().__init__()
        self.z_output_size = z_output_size
        self.sigma_output_size = sigma_output_size
        self.linear1 = nn.Linear(in_features=input_size, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=128)
        self.linear_z = nn.Linear(in_features=128, out_features=z_output_size*2)
        self.linear_sigma = nn.Linear(in_features=128, out_features=sigma_output_size*2)

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
        # sigma_loc = sigma[:, :self.sigma_output_size]
        # sigma_scale = torch.nn.functional.softplus(sigma[:, self.sigma_output_size:])
        
        # return z_loc, z_scale, sigma_loc, sigma_scale

class Decoder(nn.Module):
    def __init__(self, input_size=32, output_size=21):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=16)
        self.linear2 = nn.Linear(in_features=16, out_features=16)
        self.linear3 = nn.Linear(in_features=16, out_features=output_size)

    def forward(self, z):
        y = self.linear1(z)
        y = torch.relu(y)
        y = self.linear2(y)
        y = torch.relu(y)
        y = self.linear3(y)
        return y

class BayesianSequenceModel(nn.Module):
    def __init__(self, state_size=21, action_size=8, z_size=32, hidden_state_size=256, device=None):
        super().__init__()
        self.device = device
        self.z_size = z_size
        self.input_size = state_size+action_size
        self.output_size = state_size
        self.rnn_input_size = state_size+action_size+z_size
        self.rnn_hidden_size = hidden_state_size

        self.priors = {
            "z" : {
                "loc"   :   nn.Parameter(torch.zeros((1, self.z_size)).to(device), requires_grad=False),
                "scale" :   nn.Parameter(torch.ones((1, self.z_size)).to(device), requires_grad=False)
            },

            "sigma" : {
                "loc"   :   nn.Parameter(torch.zeros((1, self.output_size)).to(device), requires_grad=False),
                "scale" :   nn.Parameter(torch.ones((1, self.output_size)).to(device), requires_grad=False)
            }
        }

        self.init = {
            "z" :   nn.Parameter(torch.zeros(1, self.z_size).to(device)),
            "h" :   nn.Parameter(torch.zeros(1, self.rnn_hidden_size).to(device)),
            "c" :   nn.Parameter(torch.zeros(1, self.rnn_hidden_size).to(device))
        }

        self.networks = {
            "encoder"   :   Encoder(input_size=self.rnn_hidden_size, z_output_size=self.z_size, sigma_output_size=self.output_size).to(device),
            "decoder"   :   Decoder(input_size=self.z_size, output_size=self.output_size).to(device),
            "rnn"       :   nn.LSTMCell(self.rnn_input_size, self.rnn_hidden_size).to(device)
        }

        self.guide_cache = {
            "ix"    :   None,
            "z_prev":   None,
            "h_prev":   None,
            "c_prev":   None    
        }
        
        self.likelihood_sd = 0.0001 # ???


    def guide(self, X, A, Y, batch_size, prev_z=None, prev_h=None, prev_c=None):
        pyro.module('rnn', self.networks["rnn"])
        pyro.module("encoder", self.networks["encoder"])
        pyro.param('h_init', self.init["h"])
        pyro.param('c_init', self.init["c"])
        pyro.param('z_init', self.init["z"])
        
        X_ = torch.cat([X, A], dim=-1).to(self.device)

        with pyro.plate('data', X_.shape[0], subsample_size=batch_size, device=self.device) as ix:
            batch_X = X_[ix]      

            z = self.init["z"].expand(batch_X.shape[0], self.z_size).to(self.device) if prev_z is None else prev_z
            h = self.init["h"].expand(batch_X.shape[0], self.rnn_hidden_size).to(self.device) if prev_h is None else prev_h
            c = self.init["c"].expand(batch_X.shape[0], self.rnn_hidden_size).to(self.device) if prev_c is None else prev_c

            Z = []
            for t in range(batch_X.shape[1]):
                z, h, c = self.guide_step(t=t, x=batch_X[:, t, :], prev_z=z, prev_h=h, prev_c=c)
                Z.append(z)
            Z = torch.stack(Z).transpose(0,1)

            self.guide_cache = {
                "ix"    :   ix.data.cpu().numpy(),
                "z_prev":   z.data.cpu().numpy(),
                "h_prev":   h.data.cpu().numpy(),
                "c_prev":   c.data.cpu().numpy()
            }
            
            return Z.data.cpu().numpy()

    def guide_step(self, t, x, prev_z, prev_h, prev_c):
        rnn_input = torch.cat([x, prev_z], dim=-1)
        h, c = self.networks["rnn"](rnn_input, (prev_h, prev_c))
        z_loc, z_scale, sigma_loc, sigma_scale = self.networks["encoder"](h)   # could also encode(x, h)
        
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

        X_ = torch.cat([X, A], dim=-1).to(self.device)

        with pyro.plate('data', X_.shape[0], device=self.device) as ix:
            batch_X = X_[ix]
            batch_Y = Y[ix] if Y is not None else None
            Y_hat, Z = self.prior(N=batch_X.shape[0], T=batch_X.shape[1])    
            O = []
            
            for t in range(X.shape[1]):
                # sigma = pyro.sample('sigma_{}'.format(t),
                #                 dist.Normal(self.priors["sigma"]["loc"].expand(Y_hat.shape[0], self.output_size),
                #                             self.priors["sigma"]["scale"].expand(Y_hat.shape[0], self.output_size))
                #                     .to_event(1),
                #             )
                # obs = pyro.sample('obs_{}'.format(t),
                #                 dist.Normal(loc=Y_hat[:, t, :], scale=sigma.to(self.device))
                #                     .to_event(1),
                #             obs=batch_Y[:, t, :] if batch_Y is not None else None
                #             )
                obs = pyro.sample('obs_{}'.format(t),
                                dist.Normal(loc=Y_hat[:, t, :], scale=self.likelihood_sd*torch.ones(Y_hat[:, t, :].shape[0], Y_hat[:, t, :].shape[1]).to(self.device))
                                    .to_event(1),
                            obs=batch_Y[:, t, :] if batch_Y is not None else None
                            )


                O.append(obs)

            O = torch.stack(O).transpose(0,1)
        
        return Y_hat.data.cpu().numpy(), Z.data.cpu().numpy(), O.data.cpu().numpy()

    def prior(self, N, T):
        Y, Z = [], []
        for t in range(T):
            y, z = self.prior_step(N=N, t=t)
            Y.append(y)
            Z.append(z)
        return torch.stack(Y).transpose(0,1), torch.stack(Z).transpose(0,1)

    def prior_step(self, N, t):
        z = pyro.sample("z_{}".format(t), 
                        dist.Normal(self.priors["z"]["loc"].expand(N, self.z_size),
                                    self.priors["z"]["scale"].expand(N, self.z_size))
                            .to_event(1)
                        )
        y = self.networks["decoder"](z)
        return y, z

    def loss(self, X, A, Y, batch_size):
        trace = poutine.trace(self.guide).get_trace(X=X, A=A, Y=Y, batch_size=batch_size)
        batch_Y_hat, batch_Z = poutine.replay(self.prior, trace=trace)(N=X.shape[0], T=X.shape[1])
        batch_Y = Y[self.guide_cache["ix"]]
        return torch.mean(torch.abs(batch_Y-batch_Y_hat)).item(), batch_Z.data.cpu().numpy(), batch_Y_hat.data.cpu().numpy(), batch_Y.data.cpu().numpy()
        
    def predict(self, X, A, Y, lookahead=5):
        T = X.shape[1]
        Y_hat = []
        O = []
        for i in range(0, T, lookahead):
            x0 = X[:, i:i+1, :]
            z, h, c = None, None, None
            for j in range(lookahead):
                a0 = A[:, i+j:i+j+1, :]
                y0 = Y[:, i+j:i+j+1, :]
                trace = None
                if z is None:
                    trace = poutine.trace(self.guide).get_trace(X=x0, A=a0, Y=y0, batch_size=1)
                else:
                    trace = poutine.trace(self.guide).get_trace(X=x0, A=a0, Y=y0, batch_size=1, prev_z=z, prev_h=h, prev_c=c)
                z, h, c = torch.from_numpy(self.guide_cache["z_prev"]).to(self.device), torch.from_numpy(self.guide_cache["h_prev"]).to(self.device), torch.from_numpy(self.guide_cache["c_prev"]).to(self.device)
                y_hat, _, o = poutine.replay(self.model, trace=trace)(X=x0, A=a0, Y=None, batch_size=1)
                O.append(o)
                Y_hat.append(y_hat)
                x0 = torch.from_numpy(y_hat).to(self.device)
        Y_hat = np.array(Y_hat)
        O = np.array(O)
        return Y_hat.reshape((Y_hat.shape[1], Y_hat.shape[0], Y_hat.shape[-1])), O.reshape((O.shape[1], O.shape[0], O.shape[-1]))

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint)        


