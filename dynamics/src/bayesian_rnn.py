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
    def __init__(self, input_dim=256, output_dim=32):
        super().__init__()
        self.output_dim = output_dim
        self.linear1 = nn.Linear(in_features=input_dim, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=output_dim*2)

    def forward(self, h):
        z = self.linear1(h)
        z = torch.relu(z)
        z = self.linear2(z)
        z = torch.relu(z)
        z = self.linear3(z)

        z_loc = z[:, :self.output_dim]
        z_scale = torch.nn.functional.softplus(z[:, self.output_dim:])
        
        return z_loc, z_scale

class Decoder(nn.Module):
    def __init__(self, input_dim=32, output_dim=21):
        super().__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=16)
        self.linear2 = nn.Linear(in_features=16, out_features=16)
        self.linear3 = nn.Linear(in_features=16, out_features=output_dim)

    def forward(self, z):
        y = self.linear1(z)
        y = torch.relu(y)
        y = self.linear2(y)
        y = torch.relu(y)
        y = self.linear3(y)
        return y

class BayesianSequenceModel(nn.Module):
    def __init__(self, num_steps, device):
        super().__init__()

        self.num_steps = num_steps  # max timestep length of sequence
        self.device = device

        self.likelihood_sd = 0.1 # ???
        self.z_size = 32 # latent dimension size
        self.rnn_hidden_size = 256 # hidden state dimension
        self.rnn_input_size = 21 + 8 + self.z_size # state dim + action dim + latent dim

        self.decode = Decoder(input_dim=self.z_size, output_dim=21) # Decoder Neural Network
        self.encode = Encoder(input_dim=self.rnn_hidden_size, output_dim=self.z_size) # Encoder Neural Network
        self.rnn = nn.LSTMCell(self.rnn_input_size, self.rnn_hidden_size)

        # Create parameters.
        self.h_init = nn.Parameter(torch.zeros(1, self.rnn_hidden_size))
        self.c_init = nn.Parameter(torch.zeros(1, self.rnn_hidden_size))
        self.z_init = nn.Parameter(torch.zeros(1, self.z_size))

        self.z_loc_prior = nn.Parameter(
            torch.FloatTensor([0 for _ in range(self.z_size)]),
            requires_grad=False)
        self.z_scale_prior = nn.Parameter(
            torch.FloatTensor([1 for _ in range(self.z_size)]),
            requires_grad=False)

        self.ix = None

        self.z_prev, self.h_prev, self.c_prev = None, None, None

    def guide(self, X, A, Y, batch_size, z_prev=None, h_prev=None, c_prev=None):
        # print("Guide: ", end="")

        pyro.module('rnn', self.rnn)
        pyro.module("encode", self.encode)

        pyro.param('h_init', self.h_init)
        pyro.param('c_init', self.c_init)
        pyro.param('z_init', self.z_init)
        
        X_ = torch.cat([X, A], dim=-1).to(X.device)

        # print("X_ shape: ", X_.shape, end="")

        with pyro.plate('data', X_.shape[0], subsample_size = batch_size, device = X_.device) as ix:
            batch = X_[ix]      # shape (N, T, D)
            N = batch.shape[0]
            # print("batch shape: ", batch.shape, end="")

            h = self.h_init.expand(N, self.rnn_hidden_size).to(self.device) if h_prev is None else h_prev
            c = self.c_init.expand(N, self.rnn_hidden_size).to(self.device) if c_prev is None else c_prev
            z = self.z_init.expand(N, self.z_size).to(self.device) if z_prev is None else z_prev

            # print("h_init shape: ", h.shape, end="")
            # print("c_init shape: ", c.shape, end="")
            # print("z_init shape: ", z.shape, end="")

            Z = []
            for t in range(batch.shape[1]):
                x = batch[:, t, :]
                z, h, c = self.guide_step(N, t, x, z, h, c)
                Z.append(z)
            
            Z = torch.stack(Z).transpose(0,1)
            # print(Z.shape)
            # print('done guide')
            self.ix = ix.data.cpu().numpy()
            self.z_prev, self.h_prev, self.c_prev = z.data.cpu().numpy(), h.data.cpu().numpy(), c.data.cpu().numpy()
            return Z.data.cpu().numpy()

    # N = number of samples
    # t = timestep t
    def guide_step(self, N, t, x, prev_z, prev_h, prev_c):
        rnn_input = torch.cat([x, prev_z], dim=-1)
        h, c = self.rnn(rnn_input, (prev_h, prev_c))

        z_loc, z_scale = self.encode(h)  # could also encode(x, h)
        # print("z_loc shape: ", z_loc.shape, end="")
        # print("z_scale shape: ", z_scale.shape, end="")       

        z = pyro.sample("z_{}".format(t),
                        dist.Normal(z_loc, z_scale)
                            .to_event(1) 
                        )
        return z, h, c

    def model(self, X, A, Y, batch_size):
        # print("Model: ", end="")
        pyro.module("decode", self.decode)
        
        X_ = torch.cat([X, A], dim=-1).to(self.device)
        # print("X_ shape: ", X_.shape, end="")

        with pyro.plate('data', X_.shape[0], device = X_.device) as ix:
            batch_X = X_[ix]
            batch_Y = Y[ix] if Y is not None else None

            # print("batch shape: ", batch.shape, end="")
            N = batch_X.shape[0]
            T = batch_X.shape[1]
            Y_hat, Z = self.prior(N,T)    # shape of Y_hat or Z is (N, T, D)
            O = []
            # print("Z shape: ", Z.shape, "Y_hat shape: ", Y_hat.shape)
            # print(self.likelihood_sd)
            for t in range(X    .shape[1]):
                # print(Y_hat[:, t, :].shape, torch.ones(Y_hat[:, t, :].shape[0], Y_hat[:, t, :].shape[1]).shape)
                obs = pyro.sample('obs_{}'.format(t),
                            dist.Normal(Y_hat[:, t, :],
                                        self.likelihood_sd*torch.ones(Y_hat[:, t, :].shape[0], Y_hat[:, t, :].shape[1]).to(self.device))
                                .to_event(1),
                            obs=batch_Y[:, t, :] if Y is not None else None
                            )
                O.append(obs)
            O = torch.stack(O).transpose(0,1)
        
        return Y_hat, Z, O

    # N = number of samples 
    def prior(self, N, T):
        Y, Z = [], []
        for t in range(T):
            y, z = self.prior_step(N, t)
            Y.append(y)
            Z.append(z)

        return torch.stack(Y).transpose(0,1), torch.stack(Z).transpose(0,1)

    # N = number of samples
    # t = timestep t
    def prior_step(self, N, t):
        # print(self.z_loc_prior.expand(N, self.z_size).shape, self.z_scale_prior.expand(N, self.z_size).shape)
        z = pyro.sample("z_{}".format(t), 
                        dist.Normal(self.z_loc_prior.expand(N, self.z_size),
                                    self.z_scale_prior.expand(N, self.z_size))
                            .to_event(1)
                        )
        y = self.decode(z)

        return y, z

    def loss(self, X, A, Y, batch_size):
        trace = poutine.trace(self.guide).get_trace(X=X, A=A, Y=Y, batch_size=batch_size)
        batch_Y_hat, batch_Z = poutine.replay(self.prior, trace=trace)(X.shape[0], X.shape[1])
        batch_Y = Y[self.ix]
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
                    trace = poutine.trace(self.guide).get_trace(X=x0, A=a0, Y=y0, batch_size=1, z_prev=z, h_prev=h, c_prev=c)
                z, h, c = torch.from_numpy(self.z_prev).to(self.device), torch.from_numpy(self.h_prev).to(self.device), torch.from_numpy(self.c_prev).to(self.device)
                # y_hat, _ = poutine.replay(self.prior, trace=trace)(x0.shape[0], x0.shape[1])
                y_hat, _, o = poutine.replay(self.model, trace=trace)(x0, a0, None, batch_size=1)
                O.append(o.data.cpu().numpy())
                Y_hat.append(y_hat.data.cpu().numpy())
                x0 = y_hat
        Y_hat = np.array(Y_hat)
        O = np.array(O)
        return Y_hat.reshape((Y_hat.shape[1], Y_hat.shape[0], Y_hat.shape[-1])), O.reshape((O.shape[1], O.shape[0], O.shape[-1]))

class Trainer:
    def __init__(self, model, device, learing_rate=1e-5):
        self.model = model
        self.device = device
        self.learing_rate = learing_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learing_rate)
    
    def train_epoch(self, train, val, epoch=0, batch_size=64, verbose=False):
        X_train, A_train, Y_train = train
        X_val, A_val, Y_val = val

        X_train_batches, A_train_batches, Y_train_batches = DataLoader.batch(X_train, batch_size=batch_size), DataLoader.batch(A_train, batch_size=batch_size), DataLoader.batch(Y_train, batch_size=batch_size)
        X_val_batches, A_val_batches, Y_val_batches = DataLoader.batch(X_val, batch_size=batch_size), DataLoader.batch(A_val, batch_size=batch_size), DataLoader.batch(Y_val, batch_size=batch_size)

        assert X_train_batches.shape[0] == A_train_batches.shape[0] == Y_train_batches.shape[0]
        assert X_val_batches.shape[0] == A_val_batches.shape[0] == Y_val_batches.shape[0]
        
        self.optimizer.zero_grad()

        # train 
        self.model.train()
        batch_losses = []
        for i in range(X_train_batches.shape[0]):
            if verbose:
                print("EPOCH %d: Training %d/%d ... " % (epoch, i*batch_size, X_train_batches.shape[0]*batch_size), end="")
            X, A, Y = torch.from_numpy(X_train_batches[i]).to(self.device), torch.from_numpy(A_train_batches[i]).to(self.device), torch.from_numpy(Y_train_batches[i]).to(self.device)
            loss = self.model(X, A, Y)
            batch_losses.append(loss)
            loss.backward()
            if verbose:
                print("LOSS: %.5f" % (loss))
            self.optimizer.step()
        avg_loss = sum(batch_losses)/len(batch_losses)

        # val
        self.model.eval()
        val_losses = []
        for i in range(X_val_batches.shape[0]):
            X, A, Y = torch.from_numpy(X_val_batches[i]).to(self.device), torch.from_numpy(A_val_batches[i]).to(self.device), torch.from_numpy(Y_val_batches[i]).to(self.device)
            loss = self.model(X, A, Y)
            val_losses.append(loss)
        val_loss = sum(val_losses)/len(val_losses)

        print("EPOCH %d -- Train Loss: %.5f  Validation Loss: %.5f" % (epoch, avg_loss, val_loss))
        return avg_loss
    
    def train(self, train, val, epochs=10, batch_size=64, shuffle=True):
        X_train, A_train, Y_train = train
        X_val, A_val, Y_val = val

        for epoch in range(epochs):
            if shuffle:
                p_train = np.random.permutation(X_train.shape[0])
                p_val = np.random.permutation(X_val.shape[0])
                X_train, A_train, Y_train = X_train[p_train], A_train[p_train], Y_train[p_train]
                X_val, A_val, Y_val = X_val[p_val], A_val[p_val], Y_val[p_val]
            self.train_epoch(train=(X_train, A_train, Y_train), 
                             val=(X_val, A_val, Y_val), 
                             epoch=epoch, batch_size=batch_size)
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        


