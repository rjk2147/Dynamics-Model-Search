import numpy as np
import torch
from torch import nn, optim
import time, math
from torch.distributions import Normal, Uniform, OneHotCategorical
import os
from data_loader import DataLoader
from utils.visualize import *
import wandb
import warnings
from utils.tools import *

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
        state_size = state_dim
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

class Trainer:
    def __init__(self, model, device, learing_rate=1e-5, wandb=None):
        self.wandb = wandb
        self.model = model
        self.device = device
        self.learing_rate = learing_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learing_rate)

        # self.wandb.watch(self.model, log="all")     # track network topology

    
    def train_epoch(self, train, val, epoch=0, batch_size=64, verbose=True):
        X_train, A_train, Y_train = train
        X_val, A_val, Y_val = val

        X_train_batches, A_train_batches, Y_train_batches = DataLoader.batch(X_train, batch_size=batch_size), DataLoader.batch(A_train, batch_size=batch_size), DataLoader.batch(Y_train, batch_size=batch_size)
        X_val_batches, A_val_batches, Y_val_batches = DataLoader.batch(X_val, batch_size=batch_size), DataLoader.batch(A_val, batch_size=batch_size), DataLoader.batch(Y_val, batch_size=batch_size)

        assert X_train_batches.shape[0] == A_train_batches.shape[0] == Y_train_batches.shape[0]
        assert X_val_batches.shape[0] == A_val_batches.shape[0] == Y_val_batches.shape[0]
        
        self.optimizer.zero_grad()

        train_seq_batch_losses = []
        train_mae_batch_losses = []
        val_mae_batch_losses = []

        N = X_train_batches.shape[0]

        for i in range(N):
            val = i < X_val_batches.shape[0] 
            #--------------TRAIN--------------
            self.model.train()
            if verbose:
                print("EPOCH %d: Training %d/%d ... " % (epoch, i*batch_size, N*batch_size), end="")
            train_batch_X, train_batch_A, train_batch_Y = torch.from_numpy(X_train_batches[i]).to(self.device), torch.from_numpy(A_train_batches[i]).to(self.device), torch.from_numpy(Y_train_batches[i]).to(self.device)
            train_seq_loss, train_mae_loss, sd = self.model(train_batch_X, train_batch_A, None, train_batch_Y) # modify this line for the forward outputs of model
            train_seq_batch_losses.append(train_seq_loss.item())
            train_mae_batch_losses.append(train_mae_loss.item())
            train_seq_loss.backward()
            self.optimizer.step()

            #--------------VAL--------------
            self.model.eval()
            if val:
                val_batch_X, val_batch_A, val_batch_Y = torch.from_numpy(X_val_batches[i]).to(self.device), torch.from_numpy(A_val_batches[i]).to(self.device), torch.from_numpy(Y_val_batches[i]).to(self.device)
                val_seq_loss, val_mae_loss, sd = self.model(val_batch_X, val_batch_A, None, val_batch_Y)
                val_mae_batch_losses.append(val_mae_loss.item())
            
            #-----------LOG------------
            if i % 10 == 0 and val:            
                train_batch_Y_hat, _, _ = self.model(train_batch_X, train_batch_A)
                val_batch_Y_hat, _, _ = self.model(val_batch_X, val_batch_A)
                train_batch_Y_hat = train_batch_Y_hat.transpose(0, 1).data.cpu().numpy()
                val_batch_Y_hat = val_batch_Y_hat.transpose(0, 1).data.cpu().numpy()

                train_batch_Y = train_batch_Y.data.cpu().numpy()
                val_batch_Y = val_batch_Y.data.cpu().numpy()

                # train curves            
                self.wandb.log({
                    "train-x-velocity"    :   get_velocity_curve(title="x-velocity", true=train_batch_Y[0, :, 0], pred=train_batch_Y_hat[0, :, 0]),
                    "train-y-velocity"    :   get_velocity_curve(title="y-velocity", true=train_batch_Y[0, :, 1], pred=train_batch_Y_hat[0, :, 1]),
                    "train-z-velocity"    :   get_velocity_curve(title="z-velocity", true=train_batch_Y[0, :, 2], pred=train_batch_Y_hat[0, :, 2])
                }, commit=False)
                self.wandb.log({
                    "train-x-position"    :   get_position_curve(title="x-position", true=compute_position_from_velocity(train_batch_Y[0, :, 0]), pred=compute_position_from_velocity(train_batch_Y_hat[0, :, 0])),
                    "train-y-position"    :   get_position_curve(title="y-position", true=compute_position_from_velocity(train_batch_Y[0, :, 1]), pred=compute_position_from_velocity(train_batch_Y_hat[0, :, 1])),
                    "train-z-position"    :   get_position_curve(title="z-position", true=compute_position_from_velocity(train_batch_Y[0, :, 2]), pred=compute_position_from_velocity(train_batch_Y_hat[0, :, 2]))
                }, commit=False)

                self.wandb.log({
                    "val-x-velocity"    :   get_velocity_curve(title="x-velocity", true=val_batch_Y[0, :, 0], pred=val_batch_Y_hat[0, :, 0]),
                    "val-y-velocity"    :   get_velocity_curve(title="y-velocity", true=val_batch_Y[0, :, 1], pred=val_batch_Y_hat[0, :, 1]),
                    "val-z-velocity"    :   get_velocity_curve(title="z-velocity", true=val_batch_Y[0, :, 2], pred=val_batch_Y_hat[0, :, 2])
                }, commit=False)
                self.wandb.log({
                    "val-x-position"    :   get_position_curve(title="x-position", true=compute_position_from_velocity(val_batch_Y[0, :, 0]), pred=compute_position_from_velocity(val_batch_Y_hat[0, :, 0])),
                    "val-y-position"    :   get_position_curve(title="y-position", true=compute_position_from_velocity(val_batch_Y[0, :, 1]), pred=compute_position_from_velocity(val_batch_Y_hat[0, :, 1])),
                    "val-z-position"    :   get_position_curve(title="z-position", true=compute_position_from_velocity(val_batch_Y[0, :, 2]), pred=compute_position_from_velocity(val_batch_Y_hat[0, :, 2]))
                }, commit=False)
        
            if val:
                self.wandb.log({
                    "train_seq_loss":   train_seq_loss.item(),
                    "train_mae_loss":   train_mae_loss.item(),
                    "val_mae_loss"  :   val_mae_loss.item(),
                    "batch"         :   i,
                    "epoch"         :   epoch,
                    "step"          :   N*epoch + i
                })
            else:
                self.wandb.log({
                    "train_seq_loss":   train_seq_loss.item(),
                    "train_mae_loss":   train_mae_loss.item(),
                    "batch"         :   i,
                    "epoch"         :   epoch,
                    "step"          :   N*epoch + i
                })
                
            if verbose:
                print("Seq Train Loss: %.5f  Train MAE Loss: %.5f  Val MAE Loss: %.5f" % (train_seq_loss, train_mae_loss, val_mae_loss if val else 0))

        avg_train_seq_loss = sum(train_seq_batch_losses)/len(train_seq_batch_losses)
        avg_train_mae_loss = sum(train_mae_batch_losses)/len(train_mae_batch_losses)
        avg_val_mae_loss = sum(val_mae_batch_losses)/len(val_mae_batch_losses)

        print("EPOCH %d -- Train Seq Loss: %.5f  Train MAE Loss: %.5f Validation MAE Loss: %.5f" % (epoch, avg_train_seq_loss, avg_train_mae_loss, avg_val_mae_loss))
        return avg_train_seq_loss, avg_train_mae_loss, avg_val_mae_loss
    
    def train(self, train, val, epochs=10, batch_size=64, shuffle=True):
        X_train, A_train, Y_train = train
        X_val, A_val, Y_val = val

        for epoch in range(epochs):
            if shuffle:
                p_train = np.random.permutation(X_train.shape[0])
                p_val = np.random.permutation(X_val.shape[0])
                X_train, A_train, Y_train = X_train[p_train], A_train[p_train], Y_train[p_train]
                X_val, A_val, Y_val = X_val[p_val], A_val[p_val], Y_val[p_val]
            
            train_seq_loss, train_mae_loss, val_mae_loss = self.train_epoch(train=(X_train, A_train, Y_train), 
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

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    machine = "honshu"
    notes = "master_mdn_seq_dynamics_model"
    name = "mdn_seq_dynamics_model"

    wandb.init(project="bayesian_sequence_modelling", name=machine+"/"+name, tags=[machine], notes=notes, reinit=True)

    root = os.getcwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = root+"/../datasets/vanilla_rnn"
    X_train, A_train, Y_train = np.load(dataset+"/train/states.npy"), np.load(dataset+"/train/actions.npy"), np.load(dataset+"/train/next_states.npy")
    X_val, A_val, Y_val = np.load(dataset+"/val/states.npy"), np.load(dataset+"/val/actions.npy"), np.load(dataset+"/val/next_states.npy")
    X_test, A_test, Y_test = np.load(dataset+"/test/states.npy"), np.load(dataset+"/test/actions.npy"), np.load(dataset+"/test/next_states.npy")

    # X_train, Y_train = standardize_across_time(X_train), standardize_across_time(Y_train) 
    # X_val, Y_val = standardize_across_time(X_val), standardize_across_time(Y_val) 
    # X_test, Y_test = standardize_across_time(X_test), standardize_across_time(Y_test) 

    seq_model = MDNSeqModel(state_dim=21, action_size=8, z_size=128, hidden_state_size=512)

    trainer = Trainer(model=seq_model, device=device, learing_rate=1e-5, wandb=wandb)     
    # trainer.load_checkpoint(path=root+"/../models/mdn_seq_model.pth")
    trainer.train(train=(X_train, A_train, Y_train), val=(X_val, A_val, Y_val), epochs=1000, batch_size=800, shuffle=True)