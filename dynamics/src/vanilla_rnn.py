import torch
import torch.nn.functional as F
from torch import nn, optim
from data_loader import DataLoader
import numpy as np

class SequenceModel(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=256, num_recurrent_layers=1, device=None):
        super(SequenceModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_recurrent_layers = num_recurrent_layers

        self.recurrent_layer = nn.GRU(input_size=state_dim+action_dim, hidden_size=latent_dim, num_layers=self.num_recurrent_layers)
        self.layer_1 = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.layer_2 = nn.Linear(in_features=latent_dim, out_features=state_dim)

        self.norm_mean = np.array([0]).astype(np.float32)
        self.norm_std = np.array([1]).astype(np.float32)

        self.device=device
    
    # mean, std have to be set
    def normalize(self, x):
        return (x-torch.from_numpy(self.norm_mean).to(x.device)) / torch.from_numpy(self.norm_std).to(x.device)

    def unnormalize(self, x):
        return x*torch.from_numpy(self.norm_std).to(x.device) + torch.from_numpy(self.norm_mean).to(x.device)

    '''
    x: states (N, T, D=state_dim)
    a: actions (N, T, D=action_dim)
    h: initial hidden state (N, L=number of layers=1, H=latent_dim)

    return:
        y_hat: next states (N, T, D=state_dim)
        hidden_final: (N, L=number of layers=1, H=latent_dim)
    '''
    def internal_predict(self, x, a, h):
        assert x.shape[-1] == self.state_dim
        assert a.shape[-1] == self.action_dim
        assert h.shape[-1] == self.latent_dim
        assert x.shape[0] == a.shape[0] == h.shape[0]

        state = x.transpose(0,1)
        action = a.transpose(0,1)
        hidden_initial = h.transpose(0,1)
        
        output, hidden_final = self.recurrent_layer(torch.cat([state, action], dim=-1), hidden_initial)
        y_hat = self.layer_1(output)
        y_hat = torch.relu(y_hat)
        y_hat = self.layer_2(y_hat)

        return y_hat.transpose(0,1), hidden_final.transpose(0,1)

    def predict(self, x, a):
        x = torch.from_numpy(x).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        hidden_initial = torch.ones((x.shape[0], 1, self.latent_dim)).to(x.device)
        y_hat, hidden_final = self.internal_predict(x, a, hidden_initial)
        return y_hat.data.cpu().numpy()

    def foresee(self, x, a, lookahead=5):
        T = x.shape[1]
        Y_hat = []
        for i in range(0, T, lookahead):
            x0 = torch.from_numpy(x[:, i:i+1, :]).to(self.device)
            h0 = torch.ones((x0.shape[0], 1, self.latent_dim)).to(self.device)
            for j in range(lookahead):
                a0 = torch.from_numpy(a[:, i+j:i+j+1, :]).to(self.device)
                x0, h0 = self.internal_predict(x0, a0, h0)
                Y_hat.append(x0.data.cpu().numpy())
        return np.array(Y_hat).reshape((x.shape[0], T, x.shape[2]))        
    
    def loss(self, y_hat, y):
        return torch.mean(torch.abs(y-y_hat))

    def forward(self, x, a, y):
        assert y.shape[-1] == self.state_dim
        assert x.shape[0] == a.shape[0] == y.shape[0]

        hidden_initial = torch.ones((x.shape[0], 1, self.latent_dim)).to(x.device)
        y_hat, hidden_final = self.internal_predict(x, a, hidden_initial)

        return self.loss(self.normalize(y_hat), self.normalize(y))
    
    

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
        


