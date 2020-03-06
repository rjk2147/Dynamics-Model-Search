import torch
import torch.nn.functional as F
from torch import nn, optim
from data_loader import DataLoader
import numpy as np

class SequenceModel(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=256, num_recurrent_layers=1, device=None):
        super(SequenceModel, self).__init__()
        self.device=device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim            # encoder and decoder have same hidden state dim bc z_dim is latent_dim
        self.z_dim = latent_dim         
        self.num_recurrent_layers = num_recurrent_layers

        # ------ ENCODER LAYERS ------
        self.encoder_recurrent_layer = nn.GRU(input_size=state_dim+action_dim, hidden_size=latent_dim, num_layers=self.num_recurrent_layers)
        self.layer_mu = nn.Linear(in_features=self.latent_dim, out_features=self.z_dim)
        self.layer_logvar = nn.Linear(in_features=self.latent_dim, out_features=self.z_dim)
        # ------ ENCODER LAYERS ------

        # ------ IMMEDIATE STATE LAYERS ------
        self.state_layer_1 = nn.Linear(in_features=self.latent_dim, out_features=128)
        self.state_layer_2 = nn.Linear(in_features=128, out_features=64)
        self.state_layer_3 = nn.Linear(in_features=64, out_features=self.state_dim)
        # ------ IMMEDIATE STATE LAYERS ------

        # ------ DECODER LAYERS ------
        self.decoder_recurrent_layer = nn.GRU(input_size=action_dim, hidden_size=self.z_dim, num_layers=self.num_recurrent_layers)
        self.decoder_layer_1 = nn.Linear(in_features=self.z_dim, out_features=128)
        self.decoder_layer_2 = nn.Linear(in_features=128, out_features=64)
        self.decoder_layer_3 = nn.Linear(in_features=64, out_features=self.state_dim)
        # ------ DECODER LAYERS ------

        # self.layer_1 = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        # self.layer_2 = nn.Linear(in_features=latent_dim, out_features=state_dim)
        # self.norm_mean = np.array([0]).astype(np.float32)
        # self.norm_std = np.array([1]).astype(np.float32)

    def get_split_points(self, X, max_timesteps=100):
        assert X.shape[1] == max_timesteps
        assert X.shape[-1] == self.state_dim
        end_points = []
        start_points = []
        key = np.zeros((X.shape[-1], ))
        for n in range(X.shape[0]):
            for t in range(X.shape[1]):
                if (X[n, t+1, :] == key).all():
                    end_points.append(t)
                    start_points.append(t+2)
                    break
                elif not (X[n, X.shape[1]-1-t, :] == key).all():
                    end_points.append(X.shape[1]-1-t)
                    start_points.append(X.shape[1]-1-t + 2)
                    break
        return end_points, start_points
    
    def build_input_for_encoder(self, s, a):
        x = np.concatenate((s, a), axis=-1)
        h = torch.ones((x.shape[0], self.num_recurrent_layers, self.latent_dim)).to(self.device)
        x = torch.from_numpy(x).to(self.device).float()
        return x, h

    def get_final_hidden_states_from_encoder(self, output_from_encoder, end_points):
        assert output_from_encoder.shape[0] == len(end_points)
        final_hidden_states = []
        for n in range(output_from_encoder.shape[0]):
            final_hidden_state = output_from_encoder[n, end_points[n], :].data.cpu().numpy()
            final_hidden_state = final_hidden_state.reshape((1, final_hidden_state.shape[0]))
            final_hidden_states.append(final_hidden_state)
        final_hidden_states = np.array(final_hidden_states)
        return torch.from_numpy(final_hidden_states).to(self.device)
    
    def reparametrize_from_final_hidden_states(self, final_hidden_states):
        N, D = final_hidden_states.shape[0], final_hidden_states.shape[-1]
        h_f = final_hidden_states.reshape((N, D))
        mu = self.layer_mu(h_f)
        logvar = self.layer_logvar(h_f)
        return mu, logvar
    
    def sample_embeddings(self, mu, logvar):
        eps = torch.from_numpy(np.random.normal(loc=0.0, scale=1.0, size=(mu.shape[0], mu.shape[1]))).float().to(self.device)
        return mu + torch.exp(logvar/2.0)*eps
    
    def predict_final_state_from_embedding(self, embeddings):
        x = embeddings
        x = self.state_layer_1(x)
        x = torch.relu(x)
        x = self.state_layer_2(x)
        x = torch.relu(x)
        x = self.state_layer_3(x)
        return x
    
    @staticmethod
    def roll(x, n):
        return torch.cat((x[-n:, :], x[:-n, :]))

    def build_input_for_decoder(self, a, embeddings, start_points):
        assert a.shape[0] == len(start_points)
        actions = torch.from_numpy(a).to(self.device)
        # before_ = actions.clone()
        for n in range(a.shape[0]):
            # before = actions[n].clone() 
            actions[n] = SequenceModel.roll(actions[n], n=-start_points[n])
            # after = actions[n]
            # print(SequenceModel.roll(after, n=start_points[n]) == before)
            # print(before == after)
        # after_ = actions
        # print(before_ == after_)
        return actions, embeddings.reshape((embeddings.shape[0], 1, embeddings.shape[-1]))
    
    def predict_next_states_from_decoder_hidden_states(self, x):
        x = self.decoder_layer_1(x)
        x = torch.relu(x)
        x = self.decoder_layer_2(x)
        x = torch.relu(x)
        x = self.decoder_layer_3(x)
        return x.transpose(0,1)
    
    def compute_loss_on_next_states(self, s_next, s_next_hat, start_points):
        shifted_s_next = torch.from_numpy(s_next).to(self.device)
        for n in range(s_next.shape[0]):
            # before = shifted_s_next[n].clone()
            shifted_s_next[n] = SequenceModel.roll(shifted_s_next[n], n=-start_points[n])
            # after = shifted_s_next[n]
            # print(before, after)
        error = torch.abs(shifted_s_next-s_next_hat)
        loss = torch.autograd.Variable(torch.zeros(1, 1), requires_grad=True).to(self.device)
        
        for n in range(error.shape[0]):
            error_sample = error[n, :-start_points[n], :]
            loss = loss + torch.sum(error_sample)
        
        return loss

    
    def compute_loss_on_final_states(self, s_final, s_final_hat):
        pass

    def compute_distribution_loss_on_embeddings(self, mu, logvar):
        pass
    
    def forward(self, s, a, s_final, s_next):
        assert s.shape[-1] == s_final.shape[-1] == s_next.shape[-1] == self.state_dim 
        assert s.shape[0] == a.shape[0] == s_final.shape[0] == s_next.shape[0]
        assert a.shape[-1] == self.action_dim

        end_points, start_points = self.get_split_points(s)

        # ---------- ENCODER -------------
        input_to_encoder, initial_hidden_state_to_encoder = self.build_input_for_encoder(s, a)
        output_from_encoder, final_hidden_state_from_encoder = self.encoder_recurrent_layer(input_to_encoder.transpose(0,1), initial_hidden_state_to_encoder.transpose(0,1))
        output_from_encoder, final_hidden_state_from_encoder  = output_from_encoder.transpose(0,1), final_hidden_state_from_encoder.transpose(0,1)
        final_hidden_states_from_encoder = self.get_final_hidden_states_from_encoder(output_from_encoder, end_points)
        mu, logvar = self.reparametrize_from_final_hidden_states(final_hidden_states_from_encoder)
        embeddings = self.sample_embeddings(mu, logvar)
        # ---------- ENCODER -------------
        
        # ---------- IMMEDIATE STATE -------------
        s_final_hat = self.predict_final_state_from_embedding(embeddings)
        # ---------- IMMEDIATE STATE -------------

        # ---------- DECODER -------------
        input_to_decoder, initial_hidden_state_to_decoder = self.build_input_for_decoder(a, embeddings, start_points)
        # print(input_to_decoder.shape, (input_to_decoder.data.cpu().numpy()==a).all())
        output_from_decoder, final_hidden_state_from_decoder = self.decoder_recurrent_layer(input_to_decoder.transpose(0,1), initial_hidden_state_to_decoder.transpose(0,1))
        s_next_hat = self.predict_next_states_from_decoder_hidden_states(output_from_decoder)
        # ---------- DECODER -------------

        return self.compute_loss_on_next_states(s_next, s_next_hat, start_points)

        # print(output_from_encoder[:, 99:, :] == final_hidden_state_from_encoder)
        # print(output_from_encoder[:, 0:1, :] == initial_hidden_state_to_encoder)
        # print(output_from_encoder.shape, final_hidden_state_from_encoder.shape)
        # print(final_hidden_states_from_encoder.shape)
        # print(s_final_hat.shape, s_final.shape)
        # print(s_next_hat.shape)
        # return self.loss(self.normalize(y_hat), self.normalize(y))
    
    

class Trainer:
    def __init__(self, model, device, learing_rate=1e-5):
        self.model = model
        self.device = device
        self.learing_rate = learing_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learing_rate)
    
    def train_epoch(self, train, val, epoch=0, batch_size=64, verbose=False):
        S_train, A_train, S_final_train, S_next_train = train
        S_val, A_val, S_final_val, S_next_val = val

        S_train_batches = DataLoader.batch(S_train, batch_size=batch_size) 
        A_train_batches = DataLoader.batch(A_train, batch_size=batch_size)
        S_final_train_batches = DataLoader.batch(S_final_train, batch_size=batch_size)
        S_next_train_batches = DataLoader.batch(S_next_train, batch_size=batch_size)
        S_val_batches = DataLoader.batch(S_val, batch_size=batch_size) 
        A_val_batches = DataLoader.batch(A_val, batch_size=batch_size)
        S_final_val_batches = DataLoader.batch(S_final_val, batch_size=batch_size)
        S_next_val_batches = DataLoader.batch(S_next_val, batch_size=batch_size)

        self.optimizer.zero_grad()

        # train 
        self.model.train()
        batch_losses = []
        for i in range(S_train_batches.shape[0]):
            if verbose:
                print("EPOCH %d: Training %d/%d ... " % (epoch, i*batch_size, S_train_batches.shape[0]*batch_size), end="")
            # s, a, s_final, s_next = torch.from_numpy(S_train_batches[i]).to(self.device), torch.from_numpy(A_train_batches[i]).to(self.device), torch.from_numpy(S_final_train_batches[i]).to(self.device), torch.from_numpy(S_next_train_batches[i]).to(self.device)
            s, a, s_final, s_next = S_train_batches[i], A_train_batches[i], S_final_train_batches[i], S_next_train_batches[i]
            loss = self.model(s, a, s_final, s_next)
            batch_losses.append(loss)
            loss.backward()
            if verbose:
                print("LOSS: %.5f" % (loss))
            self.optimizer.step()
        avg_loss = sum(batch_losses)/len(batch_losses)

        # # val
        # self.model.eval()
        # val_losses = []
        # for i in range(S_val_batches.shape[0]):
        #     s, a, s_final, s_next = torch.from_numpy(S_val_batches[i]).to(self.device), torch.from_numpy(A_val_batches[i]).to(self.device), torch.from_numpy(S_final_val_batches[i]).to(self.device), torch.from_numpy(S_next_val_batches[i]).to(self.device)
        #     loss = self.model(s, a, s_final, s_next)
        #     val_losses.append(loss)
        # val_loss = sum(val_losses)/len(val_losses)

        # print("EPOCH %d -- Train Loss: %.5f  Validation Loss: %.5f" % (epoch, avg_loss, val_loss))
        return avg_loss
    
    def train(self, train, val, epochs=10, batch_size=64, shuffle=True, verbose=False):
        S_train, A_train, S_final_train, S_next_train = train
        S_val, A_val, S_final_val, S_next_val = val
        losses = []
        for epoch in range(epochs):
            if shuffle:
                p_train = np.random.permutation(S_train.shape[0])
                p_val = np.random.permutation(S_val.shape[0])
                S_train, A_train, S_final_train, S_next_train = S_train[p_train], A_train[p_train], S_final_train[p_train], S_next_train[p_train]
                S_val, A_val, S_final_val, S_next_val = S_val[p_val], A_val[p_val], S_final_val[p_val], S_next_val[p_val]

            losses.append(
                self.train_epoch(train=(S_train, A_train, S_final_train, S_next_train), 
                                 val=(S_val, A_val, S_final_val, S_next_val), 
                                 epoch=epoch, batch_size=batch_size, verbose=verbose)
            )
        return losses
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        


