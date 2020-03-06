import torch
import torch.nn.functional as F
from torch import nn, optim
from data_loader import DataLoader
import numpy as np
import random

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
    
    def build_input_for_encoder(self, s, a):
        x = torch.cat([s, a], dim=-1)
        h = torch.ones((x.shape[0], self.num_recurrent_layers, self.latent_dim)).to(self.device)
        # x = torch.from_numpy(x).to(self.device).float()
        return x, h
    
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

    def build_input_for_decoder(self, a, embeddings):
        return a, embeddings.reshape((embeddings.shape[0], 1, embeddings.shape[-1]))
    
    def predict_next_states_from_decoder_hidden_states(self, x):
        x = self.decoder_layer_1(x)
        x = torch.relu(x)
        x = self.decoder_layer_2(x)
        x = torch.relu(x)
        x = self.decoder_layer_3(x)
        return x.transpose(0,1)
    
    def compute_loss_on_next_states(self, s_next, s_next_hat):
        return torch.mean(torch.abs(s_next-s_next_hat))

    def compute_loss_on_final_states(self, s_final, s_final_hat):
        return torch.mean(torch.abs(s_final-s_final_hat))

    def compute_distribution_loss_on_embeddings(self, mu, logvar):
        return self.kl_loss(mu, logvar)

    def kl_loss(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld

    def forward(self, s, a, s_final, s_next, a_next):
        assert s.shape[-1] == s_final.shape[-1] == s_next.shape[-1] == self.state_dim 
        assert s.shape[0] == a.shape[0] == s_final.shape[0] == s_next.shape[0] == a_next.shape[0]
        assert a.shape[-1] == a_next.shape[-1] == self.action_dim
        # print(s.shape, a.shape, s_final.shape, s_next.shape, a_next.shape)

        # ---------- ENCODER -------------
        input_to_encoder, initial_hidden_state_to_encoder = self.build_input_for_encoder(s, a)
        output_from_encoder, final_hidden_state_from_encoder = self.encoder_recurrent_layer(input_to_encoder.transpose(0,1), initial_hidden_state_to_encoder.transpose(0,1))
        output_from_encoder, final_hidden_state_from_encoder  = output_from_encoder.transpose(0,1), final_hidden_state_from_encoder.transpose(0,1)        
        mu, logvar = self.reparametrize_from_final_hidden_states(final_hidden_state_from_encoder)
        embeddings = self.sample_embeddings(mu, logvar)
        # ---------- ENCODER -------------
        
        # ---------- IMMEDIATE STATE -------------
        s_final_hat = self.predict_final_state_from_embedding(embeddings)
        # ---------- IMMEDIATE STATE -------------

        # ---------- DECODER -------------
        input_to_decoder, initial_hidden_state_to_decoder = self.build_input_for_decoder(a_next, embeddings)
        output_from_decoder, final_hidden_state_from_decoder = self.decoder_recurrent_layer(input_to_decoder.transpose(0,1), initial_hidden_state_to_decoder.transpose(0,1))
        s_next_hat = self.predict_next_states_from_decoder_hidden_states(output_from_decoder)
        # ---------- DECODER -------------

        loss_on_embeddings = self.compute_distribution_loss_on_embeddings(mu, logvar)
        loss_on_final_states = self.compute_loss_on_final_states(s_final, s_final_hat)
        loss_on_next_states = self.compute_loss_on_next_states(s_next, s_next_hat)
        total_loss = loss_on_embeddings + loss_on_next_states + loss_on_final_states
        return total_loss, loss_on_embeddings, loss_on_final_states, loss_on_next_states

class Trainer:
    def __init__(self, model, device, learing_rate=1e-5):
        self.model = model
        self.device = device
        self.learing_rate = learing_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learing_rate)
    
    def train_epoch(self, train, val, epoch=0, batch_size=64, verbose=False):
        S_train, A_train, S_final_train, S_next_train, A_next_train = train
        S_val, A_val, S_final_val, S_next_val, A_next_val = val

        S_train_batches = DataLoader.batch(S_train, batch_size=batch_size) 
        A_train_batches = DataLoader.batch(A_train, batch_size=batch_size)
        S_final_train_batches = DataLoader.batch(S_final_train, batch_size=batch_size)
        S_next_train_batches = DataLoader.batch(S_next_train, batch_size=batch_size)
        A_next_train_batches = DataLoader.batch(A_next_train, batch_size=batch_size)

        S_val_batches = DataLoader.batch(S_val, batch_size=batch_size) 
        A_val_batches = DataLoader.batch(A_val, batch_size=batch_size)
        S_final_val_batches = DataLoader.batch(S_final_val, batch_size=batch_size)
        S_next_val_batches = DataLoader.batch(S_next_val, batch_size=batch_size)
        A_next_val_batches = DataLoader.batch(A_next_val, batch_size=batch_size)

        self.optimizer.zero_grad()

        # train 
        self.model.train()
        batch_losses = []
        batch_embedding_losses = []
        batch_final_state_losses = []
        batch_next_state_losses = []

        for i in range(S_train_batches.shape[0]):
            if verbose:
                print("EPOCH %d: Training %d/%d ... " % (epoch, i*batch_size, S_train_batches.shape[0]*batch_size), end="")
            s = torch.from_numpy(S_train_batches[i]).to(self.device)
            a = torch.from_numpy(A_train_batches[i]).to(self.device)
            s_final = torch.from_numpy(S_final_train_batches[i]).to(self.device)
            s_next = torch.from_numpy(S_next_train_batches[i]).to(self.device)
            a_next = torch.from_numpy(A_next_train_batches[i]).to(self.device)
            loss, loss_on_embeddings, loss_on_final_states, loss_on_next_states = self.model(s, a, s_final, s_next, a_next)
            batch_losses.append(loss.item())
            batch_embedding_losses.append(loss_on_embeddings.item())
            batch_final_state_losses.append(loss_on_final_states.item())
            batch_next_state_losses.append(loss_on_next_states.item())
            loss.backward()
            if verbose:
                print("LOSS: %.5f" % (loss.item()))
            self.optimizer.step()

        avg_loss = sum(batch_losses)/len(batch_losses)
        avg_embedding_loss = sum(batch_embedding_losses)/len(batch_embedding_losses)
        avg_final_state_loss = sum(batch_final_state_losses)/len(batch_final_state_losses)
        avg_next_state_loss = sum(batch_next_state_losses)/len(batch_next_state_losses)

        # val
        self.model.eval()
        val_losses = []
        for i in range(S_val_batches.shape[0]):
            s = torch.from_numpy(S_val_batches[i]).to(self.device)
            a = torch.from_numpy(A_val_batches[i]).to(self.device)
            s_final = torch.from_numpy(S_final_val_batches[i]).to(self.device)
            s_next = torch.from_numpy(S_next_val_batches[i]).to(self.device)
            a_next = torch.from_numpy(A_next_val_batches[i]).to(self.device)
            loss, _, _, _ = self.model(s, a, s_final, s_next, a_next)
            val_losses.append(loss.item())
        val_loss = sum(val_losses)/len(val_losses)

        
        print("EPOCH %d -- Train Loss: %.5f  Validation Loss: %.5f" % (epoch, avg_loss, val_loss))
        return avg_loss, avg_embedding_loss, avg_final_state_loss, avg_next_state_loss, val_loss
        # return 0, 0, 0, 0, 0
    
    def train(self, train, val, epochs=10, batch_size=64, shuffle=True, verbose=False):
        S_train, A_train, S_final_train, S_next_train, A_next_train = train["input_states"], train["input_actions"], train["intermediate_states"], train["output_states"], train["output_actions"]
        S_val, A_val, S_final_val, S_next_val, A_next_val = val["input_states"], val["input_actions"], val["intermediate_states"], val["output_states"], val["output_actions"]
        
        losses = []
        embedding_losses = []
        final_state_losses = []
        next_state_losses = []
        val_losses = []
        lengths = []

        losses_dict = {}
        embedding_losses_dict = {}
        final_state_losses_dict = {}
        next_state_losses_dict = {}
        val_losses_dict = {}

        t = 0

        for epoch in range(epochs):
            # ---- CHOOSE LENGTH OF SEQUENCE ----
            # if len(lengths) == len(S_train.keys()):
            #     lengths = []
            # t = random.choice(list(S_train.keys()))
            # while t in lengths:
            #     t = random.choice(list(S_train.keys()))
            # lengths.append(t)
            
            # if t==98:
            #     t = 0
            # t += 1
            
            t = 10
            # ---- CHOOSE LENGTH OF SEQUENCE ----

            if shuffle:
                p_train = np.random.permutation(S_train[t].shape[0])
                p_val = np.random.permutation(S_val[t].shape[0])
                S_train_, A_train_, S_final_train_, S_next_train_, A_next_train_ = S_train[t][p_train], A_train[t][p_train], S_final_train[t][p_train], S_next_train[t][p_train], A_next_train[t][p_train]
                S_val_, A_val_, S_final_val_, S_next_val_, A_next_val_ = S_val[t][p_val], A_val[t][p_val], S_final_val[t][p_val], S_next_val[t][p_val], A_next_val[t][p_val]
            else:
                S_train_, A_train_, S_final_train_, S_next_train_, A_next_train_ = S_train[t], A_train[t], S_final_train[t], S_next_train[t], A_next_train[t]
                S_val_, A_val_, S_final_val_, S_next_val_, A_next_val_ = S_val[t], A_val[t], S_final_val[t], S_next_val[t], A_next_val[t]

            print("t: %d " % (t), end="")
            avg_loss, avg_embedding_loss, avg_final_state_loss, avg_next_state_loss, val_loss = self.train_epoch(train=(S_train_, A_train_, S_final_train_, S_next_train_, A_next_train_), 
                                                                                                                 val=(S_val_, A_val_, S_final_val_, S_next_val_, A_next_val_), 
                                                                                                                 epoch=epoch, batch_size=batch_size, verbose=verbose)
            if t in embedding_losses_dict:
                embedding_losses_dict[t].append(avg_embedding_loss)
                final_state_losses_dict[t].append(avg_final_state_loss)
                next_state_losses_dict[t].append(avg_next_state_loss)
                losses_dict[t].append(avg_loss)
                val_losses_dict[t].append(val_loss)
            else:
                embedding_losses_dict[t] = [avg_embedding_loss]
                final_state_losses_dict[t] = [avg_final_state_loss]
                next_state_losses_dict[t] = [avg_next_state_loss]
                losses_dict[t] = [avg_loss]
                val_losses_dict[t] = [val_loss]
            
            embedding_losses.append(avg_embedding_loss)
            final_state_losses.append(avg_final_state_loss)
            next_state_losses.append(avg_next_state_loss)
            losses.append(avg_loss)
            val_losses.append(val_loss)

        return losses_dict, embedding_losses_dict, final_state_losses_dict, next_state_losses_dict, val_losses_dict
        # return losses, embedding_losses, final_state_losses, next_state_losses, val_losses
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        


