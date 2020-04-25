import numpy as np
import os
import torch
from torch import optim
import time

from variational_sequence_model_without_pyro import VRNN
from utils.visualize import *
from utils.tools import *
import wandb

class Trainer:
    def __init__(self, vrnn, learning_rate, device, wandb):
        self.device = device
        self.wandb = wandb

        self.vrnn = vrnn.to(device)
        self.optim = optim.Adam(self.vrnn.parameters(), lr=learning_rate)

        # self.wandb.watch(self.vrnn, log="all")     # track network topology


    def train(self, epochs, batch_size, train, val):
        X, A, Y = train
        X_train = torch.split(X, batch_size)
        A_train = torch.split(A, batch_size)
        Y_train = torch.split(Y, batch_size)
        print(X.shape)
        X_val, A_val, Y_val = val
        print(len(X_train))
        X_val = torch.split(X_val, batch_size)
        A_val = torch.split(A_val, batch_size)
        Y_val = torch.split(Y_val, batch_size)
        print(len(X_val))
        N = X.shape[0]

        t0 = time.time()
        for i in range(1, epochs+1):
            train_kl = 0.0
            train_mse = 0.0
            train_loss = 0.0
            val_kl = 0.0
            val_mse = 0.0
            val_loss = 0.0
            import sys
            start = time.time()
            for j in range(len(X_train)):
                kl, mse, loss = self.vrnn(X=X_train[j], A=A_train[j], Y=Y_train[j])
                train_kl += kl.item()
                train_mse += mse.item()
                train_loss += loss.item()
                loss.backward()
                self.optim.step()
                sys.stdout.write(str( round((100.0*j / (len(X_train)+len(X_val))), 2) )+'% done in '+str(round((time.time()-start), 3))+'s                                        \r')
            for j in range(len(X_val)):
                kl, mse, loss = self.vrnn(X=X_val[j], A=A_val[j], Y=Y_val[j])
                val_kl += kl.item()
                val_mse += mse.item()
                val_loss += loss.item()
                sys.stdout.write(str( round((100.0*(j+len(X_train)) / (len(X_train)+len(X_val))), 2) )+'% done in '+str(round((time.time()-start), 3))+'s                                        \r')
            train_kl /= len(X_train)
            train_mse /= len(X_train)
            train_loss /= len(X_train)

            val_kl /= len(X_train)
            val_mse /= len(X_val)
            val_loss /= len(X_val)
            print('Epoch '+str(i)+'/'+str(epochs+1)+'                                                                                                 ')
            print('Train: ')
            print([train_loss, train_kl, train_mse])
            print('Valid: ')
            print([val_loss, val_kl, val_mse])
            print('')


if __name__ == "__main__":
    machine = "honshu"
    notes = "attempting {a0...aT} input as opposed to {(s0,a0)...(sT,aT)} input, more expressive network"
    name = "variational_sequence_model"

    # wandb.init(project="bayesian_sequence_modelling", name=machine+"/"+name, tags=[machine], notes=notes, reinit=True)

    root = os.getcwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = root+"/../datasets/vanilla_rnn"
    X_train, A_train, Y_train = np.load(dataset+"/train/states.npy"), np.load(dataset+"/train/actions.npy"), np.load(dataset+"/train/next_states.npy")
    X_val, A_val, Y_val = np.load(dataset+"/val/states.npy"), np.load(dataset+"/val/actions.npy"), np.load(dataset+"/val/next_states.npy")
    X_test, A_test, Y_test = np.load(dataset+"/test/states.npy"), np.load(dataset+"/test/actions.npy"), np.load(dataset+"/test/next_states.npy")

    # X_train, Y_train = standardize_across_time(X_train), standardize_across_time(Y_train) 
    # X_val, Y_val = standardize_across_time(X_val), standardize_across_time(Y_val) 
    # X_test, Y_test = standardize_across_time(X_test), standardize_across_time(Y_test) 

    # X_train, Y_train = standardize_across_samples(X_train), standardize_across_samples(Y_train) 
    # X_val, Y_val = standardize_across_samples(X_val), standardize_across_samples(Y_val) 
    # X_test, Y_test = standardize_across_samples(X_test), standardize_across_samples(Y_test) 

    config = {
        "dataset"           :   dataset,
        "state_size"        :   X_train.shape[-1],
        "action_size"       :   A_train.shape[-1],
        "z_size"            :   32,
        "hidden_state_size" :   64,
        "epochs"            :   1000,
        "batch_size"        :   512,
        "learning_rate"     :   1.0e-5,
        "device"            :   device,
        "preprocessing"     :   "none"
    }

    vrnn = VRNN(state_size=config["state_size"], 
                action_size=config["action_size"], 
                z_size=config["z_size"], 
                hidden_state_size=config["hidden_state_size"], 
                device=device,
                path=None)
    # vrnn.load_checkpoint()
    trainer = Trainer(vrnn=vrnn, learning_rate=config["learning_rate"], device=device, wandb=wandb)

    trainer.train(epochs=config["epochs"], 
                  batch_size=config["batch_size"], 
                  train=(torch.from_numpy(X_train).to(device),torch.from_numpy(A_train).to(device),torch.from_numpy(Y_train).to(device)),
                  val=(torch.from_numpy(X_val).to(device),torch.from_numpy(A_val).to(device),torch.from_numpy(Y_val).to(device))
                  )

