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
        X_val, A_val, Y_val = val
        N = X.shape[0]

        t0 = time.time()
        for i in range(1, epochs+1):
            loss = self.vrnn(X=X, A=A, Y=Y)
            loss.backward()
            self.optim.step()
            print(loss)


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
        "batch_size"        :   800,
        "learning_rate"     :   1.0e-3,
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
                  train=(torch.from_numpy(X_train[:config["batch_size"]]).to(device),torch.from_numpy(A_train[:config["batch_size"]]).to(device),torch.from_numpy(Y_train[:config["batch_size"]]).to(device)),
                  val=(torch.from_numpy(X_val[:config["batch_size"]]).to(device),torch.from_numpy(A_val[:config["batch_size"]]).to(device),torch.from_numpy(Y_val[:config["batch_size"]]).to(device))
                  )

