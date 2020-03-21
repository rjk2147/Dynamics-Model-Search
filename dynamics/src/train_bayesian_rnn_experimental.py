import numpy as np
import os
import torch
import time

import pyro
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO

from bayesian_rnn_experimental import BayesianSequenceModel
from utils.visualize import *
import wandb
import warnings
from utils.tools import *

class Trainer:
    def __init__(self, vrnn, learning_rate, device, wandb):
        self.device = device
        self.wandb = wandb

        self.vrnn = vrnn.to(device)
        self.adam = optim.Adam({"lr": learning_rate})
        self.elbo = Trace_ELBO()
        self.svi = SVI(vrnn.model, vrnn.guide, self.adam, loss=self.elbo)

        self.wandb.watch(self.vrnn, log="all")     # track network topology

    def train(self, epochs, batch_size, train, val):
        pyro.poutine.util.enable_validation(True)
        # pyro.clear_param_store()
        X, A, Y = train
        X_val, A_val, Y_val = val
        N = X.shape[0]

        t0 = time.time()
        for i in range(1, epochs+1):
            elbo_loss = self.svi.step(X=X, 
                            A=A, 
                            Y=Y, 
                            batch_size=batch_size)

            train_batch_mae_loss, train_batch_Z, train_batch_Y_hat, train_batch_Y = self.vrnn.loss(X=X, A=A, Y=Y, batch_size=batch_size)
            val_batch_mae_loss, val_batch_Z, val_batch_Y_hat, val_batch_Y = self.vrnn.loss(X=X_val, A=A_val, Y=Y_val, batch_size=batch_size)
            print('i={}, epochs={:.2f}, elapsed={:.2f}, elbo={:.2f} train_loss={:.2f} val_loss={:.2f}'.format(i, (i * batch_size) / N, (time.time() - t0) / 3600, elbo_loss / N, train_batch_mae_loss, val_batch_mae_loss))
            
            if i % 10 == 0:                
                self.wandb.log({
                    "x-velocity"    :   get_velocity_curve(title="x-velocity", true=train_batch_Y[0, :, 0], pred=train_batch_Y_hat[0, :, 0]),
                    "y-velocity"    :   get_velocity_curve(title="y-velocity", true=train_batch_Y[0, :, 1], pred=train_batch_Y_hat[0, :, 1]),
                    "z-velocity"    :   get_velocity_curve(title="z-velocity", true=train_batch_Y[0, :, 2], pred=train_batch_Y_hat[0, :, 2])
                }, commit=False)
        
            self.wandb.log({
                "elbo"          :   elbo_loss / N,
                "train_mae"     :   train_batch_mae_loss,
                "train_val"     :   val_batch_mae_loss,
                "batch"         :   i,
                "epoch"         :   (i * batch_size) / N
                })
            self.vrnn.save_checkpoint()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    machine = "tesla"
    notes = "testing wandb for the first time!"
    name = "likelihood_std=1+time_standardization"

    wandb.init(project="bayesian_sequence_modelling", name=machine+"/"+name, tags=[machine], notes=notes, reinit=True)

    root = os.getcwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = root+"/../datasets/vanilla_rnn"
    X_train, A_train, Y_train = np.load(dataset+"/train/states.npy"), np.load(dataset+"/train/actions.npy"), np.load(dataset+"/train/next_states.npy")
    X_val, A_val, Y_val = np.load(dataset+"/val/states.npy"), np.load(dataset+"/val/actions.npy"), np.load(dataset+"/val/next_states.npy")
    X_test, A_test, Y_test = np.load(dataset+"/test/states.npy"), np.load(dataset+"/test/actions.npy"), np.load(dataset+"/test/next_states.npy")

    X_train, Y_train = standardize_across_time(X_train), standardize_across_time(Y_train) 
    X_val, Y_val = standardize_across_time(X_val), standardize_across_time(Y_val) 
    X_test, Y_test = standardize_across_time(X_test), standardize_across_time(Y_test) 

    config = {
        "dataset"           :   dataset,
        "state_size"        :   X_train.shape[-1],
        "action_size"       :   A_train.shape[-1],
        "z_size"            :   32,
        "hidden_state_size" :   256,
        "likelihood_std"    :   1,
        "epochs"            :   5000,
        "batch_size"        :   1600,
        "learning_rate"     :   1.0e-3,
        "device"            :   device,
        "preprocessing"     :   "standardization across time"
    }

    vrnn = BayesianSequenceModel(state_size=config["state_size"], 
                                 action_size=config["action_size"], 
                                 z_size=config["z_size"], 
                                 hidden_state_size=config["hidden_state_size"], 
                                 likelihood_std=config["likelihood_std"],
                                 device=device,
                                 path=os.path.join(wandb.run.dir, machine+"_"+name+"_bayesian_rnn_time_standardization_model.pth"))
    # vrnn.load_checkpoint()
    trainer = Trainer(vrnn=vrnn, learning_rate=config["learning_rate"], device=device, wandb=wandb)

    trainer.train(epochs=config["epochs"], 
                  batch_size=config["batch_size"], 
                  train=(torch.from_numpy(X_train[:]).to(device),torch.from_numpy(A_train[:]).to(device),torch.from_numpy(Y_train[:]).to(device)),
                  val=(torch.from_numpy(X_val[:]).to(device),torch.from_numpy(A_val[:]).to(device),torch.from_numpy(Y_val[:]).to(device))
                  )

    X_example = torch.from_numpy(X_train[0:5]).to(device)
    A_example = torch.from_numpy(A_train[0:5]).to(device)
    Y_example = torch.from_numpy(Y_train[0:5]).to(device)


