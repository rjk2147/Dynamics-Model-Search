import torch
import time
from dynamics_model_search.model_based.mpc import MPC, NullAgent
import numpy as np

if torch.cuda.is_available():
    devices = [torch.device("cuda:"+str(i)) for i in range(0, torch.cuda.device_count())]
else:
    devices = [torch.device('cpu')]

class CEM(MPC):
    def __init__(self, lookahead, dynamics_model, agent=None, N=64, Ne_ratio=0.25, epsilon=1e-3, maxits=32, initial_sd=0.1, with_hidden=False):
        MPC.__init__(self, lookahead, dynamics_model, agent)
        self.start = time.time()
        self.batch_size = 262144
        self.with_hidden = with_hidden
        self.epsilon = epsilon
        self.maxits = maxits
        self.N = N
        self.Ne = int(Ne_ratio*N)
        self.initial_sd = initial_sd
        self.discount = 0.95

    def best_move(self, obs, act=None):
        t = 0
        # Initialize parameters
        # While maxits not exceeded and not
        if not torch.is_tensor(obs[0]):
            obs = (torch.from_numpy(obs[0]).to(devices[0]), obs[1].to(devices[0]))
        if len(obs[0].shape) < 2:
            obs = (obs[0].unsqueeze(0), obs[1])
        X = (obs[0].repeat((self.N, 1)), obs[1].repeat((self.N, 1, 1)))
        mu = torch.zeros((1,self.act_dim)).to(X[0].device)
        if act is not None:
            mu = act
        if self.lookahead > 1:
            null_lookahead = torch.zeros(mu.shape).unsqueeze(0).repeat((self.lookahead-1, 1, 1)).to(mu.device)
            mu = torch.cat([mu.unsqueeze(0), null_lookahead])
        else:
            mu = mu.unsqueeze(0)
        sigma = torch.ones(mu.shape).to(mu.device)*self.initial_sd

        b = mu.shape[1]

        while t < self.maxits and (sigma > self.epsilon).any():
            # Obtain N samples from current sampling distribution
            mu = mu.repeat((1, self.N, 1))
            sigma = sigma.repeat((1, self.N, 1))
            A = (torch.randn_like(mu)*sigma + mu).clamp(-1, 1)

            # Evaluate objective function at sampled points
            S = []
            R = []
            in_s = X
            for i in range(self.lookahead):
                out_s = self.dynamics_model.step_parallel(obs_in=in_s, action_in=A[i], state=True, state_in=True)
                r = self.agent.value(in_s[0], A[i], out_s[0]).flatten()
                S.append(out_s)
                R.append(r)
                in_s = out_s

            for i in range(2, self.lookahead+1):
                R[-i] += R[-i+1]*self.discount

            # Splitting Rs and As into their initial groups
            R = [np.split(R[i], b, 0) for i in range(self.lookahead)]
            A = [torch.chunk(A[i], b, 0) for i in range(self.lookahead)]

            # Select top Ne actions based on their R score
            A = [[A[i][j][np.argsort(-R[i][j])[:self.Ne]] for j in range(b)] for i in range(self.lookahead)]

            # Update parameters of sampling distribution
            mu = torch.cat([torch.cat([torch.mean(A[i][j], 0).unsqueeze(0) for j in range(b)]).unsqueeze(0) for i in range(self.lookahead)])
            sigma = torch.cat([torch.cat([torch.std(A[i][j], 0).unsqueeze(0) for j in range(b)]).unsqueeze(0) for i in range(self.lookahead)])
            t = t + 1

        # Getting the expected Rs
        out_s = self.dynamics_model.step_parallel(obs_in=obs, action_in=mu[0], state=True, state_in=True)
        r = self.agent.value(obs[0], mu[0], out_s[0]).flatten()

        # Return mean of initial step 0 of the final sampling distribution as solution
        return mu[0], r

