import torch
from torch import nn
from torch.autograd import Variable

class VRNN(nn.Module):
    def __init__(self, state_size=21, action_size=8, z_size=32, hidden_state_size=64, device=None, path=None):
        super().__init__()
        self.device = device
        self.path = path

        self.state_size = state_size
        self.action_size = action_size
        self.z_size = z_size
        self.hidden_state_size = hidden_state_size

        # feature-extracting transformations
        self.transform_a = nn.Sequential(
            nn.Linear(action_size, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.ReLU()
        ).to(device)
        self.transform_s = nn.Sequential(
            nn.Linear(state_size, hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size, hidden_state_size),
            nn.ReLU()
        ).to(device)
        self.transform_z = nn.Sequential(
            nn.Linear(z_size, hidden_state_size),
            nn.ReLU()
        ).to(device)

        # prior
        self.prior = nn.Sequential(
            nn.Linear(hidden_state_size+hidden_state_size, hidden_state_size),
            nn.ReLU()
        ).to(device)
        self.prior_loc = nn.Linear(hidden_state_size, z_size).to(device)
        self.prior_scale = nn.Sequential(
            nn.Linear(hidden_state_size, z_size),
            nn.Softplus()
        ).to(device)

        # encoder 
        self.encoder = nn.Sequential(
            nn.Linear(hidden_state_size+hidden_state_size+hidden_state_size, hidden_state_size+hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size+hidden_state_size, hidden_state_size),
            nn.ReLU()
        ).to(device)
        self.encoder_loc = nn.Linear(hidden_state_size, z_size).to(device)
        self.encoder_scale = nn.Sequential(
            nn.Linear(hidden_state_size, z_size),
            nn.Softplus()
        ).to(device)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_state_size+hidden_state_size+hidden_state_size, hidden_state_size+hidden_state_size),
            nn.ReLU(),
            nn.Linear(hidden_state_size+hidden_state_size, hidden_state_size),
            nn.ReLU()
        ).to(device)
        self.decoder_loc = nn.Sequential(
            nn.Linear(hidden_state_size, state_size),
            nn.Tanh()
        ).to(device)
        self.decoder_scale = nn.Sequential(
            nn.Linear(hidden_state_size, state_size),
            nn.Sigmoid()
        ).to(device)

        # recurrence
        self.rnn = nn.GRUCell(hidden_state_size+hidden_state_size, hidden_state_size).to(device)
        self.hidden_state_init = nn.Parameter(torch.zeros(1, hidden_state_size)).to(device)

    def forward(self, X, A, Y):
        encoder_locs, encoder_scales = [], []
        decoder_locs, decoder_scales = [], []

        kld_loss = 0
        mle_loss = 0

        h_t = self.hidden_state_init.expand(A.shape[0], self.hidden_state_size).to(self.device)

        for t in range(A.shape[1]): 
            transform_s_t = self.transform_s(X[:, 0, :])
            transform_a_t = self.transform_a(A[:, t, :])
            
            # encoder
            encoder_t = self.encoder(torch.cat([transform_s_t, transform_a_t, h_t], dim=-1))
            encoder_loc_t = self.encoder_loc(encoder_t)
            encoder_scale_t = self.encoder_scale(encoder_t)

            # prior
            prior_t = self.prior(torch.cat([transform_s_t, h_t], dim=-1))
            prior_loc_t = self.prior_loc(prior_t)
            prior_scale_t = self.prior_scale(prior_t)

            # sampling and reparameterization
            z_t = self.reparameterized_sample(loc=encoder_loc_t, scale=encoder_scale_t)
            transform_z_t = self.transform_z(z_t)


            # decoder
            decoder_t = self.decoder(torch.cat([transform_s_t, transform_z_t, h_t], dim=-1))
            decoder_loc_t = self.decoder_loc(decoder_t)
            decoder_scale_t = self.decoder_scale(decoder_t)
            y_t = self.reparameterized_sample(loc=decoder_loc_t, scale=decoder_scale_t)

            # recurrence
            h_t = self.rnn(torch.cat([transform_a_t, transform_z_t], dim=-1), h_t)
            
            kld_loss += self.kld_gaussian(loc_0=encoder_loc_t, scale_0=encoder_scale_t, loc_1=prior_loc_t, scale_1=prior_scale_t)
            mle_loss += self.mse(true=Y[:, t, :], pred=y_t)

        loss = kld_loss + mle_loss
        return loss
    
    def reparameterized_sample(self, loc, scale):
        eps = torch.FloatTensor(scale.size()).normal_().to(self.device)
        eps = Variable(eps)
        return loc + scale*eps
    
    def kld_gaussian(self, loc_0, scale_0, loc_1, scale_1):
        kld_element =  (2 * torch.log(scale_1) - 2 * torch.log(scale_0) + 
            (scale_0.pow(2) + (loc_0 - loc_1).pow(2)) /
            scale_1.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)
    
    def mse(self, true, pred):
        return torch.mean((true-pred)**2)
    

