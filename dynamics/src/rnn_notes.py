# D = dimension of input (number of features in a single sample)
# H = dimension of hidden layer (number of features in hidden state at a single timestep)
# L = number of LSTM layers
# N = batch size or number of samples
# K = num_layers (there is an initial hidden state for each layer of the LSTM)
# T = sequence length

import torch
rnn = torch.nn.LSTM(D = 10, H = 20, L = 2)                # lstm is a layer
input = torch.randn(T = 5, N = 3, D = 10)
h0 = torch.randn(K = 2, N = 3, H = 20)
c0 = torch.randn(K = 2, N = 3, H = 20)

output, (hn, cn) = rnn(input, (h0, c0))
# output of shape (T, N, H)
# hn of shape (K, N, H)
# cn of shape (K, N, H)