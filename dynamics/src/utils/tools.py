import numpy as np

def repeat(X, repeat=100):
    Y = []
    for i in range(X.shape[0]):
        y = []
        for j in range(repeat):
            y.append(X[i])
        y = np.array(y)
        Y.append(y)
    Y = np.array(Y)
    return Y

def standardize_across_time(X):
    mu = np.mean(X, axis=1)
    sigma = np.std(X, axis=1)
    mu_repeated = repeat(mu, repeat=100)
    sigma_repeated = repeat(sigma, repeat=100)
    return (X-mu_repeated)/sigma_repeated