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

def standardize_across_samples(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    mu_repeated = repeat(mu, repeat=X.shape[0])
    mu_repeated = mu_repeated.reshape((mu_repeated.shape[1], mu_repeated.shape[0], mu_repeated.shape[2]))
    sigma_repeated = repeat(sigma, repeat=X.shape[0])
    sigma_repeated = sigma_repeated.reshape((sigma_repeated.shape[1], sigma_repeated.shape[0], sigma_repeated.shape[2]))
    return (X-mu_repeated)/sigma_repeated


def compute_position_from_velocity(V, dt=1):
    # V is an array of velocities, index=timestep
    origin = 0
    X = [origin]
    for t in range(V.shape[0]):
        X.append(X[-1]+V[t]*dt)
    X = np.array(X)
    return X