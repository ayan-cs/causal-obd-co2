import os, copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.integrate import odeint

def generateHenon(n_steps=10000, n_dim=10):
    X = np.random.rand(n_steps + 1, n_dim)
    a = 1.4
    b = 0.3
    for i in range(n_steps):
        x_next = np.zeros_like(X[i])
        for j in range(len(x_next)):
            if j != 0:
                x_next[j] = a - (b * X[i][j-1] + (1 - b) * X[i][j])**2 + b * X[i-1][j]
            else:
                x_next[j] = a - X[i][j] ** 2 + b * X[i-1][j]
        X[i+1] = x_next

    parent = os.path.abspath('')
    filename = f"henon_{n_steps}_{n_dim}_TL.npy"
    np.save(os.path.join(parent, 'datasets', filename), X.T)
    return X

def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta

def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)

    # Generate data.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += + errors[:, t-1]

    return X.T[burn_in:], beta, GC

def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F
    return dxdt

def generateLorenz96(n_dim, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000, seed=42):
    if seed is not None:
        np.random.seed(seed)

    parent = os.path.abspath('')

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=n_dim)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, n_dim))

    filename = f"lorenz_{T}_{n_dim}.npy"
    np.save(os.path.join(parent, 'datasets', filename), X.T)
    return X[burn_in:]
    
def createSplit(X_l, X_r, test_size, shuffle=True):
    X_l = np.array(X_l)
    X_r = np.array(X_r)
    num_samples = X_l.shape[0]
    test_size = int(num_samples * test_size)
    random_idx = np.random.permutation(num_samples)
    X_train_left = X_l[random_idx[test_size:]]
    X_train_right = X_r[random_idx[test_size:]]
    X_val_left = X_l[random_idx[:test_size]]
    X_val_right = X_r[random_idx[:test_size]]
    return X_train_left.tolist(), X_train_right.tolist(), X_val_left.tolist(), X_val_right.tolist()

def getCausalMatrix(n_dim, data='henon'):
    if data=='henon':
        GC = np.zeros([n_dim, n_dim])
        for i in range(n_dim):
            GC[i,i] = 1
            if i!=0:
                GC[i,i-1] = 1
        return GC

    if data=='lorenz':
        GC = np.zeros((n_dim, n_dim), dtype=int)
        for i in range(n_dim):
            GC[i, i] = 1
            GC[i, (i + 1) % n_dim] = 1
            GC[i, (i - 1) % n_dim] = 1
            GC[i, (i - 2) % n_dim] = 1
        return GC

def preprocessSyn(config, data=None):

    def createChunks(data, context):
        if context > data.shape[0]:
            context = data.shape[0]
        X_left = []
        X_right = []
        context_l = int(context/2)
        for i in range(len(data) - context + 1):
            X_left.append(data[i : i+context_l].tolist())
            X_right.append(data[i+context_l : i+context].tolist())
        return X_left, X_right

    if data is not None:
        X = data
        context = config['chunksize']
        X_left, X_right = createChunks(X, context)
        X_train_left, X_train_right, X_test_left, X_test_right = createSplit(X_left, X_right, test_size=config['test_split'], shuffle=True)
        return X_train_left, X_train_right, X_test_left, X_test_right
    
    else:
        parent = os.path.abspath('')
        dataset = os.path.join(parent, 'datasets', f"{config['dataset']}.npy")
        context = config['chunksize']
        X = np.load(dataset).T

        X_left, X_right = createChunks(X, context)
        X_train_left, X_train_right, X_test_left, X_test_right = createSplit(X_left, X_right, test_size=config['test_split'], shuffle=True)
        return X_train_left, X_train_right, X_test_left, X_test_right

def preprocessOBD(data, context):
    pass