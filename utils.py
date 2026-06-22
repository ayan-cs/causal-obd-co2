import numpy as np
import os, torch, random
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

def set_deterministic(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def epoch_time(start_time, end_time):
    elapsed = end_time - start_time
    h = int(elapsed / 3600)
    m = int((elapsed - h * 3600) / 60)
    s = elapsed - (m * 60 + h * 3600)
    return h, m, s

def getModelName(dataset, type, mode):
    now = str(datetime.now())
    date, clock = now.split()[0], now.split()[1]
    date = date.split('-')
    date.reverse()
    date = '-'.join(date)
    clock = clock.replace(':', '-')[:8]
    return f"{type}___{mode}___{dataset}___{date}_{clock}"

def getCausalMatrix(n_dim=None, data='henon'):
    if data == 'henon':
        assert n_dim is not None
        GC = np.zeros([n_dim, n_dim])
        for i in range(n_dim):
            GC[i, i] = 1
            if i != 0:
                GC[i, i - 1] = 1
        return GC
    if data == 'lorenz':
        assert n_dim is not None
        GC = np.zeros((n_dim, n_dim), dtype=int)
        for i in range(n_dim):
            GC[i, i] = 1
            GC[i, (i + 1) % n_dim] = 1
            GC[i, (i - 1) % n_dim] = 1
            GC[i, (i - 2) % n_dim] = 1
        return GC
    raise ValueError(f"No ground-truth graph for '{data}'")

def to_loader(split, batch_size, shuffle=False):
    Xl, Xr = split
    ds = TensorDataset(torch.FloatTensor(np.asarray(Xl, dtype=np.float32)), torch.FloatTensor(np.asarray(Xr, dtype=np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)