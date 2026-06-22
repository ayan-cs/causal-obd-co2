import os
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import getCausalMatrix

def load_series(path, artifact):
    parts = artifact.split('_')
    system = parts[0]
    n_dim = int(parts[-1]) if parts[-1].isdigit() else None
    X = np.asarray(np.load(path), dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2D series, got shape {X.shape}")
    # orient to (time, features)
    if n_dim is not None and X.shape[1] != n_dim and X.shape[0] == n_dim:
        X = X.T
    elif n_dim is None and X.shape[0] < X.shape[1]:
        X = X.T  # heuristic: time is the longer axis
    return X, system, X.shape[1]
 
 
def split_series(X, val_frac, test_frac):
    T = len(X)
    n_test = int(T * test_frac)
    n_val = int(T * val_frac)
    n_train = T - n_val - n_test
    return X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:]

def createChunks(data, context):
    data = np.asarray(data, dtype=np.float32)
    context_l = context // 2
    Xl, Xr = [], []
    for i in range(len(data) - context + 1):
        Xl.append(data[i:i + context_l])
        Xr.append(data[i + context_l:i + context])
    return np.asarray(Xl, dtype=np.float32), np.asarray(Xr, dtype=np.float32)

def _scaler(kind):
    if kind == 'raw':
        return None
    if kind == 'minmax':
        return MinMaxScaler(feature_range=(0, 1))
    if kind == 'std':
        return StandardScaler()
    raise ValueError(kind)
 
# --- standard variant ---
def make_variant(tr, va, te, kind, context):
    scaler = _scaler(kind)
    if scaler is not None:
        scaler.fit(tr)
        tr, va, te = scaler.transform(tr), scaler.transform(va), scaler.transform(te)
    for name, seg in (('train', tr), ('val', va), ('test', te)):
        if len(seg) < context:
            raise ValueError(f"{kind}: {name} segment ({len(seg)}) shorter than context ({context}); reduce context or fractions.")
    Xtl, Xtr = createChunks(tr, context)
    Xvl, Xvr = createChunks(va, context)
    Xtel, Xter = createChunks(te, context)
    return {
        'X_train_left': Xtl, 'X_train_right': Xtr,
        'X_val_left': Xvl,   'X_val_right': Xvr,
        'X_test_left': Xtel, 'X_test_right': Xter,
    }

# --- OBD variant ---
def make_obd_variant(kind, v1_segments, v2_segments, context):
    scaler = _scaler(kind)
    if scaler is not None:
        scaler.fit(np.concatenate([tr for _, tr in v1_segments['train']], axis=0))
    transform = (lambda a: a) if scaler is None else scaler.transform

    def chunk_pool(seg_list):
        Ls, Rs = [], []
        for pid, seg in seg_list:
            if len(seg) < context:
                print(f'  [{kind}] skip driver {pid} (len {len(seg)} < context {context})')
                continue
            l, r = createChunks(transform(seg), context)
            Ls.append(l); Rs.append(r)
        return np.concatenate(Ls, 0), np.concatenate(Rs, 0)

    arrays = {}
    for split in ['train', 'val', 'test']:
        arrays[f'X_{split}_left'], arrays[f'X_{split}_right'] = chunk_pool(v1_segments[split])
    arrays['X_ood_left'], arrays['X_ood_right'] = chunk_pool(v2_segments)
    return arrays, scaler

def preprocess_all(datasets_dir, artifact, context, val_frac=0.15, test_frac=0.15, out_subdir=None):
    path = os.path.join(datasets_dir, f"{artifact}.npy" if not artifact.endswith('.npy') else artifact)
    artifact = artifact[:-4] if artifact.endswith('.npy') else artifact
    X, system, n_dim = load_series(path, artifact)
    tr, va, te = split_series(X, val_frac, test_frac)
 
    try:
        gt = getCausalMatrix(n_dim, data=system)
    except Exception:
        gt = None
 
    out_dir = os.path.join(datasets_dir, out_subdir or f"{artifact}_ctx{context}")
    os.makedirs(out_dir, exist_ok=True)
 
    written = []
    for kind in ('raw', 'minmax', 'std'):
        arrays = make_variant(tr, va, te, kind, context)
        meta = {
            'artifact': artifact, 'system': system, 'n_dim': int(n_dim),
            'context': int(context), 'scaling': kind,
            'val_frac': val_frac, 'test_frac': test_frac,
            'split': 'chronological_series_first', 'scaler_fit_on': 'train',
            'shapes': {k: list(v.shape) for k, v in arrays.items()},
        }
        save_kwargs = dict(arrays)
        save_kwargs['gt_matrix'] = np.asarray(gt) if gt is not None else np.array([])
        save_kwargs['meta'] = np.array(json.dumps(meta))
        fname = os.path.join(out_dir, f"{artifact}_{kind}_ctx{context}.npz")
        np.savez_compressed(fname, **save_kwargs)
        written.append(fname)
        print(f"[{kind:8s}] train {arrays['X_train_left'].shape}  "
              f"val {arrays['X_val_left'].shape}  test {arrays['X_test_left'].shape}"
              f"  -> {os.path.basename(fname)}", flush=True)
    return written

# --- Standard loader ---
# def load_npz(path):
#     """Convenience loader. Returns (splits_dict, gt_matrix, meta_dict)."""
#     d = np.load(path, allow_pickle=True)
#     splits = {
#         'train': (d['X_train_left'], d['X_train_right']),
#         'val':   (d['X_val_left'],   d['X_val_right']),
#         'test':  (d['X_test_left'],  d['X_test_right']),
#     }
#     gt = d['gt_matrix']
#     gt = None if gt.size == 0 else gt
#     meta = json.loads(str(d['meta']))
#     return splits, gt, meta

# # --- OBD-specific loader ---
# def load_obd_npz(path):
#     d = np.load(path, allow_pickle=True)
#     splits = {
#         'train': (d['X_train_left'], d['X_train_right']),
#         'val':   (d['X_val_left'],   d['X_val_right']),
#         'test':  (d['X_test_left'],  d['X_test_right']),   # V1 in-distribution
#         'ood':   (d['X_ood_left'],   d['X_ood_right']),    # V2 cross-vehicle
#     }
#     return splits, list(d['feature_names']), json.loads(str(d['meta']))

def load_preprocessed(path):
    d = np.load(path, allow_pickle=True)
    files = set(d.files)
 
    def pair(split):
        return (np.asarray(d[f'X_{split}_left'], dtype=np.float32), np.asarray(d[f'X_{split}_right'], dtype=np.float32))
 
    splits = {'train': pair('train'), 'val': pair('val'), 'test': pair('test')}
    if 'X_ood_left' in files:
        splits['ood'] = pair('ood')
 
    gt = None
    if 'gt_matrix' in files:
        g = np.asarray(d['gt_matrix'])
        gt = None if g.size == 0 else g
    splits['gt_matrix'] = gt
 
    if 'feature_names' in files:
        splits['feature_names'] = [str(x) for x in d['feature_names']]
    else:
        M = splits['train'][0].shape[-1]
        splits['feature_names'] = [f"x{i}" for i in range(M)]
    return splits