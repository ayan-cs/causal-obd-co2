import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _to_np(x):
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def rmse(pred, target):
    """Root mean squared error. pred, target: same shape, any dimensionality."""
    pred = _to_np(pred).astype(np.float64)
    target = _to_np(target).astype(np.float64)
    return float(np.sqrt(np.mean((pred - target) ** 2)))

def mae(pred, target):
    """Mean absolute error."""
    pred = _to_np(pred).astype(np.float64)
    target = _to_np(target).astype(np.float64)
    return float(np.mean(np.abs(pred - target)))

def _flatten_samples(x):
    """(N, T, M) or (N, D) -> (N, D). Flattens the time and feature axes."""
    x = _to_np(x).astype(np.float64)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    return x.reshape(x.shape[0], -1)

def _pairwise_sq_dists(a, b):
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    return np.maximum(aa + bb - 2.0 * a @ b.T, 0.0)

def rbf_mmd(real, generated, bandwidths=None):
    X = _flatten_samples(real)
    Y = _flatten_samples(generated)

    if bandwidths is None:
        d = _pairwise_sq_dists(X, X)
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        med = med if med > 0 else 1.0
        bandwidths = [0.25 * med, 0.5 * med, med, 2.0 * med, 4.0 * med]

    Kxx = _pairwise_sq_dists(X, X)
    Kyy = _pairwise_sq_dists(Y, Y)
    Kxy = _pairwise_sq_dists(X, Y)

    m, n = X.shape[0], Y.shape[0]
    mmd = 0.0
    for s in bandwidths:
        g = 1.0 / (2.0 * s + 1e-12)
        kxx = np.exp(-g * Kxx)
        kyy = np.exp(-g * Kyy)
        kxy = np.exp(-g * Kxy)
        # unbiased: drop the diagonal self-similarity terms
        np.fill_diagonal(kxx, 0.0)
        np.fill_diagonal(kyy, 0.0)
        term_xx = kxx.sum() / (m * (m - 1) + 1e-12)
        term_yy = kyy.sum() / (n * (n - 1) + 1e-12)
        term_xy = kxy.mean()
        mmd += term_xx + term_yy - 2.0 * term_xy
    return float(mmd / len(bandwidths))

def crps_ensemble(samples, target):
    samples = _to_np(samples).astype(np.float64)
    target = _to_np(target).astype(np.float64)
    S = samples.shape[0]

    # term 1: mean over ensemble of |X - y|
    abs_err = np.abs(samples - target[None, ...]).mean(axis=0)

    # term 2: 0.5 * mean over ensemble pairs of |X - X'|
    # computed without forming the full S*S tensor in memory
    pair_acc = np.zeros_like(abs_err)
    for i in range(S):
        pair_acc += np.abs(samples[i][None, ...] - samples).sum(axis=0)
    pair_term = 0.5 * pair_acc / (S * S)

    return float(np.mean(abs_err - pair_term))

def prediction_interval(samples, lower=0.05, upper=0.95):
    samples = _to_np(samples).astype(np.float64)
    lo = np.quantile(samples, lower, axis=0)
    hi = np.quantile(samples, upper, axis=0)
    return lo, hi

def picp(samples, target, lower=0.05, upper=0.95):
    target = _to_np(target).astype(np.float64)
    lo, hi = prediction_interval(samples, lower, upper)
    inside = (target >= lo) & (target <= hi)
    return float(np.mean(inside))

def pinaw(samples, target, lower=0.05, upper=0.95):
    target = _to_np(target).astype(np.float64)
    lo, hi = prediction_interval(samples, lower, upper)
    width = np.mean(hi - lo)
    rng = float(np.max(target) - np.min(target))
    rng = rng if rng > 1e-8 else 1.0
    return float(width / rng)

def calibration_curve(samples, target, levels=None):
    if levels is None:
        levels = np.linspace(0.1, 0.9, 9)
    nominal, empirical = [], []
    for p in levels:
        lower = (1.0 - p) / 2.0
        upper = 1.0 - lower
        nominal.append(p)
        empirical.append(picp(samples, target, lower, upper))
    return np.array(nominal), np.array(empirical)

def auroc(scores, gt_matrix, exclude_diagonal=True):
    from sklearn.metrics import roc_auc_score
    s = _to_np(scores).astype(np.float64)
    g = _to_np(gt_matrix).astype(np.int64)
    if exclude_diagonal:
        mask = ~np.eye(s.shape[0], dtype=bool)
        s, g = s[mask], g[mask]
    else:
        s, g = s.ravel(), g.ravel()
    if g.min() == g.max():
        return float("nan")  # degenerate, undefined AUROC
    return float(roc_auc_score(g, s))

if __name__ == "__main__":
    # quick self-test on synthetic numbers
    rng = np.random.default_rng(0)
    real = rng.normal(size=(200, 15, 6))
    gen = real + rng.normal(scale=0.1, size=real.shape)
    ens = real[None] + rng.normal(scale=0.2, size=(50,) + real.shape)
    print("rmse  ", rmse(gen, real))
    print("mae   ", mae(gen, real))
    print("mmd   ", rbf_mmd(real, gen))
    print("crps  ", crps_ensemble(ens, real))
    print("picp90", picp(ens, real, 0.05, 0.95))
    print("pinaw ", pinaw(ens, real, 0.05, 0.95))
    nom, emp = calibration_curve(ens, real)
    print("calib ", np.round(emp, 3))