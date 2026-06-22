import numpy as np
import torch
from metrics import (rmse, mae, rbf_mmd, crps_ensemble, picp, pinaw, calibration_curve)

@torch.no_grad()
def _posterior(clstm, X_left):
    h0 = torch.zeros(1, X_left.shape[0], clstm.hidden_size, device=X_left.device)
    _, h = clstm.encoder(X_left, h0)
    return clstm.fc_mu(h), clstm.fc_logvar(h)

@torch.no_grad()
def _rollout(clstm, X_left, seed_hidden, future):
    hidden_list = [seed_hidden for _ in range(clstm.num_series)]
    preds = []
    for _ in range(future):
        hidden_curr, X_t = [], None
        for d, head in enumerate(clstm.decoders):
            op, h_t = head(X_left[:, -1:, :], hidden_list[d], causal_graph=clstm.causal_graph[:, d])
            X_t = op if d == 0 else torch.cat((X_t, op), dim=-1)
            hidden_curr.append(h_t)
        hidden_list = hidden_curr
        preds.append(X_t)
    return torch.stack(preds, dim=1)

@torch.no_grad()
def _error_term(clstm, errorc, X_left, future, scale=0.01):
    err = clstm(X_left, None, mode='inference', phase=1, future=future)
    err_pred = errorc(err, mode='inference', future=future)
    return scale * err_pred

@torch.no_grad()
def sample_forecast(clstm, errorc, X_left, future, n_samples=100, sample=True, use_error=True, error_scale=0.01):
    """Returns an ensemble of forecasts, shape (n_samples, B, future, M)."""
    clstm.eval()
    if errorc is not None:
        errorc.eval()
    X_left = X_left.cuda()
    mu, logvar = _posterior(clstm, X_left)
    std = torch.exp(0.5 * logvar)
    err_term = _error_term(clstm, errorc, X_left, future, error_scale) \
        if (use_error and errorc is not None) else None
    out = []
    for _ in range(n_samples):
        z = mu + std * torch.randn_like(std) if sample else mu
        yhat = _rollout(clstm, X_left, z, future)
        if err_term is not None:
            yhat = yhat + err_term[:, :future, :]
        out.append(yhat.detach().cpu())
    return torch.stack(out, dim=0).numpy()

def evaluate_probabilistic(clstm, errorc, dl, n_samples=100, use_error=True, levels=None, mmd_cap=1000):
    """Full probabilistic evaluation over a dataloader. Ensemble mean is the
    point forecast; CRPS/PICP/PINAW/calibration use the full ensemble."""
    means, targets, ens_all, draws = [], [], [], []
    for X_l, X_r in dl:
        future = X_r.shape[1]
        ens = sample_forecast(clstm, errorc, X_l, future, n_samples=n_samples, sample=True, use_error=use_error)
        means.append(ens.mean(axis=0))
        targets.append(X_r.numpy())
        ens_all.append(ens)
        draws.append(ens[0])

    mean = np.concatenate(means, axis=0)
    target = np.concatenate(targets, axis=0)
    ens = np.concatenate(ens_all, axis=1)
    draw = np.concatenate(draws, axis=0)

    idx = np.arange(target.shape[0])
    if len(idx) > mmd_cap:
        idx = np.random.default_rng(0).choice(len(idx), mmd_cap, replace=False)

    nominal, empirical = calibration_curve(ens[:, idx], target[idx], levels)
    return {
        'rmse': rmse(mean, target),
        'mae': mae(mean, target),
        'mmd': rbf_mmd(target[idx], draw[idx]),
        'crps': crps_ensemble(ens[:, idx], target[idx]),
        'picp90': picp(ens[:, idx], target[idx], 0.05, 0.95),
        'pinaw90': pinaw(ens[:, idx], target[idx], 0.05, 0.95),
        'calibration': {'nominal': nominal.tolist(), 'empirical': empirical.tolist()},
        'n_windows': int(target.shape[0]),
    }