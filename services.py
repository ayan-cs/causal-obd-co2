import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

class ProximalGradientDescent(Optimizer):
    def __init__(self, params, lr=1e-3, lam=0.1):
        super().__init__(params, dict(lr=lr))
        self.lam = lam

    def step(self, model):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data -= lr * p.grad
            if self.lam > 0:
                for head in model.decoders:
                    W = head.lstm.weight_ih_l0
                    norm = torch.norm(W, dim=0, keepdim=True)
                    W.data = ((W / torch.clamp(norm, min=(self.lam * lr))) * torch.clamp(norm - (lr * self.lam), min=0.0))
                    head.lstm.flatten_parameters()

class Criterion(nn.Module):
    """Returns (total_loss, kl). total = MSE + beta_kl * KL + ridge."""
    def __init__(self, beta_kl, lam_ridge, model='clstm'):
        super().__init__()
        self.beta_kl = beta_kl
        self.lam_ridge = lam_ridge
        self.mse = nn.MSELoss()
        self.model = model

    def forward(self, pred, gt, mu, logvar, model):
        dim = gt.shape[-1]
        mse_loss = sum(self.mse(pred[:, :, d], gt[:, :, d]) for d in range(dim))
        kl = (-0.5 * (1 + logvar - mu ** 2 - torch.exp(logvar)).sum(dim=-1).sum(dim=0)).mean(dim=0)
        if self.model == 'clstm':
            ridge = sum(self._ridge(h) for h in model.decoders)
        else:
            ridge = 0.0
        return mse_loss + self.beta_kl * kl + ridge, kl

    def _ridge(self, head):
        return self.lam_ridge * (torch.sum(head.fc.weight ** 2) + torch.sum(head.lstm.weight_hh_l0 ** 2))


class EarlyStopper:
    def __init__(self, patience=100, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.inf

    def early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False