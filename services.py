import numpy as np
import os, copy, torch
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Optimizer
import torch.nn as nn

class ProximalGradientDescent(Optimizer):
    def __init__(self, params, lr=1e-3, lam=0.1):
        defaults = dict(lr=lr)
        super(ProximalGradientDescent, self).__init__(params, defaults)
        self.lam = lam
        
    def step(self, model):
        for group in self.param_groups: # self.param_groups = [{'params' : tensors[...], 'lr':0.001, 'other_arg':val}, {...}, ...]
            lr = group['lr']
            for param in group['params']:
                if param.grad is None:
                   continue
                # Gradient descent step
                param.data -= lr * param.grad
            
            if self.lam > 0:
                for d_head in model.decoders:
                    W = d_head.lstm.weight_ih_l0
                    norm = torch.norm(W, dim=0, keepdim=True)
                    W.data = ((W / torch.clamp(norm, min=(self.lam * lr))) * torch.clamp(norm - (lr * self.lam), min=0.0))
                    d_head.lstm.flatten_parameters()

class Criterion(nn.Module):
    def __init__(self, beta_mmd, lam_ridge, model='clstm'):
        super(Criterion, self).__init__()
        self.beta_mmd = beta_mmd
        self.lam_ridge = lam_ridge
        self.mse = nn.MSELoss()
        self.model = model
    
    def forward(self, pred, gt, mu, logvar, model):
        dim = gt.shape[-1]
        mse_loss = sum([self.mse(pred[:, :, d], gt[:, :, d]) for d in range(dim)])
        mmd_loss = (-0.5*(1 + logvar - mu**2 - torch.exp(logvar)).sum(dim=-1).sum(dim=0)).mean(dim=0)
        if self.model == 'clstm':
            ridge_loss = sum([self.__ridge_regularize(decoder_head) for decoder_head in model.decoders])
        else:
            ridge_loss = 0
        return mse_loss + self.beta_mmd*mmd_loss + ridge_loss, mmd_loss

    def __ridge_regularize(self, d_head):
        ridge = self.lam_ridge * (torch.sum(d_head.fc.weight ** 2) + torch.sum(d_head.lstm.weight_hh_l0 ** 2))
        return ridge

class EarlyStopper :
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