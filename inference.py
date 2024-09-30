import torch, os
import torch.nn as nn
import numpy as np
from models import CausalLSTM, ErrorCompensation
from services import Criterion

def inference(X, dim, config):
    parent = os.path.abspath('')

    clstm = CausalLSTM(num_features=dim, hidden_size=config['hidden_size'], causal_graph=np.ones([dim, dim])).cuda()
    clstm.load_state_dict(torch.load(os.path.join(parent, 'checkpoints', f"clstm_{config['dataset']}.pth"), weights_only=True))
    
    errorc = ErrorCompensation(num_features=dim, hidden_size=config['hidden_size']).cuda()
    errorc.load_state_dict(torch.load(os.path.join(parent, 'checkpoints', f"errorc_{config['dataset']}.pth"), weights_only=True))

    clstm.eval()
    errorc.eval()
    X = X.cuda()
    err = clstm(X, None, mode='inference', phase=1, future=config['future'])
    err_pred = errorc(err, mode='inference', future=config['future'])
    X_pred = clstm(X, None, error=err_pred, mode='inference', phase=2, future=config['future'])
    # X_pred = clstm(X, None, error=None, mode='inference', phase=2, future=config['future'])

    return X_pred.squeeze(0).detach().cpu()

def batchInference(dl, dim, config):
    parent = os.path.abspath('')
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    criterion = Criterion(config['beta_mmd'], config['lam_ridge'], model='clstm')

    clstm = CausalLSTM(num_features=dim, hidden_size=config['hidden_size'], causal_graph=np.ones([dim, dim])).cuda()
    clstm.load_state_dict(torch.load(os.path.join(parent, 'checkpoints', f"clstm_{config['dataset']}.pth"), weights_only=True))
    
    errorc = ErrorCompensation(num_features=dim, hidden_size=config['hidden_size']).cuda()
    errorc.load_state_dict(torch.load(os.path.join(parent, 'checkpoints', f"errorc_{config['dataset']}.pth"), weights_only=True))

    clstm.eval()
    errorc.eval()
    # eucl_dist = 0
    total_mse = 0
    total_mae = 0
    total_mmd = 0
    total_samples = 0
    with torch.no_grad():
        for X_l, X_r in dl:
            X_l = X_l.cuda()
            X_r = X_r.cuda()

            w_size = X_l.shape[1] # Number of past observations (window size) in one sample
            num_items = X_l.shape[0] # Number of values in a batch (A sample with shape (B, 20, 6) will have B*120 items)
            pred, mu, logvar = clstm(X_l, X_r)
            err = clstm(X_l, None, mode='inference', phase=1, future=w_size)
            err_pred = errorc(err, mode='inference', future=w_size)
            X_pred = clstm(X_l, None, error=err_pred, mode='inference', phase=2, future=w_size)
            loss = mse(X_pred, X_r)
            _, mmd = criterion(pred, X_r, mu, logvar, clstm)
            l1 = mae(X_pred, X_r)
            total_samples += num_items
            total_mse += loss.item() * num_items
            total_mmd += mmd.item() * num_items
            total_mae += l1.item() * num_items

        rmse = (total_mse/total_samples) ** 0.5
        avgmmd = (total_mmd/total_samples)
        avgmae = (total_mae/total_samples)
    
    return rmse, avgmmd, avgmae # avg_eucl_dist.detach().cpu()