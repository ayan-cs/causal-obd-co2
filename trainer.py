import torch, os
import torch.nn as nn
from torch.optim import Adam

import numpy as np
from services import ProximalGradientDescent, Criterion, EarlyStopper

def trainCausalLSTM(model, train_dl, val_dl, dim, config):
    parent = os.path.abspath('')
    checkpoints_path = os.path.join(parent, 'checkpoints')
    if not os.path.exists(os.path.join(checkpoints_path)):
        os.mkdir(checkpoints_path)

    model = model.cuda()
    criterion = Criterion(config['beta_mmd'], config['lam_ridge'], model='clstm')
    prox = ProximalGradientDescent(model.parameters(), lr=config['lr'], lam=config['lambda'])
    callback = EarlyStopper(patience=config['patience'])
    
    train_loss_list = []
    val_loss_list = []
    best_loss = np.inf
    best_epoch = 0
    verbose = config['verbose']

    for epoch in range(1, config['epochs']+1):
        model.train()
        total_loss = 0
        total_samples = 0
        total_mmd = 0
        for X_l, X_r in train_dl:
            prox.zero_grad()
            model.zero_grad()
            X_l = X_l.cuda().detach()
            X_r = X_r.cuda().detach()
            pred, mu, logvar = model(X_l, X_r)

            loss, _ = criterion(pred, X_r, mu, logvar, model)
            total_samples += X_l.shape[0]

            loss.backward()
            total_loss += loss * X_l.shape[0]
            prox.step(model)

        train_loss_list.append(total_loss/total_samples)

        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for X_l, X_r in val_dl:
                X_l = X_l.cuda().detach()
                X_r = X_r.cuda().detach()
                pred, mu, logvar = model(X_l, X_r)

                loss, _ = criterion(pred, X_r, mu, logvar, model)
                total_loss += loss * X_l.shape[0]
                total_samples += X_l.shape[0]
            
            val_loss_list.append(total_loss/total_samples)

        if callback.early_stop(val_loss_list[-1]):
            print("Model is not improving. Quitting ...")
            break

        if verbose == 1:
            print(f"\nEpoch {epoch}\n")
            print(f"Train loss : {train_loss_list[-1]:.6f}\tValidation loss : {val_loss_list[-1]:.6f}")
            print(f"Variable usage : {100 * torch.mean(model.getCausalMatrix().float()):.4f}%")

        if val_loss_list[-1] < best_loss:
            torch.save(model.state_dict(), os.path.join(checkpoints_path, f"clstm_{config['dataset']}.pth"))
            best_loss = val_loss_list[-1]
            best_epoch = epoch
            print(f"CLSTM model saved with Val loss : {best_loss:.6f} on Epoch {best_epoch}")
    
    print(f"\nTraining complete! Best model saved with Val Loss : {best_loss} in Epoch {best_epoch}.\n")
    return train_loss_list, val_loss_list

def trainWithErrorCompensation(clstm, errorc, train_dl, val_dl, dim, config):
    parent = os.path.abspath('')
    checkpoints_path = os.path.join(parent, 'checkpoints')

    clstm = clstm.cuda()
    errorc = errorc.cuda()

    for params in clstm.parameters():
        params.requires_grad = False

    criterion = Criterion(config['beta_mmd'], config['lam_ridge'], model='clstm').cuda()
    criterion_e = Criterion(config['beta_e'], config['lam_ridge'], model='errorc').cuda()
    adam = Adam(errorc.parameters(), lr=config['lr'])
    prox = ProximalGradientDescent(clstm.parameters(), lr=config['lr'], lam=config['lambda'])
    callback = EarlyStopper(patience=2000)
    
    train_loss_list = []
    val_loss_list = []
    train_loss_list_e = []
    val_loss_list_e = []
    best_loss = np.inf
    best_epoch = 0
    verbose = config['verbose']

    for epoch in range(1, config['epochs']+1):
        clstm.eval()
        errorc.train()
        total_loss = 0
        total_loss_e = 0
        total_samples = 0

        for X_l, X_r in train_dl:
            adam.zero_grad()
            X_l = X_l.cuda()
            X_r = X_r.cuda()
            with torch.no_grad():
                pred, mu, logvar = clstm(X_l, X_r)
                loss = criterion(pred, X_r, mu, logvar, clstm)

            error = (X_r - pred).detach()
            pred_e, mu_e, logvar_e = errorc(error)
            loss_e, _ = criterion_e(pred_e, error, mu_e, logvar_e, errorc)

            loss_e.backward()
            adam.step()

            total_samples += X_l.shape[0]
            total_loss += loss.item() * X_l.shape[0]
            total_loss_e += loss_e.item() * X_l.shape[0]
        
        train_loss_list.append(total_loss/total_samples)
        train_loss_list_e.append(total_loss_e/total_samples)

        clstm.eval()
        errorc.eval()
        total_loss = 0
        total_loss_e = 0
        total_samples = 0
        with torch.no_grad():
            for X_l, X_r in val_dl:
                X_l = X_l.cuda()
                X_r = X_r.cuda()
                pred, mu, logvar = clstm(X_l, X_r)
                loss, mmd = criterion(pred, X_r, mu, logvar, clstm)
                
                error = (X_r - pred).detach()
                pred_e, mu_e, logvar_e = errorc(error)
                loss_e, _ = criterion_e(pred_e, error, mu_e, logvar_e, errorc)
                total_loss_e += loss_e.item() * X_l.shape[0]

                total_samples += X_l.shape[0]

            val_loss_list.append(loss/total_samples)
            val_loss_list_e.append(total_loss_e/total_samples)
        
        if callback.early_stop(val_loss_list[-1]):
            print("Model is not improving. Quitting ...")
            break

        if verbose == 1:
            print(f"\nEpoch {epoch}\n")
            print(f"ErrorC train loss : {train_loss_list_e[-1]:.6f}\tVal loss : {val_loss_list_e[-1]:.6f}")
            print(f"Variable usage : {100 * torch.mean(clstm.getCausalMatrix().float()):.4f}%")

        if val_loss_list_e[-1] < best_loss:
            torch.save(errorc.state_dict(), os.path.join(checkpoints_path, f"errorc_{config['dataset']}.pth"))
            best_loss = val_loss_list_e[-1]
            best_epoch = epoch
            print(f"ErrC model saved with Val loss : {best_loss:.6f} on Epoch {best_epoch}")
        
    return train_loss_list, val_loss_list, train_loss_list_e, val_loss_list_e