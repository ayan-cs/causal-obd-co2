import os, json, time, torch, sys
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import (to_loader, getModelName, epoch_time, set_deterministic)
from utils_data import load_preprocessed
from services import Criterion, EarlyStopper
from model import CausalLSTM, ErrorCompensation
from evaluate import evaluate_probabilistic
from train import phase1_epoch, phase2_epoch

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

def feedback_phase1_epoch(model, train_dl, val_dl, optimizer, criterion):
    model.train()
    t_loss, n = 0.0, 0
    for X_l, X_r in train_dl:
        optimizer.zero_grad()
        X_l, X_r = X_l.cuda().detach(), X_r.cuda().detach()
        pred, mu, logvar = model(X_l, X_r)
        loss, _ = criterion(pred, X_r, mu, logvar, model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        t_loss += loss.item() * X_l.shape[0]
        n += X_l.shape[0]
    train_loss = t_loss / n

    model.eval()
    v_loss, n = 0.0, 0
    with torch.no_grad():
        for X_l, X_r in val_dl:
            X_l, X_r = X_l.cuda().detach(), X_r.cuda().detach()
            pred, mu, logvar = model(X_l, X_r)
            loss, _ = criterion(pred, X_r, mu, logvar, model)
            v_loss += loss.item() * X_l.shape[0]
            n += X_l.shape[0]
    return train_loss, v_loss / n

def run_graph_feedback(splits, discovered_graph, config, epochs, out_dir):
    sys.stdout = open(os.path.join(out_dir, 'logs', f'train_hardmask_comb{config['comb']}.log'), 'w')
    
    M = splits['train'][0].shape[-1]
    graph = np.asarray(discovered_graph)
    assert graph.shape == (M, M), f"graph {graph.shape} != ({M},{M})"

    # guarantee each target keeps at least its self-edge so no head is empty
    for d in range(M):
        if graph[:, d].sum() == 0:
            graph[d, d] = 1

    train_dl = to_loader(splits['train'], config['batch_size'], shuffle=True)
    val_dl = to_loader(splits['val'], config['batch_size'], shuffle=False)

    # ---- hard-masked model: graph passed at init ----
    clstm = CausalLSTM(M, config['hidden_size'], graph).cuda()
    criterion = Criterion(config['beta_kl'], config['lam_ridge'], model='clstm')
    optimizer = AdamW(clstm.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config['step_size'])
    stopper = EarlyStopper(patience=config['patience'])

    best_val = np.inf
    start = time.time()
    for epoch in range(epochs):
        t, v = feedback_phase1_epoch(clstm, train_dl, val_dl, optimizer, criterion)
        scheduler.step(v)                       # step on validation loss, once per epoch
        print(f"[fb phase1] epoch {epoch+1}  train {t:.6f}  val {v:.6f}  lr {optimizer.param_groups[0]['lr']:.6f}", flush=True)
        if v < best_val:
            best_val = v
            torch.save(clstm.state_dict(), os.path.join(out_dir, 'checkpoints', f'clstm_hardmask_comb{config['comb']}.pt'))
        if stopper.early_stop(v):
            print("Feedback phase-1 not improving. Stopping.\n", flush=True)
            break

    clstm.load_state_dict(torch.load(os.path.join(out_dir, 'checkpoints', f'clstm_hardmask_comb{config['comb']}.pt'), weights_only=True))
    for p in clstm.parameters():
        p.requires_grad = False

    errorc = ErrorCompensation(M, config['hidden_size']).cuda()
    optimizer_e = torch.optim.AdamW(errorc.parameters(), lr=config['lr'])
    scheduler_e = ReduceLROnPlateau(optimizer_e, mode='min', factor=0.1, patience=config['step_size'])
    criterion_e = Criterion(config['beta_e'], config['lam_ridge'], model='errorc').cuda()
    stopper_e = EarlyStopper(patience=config['patience'])
    best_e = np.inf
    for epoch in range(epochs):
        te, ve = phase2_epoch(clstm, errorc, train_dl, val_dl, optimizer_e, criterion_e)
        scheduler_e.step(ve)
        print(f"[fb phase2] epoch {epoch + 1}  train_e {te:.6f}  val_e {ve:.6f}  lr_e {optimizer_e.param_groups[0]['lr']:.6f}", flush=True)
        if ve < best_e:
            best_e = ve
            torch.save(errorc.state_dict(), os.path.join(out_dir, 'checkpoints', f'errorc_hardmask_comb{config['comb']}.pt'))
        if stopper_e.early_stop(ve):
            print("Feedback phase-2 not improving. Stopping.\n", flush=True)
            break

    errorc.load_state_dict(torch.load(os.path.join(out_dir, 'checkpoints', f'errorc_comb{config['comb']}.pt'), weights_only=True))
    results = {
        'graph_used': graph.tolist(),
        'variable_usage_pct': float(100 * graph.mean()),
        'best_val_loss': float(best_val),
        'test_metrics': evaluate_probabilistic(
            clstm, errorc, to_loader(splits['test'], config['batch_size']), n_samples=100),
    }
    if 'ood' in splits:
        results['ood_metrics'] = evaluate_probabilistic(
            clstm, errorc, to_loader(splits['ood'], config['batch_size']), n_samples=100)
    h, m, s = epoch_time(start, time.time())
    results['training_time'] = {'hr': h, 'mins': m, 'sec': s}

    with open(os.path.join(out_dir, 'metadata', f'metadata_hardmark_comb{config['comb']}.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Feedback results written to {out_dir}", flush=True)

    sys.stdout = sys.__stdout__

    return results


if __name__ == "__main__":
    set_deterministic(42)
    parent = os.path.abspath('')
    datasets_dir = os.path.join(parent, 'data')

    ####################################################################
    ### Manually set the dataset, the scaling variant, and the graph path.
    ### Use the SAME variant npz that produced the discovered graph, and the
    ### GC_est saved by that variant's discovery run.
    dataset_name = 'henon'                 # ['henon', 'lorenz', 'obd']
    dataset_artifact = 'henon_5000_10'
    context = 50
    variant = 'raw'                        # 'raw' | 'minmax' | 'std'
    exp_dir = 'crvae___raw_adaptive___henon___11-06-2026_20-32-41'
    best_comb = 1
    
    # token = 'std' if dataset_name == 'obd' else 'standard'   # filename token
    if dataset_name == 'obd2':
        npz_path = os.path.join(datasets_dir, dataset_name, f'obd2_ctx{context}', f'obd_{variant}_ctx{context}.npz')
    else:
        npz_path = os.path.join(datasets_dir, dataset_name, f'{dataset_artifact}_ctx{context}', f'{dataset_artifact}_{variant}_ctx{context}.npz')
    graph_path = os.path.join(parent, f'artifacts_{dataset_name}', variant, exp_dir, 'gcs_est', f'GC_est_comb{best_comb}.npy')
    out_path = os.path.join(parent, f'artifacts_{dataset_name}', variant, exp_dir)
    ####################################################################

    splits = load_preprocessed(npz_path)
    discovered_graph = np.load(graph_path)

    config = {
        'lr': 0.01,
        'batch_size': 4096,
        'hidden_size': 256,
        'lam_ridge': 0.0,
        'beta_kl': 0.1,
        'beta_e': 1.0,
        'patience': 100,
        'step_size': 20,
        'comb': best_comb
    }

    _ = run_graph_feedback(splits, discovered_graph, config, epochs=10000, out_dir=out_path)