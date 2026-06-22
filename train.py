import os, sys, json, time, gc, torch
import numpy as np
from itertools import product

from utils_data import load_preprocessed
from utils import to_loader, getModelName, epoch_time, set_deterministic
from services import ProximalGradientDescent, Criterion, EarlyStopper
from model import CausalLSTM, ErrorCompensation
from evaluate import evaluate_probabilistic

def phase1_epoch(model, train_dl, val_dl, prox, criterion):
    model.train()
    t_loss, n = 0.0, 0
    for X_l, X_r in train_dl:
        prox.zero_grad()
        model.zero_grad()
        X_l, X_r = X_l.cuda().detach(), X_r.cuda().detach()
        pred, mu, logvar = model(X_l, X_r)
        loss, _ = criterion(pred, X_r, mu, logvar, model)
        loss.backward()
        t_loss += loss.item() * X_l.shape[0]
        n += X_l.shape[0]
        prox.step(model)
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

def phase2_epoch(clstm, errorc, train_dl, val_dl, adam, criterion_e):
    errorc.train()
    t_loss_e, n = 0.0, 0
    for X_l, X_r in train_dl:
        adam.zero_grad()
        X_l, X_r = X_l.cuda(), X_r.cuda()
        with torch.no_grad():
            pred, _, _ = clstm(X_l, X_r)
        error = (X_r - pred).detach()
        pred_e, mu_e, logvar_e = errorc(error)
        loss_e, _ = criterion_e(pred_e, error, mu_e, logvar_e, errorc)
        loss_e.backward()
        adam.step()
        t_loss_e += loss_e.item() * X_l.shape[0]
        n += X_l.shape[0]

    errorc.eval()
    v_loss_e, n = 0.0, 0
    with torch.no_grad():
        for X_l, X_r in val_dl:
            X_l, X_r = X_l.cuda(), X_r.cuda()
            pred, _, _ = clstm(X_l, X_r)
            error = (X_r - pred).detach()
            pred_e, mu_e, logvar_e = errorc(error)
            loss_e, _ = criterion_e(pred_e, error, mu_e, logvar_e, errorc)
            v_loss_e += loss_e.item() * X_l.shape[0]
            n += X_l.shape[0]
    return t_loss_e / n, v_loss_e / n

def grid_search_trainer(dataset_name, param_grid, splits, patience, step_size, threshold='adaptive', epochs=10000, variant='raw'):
    parent = os.path.abspath('')
    dataset_name, data_artifact = dataset_name
    gt_matrix = splits.get('gt_matrix', None)

    # one subfolder per scaling variant: artifacts_<dataset>/<variant>/<run>
    art_root = os.path.join(parent, f'artifacts_{dataset_name}', variant)
    os.makedirs(art_root, exist_ok=True)
    model_name = getModelName(dataset=dataset_name, type='crvae', mode=f'{variant}_{threshold}')
    artifact_path = os.path.join(art_root, model_name)
    os.makedirs(artifact_path, exist_ok=True)

    checkpoints_path = os.path.join(artifact_path, 'checkpoints')
    logs_path = os.path.join(artifact_path, 'logs')
    gc_path = os.path.join(artifact_path, 'gcs_est')
    metadata_path = os.path.join(artifact_path, 'metadata')
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(gc_path, exist_ok=True)
    os.makedirs(metadata_path, exist_ok=True)

    input_size = splits['train'][0].shape[-1]
    seq_len = splits['train'][0].shape[-2]

    combos = list(product(
        param_grid['lr'], param_grid['batch_size'], param_grid['hidden_size'],
        param_grid['lambda'], param_grid['lam_ridge'], param_grid['beta_kl'],
        param_grid['beta_e'], param_grid['future'])
    )

    metadata_list = []
    start_gs = time.time()
    combination = 0
    for lr, batch_size, hidden, lam, lam_ridge, beta_kl, beta_e, future in combos:
        combination += 1
        sys.stdout = open(os.path.join(logs_path, f'train_comb{combination}.log'), 'w')
        print(f"Dataset: {data_artifact}", flush=True)
        print(f"Combination {combination}:: lr {lr} bs {batch_size} hidden {hidden} "
              f"lambda {lam} lam_ridge {lam_ridge} beta_kl {beta_kl} beta_e {beta_e} "
              f"future {future}", flush=True)

        train_dl = to_loader(splits['train'], batch_size, shuffle=True)
        val_dl = to_loader(splits['val'], batch_size, shuffle=False)

        config = {'dataset': f"{data_artifact}_c{combination}"}
        clstm = CausalLSTM(input_size, hidden, np.ones([input_size, input_size])).cuda()
        prox = ProximalGradientDescent(clstm.parameters(), lr=lr, lam=lam)
        criterion = Criterion(beta_kl, lam_ridge, model='clstm')
        stopper = EarlyStopper(patience=patience)

        tr_loss, va_loss = [], []
        gc_raw_list, gc_list, thr_list, usage_list = [], [], [], []
        best_val, best_epoch = np.inf, 0
        start = time.time()
        for epoch in range(epochs):
            start_ep = time.time()
            t, v = phase1_epoch(clstm, train_dl, val_dl, prox, criterion)
            tr_loss.append(t)
            va_loss.append(v)

            raw, binary, thr = clstm.get_causal_matrix(threshold=threshold)
            usage = 100 * float(torch.mean(binary))
            gc_raw_list.append(raw.tolist())
            gc_list.append(binary.tolist())
            thr_list.append(thr)
            usage_list.append(usage)

            print(f"Epoch {epoch + 1}  train {t:.6f}  val {v:.6f}  "
                  f"usage {usage:.2f}% (thr={thr})", flush=True)
            _, mn, sc = epoch_time(start_ep, time.time())
            print(f"  epoch time {mn}m {sc:.2f}s", flush=True)

            if v < best_val:
                best_val, best_epoch = v, epoch + 1
                torch.save(clstm.state_dict(), os.path.join(checkpoints_path, f'clstm_comb{combination}.pt'))
                np.save(os.path.join(gc_path, f'GC_est_comb{combination}.npy'), binary.cpu().numpy())
                np.save(os.path.join(gc_path, f'GC_raw_comb{combination}.npy'), raw.cpu().numpy())
                print(f"  CLSTM recorded (val {v:.6f})", flush=True)
                step_counter = 0
            else:
                step_counter = step_counter + 1 if epoch > 0 else 1
            if step_counter >= patience:
                print("Phase-1 not improving. Stopping.", flush=True)
                break

        # ---- Phase 2: error compensation on the best phase-1 model ----
        clstm.load_state_dict(torch.load(
            os.path.join(checkpoints_path, f'clstm_comb{combination}.pt'), weights_only=True))
        for p in clstm.parameters():
            p.requires_grad = False
        errorc = ErrorCompensation(input_size, hidden).cuda()
        adam = torch.optim.Adam(errorc.parameters(), lr=lr)
        criterion_e = Criterion(beta_e, lam_ridge, model='errorc').cuda()
        stopper_e = EarlyStopper(patience=max(patience, 200))
        best_e = np.inf
        for epoch in range(epochs):
            te, ve = phase2_epoch(clstm, errorc, train_dl, val_dl, adam, criterion_e)
            print(f"[phase2] epoch {epoch + 1}  train_e {te:.6f}  val_e {ve:.6f}", flush=True)
            if ve < best_e:
                best_e = ve
                torch.save(errorc.state_dict(), os.path.join(checkpoints_path, f'errorc_comb{combination}.pt'))
            if stopper_e.early_stop(ve):
                print("Phase-2 not improving. Stopping.", flush=True)
                break

        # ---- Held-out evaluation (corrected probabilistic metrics) ----
        errorc.load_state_dict(torch.load(
            os.path.join(checkpoints_path, f'errorc_comb{combination}.pt'), weights_only=True))
        test_metrics = evaluate_probabilistic(
            clstm, errorc, to_loader(splits['test'], batch_size), n_samples=100)
        ood_metrics = None
        if 'ood' in splits:
            ood_metrics = evaluate_probabilistic(clstm, errorc, to_loader(splits['ood'], batch_size), n_samples=100)

        h, m, s = epoch_time(start, time.time())
        metadata = {
            'combination': combination,
            'dataset': data_artifact,
            'scaling': variant,
            'n_dim': input_size,
            'seq_len': seq_len,
            'artifact': model_name,
            'model': {'batch_size': batch_size, 'input_size': input_size,
                      'hidden_size': hidden, 'future': future},
            'initial_lr': lr,
            'lambda': lam, 'lam_ridge': lam_ridge,
            'beta_kl': beta_kl, 'beta_e': beta_e,
            'threshold_mode': threshold,
            'optimal_epoch': best_epoch,
            'best_val_loss': best_val,
            'training_time': {'hr': h, 'mins': m, 'sec': s},
            'final_GC_est': gc_list[-1] if gc_list else None,
            'gc_raw_list': gc_raw_list,
            'gc_est_list': gc_list,
            'threshold_list': thr_list,
            'variable_usage_list': usage_list,
            'train_loss_list': tr_loss,
            'val_loss_list': va_loss,
            'test_metrics': test_metrics,
            'ood_metrics': ood_metrics,
        }
        if gt_matrix is not None:
            from metrics import auroc
            raw, _, _ = clstm.get_causal_matrix(threshold=threshold)
            metadata['auroc'] = auroc(raw.numpy(), gt_matrix)
        sys.stdout = sys.__stdout__
        with open(os.path.join(metadata_path, f'metadata_comb{combination}.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        metadata_list.append(metadata)
        del clstm, errorc
        gc.collect()
        torch.cuda.empty_cache()

    h, m, s = epoch_time(start_gs, time.time())
    print(f"Total grid-search time: {h}h {m}m {s:.1f}s", flush=True)
    return metadata_list, artifact_path

def variant_paths(datasets_dir, dataset_name, dataset_artifact, context, kinds):
    """Build {variant_label: npz_path} for one dataset.
    The synthetic preprocessor names the standardized file 'standard'; the OBD
    notebook names it 'std'. `kinds` is a list of (label, file_token) pairs so
    the on-disk token can differ from the folder label."""
    if dataset_name == 'obd2':
        base = os.path.join(datasets_dir, dataset_name, f'obd2_ctx{context}')
        stem = 'obd'
    else:
        base = os.path.join(datasets_dir, dataset_name, f'{dataset_artifact}_ctx{context}')
        stem = dataset_artifact
    return {label: os.path.join(base, f'{stem}_{token}_ctx{context}.npz') for label, token in kinds}

def run_all_variants(dataset_name, dataset_artifact, vpaths, param_grid, patience, step_size, threshold='adaptive', epochs=10000):
    """Train every scaling variant in a single execution. Each writes to its own
    artifacts_<dataset>/<variant>/ subfolder."""
    summary = {}
    for variant, path in vpaths.items():
        if not os.path.exists(path):
            print(f"[skip] {variant}: file not found -> {path}", flush=True)
            continue
        print(f"\n{'=' * 70}\n=== VARIANT: {variant}  ({os.path.basename(path)})\n{'=' * 70}", flush=True)
        splits = load_preprocessed(path)
        meta, art = grid_search_trainer(
            dataset_name=(dataset_name, f'{dataset_artifact}_{variant}'),
            param_grid=param_grid, splits=splits,
            patience=patience, step_size=step_size,
            threshold=threshold, epochs=epochs, variant=variant)
        summary[variant] = art
        print(f"[done] {variant} -> {art}", flush=True)
    return summary


if __name__ == "__main__":
    set_deterministic(42)
    parent = os.path.abspath('')
    datasets_dir = os.path.join(parent, 'data')

    ####################################################################
    ### Manually change this block for other datasets
    dataset_name = 'henon'                 # ['henon', 'lorenz', 'obd2']
    dataset_artifact = 'henon_5000_10'
    context = 50                          # [50 for syn, 30 for obd2]
    # (label on the output folder, token in the npz filename)
    # synthetic standardized file is '<artifact>_standard_...'; OBD is 'obd_std_...'
    KINDS = ([('minmax', 'minmax'), ('std', 'std')] if dataset_name == 'obd2'
            else [('raw', 'raw'), ('minmax', 'minmax'), ('std', 'std')])
    # KINDS = [('raw', 'raw')]
    ####################################################################

    vpaths = variant_paths(datasets_dir, dataset_name, dataset_artifact, context, KINDS)

    param_grid = {
        'lr': [0.01, 0.001],
        'batch_size': [1024],
        'hidden_size': [256, 512],
        'lambda': [0.1, 0.01],
        # 'lambda': [0.0], # Sparsity Ablation
        'lam_ridge': [0.0],
        'beta_kl': [0.1, 0.05, 0.01],
        'beta_e': [1.0],
        'future': [15], # Not functional in this code
    }

    summary = run_all_variants(
        dataset_name, dataset_artifact, vpaths, param_grid,
        epochs=10000,
        patience=100, step_size=20, threshold='adaptive')

    print("\nAll variants complete:")
    for v, art in summary.items():
        print(f"  {v:8s} -> {art}")