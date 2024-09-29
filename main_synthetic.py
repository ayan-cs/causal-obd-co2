import numpy as np
import os, torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from utils import preprocessSyn, getCausalMatrix
from trainer import trainCausalLSTM, trainWithErrorCompensation
from models import ErrorCompensation, CausalLSTM
from inference import batchInference, inference

torch.manual_seed(42)
np.random.seed(42)

parent = os.path.abspath('')

# CONFIG for HENON
config = {
    'dataset' : 'henon_10000_10',
    'chunksize' : 30,
    'batchsize' : 4096,
    'test_split' : 0.15,
    'epochs' : 2500,
    'patience' : 200,
    'hidden_size' : 128,
    'lambda' : 0.1,
    'lam_ridge' : 0,
    'lr' : 0.03,
    'beta_mmd' : 0.1,
    'verbose' : 1,
    'beta_e' : 1,
    'future' : 20
}

# CONFIG for LORENZ-96
# config = {
#     'dataset' : 'lorenz_5000_10',
#     'chunksize' : 40,
#     'batchsize' : 4096,
#     'test_split' : 0.2,
#     'epochs' : 1200,
#     'hidden_size' : 64,
#     'lambda' : 10,
#     'lam_ridge' : 0.01, # 0.1
#     'lr' : 0.001,
#     'beta_mmd' : 0.1,
#     'verbose_interval' : 20,
#     'beta_e' : 1,
#     'future' : 20
# }

X_train_left, X_train_right, X_val_left, X_val_right = preprocessSyn(config)
print(torch.tensor(X_train_right).shape)

X_train_left = torch.tensor(X_train_left)
X_train_right = torch.tensor(X_train_right)
print(X_train_left.shape, X_train_right.shape)
X_val_left = torch.tensor(X_val_left)
X_val_right = torch.tensor(X_val_right)
print(X_val_left.shape, X_val_right.shape)

GC = getCausalMatrix(n_dim=X_train_left.shape[-1], data=config['dataset'].split('_')[0])

train_dl = DataLoader(TensorDataset(X_train_left, X_train_right), batch_size=config['batchsize'], shuffle=True)
val_dl = DataLoader(TensorDataset(X_val_left, X_val_right), batch_size=config['batchsize'])
print(f"Dataloader size : {len(train_dl)}")

#############################################################################
#################################  PHASE-1  #################################
#############################################################################

n_dim = X_train_left.shape[-1]
model = CausalLSTM(num_features=n_dim, hidden_size=config['hidden_size'], causal_graph=np.ones([n_dim, n_dim]))
train_loss_list, val_loss_list = trainCausalLSTM(model, train_dl, val_dl=val_dl, dim=n_dim, config=config)

model.load_state_dict(torch.load(os.path.join(parent, 'checkpoints', f"clstm_{config['dataset']}.pth"), weights_only=True))
GC_est = model.getCausalMatrix().cpu().data.numpy()

print('True variable usage = %.2f%%' % (100 * np.mean(GC)))
print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
print('Accuracy = %.2f%%' % (100 * np.mean(GC == GC_est)))
print(f"Ground truth : \n{GC}\n\nEstimated Causal matrix :\n{GC_est}")
np.save(os.path.join(parent, 'outputs', "CausalGraph.npy"), GC_est)

n_dim = X_train_left.shape[-1]

#############################################################################
#################################  PHASE-2  #################################
#############################################################################

# Reinitialize the C-LSTM model
clstm = CausalLSTM(num_features=n_dim, hidden_size=config['hidden_size'], causal_graph=np.ones([n_dim, n_dim]))
clstm.load_state_dict(torch.load(os.path.join(parent, 'checkpoints', f"clstm_{config['dataset']}.pth"), weights_only=True))
# for param in clstm.parameters():
#     param.requires_grad = False

# Initialize ErrorCompensation model
errorc = ErrorCompensation(num_features=n_dim, hidden_size=config['hidden_size'])

train_loss_list, val_loss_list, train_loss_list_e, val_loss_list_e = trainWithErrorCompensation(clstm, errorc, train_dl, val_dl=val_dl, dim=n_dim, config=config)

clstm.load_state_dict(torch.load(os.path.join(parent, 'checkpoints', f"clstm_{config['dataset']}.pth"), weights_only=True))
GC_est = clstm.getCausalMatrix().cpu().data.numpy()

print('True variable usage = %.2f%%' % (100 * np.mean(GC)))
print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
print('Accuracy = %.2f%%' % (100 * np.mean(GC == GC_est)))
print(f"Ground truth : \n{GC}\n\nEstimated Causal matrix :\n{GC_est}")
np.save(os.path.join(parent, 'outputs', "CausalGraph_err.npy"), GC_est)

#############################################################################
###############################  INFERENCE  #################################
#############################################################################

dim = X_train_left.shape[-1]
rmse, avgmmd, mae = batchInference(val_dl, dim, config)
print(f"RMSE : {rmse}\tMMD : {avgmmd}\tMAE : {mae}")

# instance_idx = 2

# X_pred = inference(X_train_left[instance_idx].unsqueeze(0), dim=X_train_left.shape[-1], config=config)
# num_items = X_train_left[instance_idx].flatten().shape[0]
# print(f"Euclidean distance of single sample : {torch.sqrt(torch.sum((X_train_right[2] - X_pred[0])**2) / num_items)}")