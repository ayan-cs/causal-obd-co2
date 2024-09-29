import numpy as np
import os, torch, json
from torch.utils.data import DataLoader, TensorDataset

from trainer import trainCausalLSTM, trainWithErrorCompensation
from models import ErrorCompensation, CausalLSTM
from inference import batchInference, inference

torch.manual_seed(42)
np.random.seed(42)

parent = os.path.abspath('')
dataset = os.path.join(parent, 'datasets', 'vehicular_modified.json')

with open(dataset, 'r') as fp:
    df = json.load(fp)
columns = list(df['columns'])
df['car1'] = np.array(df['car1'], dtype=np.float32)
df['car2'] = np.array(df['car2'], dtype=np.float32)
print(f"Columns : {columns}")

train_data = df['car1']
val_data = df['car2']
num_samples, dim = train_data.shape

# Discarding Car_Id and Person_Id
n_dim = dim-2

def preprocessOBD(X, context):
    def createChunks(data, context):
        if context > data.shape[0]:
            context = data.shape[0]
        X_left = []
        X_right = []
        context_l = int(context/2)
        for i in range(len(data) - context + 1):
            X_left.append(data[i : i+context_l].tolist())
            X_right.append(data[i+context_l : i+context].tolist())
        return X_left, X_right

    # parent = os.path.abspath('')
    # dataset = os.path.join(parent, 'datasets', f"{config['dataset']}.npy")
    # context = config['chunksize']
    # context = 30
    # X = np.load(dataset).T

    X_train_left, X_train_right = createChunks(X, context)
    return X_train_left, X_train_right

config = {
    'dataset' : 'OBD',
    'chunksize' : 30,
    'batchsize' : 4096,
    'epochs' : 2000,
    'hidden_size' : 256,
    'lambda' : 0.1,
    'lam_ridge' : 0,
    'lr' : 0.01,
    'beta_mmd' : 0.1,
    'verbose_interval' : 5,
    'beta_e' : 1,
    'future' : 20
}

# Preprocess OBD
X_train_left, X_train_right = preprocessOBD(train_data, config['chunksize'])
X_val_left, X_val_right = preprocessOBD(val_data, context=config['chunksize'])
X_train_left = torch.tensor(X_train_left)[:, :, 2:]
X_train_right = torch.tensor(X_train_right)[:, :, 2:]
X_val_left = torch.tensor(X_val_left)[:, :, 2:]
X_val_right = torch.tensor(X_val_right)[:, :, 2:]
print(f"OBD Train data size\nX_left : {X_train_left.shape}\tX_right : {X_train_right.shape}")
print(f"OBD Val data size\nX_left : {X_val_left.shape}\tX_right : {X_val_right.shape}")

train_dl = DataLoader(TensorDataset(X_train_left, X_train_right), batch_size=config['batchsize'], shuffle=True)
val_dl = DataLoader(TensorDataset(X_val_left, X_val_right), batch_size=config['batchsize'])
print(f"Dataloader size :\nrain : {len(train_dl)}\tVal : {len(val_dl)}")

#############################################################################
#################################  PHASE-1  #################################
#############################################################################

n_dim = X_train_left.shape[-1]
clstm = CausalLSTM(num_features=n_dim, hidden_size=config['hidden_size'], causal_graph=np.ones([n_dim, n_dim]))

train_loss_list, val_loss_list = trainCausalLSTM(clstm, train_dl, val_dl=val_dl, dim=n_dim, config=config)

clstm.load_state_dict(torch.load(os.path.join(parent, 'checkpoints', f"clstm_{config['dataset']}.pth"), weights_only=True))
GC_est = clstm.getCausalMatrix().cpu().data.numpy()
print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
print(f"Estimated Causal matrix :\n{GC_est}")
np.save(os.path.join(parent, 'outputs', "CausalGraph_OBD.npy"), GC_est)

causal_graph = np.load(os.path.join(parent, 'outputs', "CausalGraph_OBD.npy"))
n_dim = causal_graph.shape[-1]

#############################################################################
#################################  PHASE-2  #################################
#############################################################################

# Reinitialize the C-LSTM model
clstm = CausalLSTM(num_features=n_dim, hidden_size=config['hidden_size'], causal_graph=np.ones([n_dim, n_dim]))
clstm.load_state_dict(torch.load(os.path.join(parent, 'checkpoints', f"clstm_{config['dataset']}.pth"), weights_only=True))
for param in clstm.parameters():
    param.requires_grad = False

# Initialize ErrorCompensation model
errorc = ErrorCompensation(num_features=n_dim, hidden_size=config['hidden_size'])

train_loss_list, val_loss_list, train_loss_list_e, val_loss_list_e = trainWithErrorCompensation(clstm, errorc, train_dl, val_dl=train_dl, dim=n_dim, config=config)

clstm.load_state_dict(torch.load(os.path.join(parent, 'checkpoints', f"clstm_{config['dataset']}.pth"), weights_only=True))
GC_est_err = clstm.getCausalMatrix().cpu().data.numpy()
print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
print(f"Estimated Causal matrix :\n{GC_est}")
np.save(os.path.join(parent, 'outputs', "CausalGraph_OBD_err.npy"), GC_est)

#############################################################################
###############################  INFERENCE  #################################
#############################################################################

dim = X_train_left.shape[-1]
rmse, avgmmd, avgmae = batchInference(train_dl, dim, config)
print(f"RMSE : {rmse}\tMMD : {avgmmd}\tMAE : {avgmae}")