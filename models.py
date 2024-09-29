import torch
import torch.nn as nn
import numpy as np

class LSTMDecoder(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_series = num_features
        # self.lstm = nn.LSTM(num_features, hidden_size, batch_first=True)
        self.lstm = nn.GRU(num_features, hidden_size, batch_first=True)
        self.lstm.flatten_parameters()
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, X_right, z, causal_graph):
        X_right = X_right[:, :, np.where(causal_graph!=0)[0]]
        first = torch.zeros_like(X_right[:, 0:1, :])
        X_right = torch.cat((first, X_right[:, :-1, :]), dim=1)
        # X_right_pred, (hidden_out, _) = self.lstm(X_right, (z, z))
        X_right_pred, hidden_out = self.lstm(X_right, z)
        X_right_pred = self.fc(X_right_pred).squeeze(-1)
        return X_right_pred, hidden_out

class CausalLSTM(nn.Module):
    def __init__(self, num_features, hidden_size, causal_graph):
        super(CausalLSTM, self).__init__()
        self.device = torch.device('cuda')
        self.hidden_size = hidden_size
        self.num_series = num_features
        
        # self.encoder = nn.LSTM(num_features, hidden_size, batch_first=True)
        self.encoder = nn.GRU(num_features, hidden_size, batch_first=True)
        self.encoder.flatten_parameters()
        
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)
        self.causal_graph = causal_graph

        self.decoders = nn.ModuleList([LSTMDecoder(int(causal_graph[:, dim].sum()), hidden_size) for dim in range(num_features)])
    
    def forward(self, X_left, X_right, mode='train', phase=1, future=1, error=None):
        h_0 = torch.zeros(1, X_left.shape[0], self.hidden_size, device=self.device)
        if mode == 'train':
            # _, (hidden_out, _) = self.encoder(X_left)
            _, hidden_out = self.encoder(X_left, h_0)
            # hidden_out = hidden_out[-1]
            mu = self.fc_mu(hidden_out)
            logvar = self.fc_logvar(hidden_out)

            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps

            pred = []
            for (dim, decoder_head) in enumerate(self.decoders):
                op, _ = decoder_head(X_right, z, causal_graph=self.causal_graph[:, dim])
                pred.append(op)
            pred = torch.stack(pred, dim=2)
            
            return pred, mu, logvar

        if mode=='inference':

            # In Phase-1 Inference, the model will predict the last half of the instances to compute
            # the error and use that to pass to Error Compensation for probable error forecasting
            if phase==1:
                context = X_left.shape[1]
                context_l = int(context/2)
                _, hidden_out = self.encoder(X_left[:, :context_l, :], h_0)
                for d, d_head in enumerate(self.decoders):
                    op, h_t = d_head(X_left[:, context_l:, :], hidden_out, causal_graph=self.causal_graph[:, d])
                    if d==0:
                        X_t = op.unsqueeze(-1)
                    else:
                        X_t = torch.cat((X_t, op.unsqueeze(-1)), dim=-1)
                error = (X_left[:, context_l:, :] - X_t).detach()
                return error

            # In Phase-2 Inference, the model will forecast the actual data, compensated with the
            # forecasted error values from the Error Compensation model
            if phase==2:
                _, hidden_out = self.encoder(X_left[:, :-1, :], h_0)
                hidden_list = [hidden_out for _ in range(self.num_series)]
                X_pred = []
                for idx in range(future):
                    hidden_curr = []
                    for d, d_head in enumerate(self.decoders):
                        op, h_t = d_head(X_left[:, -1:, :], hidden_list[d], causal_graph=self.causal_graph[:, d])
                        if d==0:
                            X_t = op
                        else:
                            X_t = torch.cat((X_t, op), dim=-1)
                        hidden_curr.append(h_t)
                    hidden_list = hidden_curr
                    if error is not None:
                        X_t = X_t + 0.01*error[:, idx, :]
                    X_pred.append(X_t)
                X_pred = torch.stack(X_pred, dim=1)
                return X_pred
    
    def getCausalMatrix(self, thresh=True):
        GC = [torch.norm(decoder_head.lstm.weight_ih_l0, dim=0) for decoder_head in self.decoders]
        GC = torch.stack(GC)
        if thresh:
            return (torch.abs(GC) > 0).int()
        else:
            return GC

class ErrorCompensation(nn.Module):
    def __init__(self, num_features, hidden_size):
        super(ErrorCompensation, self).__init__()
        self.hidden_size = hidden_size
        self.num_series = num_features

        self.encoder = nn.GRU(num_features, hidden_size, batch_first=True)
        self.encoder.flatten_parameters()
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)
        self.fc_hidden = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

        self.decoder = nn.GRU(num_features, hidden_size, batch_first=True)
        self.decoder.flatten_parameters()
        self.fc = nn.Linear(hidden_size, num_features)
    
    def forward(self, E, mode='train', future=1):
        h_0 = torch.zeros(1, E.shape[0], self.hidden_size).cuda()
        
        if mode=='train':
            output, hidden_out = self.encoder(E, h_0)
            mu = self.fc_mu(hidden_out)
            logvar = self.fc_logvar(hidden_out)
            
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
            z = self.tanh(self.fc_hidden(z))

            E_pred, _ = self.decoder(E, z)
            E_pred = self.fc(E_pred)
            return E_pred, mu, logvar

        if mode=='inference': # Forecast errors
            assert future > 0
            # E = E.unsqueeze(1)
            _, hidden_out = self.encoder(E, h_0)

            for idx in range(future):
                op, hidden_out = self.decoder(E[:, -1:, :], hidden_out)
                E_t = self.fc(op)
                if idx==0:
                    E_pred = E_t
                else:
                    E_pred = torch.cat([E_pred, E_t], dim=1)
            return E_pred