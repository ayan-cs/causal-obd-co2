import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

class LSTMDecoder(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_series = num_features
        self.lstm = nn.GRU(num_features, hidden_size, batch_first=True)
        self.lstm.flatten_parameters()
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X_right, z, causal_graph):
        X_right = X_right[:, :, np.where(causal_graph != 0)[0]]
        first = torch.zeros_like(X_right[:, 0:1, :])
        X_right = torch.cat((first, X_right[:, :-1, :]), dim=1)
        X_right_pred, hidden_out = self.lstm(X_right, z)
        X_right_pred = self.fc(X_right_pred).squeeze(-1)
        return X_right_pred, hidden_out

class CausalLSTM(nn.Module):
    def __init__(self, num_features, hidden_size, causal_graph):
        super().__init__()
        self.device = torch.device('cuda')
        self.hidden_size = hidden_size
        self.num_series = num_features
        self.causal_graph = causal_graph

        self.encoder = nn.GRU(num_features, hidden_size, batch_first=True)
        self.encoder.flatten_parameters()
        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)

        # each head's input width = number of its (currently allowed) parents.
        # full-ones graph -> all M inputs; a binary graph -> hard-masked head.
        self.decoders = nn.ModuleList([
            LSTMDecoder(int(causal_graph[:, dim].sum()), hidden_size)
            for dim in range(num_features)
        ])

    def forward(self, X_left, X_right, mode='train', phase=1, future=1, error=None):
        h_0 = torch.zeros(1, X_left.shape[0], self.hidden_size, device=self.device)

        if mode == 'train':
            _, hidden_out = self.encoder(X_left, h_0)
            mu = self.fc_mu(hidden_out)
            logvar = self.fc_logvar(hidden_out)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            pred = []
            for dim, head in enumerate(self.decoders):
                op, _ = head(X_right, z, causal_graph=self.causal_graph[:, dim])
                pred.append(op)
            return torch.stack(pred, dim=2), mu, logvar

        if mode == 'inference':
            if phase == 1:
                context = X_left.shape[1]
                context_l = int(context / 2)
                _, hidden_out = self.encoder(X_left[:, :context_l, :], h_0)
                X_t = None
                for d, head in enumerate(self.decoders):
                    op, _ = head(X_left[:, context_l:, :], hidden_out,
                                 causal_graph=self.causal_graph[:, d])
                    X_t = op.unsqueeze(-1) if d == 0 else torch.cat((X_t, op.unsqueeze(-1)), dim=-1)
                return (X_left[:, context_l:, :] - X_t).detach()

            if phase == 2:
                _, hidden_out = self.encoder(X_left[:, :-1, :], h_0)
                hidden_list = [hidden_out for _ in range(self.num_series)]
                X_pred = []
                for idx in range(future):
                    hidden_curr, X_t = [], None
                    for d, head in enumerate(self.decoders):
                        op, h_t = head(X_left[:, -1:, :], hidden_list[d],
                                       causal_graph=self.causal_graph[:, d])
                        X_t = op if d == 0 else torch.cat((X_t, op), dim=-1)
                        hidden_curr.append(h_t)
                    hidden_list = hidden_curr
                    if error is not None:
                        X_t = X_t + 0.01 * error[:, idx, :]
                    X_pred.append(X_t)
                return torch.stack(X_pred, dim=1)

    def _strength_matrix(self):
        """(M, M) raw causal-strength matrix from GRU input-weight norms.
        Only well-defined when every head sees all M inputs (full connectivity,
        i.e. the discovery model). Masked heads return their fixed graph."""
        widths = [h.lstm.weight_ih_l0.shape[1] for h in self.decoders]
        if any(w != self.num_series for w in widths):
            # print(np.array(widths).shape, self.num_series, flush=True)
            return None  # masked / graph-conditioned model: no square readout
        GC = [torch.norm(h.lstm.weight_ih_l0, dim=0) for h in self.decoders]
        # print(GC, flush=True)
        return torch.stack(GC)

    def _adaptive_threshold(self, off_diagonal_norms):
        """GMM(2) crossover threshold on the off-diagonal strengths."""
        x = np.asarray(off_diagonal_norms).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42).fit(x)
        centers = sorted(gmm.means_.flatten())
        noise_idx = int(np.argmin(gmm.means_.flatten()))
        signal_idx = int(np.argmax(gmm.means_.flatten()))
        lo, hi = centers[0], centers[1]
        grid = np.linspace(lo, hi, 1000).reshape(-1, 1)
        probs = gmm.predict_proba(grid)
        dominance = probs[:, signal_idx] > probs[:, noise_idx]
        if np.any(dominance):
            return float(grid[int(np.argmax(dominance))][0])
        return float((lo + hi) / 2)

    def get_causal_matrix(self, threshold='adaptive'):
        GC = self._strength_matrix()
        if GC is None:
            g = torch.tensor(self.causal_graph).float()
            return g, g, None
        raw = GC.detach().cpu()

        M = self.num_series
        if threshold == 'adaptive':
            off = [raw[i, j].item() for i in range(M) for j in range(M) if i != j]
            thr = self._adaptive_threshold(off)
        else:
            thr = float(threshold)
        binary = (raw > thr).float()
        return raw.float(), binary, thr

class ErrorCompensation(nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
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
        h_0 = torch.zeros(1, E.shape[0], self.hidden_size, device=E.device)
        if mode == 'train':
            _, hidden_out = self.encoder(E, h_0)
            mu = self.fc_mu(hidden_out)
            logvar = self.fc_logvar(hidden_out)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            z = self.tanh(self.fc_hidden(z))
            E_pred, _ = self.decoder(E, z)
            return self.fc(E_pred), mu, logvar
        if mode == 'inference':
            assert future > 0
            _, hidden_out = self.encoder(E, h_0)
            E_pred = None
            for idx in range(future):
                op, hidden_out = self.decoder(E[:, -1:, :], hidden_out)
                E_t = self.fc(op)
                E_pred = E_t if idx == 0 else torch.cat([E_pred, E_t], dim=1)
            return E_pred