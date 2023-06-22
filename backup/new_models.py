import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, device, dropout, conv_model):
        super(GNN, self).__init__()
        self.embedding = None
        if conv_model == 'GATv2':
            self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout).to(
                device)
            self.conv2 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, concat=True, dropout=dropout).to(
                device)
        if conv_model == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout).to(device)
            self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=True, dropout=dropout).to(
                device)
        if conv_model == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels).to(device)
            self.conv2 = GCNConv(hidden_channels, out_channels).to(device)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        self.embedding = x
        return x

    def get_embedding(self):
        return self.embedding


"""
==============================================================================
Variational autoEncoder
==============================================================================
"""


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(p=0.1)

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.dropout(self.bn1(self.act(self.fc1(x))))

        mu = self.mu(h)
        log_sigma = self.log_sigma(h).clamp(-10, 10)

        return mu, log_sigma


"""
==============================================================================
Linear Decoder
==============================================================================
"""


class LDEC(nn.Module):
    def __init__(self, num_cell, num_modality, emd_size, num_topics):
        super(LDEC, self).__init__()

        # num_topics x emd_size
        self.alpha = nn.Parameter(torch.randn(num_topics, emd_size))
        # emd_size x num_modality
        self.rho = nn.Parameter(torch.randn(emd_size, num_modality))

        self.batch_bias = nn.Parameter(torch.randn(num_cell, num_modality))
        self.Topic_by_modality = None

    def forward(self, theta, rho_matrix=None, cross_prediction=True):
        # num_topics x num_modality
        if rho_matrix is None:
            self.Topic_by_modality = torch.mm(self.alpha, self.rho)
        else:
            self.rho = nn.Parameter(rho_matrix)
            self.Topic_by_modality = torch.mm(self.alpha, self.rho)
        # theta: cell x num_topics
        # recon: cell x num_modality
        recon = torch.mm(theta, self.Topic_by_modality)
        # recon += self.batch_bias[batch_indices]
        if cross_prediction:
            X_hat = F.softmax(recon, dim=-1)
        else:
            X_hat = F.log_softmax(recon, dim=-1)

        # X_hat: cell x num_modality
        return X_hat
