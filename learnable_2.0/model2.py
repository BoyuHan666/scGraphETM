import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch import optim
from torch.nn import functional as F

"""
==============================================================================
New Model
==============================================================================
"""


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(hidden_dim, eps=1e-5, momentum=0.1),
            # nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout_rate),
            # nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class DEC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(DEC, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim, eps=1e-5, momentum=0.1),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_rate),
            # nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_gene, num_peak, dropout, conv_model, node2vec=None):
        super(GNN, self).__init__()
        self.embedding = None
        if node2vec is None:
            self.initial_embedding = nn.Parameter(torch.randn((num_gene+num_peak, in_channels), requires_grad=True))
        else:
            self.initial_embedding = nn.Parameter(node2vec, requires_grad=True)
        if conv_model == 'GATv2':
            self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
            self.conv2 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, concat=True, dropout=dropout)
        if conv_model == 'Transformer':
            self.conv1 = TransformerConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
            self.conv2 = TransformerConv(hidden_channels * num_heads, out_channels, heads=1, concat=True,
                                         dropout=dropout)
        if conv_model == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
            self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=True, dropout=dropout)
        if conv_model == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.initial_embedding*(1+x)
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        # x, self.a1 = self.conv1(x, edge_index, return_attention_weights=True)
        # x = F.leaky_relu(x)
        # x, self.a2 = self.conv2(x, edge_index, return_attention_weights=True)
        # x = F.leaky_relu(x)
        self.embedding = x
        return x

    def backward(self, loss):
        self.loss = loss
        self.loss.backward()
        return self.loss

    def get_embedding(self):
        return self.embedding

    def get_attention(self):
        return self.a1, self.a2


"""
==============================================================================
Variational autoEncoder
==============================================================================
"""


class VAE(nn.Module):
    def __init__(self, num_modality, emb_size, num_topics):
        super(VAE, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_modality, emb_size),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(emb_size, eps=1e-5, momentum=0.1),
            # nn.Dropout(p=0.1),

            nn.Linear(emb_size, emb_size),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(emb_size, eps=1e-5, momentum=0.1),
            nn.Dropout(p=0.1)
        )

        self.mu = nn.Linear(emb_size, num_topics, bias=True)
        self.log_sigma = nn.Linear(emb_size, num_topics, bias=True)

    def forward(self, x):
        h = self.mlp(x)

        mu = self.mu(h)
        log_sigma = self.log_sigma(h).clamp(-10, 10)

        kl_theta = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=-1).mean()

        return mu, log_sigma, kl_theta


"""
==============================================================================
Linear Decoder
==============================================================================
"""


class LDEC(nn.Module):
    def __init__(self, num_modality, emd_size, num_topics, batch_szie):
        super(LDEC, self).__init__()

        # self.alphas = nn.Linear(emd_size, num_topics, bias=False)
        self.alphas = nn.Parameter(torch.randn(num_topics, emd_size))
        self.rho = nn.Parameter(torch.randn(num_modality, emd_size))
        self.batch_bias = nn.Parameter(torch.randn(batch_szie, num_modality))
        self.beta = None

    def forward(self, theta, matrix=None, imputation=False, scale=1):
        if matrix is None:
            # beta = F.softmax(self.alphas(self.rho), dim=0).transpose(1, 0)
            beta = self.alphas @ self.rho.T
        else:
            # beta = F.softmax(self.alphas(matrix), dim=0).transpose(1, 0)
            beta = self.alphas @ matrix.T

        # print(theta.shape, beta.shape)
        self.beta = beta
        res = torch.mm(theta, beta)
        if not imputation:
            # res = torch.log(res + 1e-6)
            res = F.log_softmax(res, dim=-1)
        else:
            # res = res + 1e-6
            res = F.softmax(res, dim=-1)

        batch_bias = F.log_softmax(self.batch_bias, dim=-1)
        # preds += batch_bias
        return res

    def get_beta(self):
        return self.beta


"""
==============================================================================
POG Decoder
==============================================================================
"""


class POG_DEC(nn.Module):
    def __init__(self, num_modality1, num_modality2, emd_size, num_topics, batch_szie):
        super(POG_DEC, self).__init__()

        self.alphas = nn.Linear(emd_size, num_topics, bias=False)
        self.rho = nn.Parameter(torch.randn(num_modality1, emd_size))
        self.batch_bias = nn.Parameter(torch.randn(batch_szie, num_modality1))
        self.beta1 = None

        self.alphas2 = nn.Linear(emd_size, num_topics, bias=False)
        self.rho2 = nn.Parameter(torch.randn(num_modality2, emd_size))
        self.batch_bias2 = nn.Parameter(torch.randn(batch_szie, num_modality2))
        self.beta2 = None

    def forward(self, theta, matrix1=None, matrix2=None):
        if matrix1 is None:
            beta = F.softmax(self.alphas(self.rho), dim=0).transpose(1, 0)
        else:
            beta = F.softmax(self.alphas(matrix1), dim=0).transpose(1, 0)
        self.beta1 = beta
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        batch_bias = F.log_softmax(self.batch_bias, dim=-1)
        # preds += batch_bias

        if matrix2 is None:
            beta2 = F.softmax(self.alphas2(self.rho2), dim=0).transpose(1, 0)
        else:
            beta2 = F.softmax(self.alphas2(matrix2), dim=0).transpose(1, 0)
        self.beta2 = beta2
        res2 = torch.mm(theta, beta2)
        preds2 = torch.log(res2 + 1e-6)
        batch_bias2 = F.log_softmax(self.batch_bias2, dim=-1)
        # preds2 += batch_bias2

        return preds, preds2

    def get_beta(self):
        return self.beta1, self.beta2
