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
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim, eps=1e-5, momentum=0.1),
            # nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            # nn.Dropout(p=dropout_rate),
            # nn.BatchNorm1d(output_dim, eps=1e-5, momentum=0.1),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, device, dropout, conv_model):
        super(GNN, self).__init__()
        self.embedding = None
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

        x = F.leaky_relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        self.embedding = x
        return x

    def backward(self, loss):
        self.loss = loss
        self.loss.backward()
        return self.loss

    def get_embedding(self):
        return self.embedding


class GAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, device, dropout, conv_model):
        super(GAE, self).__init__()
        self.embedding = None
        if conv_model == 'GATv2':
            self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
            self.conv2 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout)
            self.conv3 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout)
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
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.conv1(x, adj)
        self.embedding = hidden1
        return self.conv2(hidden1, adj), self.conv3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)

        z = self.reparameterize(mu, logvar)

        return self.dc(z), mu, logvar

    def get_embedding(self):
        return self.embedding


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward_all(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward(self, z, edge_index, sigmoid=True):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


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
            nn.ReLU(),
            nn.BatchNorm1d(emb_size, eps=1e-5, momentum=0.1),
            # nn.Dropout(p=0.1),

            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.BatchNorm1d(emb_size, eps=1e-5, momentum=0.1),
            nn.Dropout(p=0.1)
        )

        self.mu = nn.Linear(emb_size, num_topics, bias=True)
        self.log_sigma = nn.Linear(emb_size, num_topics, bias=True)

    def forward(self, x):
        h = self.mlp(x)

        mu = self.mu(h)
        log_sigma = self.log_sigma(h)
        log_sigma = 2 * log_sigma
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

        self.alphas = nn.Linear(emd_size, num_topics, bias=False)
        self.rho = nn.Parameter(torch.randn(num_modality, emd_size))
        self.batch_bias = nn.Parameter(torch.randn(batch_szie, num_modality))
        self.beta = None

    def forward(self, theta, matrix=None):
        if matrix is None:  # remove GNN
            beta = F.softmax(self.alphas(self.rho), dim=0).transpose(1, 0)
        else:
            beta = F.softmax(self.alphas(matrix), dim=0).transpose(1, 0)

        self.beta = beta
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        batch_bias = F.log_softmax(self.batch_bias, dim=-1)
        # preds += batch_bias
        return preds

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


"""
==============================================================================
Previous Model
==============================================================================
"""


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=1, concat=True, dropout=0.2).to('cuda')
        self.conv2 = GATv2Conv(hidden_channels, out_channels, heads=1, concat=True, dropout=0.2).to('cuda')
        # self.conv1 = GCNConv(in_channels, hidden_channels).to('cuda')
        # self.conv2 = GCNConv(hidden_channels, out_channels).to('cuda')
        self.embedding = None
        self.loss = None

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        # print(x.size())
        # x = F.relu(self.conv2(x, edge_index))
        x = self.conv2(x, edge_index)
        # x = F.dropout(x, training=self.training)
        self.embedding = x
        return x

    def backward(self, loss):
        self.loss = loss
        self.loss.backward()
        return self.loss

    def get_embedding(self):
        return self.embedding


class GNN_ETM(nn.Module):
    def __init__(self, num_topics, num_gene, num_peak, t_hidden_size, emb_size,
                 theta_act, enc_drop=0.5, use_gcn=False, feature_matrix=None, edge_index=None):
        super(GNN_ETM, self).__init__()
        # define hyperparameters
        self.num_topics = num_topics
        self.num_gene = num_gene
        self.num_peak = num_peak
        self.t_hidden_size = t_hidden_size
        self.rho_size = emb_size
        self.eta_size = emb_size
        self.graph_feature_size = emb_size
        self.theta_act = self.get_activation(theta_act)
        self.enc_drop = enc_drop
        self.t_drop = nn.Dropout(enc_drop)
        self.feature_matrix = feature_matrix
        self.edge_index = edge_index
        self.use_gcn = use_gcn

        ## define the word embedding matrix \rho \eta
        self.rho = nn.Parameter(torch.randn(num_gene, self.rho_size)).to('cuda')
        self.eta = nn.Parameter(torch.randn(num_peak, self.eta_size)).to('cuda')

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(self.rho_size, num_topics, bias=False)
        self.alphas_star = nn.Linear(self.eta_size, num_topics, bias=False)

        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta_gene = nn.Sequential(
            nn.Linear(num_gene, t_hidden_size),
            self.theta_act,
            nn.BatchNorm1d(t_hidden_size, eps=1e-5, momentum=0.1),
            # nn.Dropout(p=0.3),

            nn.Linear(t_hidden_size, t_hidden_size),
            self.theta_act,
            nn.BatchNorm1d(t_hidden_size, eps=1e-5, momentum=0.1),
        )

        self.q_theta_peak = nn.Sequential(
            nn.Linear(num_peak, t_hidden_size),
            self.theta_act,
            nn.BatchNorm1d(t_hidden_size, eps=1e-5, momentum=0.1),
            # nn.Dropout(p=0.3),

            nn.Linear(t_hidden_size, t_hidden_size),
            self.theta_act,
            nn.BatchNorm1d(t_hidden_size, eps=1e-5, momentum=0.1),
        )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

        # conv layer
        self.conv = GCN(self.graph_feature_size, self.graph_feature_size * 2, self.graph_feature_size).to('cuda')
        self.optimizer2 = optim.AdamW(self.conv.parameters(), lr=0.001)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act

    def split_tensor(self, tensor, num_rows):
        """
        split (gene+peak) x emb tensor to gene x emb tensor and peak x emb tensor
        """
        if num_rows >= tensor.shape[0]:
            raise ValueError("num_rows should be less than tensor's number of rows")

        top_matrix = tensor[:num_rows, :]
        bottom_matrix = tensor[num_rows:, :]

        return top_matrix, bottom_matrix

    # theta ~ mu + std N(0,1)
    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows, flag):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        if flag == "peak":
            q_theta = self.q_theta_peak(bows)
        if flag == "gene":
            q_theta = self.q_theta_gene(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)

        # KL[q(theta)||p(theta)] = lnq(theta) - lnp(theta)
        logsigma_theta = 2 * logsigma_theta

        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self, flag, matrix=None):
        ## softmax over vocab dimension
        if flag == "peak":
            if matrix == None:
                beta = F.softmax(self.alphas_star(self.eta), dim=0).transpose(1, 0)
            else:
                beta = F.softmax(self.alphas_star(matrix), dim=0).transpose(1, 0)
        if flag == "gene":
            if matrix == None:
                beta = F.softmax(self.alphas(self.rho), dim=0).transpose(1, 0)
            else:
                beta = F.softmax(self.alphas(matrix), dim=0).transpose(1, 0)
        return beta

    def get_theta(self, normalized_bows_gene, normalized_bows_peak):
        mu_theta_d, logsigma_theta_d, kld_theta_d = self.encode(normalized_bows_gene, "gene")
        mu_theta_j, logsigma_theta_j, kld_theta_j = self.encode(normalized_bows_peak, "peak")

        z_d = self.reparameterize(mu_theta_d, logsigma_theta_d)
        theta_d = F.softmax(z_d, dim=-1)

        z_j = self.reparameterize(mu_theta_j, logsigma_theta_j)
        theta_j = F.softmax(z_j, dim=-1)

        return theta_d, kld_theta_d, theta_j, kld_theta_j

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds

    def forward(self, Gene, Gene_normalized, Peak, Peak_normalized, theta=None, aggregate=True):
        ## get \theta
        # if theta is None:
        #     theta_d, kld_theta_d, theta_j, kld_theta_j = self.get_theta(Gene_normalized, Peak_normalized)
        # else:
        #     kld_theta_d = None
        #     kld_theta_j = None
        theta_d, kld_theta_d, theta_j, kld_theta_j = self.get_theta(Gene_normalized, Peak_normalized)

        kld_theta = kld_theta_d + kld_theta_j

        ## get \beta
        if self.use_gcn:
            emb_matrix = self.feature_matrix
            if self.conv.get_embedding() != None:
                emb_matrix = self.conv.get_embedding()

            emb_matrix = self.theta_act(emb_matrix)
            eta_matrix, rho_matrix = self.split_tensor(emb_matrix, self.num_peak)
            beta_Gene = self.get_beta("gene", rho_matrix)
            beta_Peak = self.get_beta("peak", eta_matrix)
        else:
            beta_Gene = self.get_beta("gene")
            beta_Peak = self.get_beta("peak")

        ## get prediction loss
        # preds_Gene = self.decode(theta_all, beta_Gene)
        # recon_loss_Gene = -(preds_Gene * Gene).sum(-1)
        #
        # preds_Peak = self.decode(theta_all, beta_Peak)
        # recon_loss_Peak = -(preds_Peak * Peak).sum(-1)

        preds_Gene = self.decode(theta_d, beta_Gene)
        recon_loss_Gene = -(preds_Gene * Gene).sum(-1)

        preds_Peak = self.decode(theta_j, beta_Peak)
        recon_loss_Peak = -(preds_Peak * Peak).sum(-1)

        recon_loss = recon_loss_Gene + recon_loss_Peak
        if aggregate:
            recon_loss_Gene = recon_loss_Gene.mean()
            recon_loss_Peak = recon_loss_Peak.mean()
            recon_loss = recon_loss.mean()

        print(recon_loss, kld_theta)
        return recon_loss, kld_theta

    def train_gcn(self):
        self.conv.train()
        self.optimizer2.zero_grad()
        # self.conv.zero_grad()
        self.conv(self.feature_matrix, self.edge_index)

    def gcn_back(self):
        # torch.nn.utils.clip_grad_norm_(self.conv.parameters(), 2.0)
        self.optimizer2.step()
