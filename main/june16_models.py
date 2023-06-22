import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool
from torch import optim
from torch.nn import functional as F
from torch_geometric.nn import BatchNorm

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean

#
# class GCN(torch.nn.Module):
#     def __init__(self, input_channels, hidden_channels, output_channels, device):
#         super(GCN, self).__init__()
#
#         self.conv1 = GCNConv(input_channels, hidden_channels).to(device)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels).to(device)
#         self.conv3 = GCNConv(hidden_channels, hidden_channels).to(device)
#         self.lin = Linear(hidden_channels, output_channels).to(device)
#
#     def forward(self, x, edge_index, batch):
#         # 1. Obtain node embeddings
#         x = F.relu(self.conv1(x, edge_index))
#         # x = F.relu(self.conv2(x, edge_index))
#         # x = self.conv3(x, edge_index)
#
#         # 2. Readout layer
#         # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#
#         # 3. Apply a final classifier
#         # x = F.dropout(x, p=0.5, training=self.training)
#         # x = self.lin(x)
#
#         return x


"""
==============================================================================
Graph neural network to learn gene embedding
==============================================================================
"""


class GNN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_heads, device, dropout, conv_model):
        super(GNN, self).__init__()
        self.embedding = None
        self.input = input_channels
        self.hidden = hidden_channels
        self.output = output_channels
        if conv_model == 'GATv2':
            self.conv1 = GATv2Conv(input_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout).to(
                device)
            self.bn1 = BatchNorm(hidden_channels)
            self.conv2 = GATv2Conv(hidden_channels * num_heads, output_channels, heads=1, concat=True,
                                   dropout=dropout).to(
                device)
        if conv_model == 'GAT':
            self.conv1 = GATConv(input_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout).to(
                device)
            self.conv2 = GATConv(hidden_channels * num_heads, output_channels, heads=1, concat=True,
                                 dropout=dropout).to(
                device)
        if conv_model == 'GCN':
            self.conv1 = GCNConv(input_channels, hidden_channels).to(device)
            self.conv2 = GCNConv(hidden_channels, hidden_channels).to(device)
        self.conv3 = None
        self.device = device

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        self.embedding = x
        # x = global_mean_pool(x, batch)
        # print(x.shape)
        # self.conv3 = GCNConv(self.hidden, self.output).to(self.device)
        # x =(self.conv3(F.relu(x),edge_index))

        return x

    def get_embedding(self):
        return self.embedding



"""
==============================================================================
MLP to transfer gene by embedding matrix to gene by cell type matrix
==============================================================================
"""


class MLP(nn.Module):
    def __init__(self, num_of_embedding, hidden_channels, num_of_cell_type, dropout=0.2):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_of_embedding, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, num_of_cell_type),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


"""
==============================================================================
PretrainModel
==============================================================================
"""


class PretrainModel(nn.Module):
    def __init__(self, gnn, mlp):
        super(PretrainModel, self).__init__()
        self.gnn = gnn
        self.mlp = mlp
        self.emb = None

    def forward(self, x, edge_index):
        x = self.gnn(x, edge_index)
        self.emb = x
        x = self.mlp(x)
        return x

    def get_embedding(self):
        return self.emb


"""
==============================================================================
GNN_ETM Model
==============================================================================
"""


class GNN_ETM(nn.Module):
    def __init__(self, num_topics, num_gene, num_peak, t_hidden_size, t_hidden_size_peak, rho_size, eta_size,
                 emb_size, graph_feature_size, theta_act, device, enc_drop=0.5,
                 use_gnn=False, gnn_model=None, feature_matrix=None, edge_index=None):
        super(GNN_ETM, self).__init__()
        # define hyperparameters
        self.num_topics = num_topics
        self.num_gene = num_gene
        self.num_peak = num_peak
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.eta_size = eta_size
        self.emb_size = emb_size
        self.graph_feature_size = graph_feature_size
        self.theta_act = self.get_activation(theta_act)
        self.enc_drop = enc_drop
        self.t_drop = nn.Dropout(enc_drop)
        self.feature_matrix = feature_matrix
        self.edge_index = edge_index
        self.use_gnn = use_gnn

        ## define the word embedding matrix \rho \eta
        self.rho = nn.Parameter(torch.randn(num_gene, rho_size)).to(device)
        self.eta = nn.Parameter(torch.randn(num_peak, eta_size)).to(device)

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)
        self.alphas_star = nn.Linear(eta_size, num_topics, bias=False)

        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta_gene = nn.Sequential(
            nn.Linear(num_gene, t_hidden_size),
            self.theta_act,
            # nn.BatchNorm1d(t_hidden_size, eps=1e-5, momentum=0.1),
            # nn.Dropout(p=0.3),

            nn.Linear(t_hidden_size, t_hidden_size),
            self.theta_act,
            # nn.BatchNorm1d(t_hidden_size, eps=1e-5, momentum=0.1),
        )

        self.q_theta_peak = nn.Sequential(
            nn.Linear(num_peak, t_hidden_size_peak),
            self.theta_act,
            # nn.BatchNorm1d(t_hidden_size_peak, eps=1e-5, momentum=0.1),
            # nn.Dropout(p=0.3),

            nn.Linear(t_hidden_size_peak, t_hidden_size),
            self.theta_act,
            # nn.BatchNorm1d(t_hidden_size, eps=1e-5, momentum=0.1),
        )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

        # conv layer in_channels, hidden_channels, out_channels, num_heads
        self.conv = gnn_model
        self.optimizer2 = optim.SGD(gnn_model.parameters(), lr=0.001, weight_decay=1.2e-6)

    def set_edge_index(self, edge_index, feature):
        self.edge_index = edge_index
        # self.feature_matrix = feature

    def get_feature(self):
        return self.feature_matrix

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
        beta = None
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
        return res

    def get_reconstruction_loss(self, raw_matrix, pred_matrix, metric):
        loss = None
        if metric == 'cos':
            pred_norm = torch.nn.functional.normalize(pred_matrix, dim=-1)
            raw_norm = torch.nn.functional.normalize(raw_matrix, dim=-1)
            loss = 1 - torch.cosine_similarity(pred_norm, raw_norm).sum(-1)
        if metric == 'focal':
            gamma = 2
            loss = -(raw_matrix * (1 - pred_matrix).pow(gamma) * pred_matrix.log()).sum(-1)
        if metric == 'kl':
            loss = F.kl_div(torch.log(pred_matrix), raw_matrix, reduction='batchmean')
        if metric == 'bce':
            loss = -(raw_matrix * torch.log(pred_matrix) + (1 - raw_matrix) * torch.log(1 - pred_matrix)).sum(-1)
        if metric == 'nll':
            loss = -(torch.log(pred_matrix) * raw_matrix).sum(-1)
        return loss

    def forward(self, Gene, Gene_normalized, Peak, Peak_normalized, theta=None, aggregate=True):
        ## get \theta
        if theta is None:
            theta_d, kld_theta_d, theta_j, kld_theta_j = self.get_theta(Gene_normalized, Peak_normalized)
        else:
            kld_theta_d = None
            kld_theta_j = None

        kld_theta = kld_theta_d + kld_theta_j

        ## get \beta
        if self.use_gnn:
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
        preds_Gene = self.decode(theta_d, beta_Gene)
        # recon_loss_Gene = -(preds_Gene * Gene).sum(-1)

        # mse, cos, euclidean_distances, ssim, kl, nll
        # recon_metric = 'nll'
        recon_loss_Gene = self.get_reconstruction_loss(raw_matrix=Gene, pred_matrix=preds_Gene, metric='nll')

        preds_Peak = self.decode(theta_j, beta_Peak)
        # recon_loss_Peak = -(preds_Peak * Peak).sum(-1)
        recon_loss_Peak = self.get_reconstruction_loss(raw_matrix=Peak, pred_matrix=preds_Peak, metric='nll')

        recon_loss = recon_loss_Gene + recon_loss_Peak
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta

    def train_gcn(self):
        self.conv.train()
        self.optimizer2.zero_grad()
        self.conv.zero_grad()
        reconstructed_adjacency = self.conv(self.feature_matrix, self.edge_index)

    def gcn_back(self):
        torch.nn.utils.clip_grad_norm_(self.conv.parameters(), 10.0)
        self.optimizer2.step()

class ETM(nn.Module):
    def __init__(self, num_topics, num_gene, num_peak, t_hidden_size, t_hidden_size_peak, rho_eta,
                 emb_size, graph_feature_size, theta_act, device, enc_drop=0.5):
        super(ETM, self).__init__()
        # define hyperparameters
        self.num_topics = num_topics
        self.num_gene = num_gene
        self.num_peak = num_peak
        self.t_hidden_size = t_hidden_size
        self.rho_eta= rho_eta
        self.graph_feature_size = graph_feature_size
        self.theta_act = self.get_activation(theta_act)
        self.enc_drop = enc_drop
        self.t_drop = nn.Dropout(enc_drop)


        ## define the word embedding matrix \rho \eta
        self.rho = nn.Parameter(torch.randn(num_gene, emb_size)).to(device)
        self.eta = nn.Parameter(torch.randn(num_peak, emb_size)).to(device)

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(emb_size, num_topics, bias=False)
        self.alphas_star = nn.Linear(emb_size, num_topics, bias=False)

        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta_gene = nn.Sequential(
            nn.Linear(num_gene, t_hidden_size),
            self.theta_act,
            # nn.BatchNorm1d(t_hidden_size, eps=1e-5, momentum=0.1),
            # nn.Dropout(p=0.3),

            nn.Linear(t_hidden_size, t_hidden_size),
            self.theta_act,
            # nn.BatchNorm1d(t_hidden_size, eps=1e-5, momentum=0.1),
        )

        self.q_theta_peak = nn.Sequential(
            nn.Linear(num_peak, t_hidden_size_peak),
            self.theta_act,
            # nn.BatchNorm1d(t_hidden_size_peak, eps=1e-5, momentum=0.1),
            # nn.Dropout(p=0.3),

            nn.Linear(t_hidden_size_peak, t_hidden_size),
            self.theta_act,
            # nn.BatchNorm1d(t_hidden_size, eps=1e-5, momentum=0.1),
        )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

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
        beta = None
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
        return res

    def get_reconstruction_loss(self, raw_matrix, pred_matrix, metric):
        loss = None
        if metric == 'cos':
            pred_norm = torch.nn.functional.normalize(pred_matrix, dim=-1)
            raw_norm = torch.nn.functional.normalize(raw_matrix, dim=-1)
            loss = 1 - torch.cosine_similarity(pred_norm, raw_norm).sum(-1)
        if metric == 'focal':
            gamma = 2
            loss = -(raw_matrix * (1 - pred_matrix).pow(gamma) * pred_matrix.log()).sum(-1)
        if metric == 'kl':
            loss = F.kl_div(torch.log(pred_matrix), raw_matrix, reduction='batchmean')
        if metric == 'bce':
            loss = -(raw_matrix * torch.log(pred_matrix) + (1 - raw_matrix) * torch.log(1 - pred_matrix)).sum(-1)
        if metric == 'nll':
            loss = -(torch.log(pred_matrix) * raw_matrix).sum(-1)
        return loss

    def forward(self, Gene, Gene_normalized, Peak, Peak_normalized, rho_eta, theta=None, aggregate=True):
        ## get \theta
        if theta is None:
            theta_d, kld_theta_d, theta_j, kld_theta_j = self.get_theta(Gene_normalized, Peak_normalized)
        else:
            kld_theta_d = None
            kld_theta_j = None

        kld_theta = kld_theta_d + kld_theta_j

        ## get \beta

        emb_matrix = rho_eta
        eta_matrix, rho_matrix = self.split_tensor(emb_matrix, self.num_peak)
        beta_Gene = self.get_beta("gene", rho_matrix)
        beta_Peak = self.get_beta("peak", eta_matrix)


        ## get prediction loss
        preds_Gene = self.decode(theta_d, beta_Gene)
        # recon_loss_Gene = -(preds_Gene * Gene).sum(-1)

        # mse, cos, euclidean_distances, ssim, kl, nll
        # recon_metric = 'nll'
        recon_loss_Gene = self.get_reconstruction_loss(raw_matrix=Gene, pred_matrix=preds_Gene, metric='nll')

        preds_Peak = self.decode(theta_j, beta_Peak)
        # recon_loss_Peak = -(preds_Peak * Peak).sum(-1)
        recon_loss_Peak = self.get_reconstruction_loss(raw_matrix=Peak, pred_matrix=preds_Peak, metric='nll')

        recon_loss = recon_loss_Gene + recon_loss_Peak
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta
