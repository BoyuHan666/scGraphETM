import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch import optim
from torch.nn import functional as F
from sklearn.metrics import adjusted_rand_score

import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

""""'
==============================================================================
hyper-parameters for training data
==============================================================================
"""
# 9631 cell × 29095 gene
rna_path = "./data/10x-Multiome-Pbmc10k-RNA.h5ad"
# 9631 cell × 107194 peak
atac_path = "./data/10x-Multiome-Pbmc10k-ATAC.h5ad"

# 32231 cell × 21478 peak (Mouse skin)
# rna_path = "./not_match_data/Ma-2020-RNA.h5ad"
# 32231 cell × 340341 peak (Mouse skin)
# atac_path = "./not_match_data/Ma-2020-ATAC.h5ad"
# 1: all cell, all gene, all peak
# 2: user_defined num_of_cell, num_of_gene, num_of_peak
match_cell = False
shuffle = False
random_seed = 1

set_number = 2
rate1 = 0.6
rate2 = -2
dis_rate = -2 # for the rate of gene-gene, peak-peak cor that not count to cor matrix

num_of_cell = 2000
num_of_gene = 2000
num_of_peak = 2000
use_pretrain = False

""""'
==============================================================================
hyper-parameters for pretraining GNN model
==============================================================================
"""
pretrain_num_epochs = 100
freq = 10
graph_feature_size = 512
mlp_hidden_channels = graph_feature_size * 2

"""
==============================================================================
hyper-parameters for training GNN_ETM model
==============================================================================
"""
num_of_topic = 60
t_hidden_size = graph_feature_size
t_hidden_size_peak = graph_feature_size * 2
rho_size = graph_feature_size
eta_size = graph_feature_size
emb_size = graph_feature_size
ari_freq = 200
num_of_epochs = 3000
# one from GATv2, GAT, GCN
conv_model = 'GCN'
# nll, kl, bce
recon_metric = 'nll'

# plot_path_rel = "./Ma_plot/"
plot_path_rel = "./han_plot/"

"""
==============================================================================
Graph neural network to learn gene embedding
==============================================================================
"""


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dropout=0.2, conv_model='GATv2'):
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
MLP to transfer gene by embedding matrix to gene by cell type matrix
==============================================================================
"""


class MLP(nn.Module):
    def __init__(self, num_of_embedding, hidden_channels, num_of_cell_type, dropout=0.2):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_of_embedding, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels, eps=1e-5, momentum=0.1),
            nn.Dropout(p=dropout),

            nn.Linear(hidden_channels, num_of_cell_type),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


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


class GNN_ETM(nn.Module):
    def __init__(self, num_topics, num_gene, num_peak, t_hidden_size, t_hidden_size_peak, rho_size, eta_size,
                 emb_size, theta_act, device, enc_drop=0.5,
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
        self.optimizer2 = optim.Adam(self.conv.parameters(), lr=0.001, weight_decay=1.2e-6)

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
        recon_loss_Gene = self.get_reconstruction_loss(raw_matrix=Gene, pred_matrix=preds_Gene, metric=recon_metric)

        preds_Peak = self.decode(theta_j, beta_Peak)
        # recon_loss_Peak = -(preds_Peak * Peak).sum(-1)
        recon_loss_Peak = self.get_reconstruction_loss(raw_matrix=Peak, pred_matrix=preds_Peak, metric=recon_metric)

        recon_loss = recon_loss_Gene + recon_loss_Peak
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta

    def train_gcn(self):
        self.conv.train()
        self.optimizer2.zero_grad()
        # self.conv.zero_grad()
        self.conv(self.feature_matrix, self.edge_index)

    def gcn_back(self):
        # torch.nn.utils.clip_grad_norm_(self.conv.parameters(), 2.0)
        self.optimizer2.step()


def evaluate_ari(cell_embed, adata):
    """
        This function is used to evaluate ARI using the lower-dimensional embedding
        cell_embed of the single-cell data
        :param cell_embed: a NxK single-cell embedding generated from NMF or scETM
        :param adata: single-cell AnnData data object (default to to mp_anndata)
        :return: ARI score of the clustering results produced by Louvain
    """
    adata.obsm['cell_embed'] = cell_embed
    sc.pp.neighbors(adata, use_rep="cell_embed", n_neighbors=30)
    sc.tl.louvain(adata, resolution=0.15)
    ari = adjusted_rand_score(adata.obs['Celltype'], adata.obs['louvain'])
    return ari


"""
==============================================================================
scGCN - ETM Helper
==============================================================================
"""


# train the VAE for one epoch
def train_GCNscETM_helper(model, optimizer, RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized):
    # initialize the model and loss
    model.train()
    optimizer.zero_grad()
    model.zero_grad()

    # forward and backward pass
    model.train_gcn()
    nll, kl_theta = model(RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized)
    loss = nll + kl_theta
    loss.backward()  # Calculate and backprop gradients w.r.t. negative ELBO

    # clip gradients to 2.0 if it gets too large
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

    # update model to minimize negative ELBO
    model.gcn_back()
    optimizer.step()

    return torch.sum(loss).item()


# get sample encoding theta from the trained encoder network
def get_theta_GCN(model, RNA_tensor_normalized, ATAC_tensor_normalized):
    model.eval()
    with torch.no_grad():
        q_theta_gene = model.q_theta_gene(RNA_tensor_normalized)
        mu_theta_gene = model.mu_q_theta(q_theta_gene)
        theta_gene = F.softmax(mu_theta_gene, dim=-1)

        q_theta_peak = model.q_theta_peak(ATAC_tensor_normalized)
        mu_theta_peak = model.mu_q_theta(q_theta_peak)
        theta_peak = F.softmax(mu_theta_peak, dim=-1)

        theta = 0.5 * theta_gene + 0.5 * theta_peak
        # theta = theta_gene * theta_peak
        return theta, theta_gene, theta_peak


def train_GCNscETM(model, optimizer, RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized, adata,
                   adata2,
                   ari_freq, niter=100):
    perf = np.ndarray(shape=(niter, 2), dtype='float')
    num_of_ari = (int)(niter / ari_freq)
    ari_perf = np.ndarray(shape=(num_of_ari, 4), dtype='float')
    best_ari = 0
    best_theta = None
    for i in range(niter):
        NELBO = train_GCNscETM_helper(model, optimizer, RNA_tensor, RNA_tensor_normalized, ATAC_tensor,
                                      ATAC_tensor_normalized)
        theta, theta_gene, theta_peak = get_theta_GCN(model, RNA_tensor_normalized, ATAC_tensor_normalized)
        perf[i, 0] = i
        perf[i, 1] = NELBO
        if (i % ari_freq == 0):
            ari = evaluate_ari(theta.to('cpu'), adata)
            ari1 = evaluate_ari(theta_gene.to('cpu'), adata)
            ari2 = evaluate_ari(theta_peak.to('cpu'), adata)
            idx = (int)(i / ari_freq)
            ari_perf[idx, 0] = idx
            ari_perf[idx, 1] = ari
            ari_perf[idx, 2] = ari1
            ari_perf[idx, 3] = ari2
            print("iter: " + str(i) + " ari: " + str(ari))
            if (best_ari < ari):
                best_ari = ari
                best_theta = theta
        else:
            if (i % 100 == 0):
                print("iter: " + str(i))
    return model, perf, ari_perf, best_ari, best_theta


"""
==============================================================================
Plotting
==============================================================================
"""


def monitor_perf2(perf, ari_perf, ari_freq, objective, path):
    niter = []
    ari_niter = []
    mse = []
    ari = []
    ari1 = []
    ari2 = []
    for i in range(len(perf)):
        niter.append(perf[i][0])
        mse.append(perf[i][1])
    for i in range(len(ari_perf)):
        ari_niter.append(ari_perf[i][0] * ari_freq)
        ari.append(ari_perf[i][1])
        ari1.append(ari_perf[i][2])
        ari2.append(ari_perf[i][3])
    if (objective == "both"):
        plt.plot(ari_niter, ari, label="GNN_scETM", color='red')
        plt.plot(ari_niter, ari1, label="scETM_RNA", color='black')
        plt.plot(ari_niter, ari2, label="scETM_ATAC", color='blue')
        plt.legend()
        plt.title("ARI comparison on GCN_scETM and scETM")
        plt.xlabel("iter")
        plt.ylabel("ARI")
        plt.ylim(0, 1)
        # plt.show()
        plt.savefig(path + 'BOTH_ARI.png')
    if (objective == "NELBO"):
        plt.plot(niter, mse)
        plt.xlabel("iter")
        plt.ylabel("NELBO")
        # plt.show()
        plt.savefig(path + 'NELBO.png')
    if (objective == "ARI"):
        plt.plot(ari_niter, ari)
        plt.xlabel("iter")
        plt.ylabel("ARI")
        plt.show()
    if (objective == "ARI1"):
        plt.plot(ari_niter, ari1)
        plt.xlabel("iter")
        plt.ylabel("ARI1")
        plt.show()
    if (objective == "ARI2"):
        plt.plot(ari_niter, ari2)
        plt.xlabel("iter")
        plt.ylabel("ARI2")
        plt.show()


"""
============================================================================== 
Helper for avg gene expression by cell type
==============================================================================
"""


def get_cell_type_index(types, element):
    for i in range(len(types)):
        if types[i] == element:
            return i


def getAvgExpression(matrix, data, dic, celltype, cell_type_list):
    for i in range(len(cell_type_list)):
        cell_type_name = cell_type_list[i]
        cell_type_index = get_cell_type_index(celltype, cell_type_name)
        row = data[i] / dic[cell_type_name]
        # print(row.shape, matrix[cell_type_index].shape)
        matrix[cell_type_index] += row
    return matrix.tocsr()


def pretrain_emb_model(model, optimizer, epochs,
                       feature_matrix, edge_index, true_gene_expression,
                       feature_matrix_val=None, edge_index_val=None, true_gene_expression_val=None, freq=freq):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(feature_matrix, edge_index)

        loss = loss_fn(output, true_gene_expression)

        loss.backward()
        optimizer.step()

        if feature_matrix_val is not None:
            model.eval()
            with torch.no_grad():
                val_output = model(feature_matrix_val, edge_index_val)

                val_loss = loss_fn(val_output, true_gene_expression_val)

        if (epoch + 1) % freq == 0:
            if feature_matrix_val is None:
                print('Epoch [{}/{}], Training Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))
            else:
                print('Epoch [{}/{}], Training Loss: {:.4f} , Validation Loss: {:.4f}'.format(epoch + 1, epochs,
                                                                                              loss.item(),
                                                                                              val_loss.item()))
    return model.get_embedding()


def match_cell_names(scRNA_data, scATAC_data):
    # Get cell names from scRNA-seq data and remove "_RNA" suffix
    scRNA_cell_names = [name.replace("_RNA", "") for name in scRNA_data.obs_names]

    # Get cell names from scATAC-seq data and remove "_ATAC" suffix
    scATAC_cell_names = [name.replace("_ATAC", "") for name in scATAC_data.obs_names]

    # Find common cell names
    common_cell_names = set(scRNA_cell_names) & set(scATAC_cell_names)

    # Filter scRNA-seq data to include only common cell names
    filtered_scRNA_data = scRNA_data[[name in common_cell_names for name in scRNA_cell_names], :]

    # Filter scATAC-seq data to include only common cell names
    filtered_scATAC_data = scATAC_data[[name in common_cell_names for name in scATAC_cell_names], :]

    # Sort the data based on the common cell names
    scRNA_indices = [scRNA_cell_names.index(name) for name in common_cell_names]
    sorted_scRNA_data = scRNA_data[scRNA_indices, :]

    scATAC_indices = [scATAC_cell_names.index(name) for name in common_cell_names]
    sorted_scATAC_data = scATAC_data[scATAC_indices, :]

    return sorted_scRNA_data, sorted_scATAC_data


def shuffle_rows_anndata(adata, random_seed=123):
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(adata.n_obs)
    adata = adata[shuffled_indices]
    return adata


"""
==============================================================================
    Main 
==============================================================================
"""

if __name__ == '__main__':
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU device found")
    else:
        device = torch.device("cpu")
        print("No GPU found")

    """# **Load data**"""
    # # 9631 cell × 29095 gene
    # scRNA_adata = ad.read_h5ad("./data/10x-Multiome-Pbmc10k-RNA.h5ad")
    # # 9631 cell × 107194 peak
    # scATAC_adata = ad.read_h5ad("./data/10x-Multiome-Pbmc10k-ATAC.h5ad")

    scRNA_adata = ad.read_h5ad(rna_path)
    scATAC_adata = ad.read_h5ad(atac_path)

    # match cell name
    if match_cell:
        scRNA_adata, scATAC_adata = match_cell_names(scRNA_adata, scATAC_adata)

    # random shuffle
    if shuffle:
        scRNA_adata = shuffle_rows_anndata(scRNA_adata, random_seed=random_seed)
        scATAC_adata = shuffle_rows_anndata(scATAC_adata, random_seed=random_seed)

    # Preprocess data
    scATAC_adata_copy = scATAC_adata

    sc.pp.normalize_total(scATAC_adata, target_sum=1e4)
    sc.pp.log1p(scATAC_adata)
    sc.pp.highly_variable_genes(scATAC_adata)

    scRNA_adata_copy = scRNA_adata

    sc.pp.normalize_total(scRNA_adata, target_sum=1e4)
    sc.pp.log1p(scRNA_adata)

    # print(sum(scRNA_adata.var["highly_variable"]))
    # print(sum(scATAC_adata.var["highly_variable"]))

    index1 = scRNA_adata.var['highly_variable'].values
    scRNA_adata = scRNA_adata[:, index1].copy()

    index2 = scATAC_adata.var['highly_variable'].values
    scATAC_adata = scATAC_adata[:, index2].copy()

    # print(scRNA_adata)
    # print(scATAC_adata)
    if set_number == 'all':
        num_of_cell = scRNA_adata.n_obs
        num_of_gene = scRNA_adata.n_vars
        num_of_peak = scATAC_adata.n_vars
    else:
        num_of_cell = num_of_cell
        num_of_gene = num_of_gene
        num_of_peak = num_of_peak
    # train_num_of_gene = 1600
    # train_num_of_peak = 6400

    print(num_of_cell, num_of_gene, num_of_peak)
    print("----------------------")
    scATAC_1000 = scATAC_adata_copy[:, index2]
    scATAC_1000 = scATAC_1000.X[:num_of_cell, :num_of_peak].toarray()
    scATAC_1000_tensor = torch.from_numpy(scATAC_1000).to(device)

    scATAC_1000_tensor_normalized = (torch.from_numpy(scATAC_adata_copy.X[:num_of_cell, :num_of_peak].toarray())).to(
        device)
    scATAC_1000_mp_anndata = ad.AnnData(X=scATAC_adata_copy.X[:num_of_cell, :num_of_peak].toarray())
    scATAC_1000_mp_anndata.obs['Celltype'] = scATAC_adata_copy.obs['cell_type'].values[:num_of_cell]
    num_topics2 = len(scATAC_1000_mp_anndata.obs['Celltype'].values.unique())
    print(num_topics2)

    scRNA_1000 = scRNA_adata_copy[:, index1]
    # scRNA_1000 = scRNA_adata_copy[:, :]
    scRNA_1000 = scRNA_1000.X[:num_of_cell, :num_of_gene].toarray()
    scRNA_1000_tensor = torch.from_numpy(scRNA_1000).to(device)

    scRNA_1000_tensor_normalized = (torch.from_numpy(scRNA_adata_copy.X[:num_of_cell, :num_of_gene].toarray())).to(
        device)
    scRNA_1000_mp_anndata = ad.AnnData(X=scRNA_adata_copy.X[:num_of_cell, :num_of_gene].toarray())
    scRNA_1000_mp_anndata.obs['Celltype'] = scRNA_adata_copy.obs['cell_type'].values[:num_of_cell]
    num_topics = len(scRNA_1000_mp_anndata.obs['Celltype'].values.unique())
    print(num_topics)

    # Compute gene-gene correlation
    print("----------------------")
    print("Compute gene-gene correlation")
    gene_expression = scRNA_adata.X[:num_of_cell, :num_of_gene].toarray()
    correlation_matrix = np.corrcoef(gene_expression, rowvar=False)
    correlation_matrix_cleaned = np.nan_to_num(correlation_matrix, nan=0, posinf=1, neginf=-1)
    print(correlation_matrix_cleaned.shape)
    print(np.max(correlation_matrix_cleaned), np.min(correlation_matrix_cleaned))
    print(sum(correlation_matrix_cleaned))

    gene_pos_dic = {}
    for i in range(num_of_gene):
        gene_names = scRNA_adata.var_names[i]
        chrom = scRNA_adata.var["chrom"][i]
        chromStart = scRNA_adata.var["chromStart"][i]
        chromEnd = scRNA_adata.var["chromEnd"][i]
        gene_pos_dic[gene_names] = [chrom, chromStart, chromEnd]

    print(len(gene_pos_dic))

    # Compute peak-peak correlation
    print("----------------------")
    print("Compute peak-peak correlation")
    peak_expression = scATAC_adata.X[:num_of_cell, :num_of_peak].toarray()
    correlation_matrix2 = np.corrcoef(peak_expression, rowvar=False)
    correlation_matrix2_cleaned = np.nan_to_num(correlation_matrix2, nan=0, posinf=1, neginf=-1)
    print(correlation_matrix2_cleaned.shape)
    print(np.max(correlation_matrix2_cleaned), np.min(correlation_matrix2_cleaned))
    print(sum(correlation_matrix2_cleaned))

    peak_pos_dic = {}
    for i in range(num_of_peak):
        peak_names = scATAC_adata.var_names[i]
        chrom = scATAC_adata.var["chrom"][i]
        chromStart = scATAC_adata.var["chromStart"][i]
        chromEnd = scATAC_adata.var["chromEnd"][i]
        peak_pos_dic[peak_names] = [chrom, chromStart, chromEnd]

    print(len(peak_pos_dic))

    print("----------------------")
    print("Compute correlation matrix")
    cor_mat = torch.zeros(num_of_peak + num_of_gene, num_of_peak + num_of_gene)
    print(cor_mat.shape)
    for i, gene in enumerate(list(gene_pos_dic.keys())):
        for j, peak in enumerate(list(peak_pos_dic.keys())):
            gene_chrom = gene_pos_dic[gene][0]
            gene_start = gene_pos_dic[gene][1]
            gene_end = gene_pos_dic[gene][2]

            peak_chrom = peak_pos_dic[peak][0]
            peak_start = peak_pos_dic[peak][1]
            peak_end = peak_pos_dic[peak][2]

            if gene_chrom == peak_chrom and abs(gene_start - peak_start) <= 2000:
                cor_mat[num_of_peak + i, j] = 1
                cor_mat[j, num_of_peak + i] = 1

    print("----------------------")
    print("Compute gene by gene correlation matrix")
    gene_cor_mat = torch.zeros(num_of_gene, num_of_gene)
    for i, gene in enumerate(list(gene_pos_dic.keys())):
        for j, gene in enumerate(list(gene_pos_dic.keys())):
            gen_cor = correlation_matrix_cleaned[i, j]
            if gen_cor > rate1 and gen_cor != dis_rate:
                cor_mat[num_of_peak + i, num_of_peak + j] = 1
                gene_cor_mat[i, j] = 1
            if gen_cor < rate2:
                cor_mat[num_of_peak + i, num_of_peak + j] = 1
                gene_cor_mat[i, j] = 1
    print("finish gene by gene cor_mat")
    print(torch.sum(gene_cor_mat))

    peak_cor_mat = torch.zeros(num_of_peak, num_of_peak)
    for i, peak in enumerate(list(peak_pos_dic.keys())):
        for j, peak in enumerate(list(peak_pos_dic.keys())):
            peak_cor = correlation_matrix2_cleaned[i, j]
            if peak_cor > rate1 and peak_cor != dis_rate:
                cor_mat[i, j] = 1
                peak_cor_mat[i, j] = 1
            if peak_cor < rate2:
                cor_mat[i, j] = 1
                peak_cor_mat[i, j] = 1
    print("finish peak by peak cor_mat")
    print(torch.sum(peak_cor_mat))

    print("finish cor_mat")
    print(torch.sum(cor_mat))

    # path = "gene_data/gene_cor_matrix.pth"
    # print(f"writing to the file {path}")
    # torch.save(gene_cor_mat, path)

    # print(f"loading to the file from {path}")
    # gene_cor_mat = torch.load(path)
    # print(torch.sum(gene_cor_mat))
    """
    ==============================================================================
    Compute gene edge index
    ==============================================================================
    """
    sparse_gene_cor_mat = csr_matrix(gene_cor_mat)
    # sparse_gene_cor_mat = csr_matrix(gene_cor_mat[:train_num_of_gene, :train_num_of_gene])
    # sparse_gene_cor_mat_val = csr_matrix(gene_cor_mat[train_num_of_gene:, train_num_of_gene:])
    # print(sparse_gene_cor_mat)
    gene_rows, gene_cols = sparse_gene_cor_mat.nonzero()
    gene_edge_index = torch.tensor(np.array([gene_rows, gene_cols]), dtype=torch.long, device=device)

    # gene_rows2, gene_cols2 = sparse_gene_cor_mat_val.nonzero()
    # gene_edge_index_val = torch.tensor(np.array([gene_rows2, gene_cols2]), dtype=torch.long, device=device)

    """
    ==============================================================================
    Compute peak edge index
    ==============================================================================
    """
    sparse_peak_cor_mat = csr_matrix(peak_cor_mat)
    # sparse_peak_cor_mat = csr_matrix(peak_cor_mat[:train_num_of_peak, :train_num_of_peak])
    # sparse_peak_cor_mat_val = csr_matrix(peak_cor_mat[train_num_of_peak:, train_num_of_peak:])
    # print(sparse_gene_cor_mat)
    peak_rows, peak_cols = sparse_peak_cor_mat.nonzero()
    peak_edge_index = torch.tensor(np.array([peak_rows, peak_cols]), dtype=torch.long, device=device)

    # peak_rows2, peak_cols2 = sparse_peak_cor_mat_val.nonzero()
    # peak_edge_index_val = torch.tensor(np.array([peak_rows2, peak_cols2]), dtype=torch.long, device=device)

    """
    ==============================================================================
    Compute peak+gene edge index
    ==============================================================================
    """
    sparse_cor_mat = csr_matrix(cor_mat)
    # sparse_cor_mat_val = csr_matrix(cor_mat)
    # print(sparse_gene_cor_mat)
    rows, cols = sparse_cor_mat.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long, device=device)

    # rows2, cols2 = sparse_cor_mat_val.nonzero()
    # edge_index_val = torch.tensor(np.array([rows2, cols2]), dtype=torch.long, device=device)

    if use_pretrain:
        """
        ==============================================================================
        Compute true training avg gene by cell_type matrix
        ==============================================================================
        """

        index3 = scRNA_adata_copy.var['highly_variable'].values
        scRNA_adata_for_pretrain = scRNA_adata_copy[:, index3].copy()

        # scRNA_adata_for_pretrain = scRNA_adata_for_pretrain[:, :train_num_of_gene]
        scRNA_adata_for_pretrain = scRNA_adata_for_pretrain[:, :num_of_gene]
        cell_type_list = np.array(list(scRNA_adata_for_pretrain.obs['cell_type']))
        cell_type = sorted(list(set(cell_type_list)))
        cell_type_number_dic = {i: 0 for i in cell_type}

        for i in range(len(cell_type_list)):
            cell_type_number_dic[cell_type_list[i]] += 1

        cell_type_number_list = list(cell_type_number_dic.values())
        # gene_by_celltype_matrix = torch.zeros(train_num_of_gene, len(cell_type))
        gene_by_celltype_matrix = torch.zeros(num_of_gene, len(cell_type))

        # true_avg_gene_matrix = csr_matrix((num_topics, train_num_of_gene)).tolil()
        true_avg_gene_matrix = csr_matrix((num_topics, num_of_gene)).tolil()
        scRNA_adata_for_pretrain_data = scRNA_adata_for_pretrain.X.tolil()

        """
        ==============================================================================
        Compute true val avg gene by cell_type matrix
        ==============================================================================
        """

        # index4 = scRNA_adata_copy.var['highly_variable'].values
        # scRNA_adata_for_val = scRNA_adata_copy[:, index4].copy()
        #
        # scRNA_adata_for_val = scRNA_adata_for_val[:, train_num_of_gene:]
        # cell_type_list_val = np.array(list(scRNA_adata_for_val.obs['cell_type']))
        # cell_type_number_dic_val = {i: 0 for i in cell_type}
        #
        # for i in range(len(cell_type_list_val)):
        #     cell_type_number_dic_val[cell_type_list_val[i]] += 1
        #
        # cell_type_number_list = list(cell_type_number_dic_val.values())
        # gene_by_celltype_matrix = torch.zeros(num_of_gene - train_num_of_gene, len(cell_type))
        #
        # true_avg_gene_matrix_val = csr_matrix((num_topics, num_of_gene - train_num_of_gene)).tolil()
        # scRNA_adata_for_val_data = scRNA_adata_for_val.X.tolil()

        """
        ==============================================================================
        Compute true training avg peak by cell_type matrix
        ==============================================================================
        """

        index5 = scATAC_adata_copy.var['highly_variable'].values
        scATAC_adata_for_pretrain = scATAC_adata_copy[:, index5].copy()

        # scATAC_adata_for_pretrain = scATAC_adata_for_pretrain[:, :train_num_of_peak]
        scATAC_adata_for_pretrain = scATAC_adata_for_pretrain[:, :num_of_peak]
        peak_cell_type_list = np.array(list(scATAC_adata_for_pretrain.obs['cell_type']))
        peak_cell_type = sorted(list(set(peak_cell_type_list)))
        peak_cell_type_number_dic = {i: 0 for i in peak_cell_type}

        for i in range(len(peak_cell_type_list)):
            peak_cell_type_number_dic[peak_cell_type_list[i]] += 1

        peak_cell_type_number_list = list(peak_cell_type_number_dic.values())
        # peak_by_celltype_matrix = torch.zeros(train_num_of_peak, len(peak_cell_type))
        peak_by_celltype_matrix = torch.zeros(num_of_peak, len(peak_cell_type))

        # true_avg_gene_matrix = csr_matrix((num_topics, train_num_of_gene)).tolil()
        true_avg_peak_matrix = csr_matrix((num_topics, num_of_peak)).tolil()
        scATAC_adata_for_pretrain_data = scATAC_adata_for_pretrain.X.tolil()

        print("----------------------")
        print("Compute true training avg gene by cell_type matrix")
        true_avg_gene_matrix = getAvgExpression(matrix=true_avg_gene_matrix,
                                                data=scRNA_adata_for_pretrain_data,
                                                dic=cell_type_number_dic,
                                                celltype=cell_type,
                                                cell_type_list=cell_type_list)

        print("----------------------")
        print("Compute true training avg peak by cell_type matrix")
        true_avg_peak_matrix = getAvgExpression(matrix=true_avg_peak_matrix,
                                                data=scATAC_adata_for_pretrain_data,
                                                dic=peak_cell_type_number_dic,
                                                celltype=peak_cell_type,
                                                cell_type_list=peak_cell_type_list)

        # print("----------------------")
        # print("Compute true val avg gene by cell_type matrix")
        # true_avg_gene_matrix_val = getAvgExpression(true_avg_gene_matrix_val,
        #                                             scRNA_adata_for_val_data,
        #                                             cell_type_number_dic_val,
        #                                             cell_type,
        #                                             cell_type_list_val)
        # print(true_avg_gene_matrix)

        gene_feature_matrix = torch.randn((num_of_gene, graph_feature_size), dtype=torch.float32).to(device)
        peak_feature_matrix = torch.randn((num_of_peak, graph_feature_size), dtype=torch.float32).to(device)
        # gene_feature_matrix = torch.randn((train_num_of_gene, graph_feature_size), dtype=torch.float32).to(device)
        # gene_feature_matrix_val = torch.randn((num_of_gene - train_num_of_gene, graph_feature_size), dtype=torch.float32).to(device)
        # target gene by cell_type matrix
        # true_avg_gene_matrix = torch.randn((num_of_gene, num_topics)).to(device)
        true_avg_gene_tensor = torch.tensor(true_avg_gene_matrix.toarray().T, dtype=torch.float32).to(device)
        true_avg_peak_tensor = torch.tensor(true_avg_peak_matrix.toarray().T, dtype=torch.float32).to(device)
        # true_avg_gene_tensor_val = torch.tensor(true_avg_gene_matrix_val.toarray().T, dtype=torch.float32).to(device)
        # print(true_avg_gene_tensor)

        gnn = GNN(
            in_channels=graph_feature_size,
            hidden_channels=graph_feature_size * 2,
            out_channels=graph_feature_size,
            num_heads=1,
            dropout=0.2,
            conv_model='GATv2'
        ).to(device)

        mlp = MLP(
            num_of_embedding=graph_feature_size,
            hidden_channels=mlp_hidden_channels,
            num_of_cell_type=num_topics,
            dropout=0.1
        ).to(device)

        gene_emb_model = PretrainModel(gnn, mlp)
        gene_emb_model.to(device)

        peak_emb_model = PretrainModel(gnn, mlp)
        peak_emb_model.to(device)

        loss_fn = nn.MSELoss()
        pretrain_optimizer = optim.Adam(gene_emb_model.parameters(), lr=0.0003, weight_decay=1.2e-6)

        # print(gene_emb_model)
        # print(peak_emb_model)

        print("====  pretraining gene_emb_model  ====")
        gene_emb = pretrain_emb_model(
            model=gene_emb_model,
            optimizer=pretrain_optimizer,
            epochs=pretrain_num_epochs,
            feature_matrix=gene_feature_matrix,
            edge_index=gene_edge_index,
            true_gene_expression=true_avg_gene_tensor
        )

        print("====  pretraining peak_emb_model  ====")
        peak_emb = pretrain_emb_model(
            model=peak_emb_model,
            optimizer=pretrain_optimizer,
            epochs=pretrain_num_epochs,
            feature_matrix=peak_feature_matrix,
            edge_index=peak_edge_index,
            true_gene_expression=true_avg_peak_tensor
        )

        print("====  generating feature_matrix ====")
        pretrain_feature_matrix = (torch.cat((peak_emb, gene_emb), dim=0)).detach()
        """ 
        ==============================================================================
        scale pretrain_feature_matrix value to [0,1] * n 
        ==============================================================================
        """
        # print("=====  scale pretrain_feature_matrix value to [0,1] * n  =====")
        # pretrain_feature_matrix = (pretrain_feature_matrix - pretrain_feature_matrix.min()) / (pretrain_feature_matrix.max()-pretrain_feature_matrix.min())
        # pretrain_feature_matrix *= 10
        """ 
        ==============================================================================
        scale pretrain_feature_matrix value to [-1,1] * n
        ==============================================================================
        """
        # print("=====  scale pretrain_feature_matrix value to [-1,1] * n  =====")
        # pretrain_feature_matrix = 2 * (pretrain_feature_matrix - pretrain_feature_matrix.min()) / (pretrain_feature_matrix.max() - pretrain_feature_matrix.min()) - 1
        # pretrain_feature_matrix *= 30
        # print(pretrain_feature_matrix)
        """
        ==============================================================================
        Normalize pretrain_feature_matrix to have mean 0 and standard deviation 1
        ==============================================================================
        """
        print("=====  sNormalize pretrain_feature_matrix to have mean 0 and standard deviation 1  =====")
        pretrain_feature_matrix = (pretrain_feature_matrix - pretrain_feature_matrix.mean()) / pretrain_feature_matrix.std()
        pretrain_feature_matrix *= 20
        # for i in range(len(pretrain_feature_matrix)):
        #     for j in range(len(pretrain_feature_matrix[0])):
        #         feature_matrix[i,j] = pretrain_feature_matrix[i,j]
        print(pretrain_feature_matrix.shape)
        # print(feature_matrix.shape)
    else:
        feature_matrix = torch.randn((num_of_peak + num_of_gene, graph_feature_size)).to(device)

    gnn_model = GNN(
        in_channels=graph_feature_size,
        hidden_channels=graph_feature_size * 2,
        out_channels=graph_feature_size,
        num_heads=1,
        dropout=0.2,
        conv_model=conv_model
    ).to(device)

    GNN_ETM_model = GNN_ETM(
        num_topics=num_of_topic,
        num_gene=num_of_gene,
        num_peak=num_of_peak,
        t_hidden_size=t_hidden_size,
        t_hidden_size_peak=t_hidden_size,
        rho_size=rho_size,
        eta_size=eta_size,
        emb_size=emb_size,
        theta_act='relu',
        device=device,
        enc_drop=0.1,
        use_gnn=True,
        gnn_model=gnn_model,
        feature_matrix=feature_matrix,
        edge_index=edge_index
    ).to(device)

    GNN_ETM_optimizer = optim.Adam(GNN_ETM_model.parameters(), lr=0.0005, weight_decay=1.2e-6)
    print("-----------------------")
    print(GNN_ETM_model)

    GNN_ETM_model, GNN_ETM_perf, ari_perf, best_ari, theta = train_GCNscETM(
        model=GNN_ETM_model,
        optimizer=GNN_ETM_optimizer,
        RNA_tensor=scRNA_1000_tensor,
        RNA_tensor_normalized=scRNA_1000_tensor_normalized,
        ATAC_tensor=scATAC_1000_tensor,
        ATAC_tensor_normalized=scATAC_1000_tensor_normalized,
        adata=scRNA_1000_mp_anndata,
        adata2=scATAC_1000_mp_anndata,
        ari_freq=ari_freq,
        niter=num_of_epochs
    )

    print(best_ari)
    """# **View result**"""
    print("-----------------------")
    print("plotting")
    # monitor_perf2(GNN_ETM_perf, ari_perf, ari_freq, "NELBO")
    monitor_perf2(GNN_ETM_perf, ari_perf, ari_freq, "both", plot_path_rel)

    """
    
    # T-SNE
    ### Theta
    """

    theta, theta_g, theta_p = get_theta_GCN(GNN_ETM_model, scRNA_1000_tensor_normalized, scATAC_1000_tensor_normalized)
    adata = ad.AnnData(np.array(theta.detach().cpu()))
    labels = scRNA_adata.obs['cell_type'][:num_of_cell]
    adata.obs["Cell_Type"] = pd.Categorical(list(labels))
    sc.tl.tsne(adata, use_rep='X')
    # sc.pl.tsne(adata, color=adata.obs)
    fig1 = sc.pl.tsne(adata, color="Cell_Type", show=False, return_fig=True)
    fig1.savefig(plot_path_rel + 'tsne.png')

    sc.pp.neighbors(adata)
    sc.tl.louvain(adata, resolution=0.5)  # Perform Louvain clustering with resolution 0.5
    sc.tl.umap(adata, spread=1.0, min_dist=0.4)
    # sc.pl.umap(adata, color="louvain", title="louvain_0.5")
    # sc.pl.umap(adata, color=adata.obs, title="Cell_Type")

    fig2 = sc.pl.umap(adata, color="louvain", title="louvain_0.5", show=False, return_fig=True)
    fig2.savefig(plot_path_rel + 'louvain.png')

    fig3 = sc.pl.umap(adata, color="Cell_Type", title="Cell_Type", show=False, return_fig=True)
    fig3.savefig(plot_path_rel + 'umap.png')

    adata = ad.AnnData(np.array(theta_g.detach().cpu()))
    labels = scRNA_adata.obs['cell_type'][:num_of_cell]
    adata.obs["Cell_Type"] = pd.Categorical(list(labels))
    sc.tl.tsne(adata, use_rep='X')
    # sc.pl.tsne(adata, color=adata.obs)
    fig4 = sc.pl.tsne(adata, color="Cell_Type", show=False, return_fig=True)
    fig4.savefig(plot_path_rel + 'gene_tsne.png')
