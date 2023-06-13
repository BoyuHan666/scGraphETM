if __name__ == '__main__':
    import torch

    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU device found")
    else:
        device = torch.device("cpu")
        print("No GPU found")

    """# **Load data**"""

    import anndata as ad

    scRNA_adata = ad.read_h5ad("./data/10x-Multiome-Pbmc10k-RNA.h5ad")
    scATAC_adata = ad.read_h5ad("./data/10x-Multiome-Pbmc10k-ATAC.h5ad")

    print(scRNA_adata)
    print(scATAC_adata)

    import scanpy as sc
    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    import scipy.sparse as sp
    from scipy.sparse import coo_matrix, block_diag
    import pandas as pd

    # Preprocess scRNA-seq data
    print("processing data")
    print('-'*30)
    print("process atac data")
    scATAC_adata_copy = scATAC_adata

    sc.pp.normalize_total(scATAC_adata, target_sum=1e4)
    sc.pp.log1p(scATAC_adata)
    sc.pp.highly_variable_genes(scATAC_adata)
    print('-' * 30)
    print("process rna data")
    scRNA_adata_copy = scRNA_adata

    sc.pp.normalize_total(scRNA_adata, target_sum=1e4)
    sc.pp.log1p(scRNA_adata)
    print('-' * 30)
    print(sum(scRNA_adata.var["highly_variable"]))
    print(sum(scATAC_adata.var["highly_variable"]))

    index1 = scRNA_adata.var['highly_variable'].values
    scRNA_adata = scRNA_adata[:, index1].copy()

    index2 = scATAC_adata.var['highly_variable'].values
    scATAC_adata = scATAC_adata[:, index2].copy()

    print(scRNA_adata)
    print(scATAC_adata)
    num_of_cell = scRNA_adata.n_obs
    num_of_gene = scRNA_adata.n_vars
    num_of_peak = scATAC_adata.n_vars
    # num_of_cell = 9000
    # num_of_gene = 2000
    # num_of_peak = 30000
    print(num_of_cell, num_of_gene, num_of_peak)
    print("----------------------")
    scATAC_1000 = scATAC_adata_copy[:, index2]
    scATAC_1000 = scATAC_1000.X[:num_of_cell, :num_of_peak].toarray()
    scATAC_1000_tensor = torch.from_numpy(scATAC_1000).to('cuda')

    scATAC_1000_tensor_normalized = (torch.from_numpy(scATAC_adata_copy.X[:num_of_cell, :num_of_peak].toarray())).to('cuda')
    scATAC_1000_mp_anndata = ad.AnnData(X=scATAC_adata_copy.X[:num_of_cell, :num_of_peak].toarray())
    scATAC_1000_mp_anndata.obs['Celltype'] = scATAC_adata_copy.obs['cell_type'].values[:num_of_cell]
    num_topics2 = len(scATAC_1000_mp_anndata.obs['Celltype'].values.unique())
    print(num_topics2)

    scRNA_1000 = scRNA_adata_copy[:, index1]
    scRNA_1000 = scRNA_1000.X[:num_of_cell, :num_of_gene].toarray()
    scRNA_1000_tensor = torch.from_numpy(scRNA_1000).to('cuda')

    scRNA_1000_tensor_normalized = (torch.from_numpy(scRNA_adata_copy.X[:num_of_cell, :num_of_gene].toarray())).to('cuda')
    scRNA_1000_mp_anndata = ad.AnnData(X=scRNA_adata_copy.X[:num_of_cell, :num_of_gene].toarray())
    scRNA_1000_mp_anndata.obs['Celltype'] = scRNA_adata_copy.obs['cell_type'].values[:num_of_cell]
    num_topics = len(scRNA_1000_mp_anndata.obs['Celltype'].values.unique())
    print(num_topics)

    """# **Generate peak-gene correlation graph**"""
    import scanpy as sc
    import numpy as np
    # import cupy as cp

    # # Compute gene-gene correlation
    # print("----------------------")
    # print("Compute gene-gene correlation")
    # gene_expression = scRNA_adata.X[:num_of_cell, :num_of_gene].toarray()
    # correlation_matrix = np.corrcoef(gene_expression, rowvar=False)
    # correlation_matrix_cleaned = np.nan_to_num(correlation_matrix, nan=0, posinf=1, neginf=-1)
    #
    # # use cupy
    # # gene_expression = cp.asarray(scRNA_adata.X[:num_of_cell, :num_of_gene].toarray())
    # # correlation_matrix = cp.corrcoef(gene_expression, rowvar=False)
    # # correlation_matrix_cleaned = cp.nan_to_num(correlation_matrix, nan=0, posinf=1, neginf=-1)
    # # correlation_matrix_cleaned = cp.asnumpy(correlation_matrix_cleaned)
    # print(correlation_matrix_cleaned.shape)
    # print(np.max(correlation_matrix_cleaned), np.min(correlation_matrix_cleaned))
    # print(sum(correlation_matrix_cleaned))
    #
    # # Compute peak-peak correlation
    # print("----------------------")
    # print("Compute peak-peak correlation")
    # peak_expression = scATAC_adata.X[:num_of_cell, :num_of_peak].toarray()
    # correlation_matrix2 = np.corrcoef(peak_expression, rowvar=False)
    # correlation_matrix2_cleaned = np.nan_to_num(correlation_matrix2, nan=0, posinf=1, neginf=-1)
    #
    # # use cupy
    # # peak_expression = cp.asarray(scATAC_adata.X[:num_of_cell, :num_of_peak].toarray())
    # # correlation_matrix2 = cp.corrcoef(peak_expression, rowvar=False)
    # # correlation_matrix2_cleaned = cp.nan_to_num(correlation_matrix2, nan=0, posinf=1, neginf=-1)
    # # correlation_matrix2_cleaned = cp.asnumpy(correlation_matrix2_cleaned)
    #
    # print(correlation_matrix2_cleaned.shape)
    # print(np.max(correlation_matrix2_cleaned), np.min(correlation_matrix2_cleaned))
    # print(sum(correlation_matrix2_cleaned))
    #
    # # peak_gene = gene_expression @ peak_expression.T
    # # correlation_matrix3 = np.corrcoef(peak_gene)
    # # correlation_matrix3_cleaned = np.nan_to_num(correlation_matrix3, nan=0, posinf=1, neginf=-1)
    # # print(np.max(correlation_matrix3_cleaned), np.min(correlation_matrix3_cleaned))
    # # print((correlation_matrix3_cleaned))
    #
    # gene_pos_dic = {}
    # for i in range(num_of_gene):
    #     gene_names = scRNA_adata.var_names[i]
    #     chrom = scRNA_adata.var["chrom"][i]
    #     chromStart = scRNA_adata.var["chromStart"][i]
    #     chromEnd = scRNA_adata.var["chromEnd"][i]
    #     gene_pos_dic[gene_names] = [chrom, chromStart, chromEnd]
    #
    # print(len(gene_pos_dic))
    #
    # peak_pos_dic = {}
    # for i in range(num_of_peak):
    #     peak_names = scATAC_adata.var_names[i]
    #     chrom = scATAC_adata.var["chrom"][i]
    #     chromStart = scATAC_adata.var["chromStart"][i]
    #     chromEnd = scATAC_adata.var["chromEnd"][i]
    #     peak_pos_dic[peak_names] = [chrom, chromStart, chromEnd]
    #
    # print(len(peak_pos_dic))
    #
    # print("----------------------")
    # print("Compute correlation matrix")
    # cor_mat = torch.zeros(num_of_peak + num_of_gene, num_of_peak + num_of_gene)
    # print(cor_mat.shape)
    # for i, gene in enumerate(list(gene_pos_dic.keys())):
    #     for j, peak in enumerate(list(peak_pos_dic.keys())):
    #         gene_chrom = gene_pos_dic[gene][0]
    #         gene_start = gene_pos_dic[gene][1]
    #         gene_end = gene_pos_dic[gene][2]
    #
    #         peak_chrom = peak_pos_dic[peak][0]
    #         peak_start = peak_pos_dic[peak][1]
    #         peak_end = peak_pos_dic[peak][2]
    #
    #         if gene_chrom == peak_chrom and abs(gene_start - peak_start) <= 2000:
    #             cor_mat[num_of_peak + i, j] = 1
    #             cor_mat[j, num_of_peak + i] = 1
    #         # cor_mat[j,j] =1
    #         # cor_mat[num_of_peak+i, num_of_peak+i] =1
    #
    # for i, gene in enumerate(list(gene_pos_dic.keys())):
    #     for j, gene in enumerate(list(gene_pos_dic.keys())):
    #         gen_cor = correlation_matrix_cleaned[i, j]
    #         # cor_mat[num_of_peak+i, num_of_peak+j] = gen_cor
    #         if gen_cor > 0.6:
    #             cor_mat[num_of_peak + i, num_of_peak + j] = 1
    #
    # for i, peak in enumerate(list(peak_pos_dic.keys())):
    #   for j, peak in enumerate(list(peak_pos_dic.keys())):
    #     peak_cor = correlation_matrix2_cleaned[i, j]
    #     if peak_cor > 0.6:
    #       cor_mat[i, j] = 1
    # print("finish cor_mat")
    # torch.sum(cor_mat)
    #
    # from scipy.sparse import csr_matrix
    #
    # sparse_cor_mat = csr_matrix(cor_mat.cpu())
    # # print(sparse_cor_mat)
    # rows, cols = sparse_cor_mat.nonzero()
    # edge_index = torch.tensor([rows, cols], dtype=torch.long).to('cuda')
    # # print(edge_index)

    import torch
    import torch.nn.functional as F
    from torch import nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """# **GCN**

    """

    import torch
    import torch.nn.functional as F
    from torch.autograd import Variable
    from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
    from torch_geometric.utils import train_test_split_edges


    # class GCN(torch.nn.Module):
    #     def __init__(self, in_channels, hidden_channels, out_channels):
    #         super(GCN, self).__init__()
    #         self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=1, concat=True, dropout=0.2).to('cuda')
    #         self.conv2 = GATv2Conv(hidden_channels, out_channels, heads=1, concat=True, dropout=0.2).to('cuda')
    #         # self.conv1 = GCNConv(in_channels, hidden_channels).to('cuda')
    #         # self.conv2 = GCNConv(hidden_channels, out_channels).to('cuda')
    #         self.embedding = None
    #         self.loss = None
    #
    #     def forward(self, x, edge_index):
    #         x = F.relu(self.conv1(x, edge_index))
    #         # print(x.size())
    #         x = F.relu(self.conv2(x, edge_index))
    #         # x = self.conv2(x, edge_index)
    #         # x = F.dropout(x, training=self.training)
    #         self.embedding = x
    #         return x
    #
    #     def backward(self, loss):
    #         self.loss = loss
    #         self.loss.backward()
    #         return self.loss
    #
    #     def get_embedding(self):
    #         return self.embedding


    class moETMdecoder(nn.Module):
        def __init__(self, mod1_dim, mod2_dim, z_dim, emd_dim):
            super(moETMdecoder, self).__init__()

            self.alpha_mod1 = nn.Parameter(torch.randn(mod1_dim, emd_dim))
            self.alpha_mod2 = nn.Parameter(torch.randn(mod2_dim, emd_dim))
            self.beta = nn.Parameter(torch.randn(z_dim, emd_dim))
            # self.mod1_batch_bias = nn.Parameter(torch.randn(num_batch, mod1_dim))
            # self.mod2_batch_bias = nn.Parameter(torch.randn(num_batch, mod2_dim))
            self.Topic_mod1 = None
            self.Topic_mod2 = None

        def forward(self, theta, cross_prediction=False):
            self.Topic_mod1 = torch.mm(self.alpha_mod1, self.beta.t()).t()
            self.Topic_mod2 = torch.mm(self.alpha_mod2, self.beta.t()).t()
            # print("------------------------------------")
            # print(type(theta), type(self.Topic_mod1))
            recon_mod1 = torch.mm(theta, self.Topic_mod1)

            # recon_mod1 += self.mod1_batch_bias[batch_indices]
            if cross_prediction == False:
                recon_log_mod1 = F.log_softmax(recon_mod1, dim=-1)
            else:
                recon_log_mod1 = F.softmax(recon_mod1, dim=-1)

            recon_mod2 = torch.mm(theta, self.Topic_mod2)
            # recon_mod2 += self.mod2_batch_bias[batch_indices]
            if cross_prediction == False:
                recon_log_mod2 = F.log_softmax(recon_mod2, dim=-1)
            else:
                recon_log_mod2 = F.softmax(recon_mod2, dim=-1)

            return recon_log_mod1, recon_log_mod2


    """# **GCN-scETM**"""


    class GCN_ETM(nn.Module):
        def __init__(self, num_topics, num_gene, num_peak, t_hidden_size, t_hidden_size_peak, rho_size, eta_size,
                     emb_size, graph_feature_size, theta_act, enc_drop=0.5,
                     use_gcn=False, feature_matrix=None, edge_index=None):
            super(GCN_ETM, self).__init__()
            # define hyperparameters
            self.moETMdecoder = moETMdecoder(mod1_dim=num_gene, mod2_dim=num_peak, z_dim=num_topics, emd_dim=emb_size)
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
            self.use_gcn = use_gcn

            ## define the word embedding matrix \rho \eta
            self.rho = nn.Parameter(torch.randn(num_gene, rho_size)).to('cuda')
            self.eta = nn.Parameter(torch.randn(num_peak, eta_size)).to('cuda')

            ## define the matrix containing the topic embeddings
            self.alphas = nn.Linear(rho_size, num_topics, bias=False)
            self.alphas_star = nn.Linear(eta_size, num_topics, bias=False)

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
                nn.Linear(num_peak, t_hidden_size_peak),
                self.theta_act,
                nn.BatchNorm1d(t_hidden_size_peak, eps=1e-5, momentum=0.1),
                # nn.Dropout(p=0.3),

                nn.Linear(t_hidden_size_peak, t_hidden_size),
                self.theta_act,
                nn.BatchNorm1d(t_hidden_size, eps=1e-5, momentum=0.1),
            )
            self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
            self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

            # conv layer
            # self.conv = GCN(graph_feature_size, graph_feature_size * 2, graph_feature_size).to('cuda')
            # self.optimizer2 = optim.AdamW(self.conv.parameters(), lr=0.001)

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
            q_theta = None
            if flag == "peak":
                q_theta = self.q_theta_peak(bows)
            if flag == "gene":
                q_theta = self.q_theta_gene(bows)
            if self.enc_drop > 0:
                q_theta = self.t_drop(q_theta)
            # print(q_theta.shape)
            # print(q_theta)
            mu_theta = self.mu_q_theta(q_theta)
            logsigma_theta = self.logsigma_q_theta(q_theta).clamp(-10,10)

            # KL[q(theta)||p(theta)] = lnq(theta) - lnp(theta)
            # --------------------------------------------------------------------
            # logsigma_theta = 2 * logsigma_theta
            logvar = 2 * logsigma_theta
            # kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
            kl_theta = -0.5 * (1 + logvar - mu_theta.pow(2) - logvar.exp()).sum(-1).mean()
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

        def prior_expert(self, size, use_cuda=False):
            """Universal prior expert. Here we use a spherical
            Gaussian: N(0, 1).
            @param size: integer
                         dimensionality of Gaussian
            @param use_cuda: boolean [default: False]
                             cast CUDA on variables
            """
            mu = Variable(torch.zeros(size))
            logvar = Variable(torch.zeros(size))
            if use_cuda:
                mu, logvar = mu.cuda(), logvar.cuda()
            return mu, logvar

        def experts(self, mu, logsigma, eps=1e-8):
            var = torch.exp(2 * logsigma) + eps
            # precision of i-th Gaussian expert at point x
            T = 1. / (var + eps)
            pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
            pd_var = 1. / torch.sum(T, dim=0)
            pd_logsigma = 0.5 * torch.log(pd_var + eps)
            return pd_mu, pd_logsigma

        def pog(self, mu, logsigma, eps=1e-8):
            variances = torch.exp(2 * logsigma) + eps
            inverse_variances = 1. / variances
            total_inverse_variance = torch.sum(inverse_variances, dim=0)
            weighted_mean = torch.sum(mu * inverse_variances, dim=0) / total_inverse_variance
            weighted_variance = 1 / total_inverse_variance
            product_log_sigma = torch.log(torch.sqrt(weighted_variance))

            return weighted_mean, product_log_sigma

        def get_theta(self, normalized_bows_gene, normalized_bows_peak):
            mu_theta_d, logsigma_theta_d, kld_theta_d = self.encode(normalized_bows_gene, "gene")
            mu_theta_j, logsigma_theta_j, kld_theta_j = self.encode(normalized_bows_peak, "peak")
            mu_prior, logsigma_prior = self.prior_expert((1, mu_theta_d.shape[0], mu_theta_d.shape[1]), use_cuda=True)

            # print("========  mu_theta_d logsigma_theta_d  =========")
            # print(mu_theta_d, logsigma_theta_d, mu_theta_d.unsqueeze(0), logsigma_theta_d.unsqueeze(0))
            # print("========  mu_theta_j logsigma_theta_j  =========")
            # print(mu_theta_j, logsigma_theta_j, mu_theta_j.unsqueeze(0), logsigma_theta_j.unsqueeze(0))
            # print("========  mu_prior logsigma_prior  =========")
            # print(mu_prior, logsigma_prior)

            Mu = torch.cat((mu_prior, mu_theta_d.unsqueeze(0), mu_theta_j.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, logsigma_theta_d.unsqueeze(0), logsigma_theta_j.unsqueeze(0)), dim=0)

            # Mu = torch.cat((mu_theta_d.unsqueeze(0), mu_theta_j.unsqueeze(0)), dim=0)
            # Log_sigma = torch.cat((logsigma_theta_d.unsqueeze(0), logsigma_theta_j.unsqueeze(0)), dim=0)
            # print(mu_prior.shape)
            mu, log_sigma = self.experts(Mu, Log_sigma)
            z = self.reparameterize(mu, log_sigma)
            theta_all = F.softmax(z, dim=-1)

            z_d = self.reparameterize(mu_theta_d, logsigma_theta_d)
            theta_d = F.softmax(z_d, dim=-1)

            z_j = self.reparameterize(mu_theta_j, logsigma_theta_j)
            theta_j = F.softmax(z_j, dim=-1)

            return theta_all, theta_d, kld_theta_d, theta_j, kld_theta_j

        def decode(self, theta, beta):
            res = torch.mm(theta, beta)
            preds = torch.log(res + 1e-6)
            return preds

        def forward(self, Gene, Gene_normalized, Peak, Peak_normalized, theta=None, aggregate=True):
            ## get \theta
            theta_all = None
            if theta is None:
                theta_all, theta_d, kld_theta_d, theta_j, kld_theta_j = self.get_theta(Gene_normalized, Peak_normalized)
            else:
                kld_theta_d = None
                kld_theta_j = None

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

            recon_log_mod1, recon_log_mod2 = self.moETMdecoder(theta_all)
            nll_mod1 = (-recon_log_mod1 * Gene).sum(-1).mean()
            nll_mod2 = (-recon_log_mod2 * Peak).sum(-1).mean()

            preds_Gene = self.decode(theta_d, beta_Gene)
            recon_loss_Gene = -(preds_Gene * Gene).sum(-1)

            preds_Peak = self.decode(theta_j, beta_Peak)
            recon_loss_Peak = -(preds_Peak * Peak).sum(-1)

            recon_loss = recon_loss_Gene + recon_loss_Peak
            if aggregate:
                recon_loss_Gene = recon_loss_Gene.mean()
                recon_loss_Peak = recon_loss_Peak.mean()
                recon_loss = recon_loss.mean()
            return recon_loss, kld_theta, nll_mod1, nll_mod2

        def train_gcn(self):
            self.conv.train()
            self.optimizer2.zero_grad()
            # self.conv.zero_grad()
            self.conv(self.feature_matrix, self.edge_index)

        def gcn_back(self):
            # torch.nn.utils.clip_grad_norm_(self.conv.parameters(), 2.0)
            self.optimizer2.step()


    """# **Helper functions**"""

    import pickle
    import numpy as np
    import pandas as pd

    import scanpy as sc
    import anndata
    import random

    import torch
    # from etm import ETM
    from torch import optim
    from torch.nn import functional as F

    import os
    from sklearn.metrics import adjusted_rand_score

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # from seaborn import heatmap, lineplot, clustermap


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


    """### scGCN - ETM Helper"""


    def calc_weight(
            epoch: int,
            n_epochs: int,
            cutoff_ratio: float = 0.,
            warmup_ratio: float = 1 / 3,
            min_weight: float = 0.,
            max_weight: float = 1e-7
    ) -> float:
        """Calculates weights.
        Args:
            epoch: current epoch.
            n_epochs: the total number of epochs to train the model.
            cutoff_ratio: ratio of cutoff epochs (set weight to zero) and
                n_epochs.
            warmup_ratio: ratio of warmup epochs and n_epochs.
            min_weight: minimum weight.
            max_weight: maximum weight.
        Returns:
            The current weight of the KL term.
        """

        fully_warmup_epoch = n_epochs * warmup_ratio

        if epoch < n_epochs * cutoff_ratio:
            return 0.
        if warmup_ratio:
            return max(min(1., epoch / fully_warmup_epoch) * max_weight, min_weight)
        else:
            return max_weight


    # train the VAE for one epoch
    def train_GCNscETM_helper(model, RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized, KL_weight):
        # initialize the model and loss
        model.train()
        optimizer.zero_grad()
        model.zero_grad()

        # forward and backward pass
        # model.train_gcn()
        nll, kl_theta, nll_mod1, nll_mod2 = model(RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized)
        # loss = nll + kl_theta
        # loss.backward()  # Calculate and backprop gradients w.r.t. negative ELBO
        loss_moETM = nll_mod1 + nll_mod2 + kl_theta
        loss_moETM.backward()

        # clip gradients to 2.0 if it gets too large
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)

        # update model to minimize negative ELBO
        # model.gcn_back()
        optimizer.step()

        return torch.sum(loss_moETM).item()


    # get sample encoding theta from the trained encoder network
    def get_theta_GCN(model, RNA_tensor_normalized, ATAC_tensor_normalized):
        model.eval()
        with torch.no_grad():
            theta_all, theta_d, kld_theta_d, theta_j, kld_theta_j = model.get_theta(RNA_tensor_normalized, ATAC_tensor_normalized)

            q_theta_gene = model.q_theta_gene(RNA_tensor_normalized)
            mu_theta_gene = model.mu_q_theta(q_theta_gene)
            theta_gene = F.softmax(mu_theta_gene, dim=-1)

            q_theta_peak = model.q_theta_peak(ATAC_tensor_normalized)
            mu_theta_peak = model.mu_q_theta(q_theta_peak)
            theta_peak = F.softmax(mu_theta_peak, dim=-1)

            theta = theta_gene + theta_peak
            # theta = theta_gene * theta_peak
            return theta_all, theta, theta_gene, theta_peak


    def train_GCNscETM(model, RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized, adata, adata2,
                       ari_freq, niter=100):
        """
            :param model: the scETM model object
            :param X_tensor: NxM raw read count matrix X
            :param X_tensor_normalized: NxM normalized read count matrix X
            :param adata: annotated single-cell data object with ground-truth cell type information for evaluation
            :param niter: maximum number of epochs
            :return:
                1. model: trained scETM model object
                2. perf: niter-by-3 ndarray with iteration index, SSE, and ARI as the 3 columns
        """
        perf = np.ndarray(shape=(niter, 2), dtype='float')
        num_of_ari = (int)(niter / ari_freq)
        ari_perf = np.ndarray(shape=(num_of_ari, 4), dtype='float')
        best_ari = 0
        best_theta = None
        best_theta_all = None
        best_theta_gene = None
        best_theta_peak = None
        # WRITE YOUR CODE HERE
        for i in range(niter):
            KL_weight = calc_weight(i, niter, 0, 1 / 3, 0, 1e-4)
            NELBO = train_GCNscETM_helper(model, RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized, KL_weight)
            theta_all, theta, theta_gene, theta_peak = get_theta_GCN(model, RNA_tensor_normalized, ATAC_tensor_normalized)
            perf[i, 0] = i
            perf[i, 1] = NELBO
            if (i % ari_freq == 0):

                ari = evaluate_ari(theta_all.to('cpu'), adata)

                ari1 = evaluate_ari(theta_gene.to('cpu'), adata)

                ari2 = evaluate_ari(theta_peak.to('cpu'), adata)

                idx = (int)(i / ari_freq)
                ari_perf[idx, 0] = idx
                ari_perf[idx, 1] = ari
                ari_perf[idx, 2] = ari1
                ari_perf[idx, 3] = ari2
                print("iter: " + str(i) + " ari: " + str(ari))
                # if (ari >= 0.75):
                #     break
                if(best_ari < ari):
                    best_ari = ari
                    best_theta = theta
                    best_theta_all = theta_all
                    best_theta_gene = theta_gene
                    best_theta_peak = theta_peak
        else:
                if (i % 100 == 0):
                    print("iter: " + str(i))
        return model, perf, ari_perf, best_ari, best_theta, best_theta_all, best_theta_gene, best_theta_peak


    def monitor_perf2(perf, ari_perf, ari_freq, objective):
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
            plt.plot(ari_niter, ari, label="GCN_scETM", color='red')
            plt.plot(ari_niter, ari1, label="scETM_RNA", color='black')
            plt.plot(ari_niter, ari2, label="scETM_ATAC", color='blue')
            plt.legend()
            plt.title("ARI comparison on GCN_scETM and scETM")
            plt.xlabel("iter")
            plt.ylabel("ARI")
            plt.ylim(0, 1)
            # plt.show()
            plt.savefig('./han_plot/BOTH_ARI.png')
        if (objective == "NELBO"):
            plt.plot(niter, mse)
            plt.xlabel("iter")
            plt.ylabel("NELBO")
            # plt.show()
            plt.savefig('./han_plot/NELBO.png')
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


    """# **Train GCN_ETM model**"""

    from torch_geometric.data import InMemoryDataset, Data

    num_of_topic = 100
    t_hidden_size = 400
    t_hidden_size_peak = 800
    rho_size = 400
    eta_size = 400
    emb_size = 400
    graph_feature_size = 400
    # graph_tensor = torch.from_numpy(cor_mat)
    feature_matrix = torch.randn((num_of_peak + num_of_gene, graph_feature_size)).to('cuda')
    ari_freq = 100
    num_of_epochs = 2000

    print(num_of_gene, num_of_peak)

    import numpy as np

    # for num_of_topic in num_of_topics:
    model = GCN_ETM(num_topics=num_of_topic,
                    num_gene=num_of_gene,
                    num_peak=num_of_peak,
                    t_hidden_size=t_hidden_size,
                    t_hidden_size_peak=t_hidden_size,
                    rho_size=rho_size,
                    eta_size=eta_size,
                    emb_size=graph_feature_size,
                    graph_feature_size=graph_feature_size,
                    theta_act='relu',
                    enc_drop=0.3,
                    use_gcn=False,
                    feature_matrix=None,
                    edge_index=None).to('cuda')

    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1.2e-6)
    print("-----------------------")
    print(model)

    # model.double()
    model, GNN_ETM_perf, ari_perf, best_ari, theta, theta_all, theta_g, theta_p = train_GCNscETM(model,
                                                   scRNA_1000_tensor, scRNA_1000_tensor_normalized,
                                                   scATAC_1000_tensor, scATAC_1000_tensor_normalized,
                                                   scRNA_1000_mp_anndata, scATAC_1000_mp_anndata, ari_freq,
                                                   num_of_epochs)

    print(best_ari)
    """# **View result**"""
    print("-----------------------")
    print("plotting")
    # monitor_perf2(GNN_ETM_perf, ari_perf, ari_freq, "NELBO")
    monitor_perf2(GNN_ETM_perf, ari_perf, ari_freq, "both")

    # genes = scRNA_adata.obs["cell_type"][:num_of_cell]
    # lut = dict(zip(genes.unique(), ['red',
    #                                 '#00FF00',
    #                                 '#0000FF',
    #                                 '#FFFF00',
    #                                 '#FF00FF',
    #                                 '#00FFFF',
    #                                 '#FFA500',
    #                                 '#800080',
    #                                 '#FFC0CB',
    #                                 '#FF69B4',
    #                                 '#00FF7F',
    #                                 '#FFD700',
    #                                 '#1E90FF',
    #                                 '#2F4F4F',
    #                                 '#808000',
    #                                 '#FF8C00',
    #                                 '#8B0000',
    #                                 '#4B0082',
    #                                 '#2E8B57',
    #                                 '#FF1493',
    #                                 '#6B8E23',
    #                                 '#48D1CC',
    #                                 '#B22222',
    #                                 '#DC143C',
    #                                 '#008080']))
    # row_colors = genes.map(lut)
    # data = [row_colors[i] for i in range(len(row_colors))]
    # row_colors = pd.Series(data)
    # theta, theta_g, theta_p = get_theta_GCN(model, scRNA_1000_tensor_normalized, scATAC_1000_tensor_normalized)
    # theta_T = (theta.detach().cpu().numpy())
    # clustermap(pd.DataFrame(theta_T), center=0, cmap="RdBu_r", row_colors=row_colors)
    #
    # from matplotlib.patches import Patch
    #
    # handles = [Patch(facecolor=lut[name]) for name in lut]
    # plt.legend(handles, lut, title='Cell Types',
    #            bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    #
    # """## Gene"""
    #
    # genes = scRNA_adata.obs["cell_type"][:num_of_cell]
    # lut = dict(zip(genes.unique(), ['red',
    #                                 '#00FF00',
    #                                 '#0000FF',
    #                                 '#FFFF00',
    #                                 '#FF00FF',
    #                                 '#00FFFF',
    #                                 '#FFA500',
    #                                 '#800080',
    #                                 '#FFC0CB',
    #                                 '#FF69B4',
    #                                 '#00FF7F',
    #                                 '#FFD700',
    #                                 '#1E90FF',
    #                                 '#2F4F4F',
    #                                 '#808000',
    #                                 '#FF8C00',
    #                                 '#8B0000',
    #                                 '#4B0082',
    #                                 '#2E8B57',
    #                                 '#FF1493',
    #                                 '#6B8E23',
    #                                 '#48D1CC',
    #                                 '#B22222',
    #                                 '#DC143C',
    #                                 '#008080']))
    # row_colors = genes.map(lut)
    # data = [row_colors[i] for i in range(len(row_colors))]
    # row_colors = pd.Series(data)
    # theta, theta_g, theta_p = get_theta_GCN(model, scRNA_1000_tensor_normalized, scATAC_1000_tensor_normalized)
    # theta_T = (theta_g.detach().cpu().numpy())
    # clustermap(pd.DataFrame(theta_T), center=0, cmap="RdBu_r", row_colors=row_colors)
    #
    # from matplotlib.patches import Patch
    #
    # handles = [Patch(facecolor=lut[name]) for name in lut]
    # plt.legend(handles, lut, title='Cell Types',
    #            bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    #
    # K = 5
    # topfeatures = np.zeros((K * num_of_topics, num_of_topics))
    # select = []
    # genes = list(scRNA_adata.var["gene_name"][:num_of_gene])
    #
    # beta_T = model.get_beta("gene").detach().cpu().numpy()
    # print(beta_T.shape)
    # top5 = []
    # for i in range(num_of_topics):
    #     top5.append(np.flip(np.concatenate((np.array(beta_T)[i, :].argsort()[-5:], select)), axis=0))
    #
    # print(np.array(top5).shape)
    # geneNames = []
    # count = 0
    # for i in range(num_of_topics):
    #     for j in range(K):
    #         topfeatures[count][i] = np.array(beta_T)[i][int(top5[i][j])]
    #         geneNames.append(genes[int(top5[i][j])])
    #         count += 1
    #
    # plt.figure(figsize=(8, 16))
    # hmp = heatmap((topfeatures), cmap='RdBu_r', vmax=0.03, center=0,
    #               xticklabels=[item for item in range(0, num_of_topics)], yticklabels=geneNames)
    # fig = hmp.get_figure()
    #
    # K = 5
    # topfeatures = np.zeros((K * num_of_topics, num_of_topics))
    # select = []
    # genes = list(scRNA_adata.var["gene_name"][:num_of_cell])
    # theta, theta_g, theta_p = get_theta_GCN(model, scRNA_1000_tensor_normalized, scATAC_1000_tensor_normalized)
    # theta_T = (theta_g.detach().cpu().numpy())
    # beta_T = model.get_beta("gene").detach().cpu().numpy()
    # top5 = []
    # for i in range(num_of_topics):
    #     top5.append(np.flip(np.concatenate((np.array(beta_T)[i, :].argsort()[-5:], select)), axis=0))
    #
    # print(np.array(top5).shape)
    # geneNames = []
    # count = 0
    # for i in range(num_of_topics):
    #     for j in range(K):
    #         topfeatures[count][i] = np.array(beta_T)[i][int(top5[i][j])]
    #         geneNames.append(genes[int(top5[i][j])])
    #         count += 1
    #
    # plt.figure(figsize=(8, 16))
    # hmp = heatmap((topfeatures), cmap='RdBu_r', vmax=0.03, center=0,
    #               xticklabels=[item for item in range(0, num_of_topics)], yticklabels=geneNames)
    # fig = hmp.get_figure()
    #
    # K = 5
    # topfeatures = np.zeros((K * num_of_topics, num_of_topics))
    # select = []
    #
    # chrom = list(scATAC_adata.var["chrom"][:num_of_peak])
    # chromStart = list(scATAC_adata.var["chromStart"][:num_of_peak])
    # chromEnd = list(scATAC_adata.var["chromEnd"][:num_of_peak])
    # peaks = []
    # for i in range(num_of_peak):
    #     peaks.append(str(chrom[i]) + " " + str(chromStart[i]) + " " + str(chromEnd[i]))
    #
    # beta_T = model.get_beta("peak").detach().cpu().numpy()
    # top5 = []
    # for i in range(num_of_topics):
    #     top5.append(np.flip(np.concatenate((np.array(beta_T)[i, :].argsort()[-5:], select)), axis=0))
    #
    # print(np.array(top5).shape)
    # peakNames = []
    # count = 0
    # for i in range(num_of_topics):
    #     for j in range(K):
    #         topfeatures[count][-(i + 1)] = np.array(beta_T)[i][int(top5[i][j])]
    #         peakNames.append(peaks[int(top5[i][j])])
    #         count += 1
    #
    # plt.figure(figsize=(8, 16))
    # hmp = heatmap(np.flip(topfeatures, axis=1), cmap='RdBu_r', vmax=0.03, center=0,
    #               xticklabels=[item for item in range(0, num_of_topics)], yticklabels=peakNames)
    # fig = hmp.get_figure()

    """# T-SNE

    ### Theta
    """

    import scanpy as sc

    # theta, theta_g, theta_p = get_theta_GCN(model, scRNA_1000_tensor_normalized, scATAC_1000_tensor_normalized)
    adata = anndata.AnnData(np.array(theta_all.detach().cpu()))
    labels = scRNA_adata.obs['cell_type'][:num_of_cell]
    adata.obs[""] = pd.Categorical(labels)
    sc.tl.tsne(adata, use_rep='X')
    # sc.pl.tsne(adata, color=adata.obs)
    fig1 = sc.pl.tsne(adata, color=adata.obs, show=False, return_fig=True)
    fig1.savefig('./han_plot/tsne.png')

    sc.pp.neighbors(adata)
    sc.tl.louvain(adata, resolution=0.5)  # Perform Louvain clustering with resolution 0.5
    sc.tl.umap(adata, spread=1.0, min_dist=0.4)
    # sc.pl.umap(adata, color="louvain", title="louvain_0.5")
    # sc.pl.umap(adata, color=adata.obs, title="Cell_Type")

    fig2 = sc.pl.umap(adata, color="louvain", title="louvain_0.5", show=False, return_fig=True)
    fig2.savefig('./han_plot/louvain.png')

    fig3 = sc.pl.umap(adata, color=adata.obs, title="Cell_Type", show=False, return_fig=True)
    fig3.savefig('./han_plot/umap.png')

    # """### Gene Theta"""
    #
    # import scanpy as sc
    #
    # theta, theta_g, theta_p = get_theta_GCN(model, scRNA_1000_tensor_normalized, scATAC_1000_tensor_normalized)
    adata = anndata.AnnData(np.array(theta_g.cpu().detach()))
    labels = scRNA_adata.obs['cell_type'][:num_of_cell]
    adata.obs[""] = pd.Categorical(list(labels))
    sc.tl.tsne(adata)
    fig4 = sc.pl.tsne(adata, color=adata.obs, show=False, return_fig=True)
    fig4.savefig('./han_plot/gene_tsne.png')

    sc.pp.neighbors(adata)
    sc.tl.louvain(adata, resolution=0.5)  # Perform Louvain clustering with resolution 0.5
    sc.tl.umap(adata, spread=1.0, min_dist=0.4)
    # sc.pl.umap(adata, color="louvain", title="louvain_0.5")
    # sc.pl.umap(adata, color=adata.obs, title="Cell_Type")

    fig5 = sc.pl.umap(adata, color="louvain", title="louvain_0.5", show=False, return_fig=True)
    fig5.savefig('./han_plot/louvain.png')

    fig6 = sc.pl.umap(adata, color=adata.obs, title="Cell_Type", show=False, return_fig=True)
    fig6.savefig('./han_plot/umap.png')
    #
    # """### Peak Theta"""
    #
    # import scanpy as sc
    #
    # theta, theta_g, theta_p = get_theta_GCN(model, scRNA_1000_tensor_normalized, scATAC_1000_tensor_normalized)
    # adata = anndata.AnnData(np.array(theta_p.cpu().detach()))
    # labels = scRNA_adata.obs['cell_type'][:num_of_cell]
    # adata.obs[""] = pd.Categorical(labels)
    # sc.tl.tsne(adata)
    # sc.pl.tsne(adata, color=adata.obs)