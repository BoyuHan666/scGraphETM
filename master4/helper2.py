import torch
from torch.nn import functional as F
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import scanpy as sc
import numpy as np
from scipy.sparse import vstack, hstack
from scipy.sparse import csr_matrix
import pickle

"""
==============================================================================
Get sub graph
==============================================================================
"""


def get_peak_index(path, top=5, threshould=None, gene_limit=None):
    with open(path, 'rb') as fp:
        gene_peak = pickle.load(fp)

    gene_index_list = []
    peak_index_list = []

    if threshould is None:
        for i, gene in enumerate(gene_peak.keys()):
            if gene_limit is not None:
                if i == gene_limit:
                    break

            gene_index_list.append(gene)
            for j, dist in gene_peak[gene][:top]:
                peak_index_list.append(j)
    else:
        for i, gene in enumerate(gene_peak.keys()):
            if gene_limit is not None:
                if i == gene_limit:
                    break

            gene_index_list.append(gene)
            for j, dist in gene_peak[gene][:top]:
                if dist < threshould:
                    peak_index_list.append(j)

    gene_index_list = list(set(gene_index_list))
    peak_index_list = list(set(peak_index_list))

    return gene_index_list, peak_index_list


def get_sub_graph(path, num_gene, num_peak, total_peak):
    if path == '':
        result = torch.zeros(num_peak + num_gene, num_peak + num_gene)
        result = csr_matrix(result.cpu())
    else:
        with open(path, 'rb') as fp:
            sp_matrix = pickle.load(fp)
        peak_peak = sp_matrix[:num_peak, :num_peak]
        peak_gene_down = sp_matrix[total_peak:(total_peak + num_gene), :num_peak]
        peak_gene_up = sp_matrix[:num_peak, total_peak:(total_peak + num_gene)]
        gene_gene = sp_matrix[total_peak:total_peak + num_gene, total_peak:total_peak + num_gene]

        top = hstack([peak_peak, peak_gene_up])
        bottom = hstack([peak_gene_down, gene_gene])

        result = vstack([top, bottom])

    rows, cols = result.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)

    return result, edge_index


def get_sub_graph_by_index(path, gene_index_list, peak_index_list, total_peak):
    with open(path, 'rb') as fp:
        sp_matrix = pickle.load(fp)

    print(f"sp_matrix.shape: {sp_matrix.shape}")
    peak_peak = sp_matrix[peak_index_list, :]
    peak_peak = peak_peak[:, peak_index_list]
    print(f"peak_peak.shape: {peak_peak.shape}")

    tmp_index_list = [total_peak + i for i in gene_index_list]

    peak_gene_down = sp_matrix[tmp_index_list, :]
    peak_gene_down = peak_gene_down[:, peak_index_list]
    print(f"peak_gene_down.shape: {peak_gene_down.shape}")

    peak_gene_up = sp_matrix[peak_index_list, :]
    peak_gene_up = peak_gene_up[:, tmp_index_list]
    print(f"peak_gene_up.shape: {peak_gene_up.shape}")

    gene_gene = sp_matrix[tmp_index_list, :]
    gene_gene = gene_gene[:, tmp_index_list]
    print(f"gene_gene.shape: {gene_gene.shape}")

    top = hstack([peak_peak, peak_gene_up])
    bottom = hstack([peak_gene_down, gene_gene])
    result = vstack([top, bottom])

    # Set diagonal values to 1
    result.setdiag(1)

    rows, cols = result.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)

    return result, edge_index


"""
==============================================================================
New Helper
==============================================================================
"""


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


def evaluate_ari2(cell_embed, adata):
    adata.obsm['cell_embed'] = cell_embed
    clustering_func, best_clustering_method, best_n_neighbor = None, None, None
    clustering_methods = ["leiden"]
    best_resolution, best_ari, best_nmi = 0, 0, 0
    resolutions = [0.15]
    n_neighbors = [30]
    # resolutions = [0.15, 0.20]
    # n_neighbors = [30, 50]
    for n_neighbor in n_neighbors:
        sc.pp.neighbors(adata, use_rep="cell_embed", n_neighbors=n_neighbor)
        for clustering_method in clustering_methods:
            if clustering_method == 'louvain':
                clustering_func = sc.tl.louvain

            if clustering_method == 'leiden':
                clustering_func = sc.tl.leiden

            for resolution in resolutions:
                col = f'{clustering_method}'
                if clustering_func is not None:
                    clustering_func(adata, resolution=resolution, key_added=col)
                ari = adjusted_rand_score(adata.obs['Celltype'], adata.obs[col])
                nmi = normalized_mutual_info_score(adata.obs['Celltype'], adata.obs[col])
                if ari > best_ari:
                    best_resolution = resolution
                    best_ari = ari
                    best_clustering_method = clustering_method
                    best_n_neighbor = n_neighbor
                if nmi > best_nmi:
                    best_nmi = nmi

    return f'{best_n_neighbor}_{best_clustering_method}_{best_resolution}', best_ari, best_nmi


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul_(std).add_(mu)


def split_tensor(tensor, num_rows):
    """
    split (gene+peak) x emb tensor to gene x emb tensor and peak x emb tensor
    """
    if num_rows >= tensor.shape[0]:
        raise ValueError("num_rows should be less than tensor's number of rows")

    tensor_len = tensor.shape[0]
    top_matrix = tensor[:num_rows, :]
    bottom_matrix = tensor[num_rows:, :]

    return top_matrix, bottom_matrix


def train_one_epoch(encoder1, encoder2, gnn, mlp1, mlp2, graph_dec, decoder1, decoder2, optimizer,
                    RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized,
                    feature_matrix, edge_index):
    encoder1.train()
    encoder2.train()
    gnn.train()
    mlp1.train()
    mlp2.train()
    graph_dec.train()
    decoder1.train()
    decoder2.train()

    optimizer.zero_grad()

    encoder1.zero_grad()
    encoder2.zero_grad()
    gnn.zero_grad()
    mlp1.zero_grad()
    mlp2.zero_grad()
    graph_dec.zero_grad()
    decoder1.zero_grad()
    decoder2.zero_grad()

    mu1, log_sigma1, kl_theta1 = encoder1(RNA_tensor_normalized)
    mu2, log_sigma2, kl_theta2 = encoder2(ATAC_tensor_normalized)

    z1 = reparameterize(mu1, log_sigma1)
    theta1 = F.softmax(z1, dim=-1)

    z2 = reparameterize(mu2, log_sigma2)
    theta2 = F.softmax(z2, dim=-1)

    # print(f"theta1:\n {theta1}")
    # print(f"theta2:\n {theta2}")

    # (gene + peak) x emb
    emb = gnn(feature_matrix, edge_index)
    act_emb = mlp1(emb)

    # num_of_peak x emb, num_of_gene x emb
    eta, rho = split_tensor(act_emb, ATAC_tensor_normalized.shape[1])

    pred_RNA_tensor = decoder1(theta1, rho)
    pred_ATAC_tensor = decoder2(theta2, eta)
    # pred_RNA_tensor = decoder1(theta1)
    # pred_ATAC_tensor = decoder2(theta2)

    """
    modify loss here
    """
    recon_loss1 = -(pred_RNA_tensor * RNA_tensor).sum(-1)
    recon_loss2 = -(pred_ATAC_tensor * ATAC_tensor).sum(-1)
    recon_loss = (recon_loss1 + recon_loss2).mean()
    kl_loss = kl_theta1 + kl_theta2

    loss = recon_loss + kl_loss
    loss.backward()

    clamp_num = 2.0
    torch.nn.utils.clip_grad_norm_(encoder1.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(encoder2.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(gnn.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(mlp1.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(mlp2.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(graph_dec.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(decoder1.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(decoder2.parameters(), clamp_num)

    optimizer.step()

    return torch.sum(recon_loss).item(), torch.sum(kl_loss).item()


# get sample encoding theta from the trained encoder network
def get_theta_GNN(encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2, RNA_tensor_normalized,
                  ATAC_tensor_normalized, metric='theta'):
    encoder1.eval()
    encoder2.eval()
    gnn.eval()
    mlp1.eval()
    mlp2.eval()
    decoder1.eval()
    decoder2.eval()

    with torch.no_grad():
        mu1, log_sigma1, kl_theta1 = encoder1(RNA_tensor_normalized)
        mu2, log_sigma2, kl_theta2 = encoder2(ATAC_tensor_normalized)

        z1 = reparameterize(mu1, log_sigma1)
        theta1 = F.softmax(z1, dim=-1)

        z2 = reparameterize(mu2, log_sigma2)
        theta2 = F.softmax(z2, dim=-1)

        """
        change metric here, theta or mu
        """
        rate1 = 0.5
        rate2 = 0.5

        theta = None
        if metric == 'theta':
            theta = rate1 * theta1 + rate2 * theta2
        if metric == 'mu':
            theta = rate1 * mu1 + rate2 * mu2

        return theta, theta1, theta2


def get_beta_GNN(encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
                 RNA_tensor_normalized, ATAC_tensor_normalized,
                 feature_matrix, edge_index, use_mlp):
    encoder1.eval()
    encoder2.eval()
    gnn.eval()
    mlp1.eval()
    mlp2.eval()
    decoder1.eval()
    decoder2.eval()

    with torch.no_grad():
        mu1, log_sigma1, kl_theta1 = encoder1(RNA_tensor_normalized)
        mu2, log_sigma2, kl_theta2 = encoder2(ATAC_tensor_normalized)

        z1 = reparameterize(mu1, log_sigma1)
        theta1 = F.softmax(z1, dim=-1)

        z2 = reparameterize(mu2, log_sigma2)
        theta2 = F.softmax(z2, dim=-1)

        emb = gnn(feature_matrix, edge_index)
        new_emb = mlp1(emb)
        # num_of_peak x emb, num_of_gene x emb
        # eta, rho = split_tensor(emb, ATAC_tensor_normalized.shape[1])
        eta, rho = split_tensor(new_emb, ATAC_tensor_normalized.shape[1])

        pred_gene_gene = torch.mm(rho, rho.t())
        pred_gene_gene = torch.log(pred_gene_gene + 1e-6)

        pred_peak_peak = torch.mm(eta, eta.t())
        pred_peak_peak = torch.log(pred_peak_peak + 1e-6)
        if use_mlp:
            rho = mlp1(rho)
            eta = mlp2(eta)
        pred_RNA_tensor = decoder1(theta1, rho)
        pred_ATAC_tensor = decoder2(theta2, eta)

        beta1 = decoder1.get_beta()
        beta2 = decoder2.get_beta()

        return beta1, beta2
