import torch
from torch.nn import functional as F
from sklearn.metrics import adjusted_rand_score

import scanpy as sc
import numpy as np

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
                   adata2, ari_freq, niter=100):
    perf = np.ndarray(shape=(niter, 2), dtype='float')
    num_of_ari = (int)(niter / ari_freq)
    ari_perf = np.ndarray(shape=(num_of_ari, 4), dtype='float')
    best_ari = 0
    best_theta = None
    for i in range(niter):
        NELBO = train_GCNscETM_helper(model, optimizer, RNA_tensor, RNA_tensor_normalized, ATAC_tensor,
                                      ATAC_tensor_normalized)
        perf[i, 0] = i
        perf[i, 1] = NELBO
        if (i % ari_freq == 0):
            theta, theta_gene, theta_peak = get_theta_GCN(model, RNA_tensor_normalized, ATAC_tensor_normalized)
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
    ari = adjusted_rand_score(adata.obs['cell_type'], adata.obs['louvain'])
    return ari


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
                       feature_matrix, edge_index, true_gene_expression, freq, loss_fn,
                       feature_matrix_val=None, edge_index_val=None, true_gene_expression_val=None):
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
