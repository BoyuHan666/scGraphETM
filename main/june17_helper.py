import torch
from torch.nn import functional as F
from sklearn.metrics import adjusted_rand_score
from torch_geometric.utils import negative_sampling

import scanpy as sc
import numpy as np
import correlation
import june16_dataloader
import models
import plot
import select_gpu
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn as nn

"""
==============================================================================
scGCN - ETM Helper
==============================================================================
"""


# train the VAE for one epoch
def train_GCNscETM_ELBO(gnn_model,
                        gnn_optimizer,
                        etm_model,
                        etm_optimizer,
                        rna,
                        rna_norm,
                        atac,
                        atac_norm,
                        device,
                        train_loader):
    # initialize the model and loss

    elbo_loss = []
    t = []
    t1 = []
    t2 = []
    epoch = 0
    rho_etas = train_GNN(gnn_model, gnn_optimizer, train_loader, device)
    for data in train_loader:  # Iterate in batches over the training dataset.\
        # print("======  convert to gpu tensor  ======")
        # gnn_model.train()
        # mse = nn.MSELoss()
        # gnn_optimizer.zero_grad()
        # out = gnn_model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        # # adjacency_matrix = torch.sigmoid(out)
        #
        # preds = torch.mm(out, out.t())
        # # preds = torch.log(preds + 1e-6)
        # loss = mse(preds, data.y)
        # loss = loss.mean().clone()
        etm_model.train()
        RNA_tensor = june13_dataloader.to_gpu_tensor(adata=rna[epoch], device=device)
        RNA_normalized_tensor = june13_dataloader.to_gpu_tensor(adata=rna_norm[epoch], device=device)
        sums = RNA_tensor.sum(1).unsqueeze(1).to('cuda')
        # RNA_normalized_tensor = (RNA_tensor / sums).to('cuda')

        ATAC_tensor = june13_dataloader.to_gpu_tensor(adata=atac[epoch], device=device)
        ATAC_normalized_tensor = june13_dataloader.to_gpu_tensor(adata=atac_norm[epoch], device=device)
        # ATAC_normalized_tensor = (ATAC_tensor / ATAC_tensor.max()).to('cuda')
        for i in range(100):
            rho_eta = rho_etas[epoch]
            etm_optimizer.zero_grad()

            nll, kl_theta = etm_model(RNA_tensor, RNA_normalized_tensor, ATAC_tensor, ATAC_normalized_tensor,
                                      rho_eta)
            elbo = nll + kl_theta
            elbo.backward(retain_graph=True)  # Calculate and backprop gradients w.r.t. negative ELBO

            # clip gradients to 2.0 if it gets too large
            torch.nn.utils.clip_grad_norm_(etm_model.parameters(), 2.0)

            # update model to minimize negative ELBO
            #  Update parameters based on gradients.

            etm_optimizer.step()

        elbo_loss.append(torch.sum(elbo).item())
        theta, theta_gene, theta_peak = get_theta_GCN(etm_model, RNA_normalized_tensor, ATAC_normalized_tensor)
        t.append(theta.detach().clone())
        t1.append(theta_gene.detach().clone())
        t2.append(theta_peak.detach().clone())
        epoch += 1
    return elbo_loss, t, t1, t2


def train_GNN(gnn_model, gnn_optimizer, train_loader, device):
    gnn_model.train()
    epoch = 0
    mse = nn.MSELoss()
    embeddings = []
    for data in train_loader:  # Iterate in batches over the training dataset.\
        gnn_optimizer.zero_grad()  # Clear gradients.
        # for i in range(100):

        out = gnn_model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        # adjacency_matrix = torch.sigmoid(out)
        embeddings.append(gnn_model.get_embedding().detach().clone())
        epsilon = 1e-8
        preds = torch.mm(out, out.t())
        # preds = torch.log(preds + 1e-6)
        loss = mse(preds, data.y)
        loss.mean().backward()  # Derive gradients.
        gnn_optimizer.step()
        print('Batch [{}/{}], Loss: {:.4f}'.format(1, 100, torch.mean(loss).item()))
    return embeddings


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


def train_GCNscETM(gnn_model,
                   gnn_optimizer,
                   etm_model,
                   etm_optimizer,
                   h5ad_dataloader,
                   graph_size,
                   mini_batch_size,
                   device,
                   ari_freq,
                   niter=100):
    all_graphs = []
    # print(total_graph.num_nodes)
    rna = []
    rna_norm = []
    atac = []
    atac_norm = []
    for i in range(5):
        rna_subsample, rna_subsample_norm, atac_subsample, atac_subsample_norm, cor_matrix, feature_matrix, indices = h5ad_dataloader.getBatch(
            mini_batch_size, i)
        sub_feature_matrix = torch.randn(cor_matrix.shape[0], 2000).to(device)
        sub_edge_index = correlation.convert_to_edge_index(cor_matrix=cor_matrix, device=device)
        rna_np = rna_subsample.X.toarray()
        row_norms = np.linalg.norm(rna_np, axis=1, keepdims=True)
        normalized_matrix1 = rna_np / row_norms

        atac_np = atac_subsample.X.toarray()
        row_norms = np.linalg.norm(rna_np, axis=1, keepdims=True)
        normalized_matrix2 = atac_np / row_norms

        # concat = np.vstack(normalized_matrix1, normalized_matrix2)
        y1 = torch.tensor(normalized_matrix1.transpose())
        y2 = torch.tensor(normalized_matrix2.transpose())
        y = torch.cat((y1, y2), dim=0)
        # y = torch.tensor(cor_matrix)
        sub_graph = Data(x=sub_feature_matrix, edge_index=sub_edge_index, y=y).to(device)
        all_graphs.append(sub_graph)

        rna.append(rna_subsample)
        rna_norm.append(rna_subsample_norm)
        atac.append(atac_subsample_norm)
        atac_norm.append(atac_subsample_norm)

    train_loader = DataLoader(all_graphs,
                              batch_size=1,
                              shuffle=True)
    for batch in train_loader:
        print(batch)

    perf = np.ndarray(shape=(niter, 2), dtype='float')
    num_of_ari = int(niter / ari_freq)
    ari_perf = np.ndarray(shape=(num_of_ari, 4), dtype='float')
    best_ari = 0
    best_theta = None

    for i in range(niter):
        # theta_list, theta_gene_list, theta_peak_list = get_theta_GCN(model, tensor1_list, tensor2_list)

        NELBO, t, t1, t2 = train_GCNscETM_ELBO(gnn_model,
                                               gnn_optimizer,
                                               etm_model,
                                               etm_optimizer,
                                               rna,
                                               rna_norm,
                                               atac,
                                               atac_norm,
                                               device,
                                               train_loader)
        for index in range(len(NELBO)):
            perf[index, 0] = index
            perf[index, 1] = NELBO[index]

            theta = t[index]
            theta_gene = t1[index]
            theta_peak = t2[index]
            ari = evaluate_ari(theta.to('cpu'), rna[index])
            ari1 = evaluate_ari(theta_gene.to('cpu'), rna[index])
            ari2 = evaluate_ari(theta_peak.to('cpu'), atac[index])
            # idx = int(index / ari_freq)
            # ari_perf[idx, 0] = idx
            # ari_perf[idx, 1] = ari
            # ari_perf[idx, 2] = ari1
            # ari_perf[idx, 3] = ari2
            print("iter: " + str(index) + " batch: " + str(i) + " ari: " + str(ari))
            if (best_ari < ari):
                best_ari = ari
                best_theta = theta
            else:
                if index % 100 == 0:
                    print("iter: " + str(i))
            if i % 1 == 0:
                print("Epoch: ", i)
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
