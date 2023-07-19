import torch
from scipy import stats
from torch.nn import functional as F
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.autograd import Variable
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch import optim
from torch_geometric.nn import InnerProductDecoder
import sys
from torch_geometric.utils import negative_sampling
from scipy.sparse import csr_matrix

import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

import select_gpu
import mini_batch
import full_batch
import helper2
import model2
import view_result
import time

if __name__ == "__main__":
    rna_path = "../data/10x-Multiome-Pbmc10k-RNA.h5ad"
    atac_path = "../data/10x-Multiome-Pbmc10k-ATAC.h5ad"
    num_of_cell = 2000
    num_of_gene = 2000
    num_of_peak = 2000
    test_num_of_cell = 2000
    batch_size = 2000
    batch_num = 10
    emb_size = 512
    emb_size2 = 256
    num_of_topic = 40
    title = 'PEAKS_' + str(num_of_peak) + '_1024'
    gnn_conv = 'GATv2'
    num_epochs = 500
    ari_freq = 10
    plot_path_rel = "./plot/"
    metric = 'theta'  # mu or theta
    lr = 0.001
    use_mlp = False
    use_mask_train = False
    use_mask_reconstruct = False  # False: one side mask for reconstructing the masked expressions
    mask_ratio = 0.2

    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index()
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

    train_set, total_training_set, test_set, scRNA_adata, scATAC_adata, mask_matrix1, mask_matrix2 = full_batch.process_full_batch_data(
        rna_path=rna_path,
        atac_path=atac_path,
        device=device,
        num_of_cell=num_of_cell,
        num_of_gene=num_of_gene,
        num_of_peak=num_of_peak,
        test_num_of_cell=test_num_of_cell,
        emb_size=emb_size,
        use_highly_variable=True,
        cor='pearson',
        use_mask=use_mask_train,
        mask_ratio=mask_ratio
    )

    (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
     scRNA_anndata, scATAC_anndata, gene_correlation_matrix,
     peak_correlation_matrix, feature_matrix, cor_mat, edge_index, X_rna_tensor_copy, X_atac_tensor_copy) = train_set
    print(cor_mat)

    rna_feature = scRNA_anndata.X.copy()
    binary_matrix1 = np.where(rna_feature != 0.0, 1.0, rna_feature)
    atac_feature = (scATAC_adata.X.toarray()).copy()
    binary_matrix2 = np.where(atac_feature != 0.0, 1.0, atac_feature)
    binary_matrix2 = binary_matrix2[:num_of_cell, :num_of_peak]

    torch.set_printoptions(threshold=sys.maxsize)
    # print(gene_correlation_matrix[gene_correlation_matrix > 0.75].shape)
    gene_correlation_matrix[gene_correlation_matrix > 0.75] = 1.0
    gene_correlation_matrix[gene_correlation_matrix < 0.75] = 0.0

    peak_correlation_matrix[peak_correlation_matrix > 0.5] = 1.0
    peak_correlation_matrix[peak_correlation_matrix < 0.5] = 0.0

    gnn = model2.GNN(emb_size, emb_size2 * 2, emb_size2, 1, device, 0, gnn_conv).to(device)
    gnn_decoder = model2.InnerProductDecoder(0).to(device)
    parameters1 = [{'params': gnn.parameters()},
                   {'params': gnn_decoder.parameters()}]
    optimizer = optim.Adam(parameters1, lr=lr, weight_decay=1.2e-6)

    for epoch in range(200):
        t = time.time()
        gnn.train()
        gnn_decoder.train()
        optimizer.zero_grad()
        # recovered, mu, logvar = model(feature_matrix2, edge_index)
        # loss = loss_function(preds=recovered, labels=cor_label,
        #                      mu=mu, logvar=logvar, n_nodes=num_of_peak + num_of_gene,
        #                      norm=norm, pos_weight=pos_weight)

        z = gnn(feature_matrix, edge_index)
        emb = z
        eta, rho = helper2.split_tensor(emb, ATAC_tensor_normalized.shape[1])
        EPS = 1e-15

        edge_index_gene = gene_correlation_matrix.nonzero().t()
        pos_loss_gene = -torch.log(gnn_decoder(rho, edge_index_gene, sigmoid=True) + EPS).mean()
        neg_edge_index = None
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(edge_index_gene, rho.size(0))
        neg_loss_gene = -torch.log(1 - gnn_decoder(rho, neg_edge_index, sigmoid=True) + EPS).mean()
        recon_loss3 = pos_loss_gene + neg_loss_gene
        print(recon_loss3)

        edge_index_peak = peak_correlation_matrix.nonzero().t()
        pos_loss_peak = -torch.log(gnn_decoder(eta, edge_index_peak, sigmoid=True) + EPS).mean()
        neg_edge_index = None
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(edge_index_peak, eta.size(0))
        neg_loss_peak = -torch.log(1 - gnn_decoder(eta, neg_edge_index, sigmoid=True) + EPS).mean()
        recon_loss4 = pos_loss_peak + neg_loss_peak
        print(recon_loss4)

        loss = (recon_loss4+recon_loss3).mean()
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              # "val_ap=", "{:.5f}".format(ap_curr),
              # "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")

