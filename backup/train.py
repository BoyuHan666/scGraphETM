import correlation
import dataloader
import helper
import models
import plot
import select_gpu

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import multiprocessing
import numpy as np
import anndata as ad

# 9631 cell × 29095 gene
rna_path = "../data/10x-Multiome-Pbmc10k-RNA.h5ad"
# 9631 cell × 107194 peak
atac_path = "../data/10x-Multiome-Pbmc10k-ATAC.h5ad"

# num_cores = 20 # max = 40
num_cores = multiprocessing.cpu_count()

# num_of_cell = 0
# num_of_gene = 0
# num_of_peak = 0

num_of_cell = 8000
num_of_gene = 2000
num_of_peak = 2000
mini_batch_size = 1000
shuffle = True
graph_feature_size = 512

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
ari_freq = 100
num_of_epochs = 2000
# one from GATv2, GAT, GCN
conv_model = 'GATv2'
# nll, kl, bce
recon_metric = 'nll'

# plot_path_rel = "./Ma_plot/"
plot_path_rel = "./han_plot/"

if __name__ == '__main__':
    # Check GPU availability
    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index(print_usage=False)
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

    """
    ==============================================================================
    load data
    ==============================================================================
    """
    h5ad_dataloader = DataLoader(
        dataloader.H5AD_Dataset(
            rna_path=rna_path,
            atac_path=atac_path,
            num_of_cell=num_of_cell,
            num_of_gene=num_of_gene,
            num_of_peak=num_of_peak
        ),
        batch_size=mini_batch_size,
        shuffle=shuffle,
        collate_fn=dataloader.custom_collate_fn,
        num_workers=num_cores
    )

    """
    ==============================================================================
    create model
    ==============================================================================
    """
    gnn_model = models.GNN(
        in_channels=graph_feature_size,
        hidden_channels=graph_feature_size * 2,
        out_channels=graph_feature_size,
        num_heads=1,
        device=device,
        dropout=0.2,
        conv_model=conv_model
    ).to(device)

    GNN_ETM_model = None
    Previous_GNN_ETM_model = None

    # GNN_ETM_optimizer = optim.Adam(GNN_ETM_model.parameters(), lr=0.0005, weight_decay=1.2e-6)
    # GNN_ETM_optimizer2 = optim.SGD(GNN_ETM_model.parameters(), lr=0.0005)

    start_time = time.time()

    all_scRNA_adata_selected = ad.read_h5ad(rna_path)[:num_of_cell, :num_of_gene]
    all_scATAC_adata_selected = ad.read_h5ad(atac_path)[:num_of_cell, :num_of_peak]
    rna_copy = all_scRNA_adata_selected.copy()
    atac_copy = all_scATAC_adata_selected.copy()
    rna_copy_normalized = dataloader.normalize(rna_copy)
    atac_copy_normalized = dataloader.normalize(atac_copy)
    rna_copy_normalized_tensor = dataloader.to_gpu_tensor(rna_copy_normalized, device)
    atac_copy_normalized_tensor = dataloader.to_gpu_tensor(atac_copy_normalized, device)

    print("======  start training  ======")

    for i, batch in enumerate(h5ad_dataloader):
        scRNA_adata, scRNA_adata_normalized, scATAC_adata, scATAC_adata_normalized = batch
        print(scRNA_adata_normalized)
        print(scATAC_adata_normalized)
        print("======  convert to gpu tensor  ======")
        scRNA_tensor = dataloader.to_gpu_tensor(adata=scRNA_adata, device=device)
        scRNA_normalized_tensor = dataloader.to_gpu_tensor(adata=scRNA_adata_normalized, device=device)
        scATAC_tensor = dataloader.to_gpu_tensor(adata=scATAC_adata, device=device)
        scATAC_normalized_tensor = dataloader.to_gpu_tensor(adata=scATAC_adata_normalized, device=device)

        print("=======  generate gene-gene cor_matrix  ======")
        gene_correlation, gene_cor_matrix = correlation.get_one_modality_cor(
            adata=scRNA_adata, rate1=0.6, rate2=-0.6, dis_rate=1
        )

        print("=======  generate peak-peak cor_matrix  ======")
        peak_correlation, peak_cor_matrix = correlation.get_one_modality_cor(
            adata=scATAC_adata, rate1=0.6, rate2=-0.6, dis_rate=1
        )

        print("=======  generate peak-gene cor_matrix  ======")
        cor_matrix = correlation.get_two_modality_cor(
            scRNA_adata=scRNA_adata, scATAC_adata=scATAC_adata,
            gene_cor_mat=gene_cor_matrix, peak_cor_mat=peak_cor_matrix)

        edge_index = correlation.convert_to_edge_index(cor_matrix=cor_matrix, device=device)

        feature_matrix = torch.randn((num_of_peak + num_of_gene, graph_feature_size)).to(device)

        if GNN_ETM_model is None:
            GNN_ETM_model = models.GNN_ETM(
                num_topics=num_of_topic,
                num_gene=num_of_gene,
                num_peak=num_of_peak,
                t_hidden_size=t_hidden_size,
                t_hidden_size_peak=t_hidden_size,
                rho_size=rho_size,
                eta_size=eta_size,
                emb_size=emb_size,
                graph_feature_size=graph_feature_size,
                theta_act='relu',
                device=device,
                enc_drop=0.1,
                use_gnn=True,
                gnn_model=gnn_model,
                feature_matrix=feature_matrix,
                edge_index=edge_index
            ).to(device)
            Previous_GNN_ETM_model = GNN_ETM_model
        else:
            GNN_ETM_model = models.GNN_ETM(
                num_topics=num_of_topic,
                num_gene=num_of_gene,
                num_peak=num_of_peak,
                t_hidden_size=t_hidden_size,
                t_hidden_size_peak=t_hidden_size,
                rho_size=rho_size,
                eta_size=eta_size,
                emb_size=emb_size,
                graph_feature_size=graph_feature_size,
                theta_act='relu',
                device=device,
                enc_drop=0.1,
                use_gnn=True,
                gnn_model=gnn_model,
                feature_matrix=feature_matrix,
                edge_index=edge_index
            ).to(device)
            GNN_ETM_model.load_state_dict(Previous_GNN_ETM_model.state_dict())

        GNN_ETM_optimizer = optim.Adam(GNN_ETM_model.parameters(), lr=0.0005, weight_decay=1.2e-6)

        print(i, ": training")
        GNN_ETM_model, GNN_ETM_perf, ari_perf, best_ari, theta = helper.train_GCNscETM(
            model=GNN_ETM_model,
            optimizer=GNN_ETM_optimizer,
            RNA_tensor=scRNA_tensor,
            RNA_tensor_normalized=scRNA_normalized_tensor,
            ATAC_tensor=scATAC_tensor,
            ATAC_tensor_normalized=scATAC_normalized_tensor,
            adata=scRNA_adata,
            adata2=scATAC_adata,
            ari_freq=ari_freq,
            niter=num_of_epochs
        )

    end_time = time.time()
    elapsed_time = end_time - start_time

    minutes = elapsed_time // 60
    seconds = elapsed_time % 60

    hours = minutes // 60
    minutes = minutes % 60

    print(f"Elapsed time: {int(hours)} hours:{int(minutes)} minutes:{int(seconds)} seconds")
    theta, theta_g, theta_p = helper.get_theta_GCN(GNN_ETM_model, rna_copy_normalized_tensor, atac_copy_normalized_tensor)
    print("======  calculate the final ari  ======")
    all_cell_ari = helper.evaluate_ari(theta.to('cpu'), all_scRNA_adata_selected)
    print(f"the final ari: {all_cell_ari}")
    print("======  plotting  ======")
    plot.monitor_perf(perf=GNN_ETM_perf, ari_perf=ari_perf, ari_freq=ari_freq, objective="both", path=plot_path_rel)
    plot.generate_cluster_plot(theta=theta, sc_adata=all_scRNA_adata_selected, plot_path_rel=plot_path_rel)
