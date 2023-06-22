import correlation
import june16_dataloader as data
import june13_helper as helper
import june13_models as models
import plot
import select_gpu

import torch
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
import time
import multiprocessing
import numpy as np
import anndata as ad

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

# 9631 cell × 29095 gene
rna_path = "../data/10x-Multiome-Pbmc10k-RNA.h5ad"
# 9631 cell × 107194 peak
atac_path = "../data/10x-Multiome-Pbmc10k-ATAC.h5ad"

# num_cores = 20 # max = 40
num_cores = multiprocessing.cpu_count()

# num_of_cell = 0
# num_of_gene = 0
# num_of_peak = 0

num_of_cell = 6000
num_of_gene = 2000
num_of_peak = 2000
graph_size = 5000
mini_batch_size = num_of_cell
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
ari_freq = 500
num_of_epochs = 510
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
    load dataset
    ==============================================================================
    """
    dataset = data.H5AD_Dataset(
            rna_path=rna_path,
            atac_path=atac_path,
            num_of_cell=num_of_cell,
            num_of_gene=num_of_gene,
            num_of_peak=num_of_peak
        )
    # feature_matrix = torch.randn((graph_size + graph_size, graph_feature_size)).to(device)
    # edge_index = correlation.convert_to_edge_index(cor_matrix=dataset.cor_matrix, device=device)
    # total_graph = Data(x=feature_matrix, edge_index=edge_index)
    # print(total_graph.num_nodes)
    #
    # total_graph.train_idx = torch.randperm(graph_size + graph_size)
    # total_graph.to(device)
    # train_loader = DataLoader([total_graph],
    #                           batch_size=50,
    #                           shuffle=True)
    # print(train_loader)
    # for batch in train_loader:
    #     print(batch)

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
    rna_test, rna_norm_test, atac_test, atac_norm_test, cor_matrix = dataset.getTrain()
    edge_index = correlation.convert_to_edge_index(cor_matrix=cor_matrix, device=device)
    print(edge_index)
    feature_matrix = torch.randn((num_of_peak + num_of_gene, graph_feature_size)).to(device)
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

    start_time = time.time()

    print("======  start training  ======")
    # GNN_ETM_optimizer = optim.Adam(GNN_ETM_model.parameters(), lr=0.0005, weight_decay=1.2e-6)
    GNN_ETM_optimizer = optim.AdamW(GNN_ETM_model.parameters(), lr=0.001, weight_decay=1.2e-6)
    GNN_ETM_model, GNN_ETM_perf, ari_perf, best_ari, theta = helper.train_GCNscETM(
        model=GNN_ETM_model,
        optimizer=GNN_ETM_optimizer,
        h5ad_dataloader=dataset,
        device=device,
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
