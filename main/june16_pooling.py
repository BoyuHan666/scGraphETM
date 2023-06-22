import correlation
import june16_dataloader as data
import june17_helper as helper
import june16_models as models
import plot
import select_gpu

import torch
from torch import optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from tqdm import tqdm
import time
import multiprocessing
import numpy as np
import anndata as ad
import scipy.sparse as sp

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

num_of_cell = 2000
num_of_gene = 1000
num_of_peak = 1000
graph_size = 5000
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
num_of_epochs = 100
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
        # selected_gpu = select_gpu.get_lowest_usage_gpu_index(print_usage=False)
        torch.cuda.set_device(8)
        # device = torch.device("cuda:{}".format(selected_gpu))
        device = torch.device("cuda")
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

    gnn_model = models.GNN(input_channels=num_of_cell,
        hidden_channels=graph_feature_size,
        output_channels=num_of_cell,
        num_heads=1,
        device=device,
        dropout=0.2,
        conv_model="GCN"
                       ).to(device)

    etm_model = models.ETM(
        num_topics=num_of_topic,
        num_gene=num_of_gene,
        num_peak=num_of_peak,
        t_hidden_size=t_hidden_size,
        t_hidden_size_peak=t_hidden_size,
        rho_eta = None,
        emb_size=emb_size,
        graph_feature_size=graph_feature_size,
        theta_act='relu',
        device=device,
        enc_drop=0.1,
    ).to(device)
    gnn_optimizer = optim.Adam(gnn_model.parameters(), lr=0.0005)
    etm_optimizer = optim.Adam(etm_model.parameters(), lr=0.0005, weight_decay=1.2e-6)

    GNN_ETM_model, GNN_ETM_perf, ari_perf, best_ari, theta = helper.train_GCNscETM(
        gnn_model=gnn_model,
        gnn_optimizer=gnn_optimizer,
        etm_model=etm_model,
        etm_optimizer=etm_optimizer,
        h5ad_dataloader=dataset,
        graph_size=graph_size,
        mini_batch_size=mini_batch_size,
        device=device,
        ari_freq=ari_freq,
        niter=num_of_epochs
    )

