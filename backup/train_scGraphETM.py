import correlation
import helper
import models
import plot
import select_gpu
import ScGraphETM
import dataloader as dl
import new_models


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


def train_scGraphETM1(trainer, batches, epochs, ari_freq):
    best_ari = 0
    best_theta = None
    num_of_ari = int(epochs / ari_freq)
    ari_perf = np.ndarray(shape=(num_of_ari, 4), dtype='float')
    for epoch in range(epochs):
        total_loss = 0
        scRNA_adata, scRNA_adata_normalized, scATAC_adata, scATAC_adata_normalized = None, None, None, None
        scRNA_tensor, scRNA_normalized_tensor, scATAC_tensor, scATAC_normalized_tensor = None, None, None, None
        for i, batch in enumerate(tqdm(batches)):
            scRNA_adata, scRNA_adata_normalized, scATAC_adata, scATAC_adata_normalized = batch
            print("======  convert to gpu tensor  ======")
            scRNA_tensor = dl.to_gpu_tensor(adata=scRNA_adata, device=device)
            scRNA_normalized_tensor = dl.to_gpu_tensor(adata=scRNA_adata_normalized, device=device)
            scATAC_tensor = dl.to_gpu_tensor(adata=scATAC_adata, device=device)
            scATAC_normalized_tensor = dl.to_gpu_tensor(adata=scATAC_adata_normalized, device=device)

            print("=======  generate gene-gene cor_matrix  ======")
            gene_correlation, gene_cor_matrix = correlation.get_one_modality_cor(
                adata=scRNA_adata, rate1=0.6, rate2=-0.6, dis_rate=1
            )

            print("=======  generate peak-peak cor_matrix  ======")
            peak_correlation, peak_cor_matrix = correlation.get_one_modality_cor(
                adata=scATAC_adata, rate1=0.6, rate2=-0.6, dis_rate=1
            )

            edge_index1 = correlation.convert_to_edge_index(cor_matrix=gene_correlation, device=device)
            edge_index2 = correlation.convert_to_edge_index(cor_matrix=peak_correlation, device=device)

            loss, nll1_mod1, nll1_mod2, nll2, kl_mod1, kl_mod2 = trainer.train(
                X_mod1=scRNA_tensor,
                X_mod2=scATAC_tensor,
                A_mod1=gene_correlation,
                edge_index1=edge_index1,
                A_mod2=peak_correlation,
                edge_index=edge_index2
            )
            total_loss += loss

        if epoch % ari_freq == 0:
            theta, theta_mu = trainer.get_theta(X_mod1=scRNA_normalized_tensor, X_mod2=scATAC_normalized_tensor)
            ari = helper.evaluate_ari(theta.to('cpu'), adata=scRNA_adata)

            i = int(epoch / ari_freq)
            ari_perf[i, 0] = i
            ari_perf[i, 1] = ari
            print("iter: " + str(epoch) + " ari: " + str(ari))
            if best_ari < ari:
                best_ari = ari
                best_theta = theta

    return trainer, ari_perf, best_ari, best_theta


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
        dl.H5AD_Dataset(
            rna_path=rna_path,
            atac_path=atac_path,
            num_of_cell=num_of_cell,
            num_of_gene=num_of_gene,
            num_of_peak=num_of_peak
        ),
        batch_size=mini_batch_size,
        shuffle=shuffle,
        collate_fn=dl.custom_collate_fn,
        num_workers=num_cores
    )

    """
    ==============================================================================
    Construct the model ScGraphETM1
        --> ScGraphETM1(encoder_mod1, encoder_mod2, decoder_mod1, decoder_mod2, gnn_mod1, gnn_mod2)
    ==============================================================================
    """
    encoder_mod1 = new_models.VAE(num_of_gene, t_hidden_size, num_of_topic)
    encoder_mod2 = new_models.VAE(num_of_peak, t_hidden_size, num_of_topic)

    decoder_mod1 = new_models.LDEC(mini_batch_size, num_of_gene, emb_size, num_of_topic)
    decoder_mod2 = new_models.LDEC(mini_batch_size, num_of_peak, emb_size, num_of_topic)

    gnn_mod1 = new_models.GNN(
        in_channels=graph_feature_size,
        hidden_channels=graph_feature_size * 2,
        out_channels=graph_feature_size,
        num_heads=1,
        device=device,
        dropout=0.2,
        conv_model=conv_model
    ).to(device)

    gnn_mod2 = new_models.GNN(
        in_channels=graph_feature_size,
        hidden_channels=graph_feature_size * 2,
        out_channels=graph_feature_size,
        num_heads=1,
        device=device,
        dropout=0.2,
        conv_model=conv_model
    ).to(device)

    scGraphETM1 = ScGraphETM.ScGraphETM1(
        encoder_mod1,
        encoder_mod2,
        decoder_mod1,
        decoder_mod2,
        gnn_mod1,
        gnn_mod2
    )

    trainer, ari_perf, best_ari, best_theta = train_scGraphETM1(
        trainer=scGraphETM1,
        batches=h5ad_dataloader,
        epochs=num_of_epochs,
        ari_freq=ari_freq
    )

