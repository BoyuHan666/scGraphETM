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
print_all_gpu_usage = False
# num_of_cell = 0
# num_of_gene = 0
# num_of_peak = 0
num_of_cell = 4000
num_of_gene = 2000
num_of_peak = 8000
mini_batch_size = 2000
shuffle = False
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
num_of_epochs = 3000
# one from GATv2, GAT, GCN
conv_model = 'GATv2'

# plot_path_rel = "./Ma_plot/"
plot_path_rel = "./han_plot/"


def train_scGraphETM(trainer, batches, epochs, ari_freq):
    best_ari = 0
    best_theta = None
    num_of_ari = int(epochs / ari_freq)
    perf = np.ndarray(shape=(epochs, 2), dtype='float')
    ari_perf = np.ndarray(shape=(num_of_ari, 4), dtype='float')
    print("======  start training  ======")
    for epoch in range(epochs):
        print(f"======  current epoch: {epoch}  ======")
        total_loss = 0
        scRNA_adata, scRNA_adata_normalized, scATAC_adata, scATAC_adata_normalized = None, None, None, None
        scRNA_tensor, scRNA_normalized_tensor, scATAC_tensor, scATAC_normalized_tensor = None, None, None, None
        for i, batch in enumerate(tqdm(batches)):
            scRNA_adata, scRNA_adata_normalized, scATAC_adata, scATAC_adata_normalized = batch
            """======  convert to gpu tensor  ======"""
            scRNA_tensor = dl.to_gpu_tensor(adata=scRNA_adata, device=device)
            scRNA_normalized_tensor = dl.to_gpu_tensor(adata=scRNA_adata_normalized, device=device)
            scATAC_tensor = dl.to_gpu_tensor(adata=scATAC_adata, device=device)
            scATAC_normalized_tensor = dl.to_gpu_tensor(adata=scATAC_adata_normalized, device=device)

            """=======  generate gene-gene cor_matrix  ======"""
            gene_correlation, gene_cor_matrix = correlation.get_one_modality_cor(
                adata=scRNA_adata, rate1=0.6, rate2=-0.6, dis_rate=1
            )
            gene_correlation_tensor = torch.tensor(gene_correlation, dtype=torch.float32).to(device)

            """=======  generate peak-peak cor_matrix  ======"""
            peak_correlation, peak_cor_matrix = correlation.get_one_modality_cor(
                adata=scATAC_adata, rate1=0.6, rate2=-0.6, dis_rate=1
            )
            peak_correlation_tensor = torch.tensor(peak_correlation, dtype=torch.float32).to(device)

            edge_index1 = correlation.convert_to_edge_index(cor_matrix=gene_cor_matrix, device=device)
            edge_index2 = correlation.convert_to_edge_index(cor_matrix=peak_cor_matrix, device=device)

            loss = trainer.train(
                X_mod1=scRNA_tensor,
                X_mod2=scATAC_tensor,
                A_mod1=gene_correlation_tensor,
                edge_index1=edge_index1,
                A_mod2=peak_correlation_tensor,
                edge_index2=edge_index2
            )
            total_loss += loss

        if (epoch) % ari_freq == 0:
            theta, theta_mu = trainer.get_theta(X_mod1=scRNA_normalized_tensor, X_mod2=scATAC_normalized_tensor)
            ari = helper.evaluate_ari(theta.to('cpu'), adata=scRNA_adata)

            idx = int(epoch / ari_freq)
            ari_perf[idx, 0] = idx
            ari_perf[idx, 1] = ari

            if best_ari < ari:
                best_ari = ari
                best_theta = theta

            print(f"epoch: {epoch}, total loss: {total_loss}, ari: {ari}")

        perf[epoch, 0] = epoch
        perf[epoch, 1] = total_loss
        torch.cuda.empty_cache()

    return trainer, perf, ari_perf, best_ari, best_theta


def train_scGraphETM_without_minibatch(
        model, RNA_tensor, RNA_tensor_normalized,
        ATAC_tensor, ATAC_tensor_normalized,
        adata1, adata2, epochs, ari_freq):
    best_ari = 0
    best_theta = None
    num_of_ari = int(epochs / ari_freq)
    perf = np.ndarray(shape=(epochs, 2), dtype='float')
    ari_perf = np.ndarray(shape=(num_of_ari, 4), dtype='float')

    print("=======  generate peak-gene cor_matrix  ======")
    cor_matrix = correlation.get_two_modality_cor(
        scRNA_adata=adata1, scATAC_adata=adata2
    )

    print("=======  generate edge_index  ======")
    edge_index = correlation.convert_to_edge_index(cor_matrix=cor_matrix, device=device)
    # edge_index1 = correlation.convert_to_edge_index(cor_matrix=gene_cor_matrix, device=device)
    # edge_index2 = correlation.convert_to_edge_index(cor_matrix=peak_cor_matrix, device=device)

    feature_matrix = torch.randn((num_of_peak + num_of_gene, num_of_peak + num_of_gene)).to(device)
    print("======  start training  ======")
    for epoch in range(epochs):
        # print(f"======  current epoch: {epoch + 1}  ======")

        loss = model.train(
            X_mod1=RNA_tensor,
            X_mod2=ATAC_tensor,
            A=feature_matrix,
            edge_index=edge_index,
            split_num=num_of_peak,
        )

        if epoch % ari_freq == 0:
            theta, theta_mu = model.get_theta(X_mod1=RNA_tensor_normalized, X_mod2=ATAC_tensor_normalized)
            ari = helper.evaluate_ari(theta.to('cpu'), adata=adata1)

            idx = int(epoch / ari_freq)
            ari_perf[idx, 0] = idx
            ari_perf[idx, 1] = ari

            if best_ari < ari:
                best_ari = ari
                best_theta = theta_mu

            print(f"epoch: {epoch}, total loss: {loss}, ari: {ari}")

        perf[epoch, 0] = epoch
        perf[epoch, 1] = loss

    return model, perf, ari_perf, best_ari, best_theta


def print_time(st, et):
    delta = et - st
    minutes = delta // 60
    seconds = delta % 60
    hours = minutes // 60
    minutes = minutes % 60
    print(f"total training time: {int(hours)} hours:{int(minutes)} minutes:{int(seconds)} seconds")


if __name__ == '__main__':
    # Check GPU availability
    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index(print_usage=print_all_gpu_usage)
        # select_gpu.get_gpu_with_most_free_memory()
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
    # h5ad_dataloader = DataLoader(
    #     dl.H5AD_Dataset(
    #         rna_path=rna_path,
    #         atac_path=atac_path,
    #         num_of_cell=num_of_cell,
    #         num_of_gene=num_of_gene,
    #         num_of_peak=num_of_peak
    #     ),
    #     batch_size=mini_batch_size,
    #     shuffle=shuffle,
    #     collate_fn=dl.custom_collate_fn,
    #     num_workers=num_cores
    # )

    """
    ==============================================================================
    anndata: all_scRNA_adata_selected, all_scATAC_adata_selected, rna_copy_normalized, atac_copy_normalized
    tensor: rna_tensor, atac_tensor, rna_copy_normalized_tensor, atac_copy_normalized_tensor
    ==============================================================================
    """
    print("=======  processing data  ======")
    start_time = time.time()
    all_scRNA_adata_selected = ad.read_h5ad(rna_path)[:num_of_cell, :]
    all_scATAC_adata_selected = ad.read_h5ad(atac_path)[:num_of_cell, :]
    rna_copy = all_scRNA_adata_selected.copy()
    atac_copy = all_scATAC_adata_selected.copy()
    rna_copy_normalized = dl.normalize(rna_copy)
    atac_copy_normalized = dl.normalize(atac_copy)

    highly_gene_index = dl.get_highly_variable_index(rna_copy_normalized)
    highly_peak_index = dl.get_highly_variable_index(atac_copy_normalized)

    all_scRNA_adata_selected = all_scRNA_adata_selected[:num_of_cell, highly_gene_index]
    all_scRNA_adata_selected = all_scRNA_adata_selected[:num_of_cell, :num_of_gene]

    all_scATAC_adata_selected = all_scATAC_adata_selected[:num_of_cell, highly_peak_index]
    all_scATAC_adata_selected = all_scATAC_adata_selected[:num_of_cell, :num_of_peak]

    rna_copy_normalized = rna_copy_normalized[:num_of_cell, highly_gene_index]
    rna_copy_normalized = rna_copy_normalized[:num_of_cell, :num_of_gene]
    atac_copy_normalized = atac_copy_normalized[:num_of_cell, highly_peak_index]
    atac_copy_normalized = atac_copy_normalized[:num_of_cell, :num_of_peak]

    rna_tensor = dl.to_gpu_tensor(all_scRNA_adata_selected, device)
    atac_tensor = dl.to_gpu_tensor(all_scATAC_adata_selected, device)
    rna_copy_normalized_tensor = dl.to_gpu_tensor(rna_copy_normalized, device)
    atac_copy_normalized_tensor = dl.to_gpu_tensor(atac_copy_normalized, device)
    end_time = time.time()
    print_time(start_time, end_time)

    """
    ==============================================================================
    Construct the model ScGraphETM1
        --> ScGraphETM1(encoder_mod1, encoder_mod2, decoder_mod1, decoder_mod2, gnn_mod1, gnn_mod2)
    ==============================================================================
    """
    encoder_mod1 = new_models.VAE(num_of_gene, t_hidden_size, num_of_topic).to(device)
    encoder_mod2 = new_models.VAE(num_of_peak, t_hidden_size, num_of_topic).to(device)

    decoder_mod1 = new_models.LDEC(mini_batch_size, num_of_gene, emb_size, num_of_topic).to(device)
    decoder_mod2 = new_models.LDEC(mini_batch_size, num_of_peak, emb_size, num_of_topic).to(device)

    gnn_mod1 = new_models.GNN(
        in_channels=num_of_gene,
        hidden_channels=graph_feature_size * 2,
        out_channels=graph_feature_size,
        num_heads=1,
        device=device,
        dropout=0.2,
        conv_model=conv_model
    ).to(device)

    gnn_mod2 = new_models.GNN(
        in_channels=num_of_peak,
        hidden_channels=graph_feature_size * 2,
        out_channels=graph_feature_size,
        num_heads=1,
        device=device,
        dropout=0.2,
        conv_model=conv_model
    ).to(device)

    gnn = new_models.GNN(
        in_channels=num_of_gene+num_of_peak,
        hidden_channels=graph_feature_size * 2,
        out_channels=graph_feature_size,
        num_heads=1,
        device=device,
        dropout=0.2,
        conv_model=conv_model
    ).to(device)

    # scGraphETM = ScGraphETM.ScGraphETM1(
    #     encoder_mod1,
    #     encoder_mod2,
    #     decoder_mod1,
    #     decoder_mod2,
    #     gnn_mod1,
    #     gnn_mod2
    # )

    scGraphETM = ScGraphETM.ScGraphETM2(
        encoder_mod1,
        encoder_mod2,
        decoder_mod1,
        decoder_mod2,
        gnn
    )


    print("=======  generate peak-gene cor_matrix  ======")
    cor_matrix = correlation.get_two_modality_cor(
        scRNA_adata=all_scRNA_adata_selected, scATAC_adata=all_scATAC_adata_selected
    )

    print("=======  generate edge_index  ======")
    edge_index = correlation.convert_to_edge_index(cor_matrix=cor_matrix, device=device)
    # edge_index1 = correlation.convert_to_edge_index(cor_matrix=gene_cor_matrix, device=device)
    # edge_index2 = correlation.convert_to_edge_index(cor_matrix=peak_cor_matrix, device=device)

    feature_matrix = torch.randn((num_of_peak + num_of_gene, num_of_peak + num_of_gene)).to(device)

    gnn_model = new_models.GNN(
        in_channels=num_of_gene + num_of_peak,
        hidden_channels=graph_feature_size * 2,
        out_channels=graph_feature_size,
        num_heads=1,
        device=device,
        dropout=0.2,
        conv_model=conv_model
    ).to(device)

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

    start_time = time.time()

    GNN_ETM_optimizer = optim.Adam(GNN_ETM_model.parameters(), lr=0.0005, weight_decay=1.2e-6)
    GNN_ETM, perf, ari_perf, best_ari, best_theta = helper.train_GNNscETM(
        model=GNN_ETM_model,
        optimizer=GNN_ETM_optimizer,
        RNA_tensor=rna_tensor,
        RNA_tensor_normalized=rna_copy_normalized_tensor,
        ATAC_tensor=atac_tensor,
        ATAC_tensor_normalized=atac_copy_normalized_tensor,
        adata=all_scRNA_adata_selected,
        adata2=all_scATAC_adata_selected,
        ari_freq=ari_freq,
        niter=num_of_epochs
    )

    # scGraphETM1, perf, ari_perf, best_ari, best_theta = train_scGraphETM(
    #     trainer=scGraphETM1,
    #     batches=h5ad_dataloader,
    #     epochs=num_of_epochs,
    #     ari_freq=ari_freq
    # )

    # scGraphETM, perf, ari_perf, best_ari, best_theta = train_scGraphETM_without_minibatch(
    #     model=scGraphETM,
    #     RNA_tensor=rna_tensor,
    #     RNA_tensor_normalized=rna_copy_normalized_tensor,
    #     ATAC_tensor=atac_tensor,
    #     ATAC_tensor_normalized=atac_copy_normalized_tensor,
    #     adata1=all_scRNA_adata_selected,
    #     adata2=all_scATAC_adata_selected,
    #     epochs=num_of_epochs,
    #     ari_freq=ari_freq
    # )

    end_time = time.time()
    print_time(start_time, end_time)

    print(f"==> the best ari is {best_ari}")
    print("======  plotting  ======")
    plot.monitor_perf(perf=perf, ari_perf=ari_perf, ari_freq=ari_freq, objective="both", path=plot_path_rel)
    plot.generate_cluster_plot(theta=best_theta, sc_adata=all_scRNA_adata_selected, plot_path_rel=plot_path_rel)
