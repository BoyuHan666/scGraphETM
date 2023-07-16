import torch
from torch import optim

import helper2
import model2
import select_gpu
import view_result

path1 = "../data/10x-Multiome-Pbmc10k-RNA.h5ad"
path2 = "../data/10x-Multiome-Pbmc10k-ATAC.h5ad"
plot_path_rel = "./plot/"
num_of_cell = 2000
num_of_gene = 2000
num_of_peak = 2000
num_of_topic = 60
emb_size = 512
emb_size2 = 512
cor_method = 'pearson'
# cor_method = 'spearman'
gnn_conv = 'GATv2'
num_of_epochs = 1000
ari_freq = 100
metric = 'mu'

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index()
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

    feature_matrix, edge_index, \
    scRNA_adata, scATAC_adata, \
    scRNA_1000_tensor, scRNA_1000_tensor_normalized, \
    scATAC_1000_tensor, scATAC_1000_tensor_normalized, \
    scRNA_1000_mp_anndata, scATAC_1000_mp_anndata, \
    correlation_matrix_cleaned, correlation_matrix2_cleaned = helper2.process_data(
        rna_path=path1,
        atac_path=path2,
        device=device,
        num_of_cell=num_of_cell,
        num_of_gene=num_of_gene,
        num_of_peak=num_of_peak,
        emb_size=emb_size,
        cor=cor_method
    )

    encoder1 = model2.VAE(num_of_gene, emb_size, num_of_topic).to(device)
    encoder2 = model2.VAE(num_of_peak, emb_size, num_of_topic).to(device)
    gnn = model2.GNN(emb_size, emb_size2 * 2, emb_size2, 1, device, 0, gnn_conv).to(device)
    mlp1 = model2.MLP(emb_size2, emb_size2 * 2, emb_size).to(device)
    mlp2 = model2.MLP(emb_size2, emb_size2 * 2, emb_size).to(device)
    decoder1 = model2.LDEC(num_of_gene, emb_size, num_of_topic).to(device)
    decoder2 = model2.LDEC(num_of_peak, emb_size, num_of_topic).to(device)

    parameters = [{'params': encoder1.parameters()},
                  {'params': encoder2.parameters()},
                  {'params': gnn.parameters()},
                  {'params': mlp1.parameters()},
                  {'params': mlp2.parameters()},
                  {'params': decoder1.parameters()},
                  {'params': decoder2.parameters()}
                  ]

    optimizer = optim.AdamW(parameters, lr=0.001, weight_decay=1.2e-6)

    print("-----------------------")
    print(encoder1)
    print(encoder2)
    print(gnn)
    print(mlp1)
    print(mlp2)
    print(decoder1)
    print(decoder2)

    encoder1, encoder2, gnn, decoder1, decoder2, GNN_ETM_perf, ari_perf, best_ari, theta, beta_gene, beta_peak = helper2.train(
        encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2, optimizer,
        scRNA_1000_tensor, scRNA_1000_tensor_normalized,
        scATAC_1000_tensor, scATAC_1000_tensor_normalized,
        feature_matrix, edge_index,
        correlation_matrix_cleaned, correlation_matrix2_cleaned,
        scRNA_1000_mp_anndata, scATAC_1000_mp_anndata, metric,
        ari_freq, num_of_epochs
    )

    print(best_ari)

    """# **View result**"""
    print("-----------------------")
    # monitor_perf2(GNN_ETM_perf, ari_perf, ari_freq, "NELBO")

    print("=========  generate_ari_plot  =========")
    helper2.monitor_perf(GNN_ETM_perf, ari_perf, ari_freq, "both", plot_path_rel)

    print("=========  generate_clustermap  =========")
    view_result.generate_clustermap(theta, scRNA_1000_mp_anndata, plot_path_rel)

    print("=========  generate_gene_heatmap  =========")
    view_result.generate_gene_heatmap(num_of_topic, scRNA_adata, beta_gene, num_of_gene, plot_path_rel)

    print("=========  generate_cell_heatmap  =========")
    view_result.generate_cell_heatmap(scRNA_adata, theta, num_of_cell, plot_path_rel)
