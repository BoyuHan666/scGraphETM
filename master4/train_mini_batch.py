import torch
from torch import optim
import time
import warnings
from numba.core.errors import NumbaDeprecationWarning
import anndata

import select_gpu
import mini_batch
import helper2
import model2


def train(model_tuple, optimizer, train_set, ari_freq, niter, edge_index):
    NELBO = None

    (encoder1, encoder2, gnn, mlp1, mlp2, graph_dec, decoder1, decoder2) = model_tuple

    for i in range(niter):
        for train_batch in train_set:
            (gene_exp, gene_exp_normalized, peak_exp, peak_exp_normalized, feature_matrix) = train_batch

            recon_loss, kl_loss = helper2.train_one_epoch(
                encoder1, encoder2, gnn, mlp1, mlp2, graph_dec, decoder1, decoder2, optimizer,
                gene_exp, gene_exp_normalized, peak_exp, peak_exp_normalized,
                feature_matrix, edge_index
            )

            NELBO = recon_loss + kl_loss

        if i % ari_freq == 0:

            print('====  Iter: {}, NELBO: {:.4f}, recon_loss: {:.4f}, kl_loss: {:.4f}  ===='.format(i, NELBO, recon_loss, kl_loss))

    return encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index()
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

    rna_path = "../data/10x-Multiome-Pbmc10k-RNA.h5ad"
    atac_path = "../data/10x-Multiome-Pbmc10k-ATAC.h5ad"
    # rna_path = "../data/BMMC_rna_filtered.h5ad"
    # atac_path = "../data/BMMC_atac_filtered.h5ad"

    gene_exp = anndata.read('../data/10x-Multiome-Pbmc10k-RNA.h5ad')
    peak_exp = anndata.read('../data/10x-Multiome-Pbmc10k-ATAC.h5ad')

    total_gene = gene_exp.shape[1]
    total_peak = peak_exp.shape[1]
    total_cell_num = gene_exp.shape[0]

    # index_path = '../data/relation/gene_peak_index_relation.pickle'
    # gene_index_list, peak_index_list = helper2.get_peak_index(index_path, top=5, threshould=None, gene_limit=12000)
    index_path = '../data/relation/highly_gene_peak_index_relation.pickle'
    gene_index_list, peak_index_list = helper2.get_peak_index(index_path, top=5, threshould=None)
    gene_exp = gene_exp[:, gene_index_list]
    gene_num = gene_exp.shape[1]
    peak_exp = peak_exp[:, peak_index_list]
    peak_num = peak_exp.shape[1]

    warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

    # mtx_path = '../data/relation_by_score/top5_peak_tf_gene.pickle'
    # mtx_path = '../data/relation2/top5_peak_tf_gene.pickle'
    # mtx_path = '../data/TF_gene/top5_peak_tf_gene.pickle'
    mtx_path = '../data/gene_peak/top5peak_gene.pickle'
    result, edge_index = helper2.get_sub_graph_by_index(
        path=mtx_path,
        gene_index_list=gene_index_list,
        peak_index_list=peak_index_list,
        total_peak=total_peak
    )

    # mtx_path = '../data/relation2/top5_peak_tf_gene.pickle'
    # result, edge_index = helper2.get_sub_graph(
    #     path=mtx_path,
    #     num_gene=num_of_gene,
    #     num_peak=num_of_peak,
    #     total_peak=total_peak
    # )
    # print(len(edge_index[0]))

    num_of_cell = 8000
    num_of_gene = gene_num
    num_of_peak = peak_num
    test_num_of_cell = total_cell_num - num_of_cell
    emb_size = 512
    emb_size2 = 512
    # emb_graph = num_of_cell
    emb_graph = emb_size2
    num_of_topic = 40
    gnn_conv = 'GATv2'
    num_epochs = 10
    ari_freq = 1
    lr = 0.001

    result_dense = result.toarray()
    result_tensor = torch.from_numpy(result_dense).float().to(device)
    print(f"result.shape: {result_tensor.shape}, result type: {type(result_tensor)}")
    print(result_tensor.sum())

    training_set = mini_batch.process_mini_batch_data(
        scRNA_adata=gene_exp,
        scATAC_adata=peak_exp,
        device=device,
        num_of_cell=num_of_cell,
        test_num_of_cell=test_num_of_cell,
        num_of_gene=num_of_gene,
        num_of_peak=num_of_peak,
        emb_size=emb_size,
    )

    encoder1 = model2.VAE(num_of_gene, emb_size, num_of_topic).to(device)
    encoder2 = model2.VAE(num_of_peak, emb_size, num_of_topic).to(device)
    gnn = model2.GNN(emb_graph, emb_size2 * 4, emb_size2, 1, device, 0, gnn_conv).to(device)
    mlp1 = model2.MLP(emb_size2, emb_size2 * 2, emb_size).to(device)
    mlp2 = model2.MLP(emb_size2, emb_size2 * 2, emb_size).to(device)
    graph_dec = model2.DEC(emb_size2, emb_size2 * 4, num_of_peak+num_of_gene).to(device)
    decoder1 = model2.LDEC(num_of_gene, emb_size, num_of_topic, batch_size).to(device)
    decoder2 = model2.LDEC(num_of_peak, emb_size, num_of_topic, batch_size).to(device)

    parameters = [{'params': encoder1.parameters()},
                  {'params': encoder2.parameters()},
                  {'params': gnn.parameters()},
                  {'params': mlp1.parameters()},
                  {'params': mlp2.parameters()},
                  {'params': graph_dec.parameters()},
                  {'params': decoder1.parameters()},
                  {'params': decoder2.parameters()}
                  ]

    optimizer = optim.Adam(parameters, lr=lr, weight_decay=1.2e-6)

    model_tuple = (encoder1, encoder2, gnn, mlp1, mlp2, graph_dec, decoder1, decoder2)
    for model in model_tuple:
        print(model)

    print(f"=========  start training {num_of_topic}  =========")
    st = time.time()
    encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2 = train(
        model_tuple=model_tuple,
        optimizer=optimizer,
        train_set=training_set,
        ari_freq=ari_freq,
        niter=num_epochs,
        edge_index=edge_index.to(device)
    )
    ed = time.time()
    print(f"training time: {ed - st}")


    # plot.plot_ari(ari_trains, ari_tests)
    # print("=========  generate_clustermap  =========")
    # (X_rna_test_tensor, X_rna_test_tensor_normalized, X_atac_test_tensor,
    #  X_atac_test_tensor_normalized, scRNA_test_anndata, scATAC_test_anndata,
    #  test_gene_correlation_matrix, test_peak_correlation_matrix,
    #  test_feature_matrix, test_edge_index) = test_set
    # view_result.generate_clustermap(best_theta, scRNA_test_anndata, plot_path_rel)

