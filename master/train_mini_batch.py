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
import view_result


def train(model_tuple, optimizer,
          train_set, total_training_set, test_set,
          metric, ari_freq, niter, use_mlp,
          use_mask, result):
    NELBO = None
    best_ari = 0
    best_train_ari = 0
    best_theta = None
    best_train_theta = None
    best_beta_gene = None
    best_beta_peak = None
    ari_trains = []
    ari_tests = []

    (X_rna_test_tensor, X_rna_test_tensor_normalized, X_atac_test_tensor,
     X_atac_test_tensor_normalized, scRNA_test_anndata, scATAC_test_anndata,
     test_feature_matrix, test_edge_index) = test_set

    (total_X_rna_tensor, total_X_rna_tensor_normalized, total_X_atac_tensor,
     total_X_atac_tensor_normalized, total_scRNA_anndata, total_scATAC_anndata,
     total_feature_matrix, total_edge_index) = total_training_set

    print(f"val set tensor dim: {X_rna_test_tensor_normalized.shape}, {X_atac_test_tensor_normalized.shape}")
    print(f"train set tensor dim: {total_X_rna_tensor_normalized.shape}, {total_X_atac_tensor_normalized.shape}")

    (encoder1, encoder2, gnn, mlp1, mlp2, graph_dec, decoder1, decoder2) = model_tuple

    for i in range(niter):
        kl_weight = helper2.calc_weight(i, niter, 0, 1 / 3, 1e-2, 4)
        for train_batch in train_set:
            (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
             scRNA_mini_batch_anndata, scATAC_mini_batch_anndata, feature_matrix, edge_index,
             mask_matrix1, mask_matrix2, X_rna_tensor_copy, X_atac_tensor_copy) = train_batch

            recon_loss, kl_loss = helper2.train_one_epoch(
                encoder1, encoder2, gnn, mlp1, mlp2, graph_dec, decoder1, decoder2, optimizer, X_rna_tensor,
                X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized, feature_matrix,
                edge_index, result, use_mlp, use_mask, mask_matrix1, mask_matrix2, X_rna_tensor_copy, X_atac_tensor_copy
            )

            NELBO = recon_loss + kl_loss

        if i % ari_freq == 0:
            # with torch.no_grad():
            theta, theta_gene, theta_peak = helper2.get_theta_GNN(
                encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
                X_rna_test_tensor_normalized, X_atac_test_tensor_normalized, metric
            )

            theta_train, theta_gene_train, theta_peak_train = helper2.get_theta_GNN(
                encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
                total_X_rna_tensor_normalized, total_X_atac_tensor_normalized, metric
            )

            # beta_gene, beta_peak = helper2.get_beta_GNN(
            #     encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
            #     X_rna_test_tensor_normalized, X_atac_test_tensor_normalized,
            #     test_feature_matrix, test_edge_index)

            # ari = helper2.evaluate_ari(theta.to('cpu'), scRNA_test_anndata)
            # ari_train = helper2.evaluate_ari(theta_train.to('cpu'), total_scRNA_anndata)
            #
            # print('Iter: {} ..  NELBO: {:.4f} .. Train ARI: {:.4f} .. Val ARI: {:.4f}'.format(i, NELBO, ari_train, ari))

            res, ari, nmi = helper2.evaluate_ari2(theta.to('cpu'), scRNA_test_anndata)
            res_train, ari_train, nmi_train = helper2.evaluate_ari2(theta_train.to('cpu'), total_scRNA_anndata)

            ari_trains.append(ari_train)
            ari_tests.append(ari)

            print('====  Iter: {}, NELBO: {:.4f}, recon_loss: {:.4f}, kl_loss: {:.4f}  ====\n'
                  'Train res: {}\t Train ARI: {:.4f}\t Train NMI: {:.4f}\n'
                  'Valid res: {}\t Valid ARI: {:.4f}\t Valid NMI: {:.4f}\n'
                  .format(i, NELBO, recon_loss, kl_loss,
                          res_train, ari_train, nmi_train,
                          res, ari, nmi)
                  )

            if best_ari < ari:
                best_ari = ari
                best_theta = theta
                # best_beta_gene = beta_gene
                # best_beta_peak = beta_peak
            if best_train_ari < ari_train:
                best_train_ari = ari_train
                best_train_theta = theta_train
        else:
            if i % 100 == 0:
                print("Iter: " + str(i))

    return (encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
            best_ari, best_theta, best_beta_gene, best_beta_peak,
            best_train_ari, best_train_theta, ari_trains, ari_tests
            )


if __name__ == "__main__":
    rna_path = "../data/10x-Multiome-Pbmc10k-RNA.h5ad"
    atac_path = "../data/10x-Multiome-Pbmc10k-ATAC.h5ad"
    # rna_path = "../data/BMMC_rna_filtered.h5ad"
    # atac_path = "../data/BMMC_atac_filtered.h5ad"

    gene_exp = anndata.read('../data/10x-Multiome-Pbmc10k-RNA.h5ad')
    total_gene = gene_exp.shape[1]

    peak_exp = anndata.read('../data/10x-Multiome-Pbmc10k-ATAC.h5ad')
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

    num_of_cell = 8000
    # num_of_gene = 20000
    # num_of_peak = 60000
    num_of_gene = gene_num
    num_of_peak = peak_num
    test_num_of_cell = total_cell_num - num_of_cell
    batch_num = 10
    batch_size = int(num_of_cell / batch_num)
    emb_size = 600
    emb_size2 = 600
    emb_graph = num_of_cell
    num_of_topic = 40
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
    use_noise = False
    noise_ratio = 0.2

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


    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index()
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

    result_dense = result.toarray()
    result_tensor = torch.from_numpy(result_dense).float().to(device)
    print(f"result.shape: {result_tensor.shape}, result type: {type(result_tensor)}")
    print(result_tensor.sum())

    training_set, total_training_set, test_set, scRNA_adata, scATAC_adata = mini_batch.process_mini_batch_data(
        scRNA_adata=gene_exp,
        scATAC_adata=peak_exp,
        device=device,
        num_of_cell=num_of_cell,
        num_of_gene=num_of_gene,
        num_of_peak=num_of_peak,
        test_num_of_cell=test_num_of_cell,
        batch_size=batch_size,
        batch_num=batch_num,
        emb_size=emb_size,
        edge_index=edge_index,
        use_mask=use_mask_train,
        mask_ratio=mask_ratio,
        use_noise=use_noise,
        noise_ratio=noise_ratio
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
    (encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
     best_ari, best_theta, best_beta_gene, best_beta_peak,
     best_train_ari, best_train_theta, ari_trains, ari_tests) = train(
        model_tuple=model_tuple,
        optimizer=optimizer,
        train_set=training_set,
        total_training_set=total_training_set,
        test_set=test_set,
        metric=metric,
        ari_freq=ari_freq,
        niter=num_epochs,
        use_mlp=use_mlp,
        use_mask=use_mask_reconstruct,
        result=result_tensor,
    )
    ed = time.time()
    print(f"training time: {ed - st}")

    print(f"best_train_ari: {best_train_ari}, best_val_ari: {best_ari}")
    print(ari_trains)
    print(ari_tests)
    # print("=========  generate_clustermap  =========")
    # (X_rna_test_tensor, X_rna_test_tensor_normalized, X_atac_test_tensor,
    #  X_atac_test_tensor_normalized, scRNA_test_anndata, scATAC_test_anndata,
    #  test_gene_correlation_matrix, test_peak_correlation_matrix,
    #  test_feature_matrix, test_edge_index) = test_set
    # view_result.generate_clustermap(best_theta, scRNA_test_anndata, plot_path_rel)
