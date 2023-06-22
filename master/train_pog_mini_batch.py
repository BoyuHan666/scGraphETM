import torch
from torch import optim
import time

import select_gpu
import mini_batch
import helper2
import model2
import view_result


def train(model_tuple, optimizer, train_set, test_set, metric, ari_freq, niter):
    NELBO = None
    best_ari = 0
    best_theta = None
    best_beta_gene = None
    best_beta_peak = None

    (X_rna_test_tensor, X_rna_test_tensor_normalized, X_atac_test_tensor,
     X_atac_test_tensor_normalized, scRNA_test_anndata, scATAC_test_anndata,
     test_gene_correlation_matrix, test_peak_correlation_matrix,
     test_feature_matrix, test_edge_index) = test_set

    (encoder1, encoder2, gnn, mlp1, mlp2, pog_decoder) = model_tuple

    for i in range(niter):
        kl_weight = helper2.calc_weight(i, niter, 0, 1 / 3, 1e-2, 4)
        for train_batch in train_set:
            (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
             scRNA_mini_batch_anndata, scATAC_mini_batch_anndata, gene_correlation_matrix,
             peak_correlation_matrix, feature_matrix, edge_index) = train_batch

            NELBO = helper2.train_one_epoch_pog(
                encoder1, encoder2, gnn, mlp1, mlp2, pog_decoder, optimizer, X_rna_tensor,
                X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized, feature_matrix,
                edge_index, gene_correlation_matrix, peak_correlation_matrix, kl_weight
            )

        if i % ari_freq == 0:
            # with torch.no_grad():
            theta = helper2.get_theta_GNN_pog(
                encoder1, encoder2, gnn, mlp1, mlp2, pog_decoder,
                X_rna_test_tensor_normalized, X_atac_test_tensor_normalized, metric
            )

            # beta_gene, beta_peak = helper2.get_beta_GNN(
            #     encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
            #     X_rna_test_tensor_normalized, X_atac_test_tensor_normalized,
            #     test_feature_matrix, test_edge_index)

            ari = helper2.evaluate_ari(theta.to('cpu'), scRNA_test_anndata)

            print('Iter: {} ..  NELBO: {:.4f} .. Val ARI: {:.4f}'.format(i, NELBO, ari))

            if best_ari < ari:
                best_ari = ari
                best_theta = theta
                # best_beta_gene = beta_gene
                # best_beta_peak = beta_peak
        else:
            if i % 100 == 0:
                print("Iter: " + str(i))

    return (encoder1, encoder2, gnn, mlp1, mlp2, pog_decoder,
            best_ari, best_theta, best_beta_gene, best_beta_peak)


if __name__ == "__main__":

    rna_path = "../data/10x-Multiome-Pbmc10k-RNA.h5ad"
    atac_path = "../data/10x-Multiome-Pbmc10k-ATAC.h5ad"
    num_of_cell = 6000
    num_of_gene = 2000
    num_of_peak = 2000
    test_num_of_cell = 2000
    batch_size = 2000
    batch_num = 4
    emb_size = 512
    emb_size2 = 512
    num_of_topic = 20
    gnn_conv = 'GATv2'
    num_epochs = 500
    ari_freq = 10
    plot_path_rel = "./plot/"
    metric = 'theta'  # mu or theta
    lr = 0.001

    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index()
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

    training_set, test_set, scRNA_adata, scATAC_adata = mini_batch.process_mini_batch_data(
        rna_path=rna_path,
        atac_path=atac_path,
        device=device,
        num_of_cell=num_of_cell,
        num_of_gene=num_of_gene,
        num_of_peak=num_of_peak,
        test_num_of_cell=test_num_of_cell,
        batch_size=batch_size,
        batch_num=batch_num,
        emb_size=emb_size,
        use_highly_variable=True,
        cor='pearson',
        val=1 # 0 for using a new test set other than train set, 1 for using train set to calculate ari
    )

    encoder1 = model2.VAE(num_of_gene, emb_size, num_of_topic).to(device)
    encoder2 = model2.VAE(num_of_peak, emb_size, num_of_topic).to(device)
    gnn = model2.GNN(emb_size, emb_size2 * 2, emb_size2, 1, device, 0, gnn_conv).to(device)
    mlp1 = model2.MLP(emb_size2, emb_size2 * 2, emb_size).to(device)
    mlp2 = model2.MLP(emb_size2, emb_size2 * 2, emb_size).to(device)
    pog_decoder = model2.POG_DEC(num_of_gene, num_of_peak, emb_size, num_of_topic, batch_size).to(device)

    parameters = [{'params': encoder1.parameters()},
                  {'params': encoder2.parameters()},
                  {'params': gnn.parameters()},
                  {'params': mlp1.parameters()},
                  {'params': mlp2.parameters()},
                  {'params': pog_decoder.parameters()}
                  ]

    optimizer = optim.Adam(parameters, lr=lr, weight_decay=1.2e-6)

    model_tuple = (encoder1, encoder2, gnn, mlp1, mlp2, pog_decoder)
    for model in model_tuple:
        print(model)

    print(f"=========  start training {num_of_topic}  =========")
    st = time.time()
    (encoder1, encoder2, gnn, mlp1, mlp2, pog_decoder,
     best_ari, best_theta, best_beta_gene, best_beta_peak) = train(
        model_tuple=model_tuple,
        optimizer=optimizer,
        train_set=training_set,
        test_set=test_set,
        metric=metric,
        ari_freq=ari_freq,
        niter=num_epochs
    )
    ed = time.time()
    print(f"training time: {ed - st}")

    print(best_ari)
    print("=========  generate_clustermap  =========")
    (X_rna_test_tensor, X_rna_test_tensor_normalized, X_atac_test_tensor,
     X_atac_test_tensor_normalized, scRNA_test_anndata, scATAC_test_anndata,
     test_gene_correlation_matrix, test_peak_correlation_matrix,
     test_feature_matrix, test_edge_index) = test_set
    view_result.generate_clustermap(best_theta, scRNA_test_anndata, plot_path_rel)
