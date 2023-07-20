import torch
from torch import optim
import time
import warnings
from numba.core.errors import NumbaDeprecationWarning
import numpy as np
import pickle
import anndata

import select_gpu
import mini_batch
import helper2
import model2
import view_result

from scipy.sparse import vstack, hstack


def get_sub_graph(path, num_gene, num_peak, total_peak):
    with open(path, 'rb') as fp:
        sp_matrix = pickle.load(fp)
    peak_peak = sp_matrix[:num_peak, :num_peak]
    peak_gene_down = sp_matrix[total_peak:(total_peak + num_gene), :num_peak]
    peak_gene_up = sp_matrix[:num_peak, total_peak:(total_peak + num_gene)]
    gene_gene = sp_matrix[total_peak:total_peak + num_gene, total_peak:total_peak + num_gene]

    top = hstack([peak_peak, peak_gene_up])
    bottom = hstack([peak_gene_down, gene_gene])

    result = vstack([top, bottom])
    rows, cols = result.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)

    return result, edge_index


def train(model_tuple, optimizer,
          train_set, total_training_set, test_set,
          metric, ari_freq, niter,
          use_mlp, use_mask):
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

    (encoder1, encoder2, gnn, mlp1, mlp2, pog_decoder) = model_tuple

    for i in range(niter):
        kl_weight = helper2.calc_weight(i, niter, 0, 1 / 3, 1e-2, 4)
        for train_batch in train_set:
            (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
             scRNA_mini_batch_anndata, scATAC_mini_batch_anndata, feature_matrix, edge_index,
             mask_matrix1, mask_matrix2, X_rna_tensor_copy, X_atac_tensor_copy) = train_batch

            NELBO = helper2.train_one_epoch_pog(
                encoder1, encoder2, gnn, mlp1, mlp2, pog_decoder, optimizer, X_rna_tensor,
                X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized, feature_matrix,
                edge_index, use_mlp, use_mask, mask_matrix1, mask_matrix2, X_rna_tensor_copy, X_atac_tensor_copy
            )

        if i % ari_freq == 0:
            # with torch.no_grad():
            theta = helper2.get_theta_GNN_pog(
                encoder1, encoder2, gnn, mlp1, mlp2, pog_decoder,
                X_rna_test_tensor_normalized, X_atac_test_tensor_normalized, metric
            )

            theta_train = helper2.get_theta_GNN_pog(
                encoder1, encoder2, gnn, mlp1, mlp2, pog_decoder,
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

            print('====  Iter: {},  NELBO: {:.4f}  ====\n'
                  'Train res: {}\t Train ARI: {:.4f}\t Train NMI: {:.4f}\n'
                  'Valid res: {}\t Valid ARI: {:.4f}\t Valid NMI: {:.4f}\n'
                  .format(i, NELBO,
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

    return (encoder1, encoder2, gnn, mlp1, mlp2, pog_decoder,
            best_ari, best_theta, best_beta_gene, best_beta_peak,
            best_train_ari, best_train_theta, ari_trains, ari_tests)


if __name__ == "__main__":

    rna_path = "../data/10x-Multiome-Pbmc10k-RNA.h5ad"
    atac_path = "../data/10x-Multiome-Pbmc10k-ATAC.h5ad"
    cor_path = '../data/TF_gene/top1_peak_tf_gene.pickle'

    gene_exp = anndata.read('../data/10x-Multiome-Pbmc10k-RNA.h5ad')
    total_gene_num = gene_exp.shape[1]

    peak_exp = anndata.read('../data/10x-Multiome-Pbmc10k-ATAC.h5ad')
    total_peak_num = peak_exp.shape[1]

    total_cell_num = gene_exp.shape[0]

    # rna_path = "../data/BMMC_rna_filtered.h5ad"
    # atac_path = "../data/BMMC_atac_filtered.h5ad"

    warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

    num_of_cell = 8000
    num_of_gene = 3000
    num_of_peak = 8000
    # num_of_gene = total_gene_num
    # num_of_peak = total_peak_num - 50000
    test_num_of_cell = total_cell_num - num_of_cell
    batch_num = 10
    batch_size = int(num_of_cell / batch_num)
    emb_size = 600
    emb_size2 = 600
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

    result, edge_index = get_sub_graph(
        path=cor_path,
        num_gene=num_of_gene,
        num_peak=num_of_peak,
        total_peak=total_peak_num
    )

    print(result.shape)
    print(result.sum())

    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index()
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

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
