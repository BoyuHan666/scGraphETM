import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
import time

import select_gpu
import full_batch
import helper2
import model2
import view_result


def train(model_tuple, optimizer1, optimizer2,
          train_set, total_training_set, test_set, metric,
          ari_freq, niter, use_mlp, use_mask, mask1, mask2):
    NELBO = None
    best_ari = 0
    best_train_ari = 0
    best_theta = None
    best_train_theta = None
    best_beta_gene = None
    best_beta_peak = None

    (X_rna_test_tensor, X_rna_test_tensor_normalized, X_atac_test_tensor,
     X_atac_test_tensor_normalized, scRNA_test_anndata, scATAC_test_anndata,
     test_gene_correlation_matrix, test_peak_correlation_matrix,
     test_feature_matrix, test_cor_mat, test_edge_index) = test_set

    (total_X_rna_tensor, total_X_rna_tensor_normalized, total_X_atac_tensor,
     total_X_atac_tensor_normalized, total_scRNA_anndata, total_scATAC_anndata,
     total_gene_correlation_matrix, total_peak_correlation_matrix,
     total_feature_matrix, train_cor_mat, total_edge_index) = total_training_set

    print(f"val set tensor dim: {X_rna_test_tensor_normalized.shape}")
    print(f"train set tensor dim: {total_X_rna_tensor_normalized.shape}")

    (encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2) = model_tuple
    train_ari_list = []
    train_loss_list = []
    train_ari_gene_list = []
    train_ari_peak_list = []
    test_ari_list = []
    test_loss_list = []
    test_ari_gene_list = []
    test_ari_peak_list = []
    for i in range(niter):
        kl_weight = helper2.calc_weight(i, niter, 0, 1 / 3, 1e-2, 4)

        (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
         scRNA_mini_batch_anndata, scATAC_mini_batch_anndata, gene_correlation_matrix,
         peak_correlation_matrix, feature_matrix, cor_mat, edge_index, X_rna_tensor_copy, X_atac_tensor_copy) = train_set

        # rna_feature = scRNA_mini_batch_anndata.X.copy()/ scRNA_mini_batch_anndata.X.max()
        # binary_matrix1 = np.where(rna_feature != 0.0, 1.0, rna_feature)
        # atac_feature = (scATAC_mini_batch_anndata.X).copy()/ scATAC_mini_batch_anndata.X.max()
        # binary_matrix2 = np.where(atac_feature != 0.0, 1.0, atac_feature)
        # binary_matrix2 = binary_matrix2[:6000, :2000]
        # feature_matrix = torch.tensor(np.vstack((rna_feature.T, atac_feature.T))).to(device)

        if i < 500:
            optimizer = optimizer1
        else:
            optimizer = optimizer2
        print(feature_matrix)
        NELBO = helper2.train_one_epoch(
            encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2, optimizer1, X_rna_tensor,
            X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized, feature_matrix,
            cor_mat, edge_index, gene_correlation_matrix, peak_correlation_matrix, kl_weight, use_mlp,
            use_mask, mask1, mask2, X_rna_tensor_copy, X_atac_tensor_copy
        )

        if i % ari_freq == 0:
            # with torch.no_grad():
            theta, theta_gene, theta_peak = helper2.get_theta_GNN(
                encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2, X_rna_tensor,
                X_rna_tensor_normalized,
                X_rna_test_tensor_normalized, X_atac_test_tensor_normalized, metric
            )

            theta_train, theta_gene_train, theta_peak_train = helper2.get_theta_GNN(
                encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2, X_rna_tensor,
                X_rna_tensor_normalized,
                total_X_rna_tensor_normalized, total_X_atac_tensor_normalized, metric
            )

            # beta_gene, beta_peak = helper2.get_beta_GNN(
            #     encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
            #     X_rna_test_tensor_normalized, X_atac_test_tensor_normalized,
            #     test_feature_matrix, test_edge_index, use_mlp)

            # ari = helper2.evaluate_ari(theta.to('cpu'), scRNA_test_anndata)
            # ari_train = helper2.evaluate_ari(theta_train.to('cpu'), total_scRNA_anndata)
            #
            # print('Iter: {} ..  NELBO: {:.4f} .. Train ARI: {:.4f} .. Val ARI: {:.4f}'.format(i, NELBO, ari_train, ari))

            res_val_gene, ari_val_gene, nmi_val_gene = helper2.evaluate_ari2(theta_gene.to('cpu'),
                                                                             scRNA_test_anndata)
            res_val_peak, ari_val_peak, nmi_val_peak = helper2.evaluate_ari2(theta_peak.to('cpu'),
                                                                             scATAC_test_anndata)
            res, ari, nmi = helper2.evaluate_ari2(theta.to('cpu'), scRNA_test_anndata)

            res_train_gene, ari_train_gene, nmi_train_gene = helper2.evaluate_ari2(theta_gene_train.to('cpu'),
                                                                                   total_scRNA_anndata)
            res_train_peak, ari_train_peak, nmi_train_peak = helper2.evaluate_ari2(theta_peak_train.to('cpu'),
                                                                                   total_scATAC_anndata)
            res_train, ari_train, nmi_train = helper2.evaluate_ari2(theta_train.to('cpu'), total_scRNA_anndata)
            train_loss_list.append(NELBO)
            # test_loss_list.append(val_loss2)
            train_ari_list.append(ari_train)
            train_ari_gene_list.append(ari_train_gene)
            train_ari_peak_list.append(ari_train_peak)
            test_ari_list.append(ari)
            test_ari_gene_list.append(ari_val_gene)
            test_ari_peak_list.append(ari_val_peak)
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

    return (encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
            best_ari, best_theta, best_beta_gene, best_beta_peak,
            best_train_ari, best_train_theta, train_ari_list, test_ari_list, train_loss_list, test_loss_list,
            train_ari_gene_list, train_ari_peak_list, test_ari_gene_list, test_ari_peak_list)


if __name__ == "__main__":

    rna_path = "../data/10x-Multiome-Pbmc10k-RNA.h5ad"
    atac_path = "../data/10x-Multiome-Pbmc10k-ATAC.h5ad"

    num_of_cell = 6000
    num_of_gene = 2000
    num_of_peak = 2000
    title = 'Full_' + str(num_of_peak)
    test_num_of_cell = 2000
    emb_size = 512
    emb_size2 = 512
    num_of_topic = 40
    gnn_conv = 'GATv2'
    num_epochs = 2000
    ari_freq = 40
    plot_path_rel = "./plot/"
    metric = 'theta'  # mu or theta
    lr = 0.001
    use_mlp = False
    use_mask_train = False
    use_mask_reconstruct = False  # False: one side mask for reconstructing the masked expressions
    mask_ratio = 0.2

    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index()
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

    training_set, total_training_set, test_set, scRNA_adata, scATAC_adata, mask_matrix1, mask_matrix2 = full_batch.process_full_batch_data(
        rna_path=rna_path,
        atac_path=atac_path,
        device=device,
        num_of_cell=num_of_cell,
        num_of_gene=num_of_gene,
        num_of_peak=num_of_peak,
        test_num_of_cell=test_num_of_cell,
        emb_size=emb_size,
        use_highly_variable=True,
        cor='pearson',
        use_mask=use_mask_train,
        mask_ratio=mask_ratio
    )

    encoder1 = model2.VAE(num_of_gene, emb_size, num_of_topic).to(device)
    encoder2 = model2.VAE(num_of_peak, emb_size, num_of_topic).to(device)
    gnn = model2.GNN(emb_size, emb_size2 * 2, emb_size2, 1, device, 0, gnn_conv).to(device)
    mlp1 = model2.MLP(emb_size2, emb_size2 * 2, emb_size).to(device)
    mlp2 = model2.MLP(emb_size2, emb_size2 * 2, emb_size).to(device)
    decoder1 = model2.LDEC(num_of_gene, emb_size, num_of_topic, num_of_cell).to(device)
    decoder2 = model2.LDEC(num_of_peak, emb_size, num_of_topic, num_of_cell).to(device)

    parameters = [{'params': encoder1.parameters()},
                  {'params': encoder2.parameters()},
                  {'params': gnn.parameters()},
                  {'params': mlp1.parameters()},
                  {'params': mlp2.parameters()},
                  {'params': decoder1.parameters()},
                  {'params': decoder2.parameters()}
                  ]

    optimizer_sgd = optim.SGD(parameters, lr=lr, momentum=0.01)
    optimizer_adamW = optim.AdamW(parameters, lr=lr, weight_decay=1.2e-6)
    optimizer_adam = optim.Adam(parameters, lr=lr, weight_decay=1.2e-6)
    optimizer_adam2 = optim.Adam(parameters, lr=lr * 0.1, weight_decay=1.2e-6)

    model_tuple = (encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2)
    for model in model_tuple:
        print(model)

    print(f"=========  start training {num_of_topic} =========")
    st = time.time()
    (encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
     best_ari, best_theta, best_beta_gene, best_beta_peak,
     best_train_ari, best_train_theta,
     train_ari_list, test_ari_list, train_loss_list, test_loss_list,
     train_ari_gene_list, train_ari_peak_list, test_ari_gene_list, test_ari_peak_list
     ) = train(
        model_tuple=model_tuple,
        optimizer1=optimizer_adam,
        optimizer2=optimizer_adam,
        train_set=training_set,
        total_training_set=total_training_set,
        test_set=test_set,
        metric=metric,
        ari_freq=ari_freq,
        niter=num_epochs,
        use_mlp=use_mlp,
        use_mask=use_mask_reconstruct,
        mask1=mask_matrix1,
        mask2=mask_matrix2,
    )
    ed = time.time()
    print(f"training time: {ed - st}")

    print(f"best_train_ari: {best_train_ari}, best_val_ari: {best_ari}")
    print("=========  generate_Trends  =========")
    print("Train ARI: ", train_ari_list)
    print("Val ARI: ", test_ari_list)
    # view_result.ari_plot2(train_ari_list, test_ari_list, train_ari_gene_list, train_ari_peak_list, plot_path_rel, title)
    # print("=========  generate_clustermap  =========")
    # (X_rna_test_tensor, X_rna_test_tensor_normalized, X_atac_test_tensor,
    #  X_atac_test_tensor_normalized, scRNA_test_anndata, scATAC_test_anndata,
    #  test_gene_correlation_matrix, test_peak_correlation_matrix,
    #  test_feature_matrix, test_edge_index) = test_set
    # view_result.generate_clustermap(best_theta, scRNA_test_anndata, plot_path_rel, title)
