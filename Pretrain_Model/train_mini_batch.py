import torch
from torch import optim
import time
from torch_geometric.utils import negative_sampling

import select_gpu
import mini_batch
import helper2
import model2
import view_result

def pretrain(gnn, gnn_decoder,optimizer, train_set, num_epochs=500):
    (total_X_rna_tensor, total_X_rna_tensor_normalized, total_X_atac_tensor,
     total_X_atac_tensor_normalized, total_scRNA_anndata, total_scATAC_anndata,
     total_gene_correlation_matrix, feature_matrix, edge_index) = train_set

    for epochs in range(num_epochs):
        gnn.train()
        gnn_decoder.train()
        optimizer.zero_grad()
        gnn.zero_grad()
        gnn_decoder.zero_grad()

        z = gnn(feature_matrix, edge_index)
        emb = z
        # eta, rho = split_tensor(emb, ATAC_tensor_normalized.shape[1])

        EPS = 1e-15
        pos_loss = -torch.log(gnn_decoder(z, edge_index, sigmoid=True) + EPS).mean()
        neg_edge_index = None
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(edge_index, z.size(0))
        neg_loss = -torch.log(1 - gnn_decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        recon_loss = pos_loss + neg_loss
        loss = recon_loss * 100
        loss.backward()
        optimizer.step()
        if epochs % 10 == 0:
            print(torch.sum(loss).item())
    return gnn, gnn_decoder

def train(model_tuple, optimizer,
          train_set, total_training_set, test_set,
          metric, ari_freq, niter, use_mlp,
          use_mask):
    NELBO = None
    best_ari = 0
    best_train_ari = 0
    best_theta = None
    best_train_theta = None
    best_beta_gene = None
    best_beta_peak = None

    (X_rna_test_tensor, X_rna_test_tensor_normalized, X_atac_test_tensor,
     X_atac_test_tensor_normalized, scRNA_test_anndata, scATAC_test_anndata,
     test_gene_correlation_matrix,
     test_feature_matrix, test_edge_index) = test_set

    (total_X_rna_tensor, total_X_rna_tensor_normalized, total_X_atac_tensor,
     total_X_atac_tensor_normalized, total_scRNA_anndata, total_scATAC_anndata,
     total_gene_correlation_matrix, total_feature_matrix, total_edge_index) = total_training_set

    print(f"val set tensor dim: {X_rna_test_tensor_normalized.shape}")
    print(f"train set tensor dim: {total_X_rna_tensor_normalized.shape}")

    (encoder1, encoder2, gnn, gnn_decoder, mlp1, mlp2, decoder1, decoder2) = model_tuple
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
        for train_batch in train_set:
            (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
             scRNA_mini_batch_anndata, scATAC_mini_batch_anndata, gene_correlation_matrix, feature_matrix, edge_index) = train_batch

            NELBO = helper2.train_one_epoch(
                encoder1, encoder2, gnn, gnn_decoder, mlp1, mlp2, decoder1, decoder2, optimizer, X_rna_tensor,
                X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized, total_feature_matrix,
                total_edge_index, kl_weight, use_mlp,use_mask)

        if i % ari_freq == 0:
            # with torch.no_grad():

            theta, theta_gene, theta_peak = helper2.get_theta_GNN(
                encoder1, encoder2, gnn, gnn_decoder,mlp1, mlp2, decoder1, decoder2, X_rna_tensor, X_atac_tensor,
                X_rna_test_tensor_normalized, X_atac_test_tensor_normalized, metric
            )

            theta_train, theta_gene_train, theta_peak_train = helper2.get_theta_GNN(
                encoder1, encoder2, gnn, gnn_decoder, mlp1, mlp2, decoder1, decoder2, X_rna_test_tensor, X_atac_test_tensor,
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
                  'Train res: {}\t Gene ARI: {:.4f}\n'
                  'Train res: {}\t Peak ARI: {:.4f}\n'
                  'Valid res: {}\t Valid ARI: {:.4f}\t Valid NMI: {:.4f}\n'
                  .format(i, NELBO,
                          res_train, ari_train, nmi_train,
                          res_train_gene, ari_train_gene,
                          res_train_peak, ari_train_peak,
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
            train_ari_gene_list, train_ari_peak_list, test_ari_gene_list, test_ari_peak_list
            )


if __name__ == "__main__":
    rna_path = "../data/10x-Multiome-Pbmc10k-RNA.h5ad"
    atac_path = "../data/10x-Multiome-Pbmc10k-ATAC.h5ad"
    num_of_cell = 6000
    num_of_gene = 20000
    num_of_peak = 20000
    test_num_of_cell = 2000
    batch_size = 2000
    batch_num = 10
    emb_size = 512 * 2
    emb_size2 = 512 * 2
    num_of_topic = 40
    title = 'PEAKS1_' + str(num_of_peak)
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

    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index()
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

    training_set, total_training_set, test_set, scRNA_adata, scATAC_adata = mini_batch.process_mini_batch_data(
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
        use_highly_variable=False,
        cor='pearson',
        use_mask=use_mask_train,
        mask_ratio=mask_ratio
    )

    encoder1 = model2.VAE(num_of_gene, emb_size, num_of_topic).to(device)
    encoder2 = model2.VAE(num_of_peak, emb_size, num_of_topic).to(device)
    gnn = model2.GNN(emb_size, emb_size2 * 2, emb_size2, 1, device, 0, gnn_conv).to(device)
    gnn_decoder = model2.InnerProductDecoder(0).to(device)

    mlp1 = model2.MLP(emb_size2, emb_size2 * 2, emb_size).to(device)
    mlp2 = model2.MLP(emb_size2, emb_size2 * 2, emb_size).to(device)
    decoder1 = model2.LDEC(num_of_gene, emb_size, num_of_topic, batch_size).to(device)
    decoder2 = model2.LDEC(num_of_peak, emb_size, num_of_topic, batch_size).to(device)
    parameters_pretrain = [{'params': gnn.parameters()},
                  {'params': gnn_decoder.parameters()}]

    optimizer_pretrain = optim.Adam(parameters_pretrain, lr=0.0005, weight_decay=1.2e-6)
    gnn, gnn_decoder = pretrain(gnn, gnn_decoder, optimizer_pretrain, total_training_set)
    parameters = [{'params': encoder1.parameters()},
                  {'params': encoder2.parameters()},
                  {'params': gnn.parameters()},
                  {'params': gnn_decoder.parameters()},
                  {'params': mlp1.parameters()},
                  {'params': mlp2.parameters()},
                  {'params': decoder1.parameters()},
                  {'params': decoder2.parameters()}
                  ]

    optimizer = optim.Adam(parameters, lr=lr, weight_decay=1.2e-6)

    model_tuple = (encoder1, encoder2, gnn, gnn_decoder, mlp1, mlp2, decoder1, decoder2)
    for model in model_tuple:
        print(model)

    print(f"=========  start training {num_of_topic}  =========")
    st = time.time()
    (encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
     best_ari, best_theta, best_beta_gene, best_beta_peak,
     best_train_ari, best_train_theta,
     train_ari_list, test_ari_list, train_loss_list, test_loss_list,
     train_ari_gene_list, train_ari_peak_list, test_ari_gene_list, test_ari_peak_list) = train(
        model_tuple=model_tuple,
        optimizer=optimizer,
        train_set=training_set,
        total_training_set=total_training_set,
        test_set=test_set,
        metric=metric,
        ari_freq=ari_freq,
        niter=num_epochs,
        use_mlp=use_mlp,
        use_mask=use_mask_reconstruct
    )
    ed = time.time()
    print(f"training time: {ed - st}")

    print(f"best_train_ari: {best_train_ari}, best_val_ari: {best_ari}")
    print("=========  generate_Trends  =========")
    # view_result.ari_plot(train_ari_list, test_ari_list, plot_path_rel, title)
    # view_result.loss_plot(train_loss_list, test_loss_list, plot_path_rel, title)
    print("Train ARI: ", train_ari_list)
    print("Val ARI: ", test_ari_list)
    # print("=========  generate_clustermap  =========")
    # (X_rna_test_tensor, X_rna_test_tensor_normalized, X_atac_test_tensor,
    #  X_atac_test_tensor_normalized, scRNA_test_anndata, scATAC_test_anndata,
    #  test_gene_correlation_matrix, test_peak_correlation_matrix,
    #  test_feature_matrix, test_edge_index) = test_set
    # view_result.generate_clustermap(best_theta, scRNA_test_anndata, plot_path_rel, title)
