import torch
from torch import optim
from torch_geometric.nn import Node2Vec
import time
import warnings
from numba.core.errors import NumbaDeprecationWarning
import anndata
import numpy as np
import pickle
import random

import select_gpu
import mini_batch
import helper2
import model2
import view_results
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def train(model_tuple, optimizer,
          train_set, total_training_set, test_set,
          lookup, random_matrix, edge_index, ari_freq, niter,
          device, param_savepath=None, best_ari_path=None):
    NELBO = None
    best_ari = 0
    best_train_ari = 0
    best_theta = None
    best_train_theta = None
    best_beta_gene = None
    best_beta_peak = None
    ari_trains = []
    ari_tests = []
    loss_set = []
    recon_list = []
    kl_list = []
    pear_list = []
    spear_list = []
    (X_rna_test_tensor, X_rna_test_tensor_normalized, X_atac_test_tensor,
     X_atac_test_tensor_normalized, scRNA_test_anndata, scATAC_test_anndata,
     test_feature_matrix, test_edge_index) = test_set

    (total_X_rna_tensor, total_X_rna_tensor_normalized, total_X_atac_tensor,
     total_X_atac_tensor_normalized, total_scRNA_anndata, total_scATAC_anndata,
     total_feature_matrix, total_edge_index) = total_training_set

    print(f"val set tensor dim: {X_rna_test_tensor_normalized.shape}, {X_atac_test_tensor_normalized.shape}")
    print(f"train set tensor dim: {total_X_rna_tensor_normalized.shape}, {total_X_atac_tensor_normalized.shape}")

    (encoder1, encoder2, gnn, mlp1, mlp2, graph_dec, decoder1, decoder2) = model_tuple

    if param_savepath is not None and os.path.exists(param_savepath):
        state_dicts = torch.load(param_savepath)
        encoder1.load_state_dict(state_dicts['encoder1'])
        encoder2.load_state_dict(state_dicts['encoder2'])
        gnn.load_state_dict(state_dicts['gnn'])
        mlp1.load_state_dict(state_dicts['mlp1'])
        mlp2.load_state_dict(state_dicts['mlp2'])
        graph_dec.load_state_dict(state_dicts['graph_dec'])
        decoder1.load_state_dict(state_dicts['decoder1'])
        decoder2.load_state_dict(state_dicts['decoder2'])
        print("load params successful")
    if best_ari_path is not None and os.path.exists(best_ari_path):
        with open(best_ari_path, 'rb') as file:
            best_train_ari = pickle.load(file)
        print(f"previous best_train_ari is {best_train_ari}")

    for i in range(niter):

        for train_batch in tqdm(train_set):
            (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
             scRNA_mini_batch_anndata, scATAC_mini_batch_anndata, emb_size) = train_batch

            recon_loss, kl_loss = helper2.finetune_one_epoch(
                encoder1, encoder2, gnn, mlp1, mlp2, graph_dec, decoder1, decoder2, optimizer, X_rna_tensor,
                X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
                edge_index, emb_size, lookup, random_matrix, device, i, niter
            )

            NELBO = recon_loss + kl_loss

        if (i + 1) % ari_freq == 0:
            # with torch.no_grad():
            theta, theta_gene, theta_peak = helper2.get_theta_GNN(
                encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
                X_rna_test_tensor_normalized, X_atac_test_tensor_normalized
            )

            theta_train, theta_gene_train, theta_peak_train = helper2.get_theta_GNN(
                encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
                total_X_rna_tensor_normalized, total_X_atac_tensor_normalized
            )

            res, ari, nmi = helper2.evaluate_ari2(theta.to('cpu'), scRNA_test_anndata)
            res_train, ari_train, nmi_train = helper2.evaluate_ari2(theta_train.to('cpu'), total_scRNA_anndata)

            ari_trains.append(ari_train)
            ari_tests.append(ari)
            loss_set.append(NELBO)
            recon_list.append(recon_loss)
            kl_list.append(kl_loss)

            impute_title = "Finetune_nll"

            impute_mod = "ATAC"
            if impute_mod == "RNA":
                from_mod = "ATAC"
            else:
                from_mod = "RNA"
            correlation_pear, correlation_spear = helper2.impute_one_epoch(encoder1, encoder2, gnn, mlp1, mlp2,
                                                                           graph_dec, decoder1, decoder2, impute_mod,
                                                                           X_rna_test_tensor,
                                                                           X_rna_test_tensor_normalized,
                                                                           X_atac_test_tensor,
                                                                           X_atac_test_tensor_normalized,
                                                                           edge_index, 512, lookup, random_matrix, device,
                                                                           impute_title, niter)
            pear_list.append(correlation_pear)
            spear_list.append(correlation_spear)
            print(
                f'==== {from_mod} to {impute_mod} Imputation  ====\n'
                'Pearson correlation: {:.4f}\t Spearman correlation: {:.4f}\t \n'
                .format(
                    correlation_pear, correlation_spear)
            )

            impute_mod = "RNA"
            if impute_mod == "RNA":
                from_mod = "ATAC"
            else:
                from_mod = "RNA"
            correlation_pear, correlation_spear = helper2.impute_one_epoch(encoder1, encoder2, gnn, mlp1, mlp2,
                                                                           graph_dec, decoder1, decoder2, impute_mod,
                                                                           X_rna_test_tensor,
                                                                           X_rna_test_tensor_normalized,
                                                                           X_atac_test_tensor,
                                                                           X_atac_test_tensor_normalized,
                                                                           edge_index, 512, lookup, random_matrix, device,
                                                                           impute_title, niter)
            pear_list.append(correlation_pear)
            spear_list.append(correlation_spear)

            # correlation_pear = 1
            # correlation_spear = 1
            print('====  Iter: {}, NELBO: {:.4f}, recon_loss: {:.4f}, kl_loss: {:.4f}  ====\n'
                  
                  f'==== {from_mod} to {impute_mod} Imputation  ====\n'
                  'Pearson correlation: {:.4f}\t Spearman correlation: {:.4f}\t \n'
                  .format(i + 1, NELBO, recon_loss, kl_loss,
                          correlation_pear, correlation_spear)
                  )
            # if best_cor < epoch_cor:
            #     best_cor = ari
            #     if param_savepath is not None and best_ari_path is not None:
            #         torch.save({
            #             'encoder1': encoder1.state_dict(),
            #             'encoder2': encoder2.state_dict(),
            #             'gnn': gnn.state_dict(),
            #             'mlp1': mlp1.state_dict(),
            #             'mlp2': mlp2.state_dict(),
            #             'graph_dec': graph_dec.state_dict(),
            #             'decoder1': decoder1.state_dict(),
            #             'decoder2': decoder2.state_dict()
            #         }, param_savepath)

    return (encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
            best_ari, best_theta, best_beta_gene, best_beta_peak,
            best_train_ari, best_train_theta, ari_trains, ari_tests, loss_set,
            recon_list, kl_list, pear_list, spear_list)


if __name__ == "__main__":
    rna_path = "../../data/10x-Multiome-Pbmc10k-RNA.h5ad"
    atac_path = "../../data/10x-Multiome-Pbmc10k-ATAC.h5ad"
    # rna_path = "../data/BMMC_rna_filtered.h5ad"
    # atac_path = "../data/BMMC_atac_filtered.h5ad"

    gene_exp = anndata.read('../../data/10x-Multiome-Pbmc10k-RNA.h5ad')
    peak_exp = anndata.read('../../data/10x-Multiome-Pbmc10k-ATAC.h5ad')
    total_gene = gene_exp.shape[1]
    total_peak = peak_exp.shape[1]
    total_cell_num = gene_exp.shape[0]

    # index_path = '../data/relation/gene_peak_index_relation.pickle'
    # gene_index_list, peak_index_list = helper2.get_peak_index(index_path, top=5, threshould=None, gene_limit=12000)
    index_path = '../../data/highly_gene_peak_index_relation.pickle'
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
    batch_num = 500
    batch_size = int(num_of_cell / batch_num)
    emb_size = 512
    # emb_graph = num_of_cell
    # emb_graph = emb_size
    num_of_topic = 100
    gnn_conv = 'GATv2'
    num_epochs = 40
    ari_freq = 1
    plot_path_rel = "./plot/"
    lr = 0.001
    # param_savepath = f"./model_params/best_model_{emb_size}.pth"
    # best_ari_path = f"./model_params/best_ari_{emb_size}.pkl"
    param_savepath = f"./model_params/model_node2vec_{emb_size}.pth"
    best_ari_path = f"./model_params/ari_node2vec_mean_{emb_size}.pth"
    # param_savepath = None
    # best_ari_path = None

    seed = 123

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU随机种子确定
    torch.cuda.manual_seed(seed)  # GPU随机种子确定
    torch.cuda.manual_seed_all(seed)  # 所有的GPU设置种子

    print(f"num_of_topic: {num_of_topic}")
    # mtx_path = '../data/relation_by_score/top5_peak_tf_gene.pickle'
    # mtx_path = '../data/relation2/top5_peak_tf_gene.pickle'
    # mtx_path = '../data/TF_gene/top5_peak_tf_gene.pickle'
    mtx_path = '../../data/notf_gene_peak/top5_peak_gene.pickle'
    result, edge_index = helper2.get_sub_graph_by_index(
        path=mtx_path,
        gene_index_list=gene_index_list,
        peak_index_list=peak_index_list,
        total_peak=total_peak
    )


    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index()
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

    # use node2vec to get lookup table
    Node2Vec_model = Node2Vec(edge_index, emb_size, walk_length=10, context_size=10, walks_per_node=10)
    Node2Vec_model = Node2Vec_model.to(device)

    Node2Vec_model.train()
    optimizer = torch.optim.Adam(Node2Vec_model.parameters(), lr=0.01)
    loader = Node2Vec_model.loader(batch_size=256, shuffle=True, num_workers=0)
    for epoch in tqdm(range(200)):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = Node2Vec_model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

    node_embeddings = Node2Vec_model.embedding.weight
    print(node_embeddings.shape)
    print(node_embeddings.sum())

    result_dense = result.toarray()
    result_tensor = torch.from_numpy(result_dense).float().to(device)
    print(f"result.shape: {result_tensor.shape}, result type: {type(result_tensor)}")
    print(result_tensor.sum())

    training_set, total_training_set, test_set, fm = mini_batch.process_mini_batch_data(
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
    )

    encoder1 = model2.VAE(num_of_gene, emb_size, num_of_topic).to(device)
    encoder2 = model2.VAE(num_of_peak, emb_size, num_of_topic).to(device)
    gnn = model2.GNN(emb_size, emb_size * 2, emb_size, 1, node_embeddings.to(device), device, 0, gnn_conv).to(device)
    mlp1 = model2.MLP(emb_size, emb_size * 2, emb_size).to(device)
    mlp2 = model2.MLP(emb_size, emb_size * 2, emb_size).to(device)
    graph_dec = model2.DEC(emb_size, emb_size * 4, num_of_peak + num_of_gene).to(device)
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

    optimizer = optim.AdamW(parameters, lr=lr, weight_decay=1.2e-6)

    model_tuple = (encoder1, encoder2, gnn, mlp1, mlp2, graph_dec, decoder1, decoder2)
    for model in model_tuple:
        print(model)

    print(f"=========  start training {num_of_topic}  =========")
    st = time.time()
    (encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
     best_ari, best_theta, best_beta_gene, best_beta_peak,
     best_train_ari, best_train_theta, ari_trains, ari_tests, loss_set,
     recon_list, kl_list, pear_list, spear_list) = train(
        model_tuple=model_tuple,
        optimizer=optimizer,
        train_set=training_set,
        total_training_set=total_training_set,
        test_set=test_set,
        lookup=node_embeddings,  # node_embeddings or fm
        random_matrix=node_embeddings,  # node_embeddings or fm
        edge_index=edge_index.to(device),
        ari_freq=ari_freq,
        niter=num_epochs,
        device=device,
        param_savepath=param_savepath,
        best_ari_path=best_ari_path,
    )
    ed = time.time()
    print(f"training time: {ed - st}")

    print(f"best_train_ari: {best_train_ari}, best_val_ari: {best_ari}")
    print(ari_trains)
    print(ari_tests)
    print(loss_set)

    path = './plot_Oct24/'
    title = f'{emb_size}_Normalized_ATAC_to_RNA_correlation'
    view_results.plot_graph2(path, title, pear_list, 'Pearson', spear_list, 'Spearman')

    title = f'{emb_size}_Normalized_ATAC_to_RNA_ARI'
    view_results.plot_graph2(path, title, ari_trains, 'ARI_Train', ari_tests, 'ARI_Val')

    title = f'{emb_size}_Nomralized_Training_Loss'
    view_results.plot_half_graph(path, title, recon_list, 'Recon_Loss', kl_list, 'KL_Loss')
