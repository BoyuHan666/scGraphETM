import math

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
import scanpy as sc

import select_gpu
import mini_batch
import helper2
import model2
import model
import view_results
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json


def set_device(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index()
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

    return device


def plot_umap(train_adata, test_adata, train_theta, test_theta, mode, iter, plot_dir, tissue='PBMC'):
    # Plot UMAP
    train_adata.obsm['delta'] = train_theta.to('cpu')
    res1 = utils.evaluate(adata=train_adata, plot_fname=f"{mode}_umap", n_epoch=f'_train_iter{iter}',
                          cell_type_col="Celltype", batch_col=None, return_fig=True,
                          plot_dir=f"{plot_dir}{mode}")
    print(res1)

    res_train = {}
    res_train['asw'] = float(res1['asw'])
    res_train['asw_2'] = float(res1['asw_2'])
    res_train['ari'] = float(res1['ari'])
    res_train['nmi'] = float(res1['nmi'])
    res_train['avg_bio'] = (float(res1['asw_2']) + float(res1['ari']) + float(res1['nmi'])) / 3

    # train_gene_exp.obsm['delta'] = best_z_train.to('cpu')
    # res1 = utils.evaluate(adata=train_gene_exp, plot_fname=f"{mode}_umap", n_epoch='_train', cell_type_col="cell_type", batch_col = None,return_fig=True,
    #                       plot_dir=path)

    test_adata.obsm['delta'] = test_theta.to('cpu')
    res2 = utils.evaluate(adata=test_adata, plot_fname=f"{mode}_umap", n_epoch=f'_test_iter{iter}',
                          cell_type_col="Celltype",
                          batch_col=None, return_fig=True,
                          plot_dir=f"{plot_dir}{mode}")
    res_test = {}
    res_test['asw'] = float(res2['asw'])
    res_test['asw_2'] = float(res2['asw_2'])
    res_test['ari'] = float(res2['ari'])
    res_test['nmi'] = float(res2['nmi'])
    res_test['avg_bio'] = (float(res2['asw_2']) + float(res2['ari']) + float(res2['nmi'])) / 3

    # test_gene_exp.obsm['delta'] = best_z_test.to('cpu')
    # res2 = utils.evaluate(adata=test_gene_exp, plot_fname=f"{mode}_umap", n_epoch='_test', cell_type_col="cell_type", batch_col = None, return_fig=True,
    #                       plot_dir=path)
    print(res2)

    with open(f"./numerical_result/{tissue}/{mode}_train_result.json", 'w') as file:
        json.dump(res_train, file, indent=4)

    with open(f"./numerical_result/{tissue}/{mode}_test_result.json", 'w') as file:
        json.dump(res_test, file, indent=4)

    view_results.generate_tsne(train_theta, train_adata, plot_path_rel=f"{plot_dir}{mode}/", title=f"tsne_train_iter{iter}")
    view_results.generate_tsne(test_theta, test_adata, plot_path_rel=f"{plot_dir}{mode}/", title=f"tsne_test_iter{iter}")

def train(model_tuple, optimizer,
          train_set, total_training_set, test_set, emb_size,
          lookup, random_matrix, edge_index, ari_freq, niter,
          device, param_savepath=None, best_ari_path=None, plot_rel_path=None, mode=None, tissue="PBMC", umap_freq=1):
    NELBO = None
    best_ari = 0
    best_train_ari = 0
    best_theta = None
    best_train_theta = None
    best_beta_gene = None
    best_beta_peak = None
    best_z_test = None
    best_z_train = None
    ari_trains = []
    ari_tests = []
    loss_set = []
    recon_list = []
    kl_list = []
    pear_list = []
    spear_list = []

    (encoder1, encoder2, gnn, mlp1, mlp2, graph_dec, decoder1, decoder2) = model_tuple

    (X_rna_test_tensor, X_rna_test_tensor_normalized, X_atac_test_tensor,
     X_atac_test_tensor_normalized, scRNA_test_anndata, scATAC_test_anndata,
     test_feature_matrix, test_edge_index) = test_set

    (total_X_rna_tensor, total_X_rna_tensor_normalized, total_X_atac_tensor,
     total_X_atac_tensor_normalized, total_scRNA_anndata, total_scATAC_anndata,
     total_feature_matrix, total_edge_index) = total_training_set

    print(f"val set tensor dim: {X_rna_test_tensor_normalized.shape}, {X_atac_test_tensor_normalized.shape}")
    print(f"train set tensor dim: {total_X_rna_tensor_normalized.shape}, {total_X_atac_tensor_normalized.shape}")

    if load_prev:
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
                                                                   edge_index, emb_size, lookup, random_matrix, device,
                                                                   1, niter)
    # correlation_pear = 1
    # correlation_spear = 1
    pear_list.append(correlation_pear)
    spear_list.append(correlation_spear)
    print(
          f'==== {from_mod} to {impute_mod} Imputation  ====\n'
          'Pearson correlation: {:.4f}\t Spearman correlation: {:.4f}\t \n'
          .format(
              correlation_pear, correlation_spear)
          )

    for i in range(niter):

        for train_batch in tqdm(train_set):
            (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
             scRNA_mini_batch_anndata, scATAC_mini_batch_anndata, emb_size) = train_batch

            recon_loss, kl_loss = helper2.train_one_epoch(
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

            z_test, z_gene_test, z_peak_test = helper2.get_z_GNN(
                encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
                X_rna_test_tensor_normalized, X_atac_test_tensor_normalized
            )

            z_train, z_gene_train, z_peak_train = helper2.get_z_GNN(
                encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
                total_X_rna_tensor_normalized, total_X_atac_tensor_normalized
            )

            if i % umap_freq == 0:
                plot_umap(total_scRNA_anndata, scRNA_test_anndata, theta_train, theta, mode, i, plot_dir=plot_rel_path,
                          tissue=tissue)

            ari_trains.append(ari_train)
            ari_tests.append(ari)
            loss_set.append(NELBO)
            recon_list.append(recon_loss)
            kl_list.append(kl_loss)

            # mod = "ATAC"
            # if impute_mod == "RNA":
            #     from_mod = "ATAC"
            # else:
            #     from_mod = "RNA"
            # correlation_pear, correlation_spear = helper2.impute_one_epoch(encoder1, encoder2, gnn, mlp1, mlp2,
            #                                                                graph_dec, decoder1, decoder2, impute_mod,
            #                                                                X_rna_test_tensor,
            #                                                                X_rna_test_tensor_normalized,
            #                                                                X_atac_test_tensor,
            #                                                                X_atac_test_tensor_normalized,
            #                                                                edge_index, emb_size, lookup, random_matrix, device,
            #                                                                i, niter)
            correlation_pear = 1
            correlation_spear = 1
            pear_list.append(correlation_pear)
            spear_list.append(correlation_spear)
            print('====  Iter: {}, NELBO: {:.4f}, recon_loss: {:.4f}, kl_loss: {:.4f}  ====\n'
                  'Train res: {}\t Train ARI: {:.4f}\t Train NMI: {:.4f}\n'
                  'Valid res: {}\t Valid ARI: {:.4f}\t Valid NMI: {:.4f}\n'
                  # f'==== {from_mod} to {impute_mod} Imputation  ====\n'
                  'Pearson correlation: {:.4f}\t Spearman correlation: {:.4f}\t \n'
                  .format(i + 1, NELBO, recon_loss, kl_loss,
                          res_train, ari_train, nmi_train,
                          res, ari, nmi,
                          correlation_pear, correlation_spear)
                  )

            # impute_mod = "RNA"
            # if impute_mod == "RNA":
            #     from_mod = "ATAC"
            # else:
            #     from_mod = "RNA"
            # correlation_pear, correlation_spear = helper2.impute_one_epoch(encoder1, encoder2, gnn, mlp1, mlp2,
            #                                                                graph_dec, decoder1, decoder2, impute_mod,
            #                                                                X_rna_test_tensor,
            #                                                                X_rna_test_tensor_normalized,
            #                                                                X_atac_test_tensor,
            #                                                                X_atac_test_tensor_normalized,
            #                                                                edge_index, emb_size, lookup, random_matrix,
            #                                                                device,
            #                                                                1, niter)
            # # correlation_pear = 1
            # # correlation_spear = 1
            # pear_list.append(correlation_pear)
            # spear_list.append(correlation_spear)
            # print(
            #     f'==== {from_mod} to {impute_mod} Imputation  ====\n'
            #     'Pearson correlation: {:.4f}\t Spearman correlation: {:.4f}\t \n'
            #     .format(
            #         correlation_pear, correlation_spear)
            # )

            if best_ari < ari:
                best_ari = ari
                best_theta = theta
                best_z_test = z_test

            if best_train_ari < ari_train:
                best_train_ari = ari_train
                best_train_theta = theta_train
                best_z_train = z_train
                if param_savepath is not None and best_ari_path is not None:
                    torch.save({
                        'encoder1': encoder1.state_dict(),
                        'encoder2': encoder2.state_dict(),
                        'gnn': gnn.state_dict(),
                        'mlp1': mlp1.state_dict(),
                        'mlp2': mlp2.state_dict(),
                        'graph_dec': graph_dec.state_dict(),
                        'decoder1': decoder1.state_dict(),
                        'decoder2': decoder2.state_dict()
                    }, param_savepath)
                    with open(best_ari_path, 'wb') as file:
                        pickle.dump(best_train_ari, file)
                    print("save params successful!")

    return (encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
            best_ari, theta, z_test, best_beta_gene, best_beta_peak,
            best_train_ari, theta_train, z_train, ari_trains, ari_tests, loss_set,
            recon_list, kl_list, pear_list, spear_list)


def node2vec_emb(edge_index, emb_size, num_nodes, device, walk_length=10, context_size=10, walks_per_node=10):
    # use node2vec to get lookup table
    Node2Vec_model = Node2Vec(edge_index, emb_size, num_nodes=num_nodes, walk_length=walk_length,
                              context_size=context_size,
                              walks_per_node=walks_per_node)
    Node2Vec_model = Node2Vec_model.to(device)

    Node2Vec_model.train()
    optimizer = torch.optim.Adam(Node2Vec_model.parameters(), lr=0.01)
    loader = Node2Vec_model.loader(batch_size=256, shuffle=True, num_workers=0)
    for epoch in tqdm(range(200)):
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader):
            optimizer.zero_grad()  # set the gradients to 0
            loss = Node2Vec_model.loss(pos_rw.to(device), neg_rw.to(device))  # compute the loss for the batch
            loss.backward()
            optimizer.step()  # optimize the parameters

    node_embeddings = Node2Vec_model.embedding.weight
    print(f"Shape of Node2Vec emb: {node_embeddings.shape}")
    # print(node_embeddings.sum())
    return node_embeddings


def matrix2tensor(gcn_matrix, device):
    sparse_matrix_coo = gcn_matrix.tocoo()
    values = sparse_matrix_coo.data
    indices = np.vstack((sparse_matrix_coo.row, sparse_matrix_coo.col))

    # Convert indices and values to PyTorch tensors
    indices_tensor = torch.LongTensor(indices)
    values_tensor = torch.FloatTensor(values)

    gcn_tensor = torch.sparse.FloatTensor(indices_tensor, values_tensor, torch.Size(sparse_matrix_coo.shape)).to(device)
    return gcn_tensor, indices_tensor.to(device)


if __name__ == "__main__":

    seed = 123
    device = set_device(seed)

    dataset = "PBMC"
    tissue = "PBMC"

    # data_folder = "../../data/"
    # dataset = "BMMC"
    # tissue = "BMMC"
    #
    # # dataset = "3_mouse_data"
    # # tissue = "mouse_kidney"
    #
    # cistarget_threshold = 3
    # tss_distance = 2000
    # rna_path = f"{data_folder}{dataset}/{tissue}_processed/HV_{tissue}_rna_count.h5ad"
    # atac_path = f"{data_folder}{dataset}/{tissue}_processed/TSS_{tissue}_atac_count.h5ad"
    #
    # rna_path = f"{data_folder}{dataset}/BMMC_rna_filtered.h5ad"
    # atac_path = f"{data_folder}{dataset}/BMMC_atac_filtered.h5ad"

    # gene_exp = anndata.read(rna_path)
    # sc.pp.normalize_total(gene_exp, target_sum=1e4)
    # sc.pp.log1p(gene_exp)
    #
    # sc.pp.highly_variable_genes(gene_exp)
    # gene_exp = gene_exp[:, gene_exp.var['highly_variable']]
    # gene_exp.var['gex_name'] = gene_exp.var_names
    # gene_exp.var.reset_index(drop=True, inplace=True)
    #
    # peak_exp = anndata.read(atac_path)
    # sc.pp.normalize_total(peak_exp, target_sum=1e4)
    # sc.pp.log1p(peak_exp)
    #
    # sc.pp.highly_variable_genes(peak_exp)
    # peak_exp = peak_exp[:, peak_exp.var['highly_variable']]
    # peak_exp.var['peak_name'] = peak_exp.var_names
    # peak_exp.var.reset_index(drop=True, inplace=True)
    #
    # gene_num = gene_exp.shape[1]
    # peak_num = peak_exp.shape[1]
    # total_cell_num = gene_exp.shape[0]
    #
    # gcn_path = f"{data_folder}{dataset}/{tissue}_processed/HV_{tissue}_gcn_threshold{cistarget_threshold}_{tss_distance}bp.pkl"
    # with open(gcn_path, 'rb') as file:
    #     gcn_matrix = pickle.load(file)
    #
    # warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

    mode = "learnable6.0"
    load_prev = False

    rna_path = "../../data/10x-Multiome-Pbmc10k-RNA.h5ad"
    atac_path = "../../data/10x-Multiome-Pbmc10k-ATAC.h5ad"

    gene_exp = anndata.read('../../data/10x-Multiome-Pbmc10k-RNA.h5ad')
    peak_exp = anndata.read('../../data/10x-Multiome-Pbmc10k-ATAC.h5ad')
    total_gene = gene_exp.shape[1]
    total_peak = peak_exp.shape[1]
    total_cell_num = gene_exp.shape[0]

    index_path = '../../data/highly_gene_peak_index_relation.pickle'
    gene_index_list, peak_index_list = helper2.get_peak_index(index_path, top=5, threshould=None)
    gene_exp = gene_exp[:, gene_index_list]
    gene_num = gene_exp.shape[1]
    peak_exp = peak_exp[:, peak_index_list]
    peak_num = peak_exp.shape[1]

    mtx_path = '../../data/notf_gene_peak/top5_peak_gene.pickle'
    gcn_tensor, edge_index = helper2.get_sub_graph_by_index(
        path=mtx_path,
        gene_index_list=gene_index_list,
        peak_index_list=peak_index_list,
        total_peak=total_peak
    )

    warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

    plot_path_rel = f"./plot/{tissue}/Jan10/"
    if not os.path.exists(plot_path_rel):
        os.makedirs(plot_path_rel)
        print(f"Folder created at {plot_path_rel}")
    else:
        print(f"Folder already exists at {plot_path_rel}")

    num_of_cell = 8000
    # num_of_cell = math.floor(total_cell_num * 0.8)
    # num_of_gene = 20000
    # num_of_peak = 60000
    num_of_gene = gene_num
    num_of_peak = peak_num
    test_num_of_cell = 1500
    # test_num_of_cell = total_cell_num - num_of_cell
    batch_num = 500
    batch_size = int(num_of_cell / batch_num)
    emb_size = 512
    # emb_graph = num_of_cell
    # emb_graph = emb_size
    num_of_topic = 40
    gnn_conv = 'GATv2'
    num_epochs = 41
    ari_freq = 1
    umap_freq = 5

    lr = 0.001
    save_path = f"./model_params/{tissue}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Folder created at {save_path}")
    else:
        print(f"Folder already exists at {save_path}")

    param_savepath = f"./model_params/{tissue}/best_model_{mode}_GCN_{num_of_cell}_{num_of_topic}_{emb_size}.pth"
    best_ari_path = f"./model_params/{tissue}/best_ari_{mode}_GCN_{num_of_cell}_{num_of_topic}_{emb_size}.pth"

    print(
        f"Num of training cells: {num_of_cell}\nNum of genes: {num_of_gene}\nNum of peaks: {num_of_peak}\nNum_of_topic: {num_of_topic}")

    # gcn_tensor, edge_index = matrix2tensor(gcn_matrix, device)
    node_embeddings = node2vec_emb(edge_index, emb_size, num_of_peak + num_of_gene, device)

    print(f"GCN matrix shape: {gcn_tensor.shape}, result type: {type(gcn_tensor)}")
    print(f"Number of edges in GCN: {gcn_tensor.sum()}")

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

    # encoder1 = model.VAE(num_of_gene, emb_size, num_of_topic).to(device)
    # encoder2 = model.VAE(num_of_peak, emb_size, num_of_topic).to(device)
    # gnn = model.GNN(emb_size, emb_size * 2, emb_size, 1, num_of_gene, num_of_peak, 0, gnn_conv).to(device)
    # mlp1 = model.MLP(emb_size, emb_size * 2, emb_size).to(device)
    # mlp2 = model.MLP(emb_size, emb_size * 2, emb_size).to(device)
    # graph_dec = model.DEC(emb_size, emb_size * 4, num_of_peak + num_of_gene).to(device)
    # decoder1 = model.LDEC(num_of_gene, emb_size, num_of_topic, batch_size).to(device)
    # decoder2 = model.LDEC(num_of_peak, emb_size, num_of_topic, batch_size).to(device)

    encoder1 = model2.VAE(num_of_gene, emb_size, num_of_topic).to(device)
    encoder2 = model2.VAE(num_of_peak, emb_size, num_of_topic).to(device)
    gnn = model2.GNN(emb_size, emb_size * 2, emb_size, 1, num_of_gene, num_of_peak, 0, gnn_conv).to(device)
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
     best_ari, best_theta, best_z_test, best_beta_gene, best_beta_peak,
     best_train_ari, best_train_theta, best_z_train, ari_trains, ari_tests, loss_set,
     recon_list, kl_list, pear_list, spear_list) = train(
        model_tuple=model_tuple,
        optimizer=optimizer,
        train_set=training_set,
        total_training_set=total_training_set,
        test_set=test_set,
        emb_size=emb_size,
        lookup=node_embeddings,  # node_embeddings or fm
        random_matrix=torch.randn((num_of_gene + num_of_peak, emb_size)).to(device),  # random matrix
        edge_index=edge_index.to(device),
        ari_freq=ari_freq,
        niter=num_epochs,
        device=device,
        param_savepath=param_savepath,
        best_ari_path=best_ari_path,
        plot_rel_path=plot_path_rel,
        mode=mode,
        tissue=tissue,
        umap_freq=umap_freq
    )
    ed = time.time()
    print(f"training time: {ed - st}")

    print(f"best_train_ari: {best_train_ari}, best_val_ari: {best_ari}")
    print(ari_trains)
    print(ari_tests)
    print(loss_set)

    path = plot_path_rel

    title = f'{emb_size}_Normalized_ATAC_to_RNA_correlation_{mode}'
    view_results.plot_graph2(path, title, pear_list, 'Pearson', spear_list, 'Spearman')

    title = f'{emb_size}_Normalized_ATAC_to_RNA_ARI_{mode}'
    view_results.plot_graph2(path, title, ari_trains, 'ARI_Train', ari_tests, 'ARI_Val')

    title = f'{emb_size}_Nomralized_Training_Loss_{mode}'
    view_results.plot_half_graph(path, title, recon_list, 'Recon_Loss', kl_list, 'KL_Loss')
