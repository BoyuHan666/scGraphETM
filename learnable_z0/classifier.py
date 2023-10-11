import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch import optim
import time
import warnings
from numba.core.errors import NumbaDeprecationWarning
import anndata
import torch.nn as nn
from tqdm import tqdm
import os

import select_gpu
import mini_batch
import helper2
import model2

import torch


def sample_from_encoder(encoder, data):
    with torch.no_grad():
        mu, log_var, _ = encoder(data)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sampled_emb = mu + eps * std
    return sampled_emb

def train_one_batch(gnn, mlp_gat, mlp_vae, encoder1, encoder2, optimizer, RNA_tensor_normalized, ATAC_tensor_normalized,
                    emb_size, random_matrix, edge_index, true_celltype, device):
    gnn.train()
    mlp_gat.train()
    mlp_vae.train()

    gnn.zero_grad()
    mlp_gat.zero_grad()
    mlp_vae.zero_grad()
    optimizer.zero_grad()

    cell_num = len(RNA_tensor_normalized)

    pred_celltype_list = []
    one_hot_encoded_pred = []
    gat_list = []
    one_hot_encoded = F.one_hot(true_celltype, num_classes=20)
    batch_loss = 0
    for cell in range(cell_num):
        one_cell_RNA_tensor_normalized = RNA_tensor_normalized[cell].unsqueeze(0)
        one_cell_ATAC_tensor_normalized = ATAC_tensor_normalized[cell].unsqueeze(0)
        feature_matrix = helper2.generate_feature_matrix(one_cell_RNA_tensor_normalized,
                                                         one_cell_ATAC_tensor_normalized,
                                                         emb_size, random_matrix, device)
        # gene+peak x emb
        cell_emb = gnn(feature_matrix.to(device), edge_index.to(device))
        # 1 x emb
        # cell_emb = cell_emb.mean(dim=1, keepdim=True).T
        cell_emb, _ = torch.max(cell_emb, dim=1)

        gat_list.append(cell_emb[0])
        cell_emb1 = sample_from_encoder(encoder1, one_cell_RNA_tensor_normalized)
        cell_emb2 = sample_from_encoder(encoder2, one_cell_ATAC_tensor_normalized)
        gat_emb = mlp_gat(cell_emb)
        vae_emb = mlp_vae(cell_emb2)
        pred_emb =gat_emb

        # print(pred_emb)
    # gat_list = torch.stack(gat_list).to(device)
    # gat_emb = mlp_gat(RNA_tensor_normalized)
    # pred_emb = (gat_emb + cell_emb1 + cell_emb2) / 3.0
    # pred_emb = mlp_vae(cell_emb1)

    # vae_emb2 = mlp_vae(cell_emb2)
    # print(pred_emb)
    # pred_emb = F.softmax(pred_emb, dim=1)
    # print(pred_emb)
        pred_celltype = torch.argmax(pred_emb)
    # pred_celltype = pred_celltype.clone().detach().requires_grad_(True)

        pred_celltype_list.append(pred_celltype)
        one_hot_encoded_pred.append(pred_emb[0])

    # loss_func = nn.MSELoss()

    # optimizer.zero_grad()
    # print(pred_emb[0])
    # loss = loss_func(pred_emb[0], one_hot_encoded[cell].to(torch.float32))
    # batch_loss += loss

    # one_hot_encoded_pred = torch.stack(one_hot_encoded_pred, dim=0).detach()
    # one_hot_encoded_pred = one_hot_encoded_pred.requires_grad_(True).to(device)
    # print(one_hot_encoded_pred)
    # one_hot_encoded_pred = F.one_hot(one_hot_encoded_pred, num_classes=20).to(torch.float32).requires_grad_(True)
    # one_hot_encoded_pred = torch.concat(one_hot_encoded_pred, dim=0)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(pred_emb, true_celltype[cell])
        # loss = batch_loss/cell_num

        batch_loss += loss
        loss.backward()
        optimizer.step()

    pred_celltype = torch.tensor(pred_celltype_list).to(device)
    # print(pred_celltype)
    batch_loss = batch_loss / cell_num
    return batch_loss.item(), pred_celltype


def train(gnn, mlp_gat, mlp_vae, encoder1, encoder2, optimizer, train_set, labels, batch_size, random_matrix, edge_index, epochs, device):

    # labels = F.one_hot(labels, num_classes=2).float().clone().detach().requires_grad_(True)
    # print(labels)

    if os.path.exists(param_savepath):
        state_dicts = torch.load(param_savepath)
        # gnn.load_state_dict(state_dicts['gnn'])
        encoder1.load_state_dict(state_dicts['encoder1'])
        encoder2.load_state_dict(state_dicts['encoder2'])
        print("load params successful")

    for epoch in range(epochs):
        pred_labels_list = []
        total_loss = 0
        index = 0
        for train_batch in tqdm(train_set):
            (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
             scRNA_mini_batch_anndata, scATAC_mini_batch_anndata, emb_size) = train_batch

            start = index*batch_size
            end = start+batch_size
            true_celltype = labels[start:end]

            loss, pred_celltype = train_one_batch(gnn, mlp_gat, mlp_vae, encoder1, encoder2, optimizer, X_rna_tensor_normalized, X_atac_tensor_normalized,
                                   emb_size, random_matrix, edge_index, true_celltype, device)
            total_loss+=loss

            pred_labels_list.append(pred_celltype)
            index += 1

        pred_labels = torch.cat(pred_labels_list, dim=0)
        acc = 0.0
        for i in range(len(pred_labels)):
            if int(labels[i]) == int(pred_labels[i]):
                acc += 1.0
        acc /= len(pred_labels)

        print(f"training Loss: {total_loss}, training Accuracy: {acc * 100:.2f}%")


    model_tuple = (gnn, mlp)
    return model_tuple


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
    emb_size = 256
    emb_size2 = 256
    # emb_graph = num_of_cell
    emb_graph = emb_size2
    num_of_topic = 20
    gnn_conv = 'GATv2'
    num_epochs = 40
    ari_freq = 2
    plot_path_rel = "./plot/"
    lr = 0.01
    param_savepath = f"./model_params/best_model_{emb_size2}.pth"

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

    cell_type = (gene_exp.obs['cell_type'].values)
    celltype_to_numeric = {category: index for index, category in enumerate(sorted(set(cell_type)))}
    numeric_labels = [celltype_to_numeric[category] for category in cell_type]
    labels = torch.tensor(numeric_labels).to(device)
    print(labels)

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

    # gene+peak x emb
    gnn = model2.GNN(emb_graph, emb_size2 * 2, emb_size2, 1, device, 0, gnn_conv).to(device)
    encoder1 = model2.VAE(num_of_gene, emb_size, num_of_topic).to(device)
    encoder2 = model2.VAE(num_of_peak, emb_size, num_of_topic).to(device)
    # 1 x emb
    mlp_gat = model2.MLP(num_of_gene+num_of_peak, emb_size2, num_of_topic).to(device)
    mlp_vae = model2.MLP(20, 64, num_of_topic).to(device)

    model_tuple = (gnn, mlp_gat, mlp_vae, encoder1, encoder2)


    parameters = [
                  {'params': gnn.parameters()},
                  {'params': mlp_gat.parameters()},
                  {'params': mlp_vae.parameters()},
                  {'params': encoder1.parameters()},
                  {'params': encoder2.parameters()},
                ]

    optimizer = optim.Adam(parameters, lr=lr)
    # optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9)

    print(model_tuple)
    print(labels)

    st = time.time()
    # gnn, mlp, optimizer, train_set, labels, emb_size, random_matrix, edge_index, epochs, device
    model = train(
        gnn, mlp_gat, mlp_vae, encoder1, encoder2, optimizer, training_set, labels, batch_size, fm, edge_index, num_epochs, device
    )
    ed = time.time()
    print(f"training time: {ed - st}")