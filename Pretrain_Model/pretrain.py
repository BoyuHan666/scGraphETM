import torch
from scipy import stats
from torch.nn import functional as F
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.autograd import Variable
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch import optim
from torch_geometric.nn import InnerProductDecoder

import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

import select_gpu
import mini_batch
import full_batch
import helper2
import model2
import view_result
import time


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, device, dropout, conv_model):
        super(GNN, self).__init__()
        self.embedding = None
        if conv_model == 'GATv2':
            self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
            self.conv2 = GATv2Conv(hidden_channels * num_heads, hidden_channels * num_heads, heads=1, concat=True,
                                   dropout=dropout)
            self.conv3 = GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, concat=True, dropout=dropout)
        if conv_model == 'Transformer':
            self.conv1 = TransformerConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
            self.conv2 = TransformerConv(hidden_channels * num_heads, out_channels, heads=1, concat=True,
                                         dropout=dropout)
        if conv_model == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
            self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=True, dropout=dropout)
        if conv_model == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        self.embedding = x
        return x

    def backward(self, loss):
        self.loss = loss
        self.loss.backward()
        return self.loss

    def get_embedding(self):
        return self.embedding


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

if __name__ == "__main__":
    rna_path = "../data/10x-Multiome-Pbmc10k-RNA.h5ad"
    atac_path = "../data/10x-Multiome-Pbmc10k-ATAC.h5ad"
    num_of_cell = 2000
    num_of_gene = 2000
    num_of_peak = 2000
    test_num_of_cell = 2000
    batch_size = 2000
    batch_num = 10
    emb_size = 512
    emb_size2 = 256
    num_of_topic = 40
    title = 'PEAKS_' + str(num_of_peak) + '_1024'
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

    train_set, total_training_set, test_set, scRNA_adata, scATAC_adata, mask_matrix1, mask_matrix2 = full_batch.process_full_batch_data(
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

    (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
     scRNA_anndata, scATAC_anndata, gene_correlation_matrix,
     peak_correlation_matrix, feature_matrix, cor_mat, edge_index, X_rna_tensor_copy, X_atac_tensor_copy) = train_set
    print(X_rna_tensor_normalized.shape)

    # rna_np = scRNA_anndata.X.toarray()
    # row_norms = np.linalg.norm(rna_np, axis=1, keepdims=True)
    # normalized_matrix1 = rna_np / row_norms
    #
    # atac_np = scATAC_anndata.X.toarray()
    # row_norms = np.linalg.norm(rna_np, axis=1, keepdims=True)
    # normalized_matrix2 = atac_np / row_norms

    # concat = np.vstack(normalized_matrix1, normalized_matrix2)
    # y1 = torch.tensor(normalized_matrix1.transpose())
    # y2 = torch.tensor(normalized_matrix2.transpose())
    rna_feature = scRNA_anndata.X.copy()
    binary_matrix1 = np.where(rna_feature != 0.0, 1.0, rna_feature)
    atac_feature = (scATAC_adata.X.toarray()).copy()
    binary_matrix2 = np.where(atac_feature != 0.0, 1.0, atac_feature)
    binary_matrix2 = binary_matrix2[:num_of_cell, :num_of_peak]
    feature_matrix1 = torch.tensor(np.vstack((binary_matrix1.T, binary_matrix2.T))).to(device)
    labels = torch.vstack((X_rna_tensor_normalized.T, X_atac_tensor_normalized.T)).to(device)

    feature_matrix2 = torch.rand((num_of_peak + num_of_gene, emb_size)).to(device)

    print(cor_mat)
    gnn = GNN(emb_size, emb_size2 * 2, emb_size, 1, device, 0, gnn_conv).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.005, weight_decay=5e-4)

    # mse = nn.MSELoss()
    # for epoch in range(2):
    #     gnn.train()
    #     optimizer.zero_grad()
    #     out = gnn(feature_matrix1,edge_index)
    #     out = torch.mm(out, out.T)
    #     print(torch.sigmoid(out))

    model = model2.GAE(emb_size, emb_size2 * 2, emb_size, 1, device, 0, gnn_conv).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    hidden_emb = None

    # Store original adjacency matrix (without diagonal entries) for later
    cor_orig = cor_mat
    cor_orig = cor_orig - sp.dia_matrix((cor_orig.diagonal()[np.newaxis, :], [0]), shape=cor_orig.shape)
    cor_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(cor_mat)
    cor = adj_train

    # Some preprocessing
    cor_norm = preprocess_graph(cor)
    cor_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    cor_label = torch.FloatTensor(cor_label.toarray()).to(device)

    pos_weight = float(cor.shape[0] * cor.shape[0] - cor.sum()) / cor.sum()
    pos_weight = torch.tensor(pos_weight).to(device)
    norm = cor.shape[0] * cor.shape[0] / float((cor.shape[0] * cor.shape[0] - cor_mat.sum()) * 2)
    edge_index = cor_norm.coalesce()
    edge_index = edge_index.indices().to(device)
    for epoch in range(200):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(feature_matrix2, edge_index)
        loss = loss_function(preds=recovered, labels=cor_label,
                             mu=mu, logvar=logvar, n_nodes=num_of_peak + num_of_gene,
                             norm=norm, pos_weight=pos_weight)

        loss.backward()
        cur_loss = loss.item()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        optimizer.step()

        hidden_emb = mu.data.cpu().numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, cor_orig, val_edges, val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              # "val_ap=", "{:.5f}".format(ap_curr),
              # "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, cor_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))
