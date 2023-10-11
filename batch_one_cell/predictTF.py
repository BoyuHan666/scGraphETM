import anndata
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import Node2Vec
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import random
from numba.core.errors import NumbaDeprecationWarning
from tqdm import tqdm
import os
import warnings


import select_gpu
import helper2
import model2
import mini_batch


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CombinedModel(nn.Module):
    def __init__(self, gnn, mlp):
        super(CombinedModel, self).__init__()
        self.gnn = gnn
        self.mlp = mlp

    def forward(self, x, edge_index, num, tf_index_list, repeat):
        emb = self.gnn(x, edge_index)
        pX, gX = helper2.split_tensor(emb, num)

        tf_embeddings = gX[tf_index_list]
        repeat_times = repeat // len(tf_index_list)
        remainder = repeat % len(tf_index_list)

        augmented_data = tf_embeddings.repeat(repeat_times, 1)
        augmented_data = augmented_data.to(device)
        if remainder > 0:
            extra_data = tf_embeddings[:remainder]
            augmented_data = torch.cat((augmented_data, extra_data), dim=0)

        assert augmented_data.size(0) == repeat, "Augmented data size is not correct"
        gX = torch.cat((gX, augmented_data), dim=0)

        gX = self.mlp(gX)
        return gX


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

    TF_index_path = '../data/TF/TF.pickle'
    with open(TF_index_path, 'rb') as file:
        tf_data = pickle.load(file)

    is_tf = pd.Series(0, index=gene_exp.var.index)
    gene_indices = [item[0] for item in tf_data]
    print(gene_indices)
    for i in range(len(is_tf)):
        if i in gene_indices:
            is_tf[i] = 1

    gene_exp.var['is_TF'] = is_tf
    print(sum(gene_exp.var['is_TF']))

    # index_path = '../data/relation/gene_peak_index_relation.pickle'
    # gene_index_list, peak_index_list = helper2.get_peak_index(index_path, top=5, threshould=None, gene_limit=12000)
    index_path = '../data/relation/highly_gene_peak_index_relation.pickle'
    gene_index_list, peak_index_list = helper2.get_peak_index(index_path, top=5, threshould=None)
    gene_exp = gene_exp[:, gene_index_list]
    gene_num = gene_exp.shape[1]
    peak_exp = peak_exp[:, peak_index_list]
    peak_num = peak_exp.shape[1]

    print(sum(gene_exp.var['is_TF']))


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
    num_of_topic = 40
    gnn_conv = 'GATv2'
    num_epochs = 30
    ari_freq = 2
    plot_path_rel = "./plot/"
    lr = 0.001
    # param_savepath = f"./model_params/best_model_{emb_size}.pth"
    # best_ari_path = f"./model_params/best_ari_{emb_size}.pkl"
    param_savepath = f"./model_params/best_model_node2vec_{emb_size}2.pth"
    best_ari_path = f"./model_params/best_ari_node2vec_{emb_size}2.pth"
    # param_savepath = None
    # best_ari_path = None

    seed = 11

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU随机种子确定
    torch.cuda.manual_seed(seed)  # GPU随机种子确定
    torch.cuda.manual_seed_all(seed)  # 所有的GPU设置种子

    print(f"num_of_topic: {num_of_topic}")
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

    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index()
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

    # use node2vec to get lookup table
    # Node2Vec_model = Node2Vec(edge_index, emb_size, walk_length=10, context_size=10, walks_per_node=10)
    # Node2Vec_model = Node2Vec_model.to(device)
    #
    # Node2Vec_model.train()
    # optimizer = torch.optim.Adam(Node2Vec_model.parameters(), lr=0.01)
    # loader = Node2Vec_model.loader(batch_size=256, shuffle=True, num_workers=0)
    # for epoch in tqdm(range(200)):
    #     for pos_rw, neg_rw in loader:
    #         optimizer.zero_grad()
    #         loss = Node2Vec_model.loss(pos_rw.to(device), neg_rw.to(device))
    #         loss.backward()
    #         optimizer.step()
    #
    # node_embeddings = Node2Vec_model.embedding.weight
    # print(node_embeddings.shape)
    # print(node_embeddings.sum())

    fm = torch.randn((num_of_peak + num_of_gene, emb_size)).to(device)
    edge_index = edge_index.to(device)

    result_dense = result.toarray()
    result_tensor = torch.from_numpy(result_dense).float().to(device)
    print(f"result.shape: {result_tensor.shape}, result type: {type(result_tensor)}")
    print(result_tensor.sum())

    true_tf_label = list(gene_exp.var['is_TF'])
    true_tf_label_tensor = torch.tensor(true_tf_label, dtype=torch.long).to(device)

    aug = []
    for i in range(len(gene_exp.var['is_TF'])):
        if gene_exp.var['is_TF'][i] == 1:
            aug.append(i)

    print(aug)


    gnn = model2.GNN(emb_size, emb_size * 2, emb_size, 1, device, 0, gnn_conv).to(device)
    mlp = MLP(emb_size).to(device)
    model = CombinedModel(gnn, mlp).to(device)

    param_savepath = f"./model_params/best_model_{emb_size}.pth"
    if os.path.exists(param_savepath):
        checkpoint = torch.load(param_savepath, map_location=torch.device(device))
        model.gnn.load_state_dict(checkpoint['gnn'])
    else:
        print(f"Warning: Checkpoint not found at {param_savepath}. Skipping parameter loading.")

    repeat = 100
    augmented_labels = torch.ones(repeat, dtype=torch.long).to(device)
    true_tf_label_tensor = torch.cat((true_tf_label_tensor, augmented_labels))

    class_counts = [sum(true_tf_label_tensor == 0), sum(true_tf_label_tensor == 1)]
    total_samples = sum(class_counts)
    class_weights = [total_samples / count for count in class_counts]

    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    for epoch in range(num_epochs):

        optimizer.zero_grad()
        output = model(fm, edge_index, num_of_peak, aug, repeat)
        softmax_output = F.softmax(output, dim=1)
        pred = torch.argmax(softmax_output, dim=1)
        print(pred.sum())


        loss = criterion(output, true_tf_label_tensor)
        loss.backward()
        optimizer.step()

        total_loss = loss.item()
        _, predicted = torch.max(output, 1)
        correct = (predicted == true_tf_label_tensor).sum().item()
        total_correct = correct
        total_samples = true_tf_label_tensor.size(0)

        accuracy = total_correct / total_samples
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}')








