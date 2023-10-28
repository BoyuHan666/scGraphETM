import anndata
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import Node2Vec
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import json
import random
from numba.core.errors import NumbaDeprecationWarning
from tqdm import tqdm
import os
import warnings
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
from sklearn.metrics import roc_auc_score, average_precision_score


import select_gpu
import helper2
import model2
import mini_batch


class MLP(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, out_dim)

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
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, num, tf_index_list, gene_index_list=None):
        emb = self.gnn(x, edge_index)
        pX, gX = helper2.split_tensor(emb, num)

        tf_embeddings = gX[tf_index_list]
        if gene_index_list is not None:
            gX = gX[gene_index_list]

        combined_tensors = []
        for tf_emb in tf_embeddings:
            for g in gX:
                combined = torch.cat((tf_emb, g), dim=-1)
                combined_tensors.append(combined)

        combined_matrix = torch.stack(combined_tensors, dim=0)

        out = self.mlp(combined_matrix)
        out = self.softmax(out)
        return out


def mask_train_test(input_list, ratio):
    size_first_list = int(len(input_list) * ratio)
    first_list = random.sample(input_list, size_first_list)
    second_list = [item for item in input_list if item not in first_list]

    return first_list, second_list


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

    result = {}
    # TODO: Here to add cell type specific adata
    unique_cell_types = gene_exp.obs['cell_type'].unique()
    split_adatas = {cell_type: gene_exp[gene_exp.obs['cell_type'] == cell_type] for cell_type in unique_cell_types}
    for i in range(len(split_adatas)):
        specific_cell_type = unique_cell_types[i]
        gene_exp = split_adatas[specific_cell_type]
        print("="*20+f"  {specific_cell_type}  "+"="*20)
        print(gene_exp)

        # rna
        X_rna = gene_exp.X.toarray()
        X_rna_tensor = torch.from_numpy(X_rna)
        X_rna_tensor = X_rna_tensor.to(torch.float32)
        sums_rna = X_rna_tensor.sum(1).unsqueeze(1)
        X_rna_tensor_normalized = X_rna_tensor / sums_rna

        # atac
        X_atac = peak_exp.X.toarray()
        X_atac_tensor = torch.from_numpy(X_atac)
        X_atac_tensor = X_atac_tensor.to(torch.float32)
        sums_atac = X_atac_tensor.sum(1).unsqueeze(1)
        X_atac_tensor_normalized = X_atac_tensor / sums_atac

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
        num_epochs = 200
        freq = 5
        step = 16
        plot_path_rel = "./plot/"
        lr = 0.001
        param_savepath = f"./model_params/best_model_{emb_size}.pth"
        best_ari_path = f"./model_params/best_ari_{emb_size}.pkl"
        # param_savepath = f"./model_params/best_model_node2vec_{emb_size}2.pth"
        # best_ari_path = f"./model_params/best_ari_node2vec_{emb_size}2.pth"
        # param_savepath = None
        # best_ari_path = None

        seed = 11
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # CPU随机种子确定
        torch.cuda.manual_seed(seed)  # GPU随机种子确定
        torch.cuda.manual_seed_all(seed)  # 所有的GPU设置种子

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
        # Node2Vec_model = Node2Vec(edge_index, emb_size, walk_length=5, context_size=5, walks_per_node=10)
        # Node2Vec_model = Node2Vec_model.to(device)
        #
        # Node2Vec_model.train()
        # optimizer = torch.optim.Adam(Node2Vec_model.parameters(), lr=0.01)
        # loader = Node2Vec_model.loader(batch_size=256, shuffle=False, num_workers=0)
        # for epoch in tqdm(range(200)):
        #     for pos_rw, neg_rw in loader:
        #         optimizer.zero_grad()
        #         loss = Node2Vec_model.loss(pos_rw.to(device), neg_rw.to(device))
        #         loss.backward()
        #         optimizer.step()
        #
        # node_embeddings = Node2Vec_model.embedding.weight
        # print(f"node_embeddings shape: {node_embeddings.shape}")

        fm = torch.randn((num_of_peak + num_of_gene, emb_size)).to(device)
        # fm = node_embeddings.detach().clone().to(device)
        print(f"fm shape: {fm.shape}")

        train_fm = fm
        test_fm = fm
        edge_index = edge_index.to(device)

        result_dense = result.toarray()
        result_tensor = torch.from_numpy(result_dense).float().to(device)
        print(f"result.shape: {result_tensor.shape}, result type: {type(result_tensor)}")
        print(result_tensor.sum())


        gnn = model2.GNN(emb_size, emb_size * 2, emb_size, 1, device, 0, gnn_conv).to(device)
        mlp = MLP(emb_size*2, 2).to(device)
        model = CombinedModel(gnn, mlp).to(device)

        param_savepath = f"./model_params/best_model_{emb_size}.pth"
        if os.path.exists(param_savepath):
            checkpoint = torch.load(param_savepath, map_location=torch.device(device))
            model.gnn.load_state_dict(checkpoint['gnn'])
        else:
            print(f"Warning: Checkpoint not found at {param_savepath}. Skipping parameter loading.")

        result = []
        tf_gene_path = '../data/TF/tf_target_in_vocab_update.txt'
        with open(tf_gene_path, "r") as file:
            for line in file:
                gene_ids = line.strip().split()
                result.append(gene_ids)

        gene_id_to_index = {gene_id: index for index, gene_id in enumerate(gene_exp.var['gene_ids'])}

        new_result = []
        for pair in tqdm(result):
            new_pair = [gene_id_to_index.get(gene_id, None) for gene_id in pair]
            if all(index is not None for index in new_pair):
                new_result.append(new_pair)

        tf_indices = [pair[0] for pair in new_result]
        unique_tf_indices = sorted(list(set(tf_indices)))
        train_tf_list, test_tf_list = mask_train_test(input_list=unique_tf_indices, ratio=0.8)

        train_tf_num = len(train_tf_list)
        test_tf_num = len(test_tf_list)

        train_gene_index_list = []
        test_gene_index_list = []
        gene_index_list = []
        for tf_gene in new_result:
            tf_idx = tf_gene[0]
            gene_idx = tf_gene[1]
            if tf_idx in train_tf_list and gene_idx not in train_gene_index_list:
                train_gene_index_list.append(gene_idx)
            if tf_idx in test_tf_list and gene_idx not in test_gene_index_list:
                test_gene_index_list.append(gene_idx)
            if gene_idx not in gene_index_list:
                gene_index_list.append(gene_idx)

        train_gene_index_list = sorted(train_gene_index_list)
        test_gene_index_list = sorted(test_gene_index_list)
        gene_index_list = sorted(gene_index_list)

        train_gene_num = len(train_gene_index_list)
        test_gene_num = len(test_gene_index_list)
        new_gene_num = len(gene_index_list)

        print(train_gene_num)
        print(test_gene_num)
        print(new_gene_num)


        tf_index_to_unique_index = {old_index: new_index for new_index, old_index in enumerate(train_tf_list + test_tf_list)}
        gene_index_to_unique_index = {old_index: new_index for new_index, old_index in enumerate(gene_index_list)}
        for i, pair in enumerate(new_result):
            new_tf_index = tf_index_to_unique_index[pair[0]]
            new_gene_index = gene_index_to_unique_index[pair[1]]
            new_result[i].append(new_tf_index)
            new_result[i].append(new_gene_index)

        train_tf_label = [0 for _ in range(train_tf_num * new_gene_num)]
        test_tf_label = [0 for _ in range(test_tf_num * new_gene_num)]

        for rel in new_result:
            tf_index = rel[2]
            gene_index = rel[3]
            if tf_index >= train_tf_num:
                tf_index -= train_tf_num  # test_tf_index
            org_tf_index = rel[0]  # to mask fm
            org_gene_index = rel[1]
            flag = 'train'
            if org_tf_index in train_tf_list:
                test_fm[peak_num + org_tf_index] = 0  # TODO: this can be comment
                train_tf_label[tf_index * new_gene_num + gene_index] = 1
            else:  # if tf_index is test tf index, mask the tf in feature matrix to all zero
                train_fm[peak_num + org_tf_index] = 0  # TODO: this can be comment
                test_tf_label[tf_index * new_gene_num + gene_index] = 1
                flag = 'test'

            # try:
            #     if org_tf_index in train_tf_list:
            #         test_fm[peak_num + org_tf_index] = 0
            #         train_tf_label[tf_index * gene_num + gene_index] = 1
            #     else:  # if tf_index is test tf index, mask the tf in feature matrix to all zero
            #         train_fm[peak_num + org_tf_index] = 0
            #         test_tf_label[tf_index * gene_num + gene_index] = 1
            #         flag = 'test'
            # except:
            #     if flag == 'train':
            #         print(len(train_tf_label), tf_index * gene_num + gene_index, tf_index, gene_index)
            #     else:
            #         print(len(test_tf_list), tf_index * gene_num + gene_index, tf_index, gene_index)

        train_tf_label_tensor = torch.tensor(train_tf_label, dtype=torch.long).to(device)
        test_tf_label_tensor = torch.tensor(test_tf_label, dtype=torch.long).to(device)
        print(f"train_tf_label_tensor shape: {train_tf_label_tensor.shape}")
        print(f"train_tf_label sum: {sum(train_tf_label)}")

        print(f"test_tf_label_tensor shape: {test_tf_label_tensor.shape}")
        print(f"test_tf_label sum: {sum(test_tf_label)}")

        # train weighted label
        class_counts = [sum(train_tf_label_tensor == 0), sum(train_tf_label_tensor == 1)]
        total_samples = sum(class_counts)
        class_weights = [total_samples / count for count in class_counts]

        train_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # test weighted label
        class_counts = [sum(test_tf_label_tensor == 0), sum(test_tf_label_tensor == 1)]
        total_samples = sum(class_counts)
        class_weights = [total_samples / count for count in class_counts]

        test_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss()
        criterion_train = nn.CrossEntropyLoss(weight=train_weights)
        criterion_test = nn.CrossEntropyLoss(weight=test_weights)
        # criterion = nn.BCEWithLogitsLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        loss_list = []
        acc_list = []
        test_loss_list = []
        test_acc_list = []

        train_auroc_list = []
        train_auprc_list = []
        test_auroc_list = []
        test_auprc_list = []

        start = 0
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            output_list = []
            train_pred_probabilities_list = []

            for cell in tqdm(range(step)):
                one_cell_RNA_tensor_normalized = X_rna_tensor_normalized[cell].unsqueeze(0)
                one_cell_ATAC_tensor_normalized = X_atac_tensor_normalized[cell].unsqueeze(0)
                train_feature_matrix = helper2.generate_feature_matrix(
                    gene_exp_normalized=one_cell_RNA_tensor_normalized,
                    peak_exp_normalized=one_cell_ATAC_tensor_normalized,
                    emb_size=emb_size,
                    lookup=train_fm,
                    random_matrix=train_fm,
                    device=device
                )


                output = model(train_feature_matrix, edge_index, num_of_peak, train_tf_list)
                train_pred_probabilities = output[:, 1].cpu().detach().numpy()
                pred = torch.argmax(output, dim=1)

                output_list.append(output)
                train_pred_probabilities_list.append(train_pred_probabilities)

                # loss = criterion_train(output, train_tf_label_tensor)
                # loss.backward()
                # optimizer.step()

                # if start + step <= num_of_cell:
                #     start += step
                # else:
                #     start = 0

            stacked_output = torch.stack(output_list, dim=0)
            average_output = torch.mean(stacked_output, dim=0)

            stacked_train_pred_probabilities = np.stack(train_pred_probabilities_list, axis=0)
            average_train_pred_probabilities = np.mean(stacked_train_pred_probabilities, axis=0)

            loss = criterion_train(average_output, train_tf_label_tensor)
            loss.backward()
            optimizer.step()

            total_loss = loss.item()

            _, predicted = torch.max(average_output, 1)
            correct = (predicted == train_tf_label_tensor).sum().item()
            total_correct = correct
            total_samples = train_tf_label_tensor.size(0)

            train_auroc = roc_auc_score(train_tf_label_tensor.cpu().numpy(), average_train_pred_probabilities)
            train_auprc = average_precision_score(train_tf_label_tensor.cpu().numpy(), average_train_pred_probabilities)

            accuracy = total_correct / total_samples
            if epoch == 0:
                accuracy = 0.00
            if (epoch + 1) % freq == 0:
                model.eval()
                with torch.no_grad():
                    test_output_list = []
                    test_pred_probabilities_list = []
                    for cell in tqdm(range(step)):
                        one_cell_RNA_tensor_normalized = X_rna_tensor_normalized[cell].unsqueeze(0)
                        one_cell_ATAC_tensor_normalized = X_atac_tensor_normalized[cell].unsqueeze(0)

                        test_feature_matrix = helper2.generate_feature_matrix(
                            gene_exp_normalized=one_cell_RNA_tensor_normalized,
                            peak_exp_normalized=one_cell_ATAC_tensor_normalized,
                            emb_size=emb_size,
                            lookup=test_fm,
                            random_matrix=test_fm,
                            device=device
                        )

                        test_output = model(test_fm, edge_index, num_of_peak, test_tf_list)
                        test_pred_probabilities = test_output[:, 1].cpu().detach().numpy()
                        test_pred = torch.argmax(test_output, dim=1)

                        test_output_list.append(test_output)
                        test_pred_probabilities_list.append(test_pred_probabilities)

                    stacked_test_output = torch.stack(test_output_list, dim=0)
                    average_test_output = torch.mean(stacked_test_output, dim=0)

                    stacked_test_pred_probabilities = np.stack(test_pred_probabilities_list, axis=0)
                    average_test_pred_probabilities = np.mean(stacked_test_pred_probabilities, axis=0)

                    # if start + step <= num_of_cell:
                    #     start += step
                    # else:
                    #     start = 0

                    test_loss = criterion_test(average_test_output, test_tf_label_tensor)
                    _, test_predicted = torch.max(average_test_output, 1)
                    test_correct = (test_predicted == test_tf_label_tensor).sum().item()
                    test_total_samples = test_tf_label_tensor.size(0)
                    test_accuracy = test_correct / test_total_samples

                    # Compute AUROC and AUPRC
                    # 评估二分类模型的性能
                    test_auroc = roc_auc_score(test_tf_label_tensor.cpu().numpy(), average_test_pred_probabilities)
                    # 常用来评估正样本（通常是少数类）在不平衡数据集中的分类性能
                    test_auprc = average_precision_score(test_tf_label_tensor.cpu().numpy(), average_test_pred_probabilities)

                    # Store metrics
                    test_loss_list.append(test_loss.item())
                    test_acc_list.append(test_accuracy)
                    test_auroc_list.append(test_auroc)
                    test_auprc_list.append(test_auprc)

                print(f'==========  Epoch {epoch + 1}/{num_epochs}  ==========\n'
                      f'Train Loss: {total_loss:.4f}, Train Accuracy: {accuracy:.4f}, train_sum: {pred.sum()}\n'
                      f'Test Loss:  {test_loss:.4f}, Test Accuracy:  {test_accuracy:.4f}, test_sum:  {test_pred.sum()}\n'
                      f'Train AUROC: {train_auroc:.4f}, Train AUPRC: {train_auprc:.4f}\n'
                      f'Test AUROC:  {test_auroc:.4f}, Test AUPRC:  {test_auprc:.4f}')

                loss_list.append(loss.item())
                acc_list.append(accuracy)
                train_auroc_list.append(train_auroc)
                train_auprc_list.append(train_auprc)

        print(loss_list)
        print(acc_list)
        print(train_auroc_list)
        print(train_auprc_list)

        print(test_loss_list)
        print(test_acc_list)
        print(test_auroc_list)
        print(test_auprc_list)

        result[specific_cell_type] = {
            'train_loss': loss_list,
            'train_acc_list': acc_list,
            'train_auroc_list': train_auroc_list,
            'train_auprc_list': train_auprc_list,
            'test_loss_list': test_loss_list,
            'test_acc_list': test_acc_list,
            'test_auroc_list': test_auroc_list,
            'test_auprc_list': test_auprc_list,
        }

    with open("./numerical_result/result.json", "w") as outfile:
        json.dump(result, outfile, indent=4)