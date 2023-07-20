import torch
from scipy import stats
from torch.nn import functional as F
from sklearn.metrics import adjusted_rand_score
from scipy.sparse import csr_matrix
import anndata
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import select_gpu
from scipy.sparse import vstack, hstack


def add_noise(tensor, noise_rate):
    print("==== add noise ====")
    num_cells, num_genes = tensor.shape

    num_to_select = int(num_genes * noise_rate)

    # for i in range(num_cells):
    #     while True:
    #         selected_indices = torch.randint(0, num_genes, (num_to_select,))
    #         total_expression = tensor[i, selected_indices].sum().item()
    #         if total_expression == 0:
    #             continue
    #
    #         total_expression = int(total_expression)
    #         probabilities = tensor[i, selected_indices].float() / total_expression
    #
    #         new_expression = torch.multinomial(probabilities, num_samples=total_expression, replacement=True)
    #         new_expression_counts = torch.bincount(new_expression, minlength=num_to_select)
    #         new_expression_counts = new_expression_counts.to(torch.float32)
    #         tensor[i, selected_indices] = new_expression_counts
    #         break

    for i in range(num_cells):
        selected_indices = torch.randint(0, num_genes, (num_to_select,))
        total_expression = tensor[i, selected_indices].sum().item()
        if total_expression == 0:
            continue

        total_expression = int(total_expression)
        probabilities = tensor[i, selected_indices].float() / total_expression

        new_expression = torch.multinomial(probabilities, num_samples=total_expression, replacement=True)
        new_expression_counts = torch.bincount(new_expression, minlength=num_to_select)
        new_expression_counts = new_expression_counts.to(torch.float32)
        tensor[i, selected_indices] = new_expression_counts

    print("==== finish noise ====")
    return tensor


def get_val_data(start, end, num_of_gene, num_of_peak, scRNA_adata, scATAC_adata, feature_matrix, edge_index, device):
    """
   =====================================================================================
   Generate: X_rna_test_tensor, X_rna_test_tensor_normalized, scRNA_test_anndata
   =====================================================================================
   """
    test_end = end
    test_start = start

    scRNA_adata_test = scRNA_adata[test_start:test_end, :num_of_gene]

    X_rna_test = scRNA_adata.X.toarray()[test_start:test_end, :num_of_gene]
    X_rna_test_tensor = torch.from_numpy(X_rna_test)
    sums_test_rna = X_rna_test_tensor.sum(1).unsqueeze(1)
    X_rna_test_tensor_normalized = X_rna_test_tensor / sums_test_rna

    scRNA_test_anndata = anndata.AnnData(
        X=scRNA_adata.X[test_start:test_end, :num_of_gene].toarray())
    scRNA_test_anndata.obs['Celltype'] = scRNA_adata.obs['cell_type'].values[
                                         test_start:test_end]

    """
    =====================================================================================
    Generate: X_atac_test_tensor, X_atac_test_tensor_normalized, scATAC_test_anndata
    =====================================================================================
    """
    scATAC_adata_test = scATAC_adata[test_start:test_end, :num_of_peak]

    X_atac_test = scATAC_adata.X.toarray()[test_start:test_end, :num_of_peak]
    X_atac_test_tensor = torch.from_numpy(X_atac_test)
    sums_test_atac = X_atac_test_tensor.sum(1).unsqueeze(1)
    X_atac_test_tensor_normalized = X_atac_test_tensor / sums_test_atac

    scATAC_test_anndata = anndata.AnnData(
        X=scATAC_adata.X[test_start:test_end, :num_of_peak].toarray())
    scATAC_test_anndata.obs['Celltype'] = scATAC_adata.obs['cell_type'].values[
                                          test_start:test_end]

    """
    =====================================================================================
    Generate: test_edge_index, test_gene_correlation_matrix, test_peak_correlation_matrix
    =====================================================================================
    """
    test_edge_index = edge_index

    X_rna_test_tensor = X_rna_test_tensor.to(device)
    X_rna_test_tensor_normalized = X_rna_test_tensor_normalized.to(device)
    X_atac_test_tensor = X_atac_test_tensor.to(device)
    X_atac_test_tensor_normalized = X_atac_test_tensor_normalized.to(device)
    feature_matrix = feature_matrix.to(device)
    test_edge_index = test_edge_index.to(device)

    test_set = (X_rna_test_tensor, X_rna_test_tensor_normalized, X_atac_test_tensor,
                X_atac_test_tensor_normalized, scRNA_test_anndata, scATAC_test_anndata,
                feature_matrix, test_edge_index)

    return test_set


def process_mini_batch_data(scRNA_adata, scATAC_adata, device,
                            num_of_cell, num_of_gene, num_of_peak,
                            test_num_of_cell, batch_size, batch_num,
                            emb_size, edge_index,
                            use_mask=False, mask_ratio=0.2,
                            use_noise=True, noise_ratio=0.2):
    print("======  start processing data  ======")
    feature_matrix = torch.randn((num_of_peak + num_of_gene, emb_size))
    training_set = []

    for i in range(batch_num):
        print(f"process batches [{i + 1} / {batch_num}]")
        start = i * batch_size
        end = start + batch_size
        selected_cells = np.array([i for i in range(start, end, 1)])
        # selected_cells = np.random.choice(num_of_cell, size=batch_size, replace=False)
        # print(selected_cells)

        """
        =====================================================================================
        Generate: X_rna_tensor, X_rna_tensor_normalized, scRNA_mini_batch_anndata
        =====================================================================================
        """
        scRNA_adata_mini_batch = scRNA_adata[:num_of_cell, :]
        scRNA_adata_mini_batch = scRNA_adata_mini_batch[selected_cells, :]

        gene_expression = scRNA_adata_mini_batch.X[:num_of_cell, :num_of_gene]
        mask_matrix1 = np.random.choice([0, 1], size=gene_expression.shape, p=[mask_ratio, 1 - mask_ratio])

        X_rna = scRNA_adata_mini_batch.X.toarray()[:num_of_cell, :num_of_gene]
        X_rna_tensor_copy = torch.from_numpy(X_rna)
        X_rna_tensor_copy = X_rna_tensor_copy.to(torch.float32)

        if use_mask:
            X_rna = scRNA_adata_mini_batch.X.toarray()[:num_of_cell, :num_of_gene] * mask_matrix1
            mask_matrix1 = torch.from_numpy(mask_matrix1)
            mask_matrix1 = torch.tensor(mask_matrix1, dtype=torch.float32)
            mask_matrix1 = mask_matrix1.to(device)
            # print(sum(mask_matrix1))
        else:
            X_rna = scRNA_adata_mini_batch.X.toarray()[:num_of_cell, :num_of_gene]

        X_rna_tensor = torch.from_numpy(X_rna)
        # X_rna_tensor = torch.tensor(X_rna_tensor, dtype=torch.float32)
        X_rna_tensor = X_rna_tensor.clone().detach().float()

        if use_noise:
            X_rna_tensor = add_noise(X_rna_tensor, noise_ratio)

        sums_rna = X_rna_tensor.sum(1).unsqueeze(1)
        X_rna_tensor_normalized = X_rna_tensor / sums_rna

        scRNA_mini_batch_anndata = anndata.AnnData(X=scRNA_adata_mini_batch.X[:num_of_cell, :num_of_gene].toarray())
        scRNA_mini_batch_anndata.obs['Celltype'] = scRNA_adata_mini_batch.obs['cell_type'].values[:num_of_cell]

        """
        =====================================================================================
        Generate: X_atac_tensor, X_atac_tensor_normalized, scATAC_mini_batch_anndata
        =====================================================================================
        """
        scATAC_adata_mini_batch = scATAC_adata[:num_of_cell, :]
        scATAC_adata_mini_batch = scATAC_adata_mini_batch[selected_cells, :]

        peak_expression = scATAC_adata_mini_batch.X[:num_of_cell, :num_of_peak]
        mask_matrix2 = np.random.choice([0, 1], size=peak_expression.shape, p=[mask_ratio, 1 - mask_ratio])

        X_atac = scATAC_adata_mini_batch.X.toarray()[:num_of_cell, :num_of_peak]
        X_atac_tensor_copy = torch.from_numpy(X_atac)
        # X_atac_tensor_copy = torch.tensor(X_atac_tensor_copy, dtype=torch.float32)
        X_atac_tensor_copy = X_atac_tensor_copy.clone().detach().float()

        if use_mask:
            X_atac = scATAC_adata_mini_batch.X.toarray()[:num_of_cell, :num_of_peak] * mask_matrix2
            mask_matrix2 = torch.from_numpy(mask_matrix2)
            mask_matrix2 = torch.tensor(mask_matrix2, dtype=torch.float32)
            mask_matrix2 = mask_matrix2.to(device)
            # print(sum(mask_matrix2))
        else:
            X_atac = scATAC_adata_mini_batch.X.toarray()[:num_of_cell, :num_of_peak]

        X_atac_tensor = torch.from_numpy(X_atac)
        X_atac_tensor = X_atac_tensor.to(torch.float32)

        if use_noise:
            X_atac_tensor = add_noise(X_atac_tensor, noise_ratio)

        sums_atac = X_atac_tensor.sum(1).unsqueeze(1)
        X_atac_tensor_normalized = X_atac_tensor / sums_atac

        scATAC_mini_batch_anndata = anndata.AnnData(X=scATAC_adata_mini_batch.X[:num_of_cell, :num_of_peak].toarray())
        scATAC_mini_batch_anndata.obs['Celltype'] = scATAC_adata_mini_batch.obs['cell_type'].values[:num_of_cell]

        """
        =====================================================================================
        Generate: edge_index, gene_correlation_matrix, peak_correlation_matrix
        =====================================================================================
        """
        edge_index = edge_index

        X_rna_tensor = X_rna_tensor.to(device)
        X_rna_tensor_normalized = X_rna_tensor_normalized.to(device)
        X_atac_tensor = X_atac_tensor.to(device)
        X_atac_tensor_normalized = X_atac_tensor_normalized.to(device)
        feature_matrix = feature_matrix.to(device)
        edge_index = edge_index.to(device)
        X_rna_tensor_copy = X_rna_tensor_copy.to(device)
        X_atac_tensor_copy = X_atac_tensor_copy.to(device)

        training_batch = (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
                          scRNA_mini_batch_anndata, scATAC_mini_batch_anndata,
                          feature_matrix, edge_index, mask_matrix1, mask_matrix2,
                          X_rna_tensor_copy, X_atac_tensor_copy)

        training_set.append(training_batch)

    total_training_set = get_val_data(
        start=0,
        end=num_of_cell,
        num_of_gene=num_of_gene,
        num_of_peak=num_of_peak,
        scRNA_adata=scRNA_adata,
        scATAC_adata=scATAC_adata,
        edge_index=edge_index,
        feature_matrix=feature_matrix,
        device=device,
    )

    test_set = get_val_data(
        start=num_of_cell,
        end=num_of_cell + test_num_of_cell,
        num_of_gene=num_of_gene,
        num_of_peak=num_of_peak,
        scRNA_adata=scRNA_adata,
        scATAC_adata=scATAC_adata,
        edge_index=edge_index,
        feature_matrix=feature_matrix,
        device=device,
    )

    return training_set, total_training_set, test_set, scRNA_adata, scATAC_adata
