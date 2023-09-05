import torch
import anndata
import numpy as np


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
                            emb_size, edge_index):
    print("======  start processing data  ======")
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

        X_rna = scRNA_adata_mini_batch.X.toarray()[:num_of_cell, :num_of_gene]

        X_rna_tensor = torch.from_numpy(X_rna)
        X_rna_tensor = X_rna_tensor.clone().detach().float()


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


        X_atac = scATAC_adata_mini_batch.X.toarray()[:num_of_cell, :num_of_peak]

        X_atac_tensor = torch.from_numpy(X_atac)
        X_atac_tensor = X_atac_tensor.to(torch.float32)

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
        edge_index = edge_index.to(device)

        training_batch = (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
                          scRNA_mini_batch_anndata, scATAC_mini_batch_anndata, edge_index, emb_size)

        training_set.append(training_batch)

    fm = torch.randn((num_of_peak + num_of_gene, emb_size))

    total_training_set = get_val_data(
        start=0,
        end=num_of_cell,
        num_of_gene=num_of_gene,
        num_of_peak=num_of_peak,
        scRNA_adata=scRNA_adata,
        scATAC_adata=scATAC_adata,
        edge_index=edge_index,
        feature_matrix=fm,
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
        feature_matrix=fm,
        device=device,
    )

    return training_set, total_training_set, test_set, fm
