import torch
from scipy import stats
import scipy.sparse as sp
from torch.nn import functional as F
from sklearn.metrics import adjusted_rand_score
from scipy.sparse import csr_matrix
import anndata
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import select_gpu
import torch.distributions as dist


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


def cal_cor(scRNA_adata, scATAC_adata, num_of_gene, num_of_peak, cor):

    gene_correlation_matrix = torch.zeros(num_of_gene, num_of_gene)
    peak_correlation_matrix = torch.zeros(num_of_peak, num_of_peak)
    cor_mat = torch.zeros(num_of_peak + num_of_gene, num_of_peak + num_of_gene)
    gene_cor_mat = torch.zeros(num_of_gene, num_of_gene)
    peak_cor_mat = torch.zeros(num_of_peak, num_of_peak)

    if "chrom" in scRNA_adata.var and "chromStart" in scRNA_adata.var and "chromEnd" in scRNA_adata.var \
            and "chrom" in scATAC_adata.var and "chromStart" in scATAC_adata.var and "chromEnd" in scATAC_adata.var:
        print("======= match gene-peak relation ======")
        gene_pos_dic = {}
        for i in range(num_of_gene):
            gene_names = scRNA_adata.var_names[i]
            chrom = scRNA_adata.var["chrom"][i]
            chromStart = scRNA_adata.var["chromStart"][i]
            chromEnd = scRNA_adata.var["chromEnd"][i]
            gene_pos_dic[gene_names] = [chrom, chromStart, chromEnd]

        peak_pos_dic = {}
        for i in range(num_of_peak):
            peak_names = scATAC_adata.var_names[i]
            chrom = scATAC_adata.var["chrom"][i]
            chromStart = scATAC_adata.var["chromStart"][i]
            chromEnd = scATAC_adata.var["chromEnd"][i]
            peak_pos_dic[peak_names] = [chrom, chromStart, chromEnd]

        # print(cor_mat.shape)
        for i, gene in enumerate(list(gene_pos_dic.keys())):
            for j, peak in enumerate(list(peak_pos_dic.keys())):
                gene_chrom = gene_pos_dic[gene][0]
                gene_start = gene_pos_dic[gene][1]
                gene_end = gene_pos_dic[gene][2]

                peak_chrom = peak_pos_dic[peak][0]
                peak_start = peak_pos_dic[peak][1]
                peak_end = peak_pos_dic[peak][2]

                if gene_chrom == peak_chrom and abs(gene_start - peak_start) <= 2000:
                    cor_mat[num_of_peak + i, j] = 1
                    cor_mat[j, num_of_peak + i] = 1

        print(cor_mat.sum())

    # print(type(scRNA_adata_mini_batch.X[:, :num_of_gene]))

    # gene_expression = scRNA_adata.X[:, :num_of_gene].toarray()
    # if cor == 'pearson':
    #     correlation_matrix = np.corrcoef(gene_expression + 1e-6, rowvar=False)
    # if cor == 'spearman':
    #     correlation_matrix = stats.spearmanr(gene_expression + 1e-6).correlation
    # gene_correlation_matrix = np.nan_to_num(correlation_matrix, nan=0, posinf=1, neginf=-1)

    print("==== Compute gene-gene correlation ====")
    # for i in range(num_of_gene):
    #     for j in range(num_of_gene):
    #         gen_cor = gene_correlation_matrix[i, j]
    #         if gen_cor > 0.6:
    #         # if gen_cor > 0.6 and i != j:
    #         # if i == j:
    #             cor_mat[num_of_peak + i, num_of_peak + j] = 1
    #             gene_cor_mat[i, j] = 1

    for i in range(num_of_gene):
        cor_mat[num_of_peak + i, num_of_peak + i] = 1
        gene_cor_mat[i, i] = 1

    # peak_expression = scATAC_adata.X[:, :num_of_peak].toarray()
    # if cor == 'pearson':
    #     correlation_matrix2 = np.corrcoef(peak_expression + 1e-6, rowvar=False)
    # if cor == 'spearman':
    #     correlation_matrix2 = stats.spearmanr(peak_expression + 1e-6).correlation
    # peak_correlation_matrix = np.nan_to_num(correlation_matrix2, nan=0, posinf=1, neginf=-1)

    print("==== Compute peak-peak correlation ====")
    # for i in range(num_of_peak):
    #     for j in range(num_of_peak):
    #         peak_cor = peak_correlation_matrix[i, j]
    #         if peak_cor > 0.3:
    #         # if peak_cor > 0.6 and i != j:
    #         # if i == j:
    #             cor_mat[i, j] = 1
    #             peak_cor_mat[i, j] = 1
    for i in range(num_of_peak):
        cor_mat[i, i] = 1
        peak_cor_mat[i, i] = 1

    print("finish cor_mat")

    sparse_cor_mat = csr_matrix(cor_mat.cpu())
    rows, cols = sparse_cor_mat.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    gene_correlation_matrix = gene_correlation_matrix.to(torch.float32)
    peak_correlation_matrix = peak_correlation_matrix.to(torch.float32)

    return gene_correlation_matrix, peak_correlation_matrix, edge_index


def get_val_data(start, end, num_of_gene, num_of_peak, scRNA_adata, scATAC_adata, cor, feature_matrix, device):
    test_end = end
    test_start = start

    """
    =====================================================================================
    Generate: X_rna_test_tensor, X_rna_test_tensor_normalized, scRNA_test_anndata
    =====================================================================================
    """

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
    test_gene_correlation_matrix, test_peak_correlation_matrix, test_edge_index = cal_cor(scRNA_adata_test,
                                                                                          scATAC_adata_test,
                                                                                          num_of_gene, num_of_peak, cor)
    X_rna_test_tensor = X_rna_test_tensor.to(device)
    X_rna_test_tensor_normalized = X_rna_test_tensor_normalized.to(device)
    X_atac_test_tensor = X_atac_test_tensor.to(device)
    X_atac_test_tensor_normalized = X_atac_test_tensor_normalized.to(device)
    test_gene_correlation_matrix = test_gene_correlation_matrix.to(device)
    test_peak_correlation_matrix = test_peak_correlation_matrix.to(device)
    feature_matrix = feature_matrix.to(device)
    test_edge_index = test_edge_index.to(device)

    test_set = (X_rna_test_tensor, X_rna_test_tensor_normalized, X_atac_test_tensor,
                X_atac_test_tensor_normalized, scRNA_test_anndata, scATAC_test_anndata,
                test_gene_correlation_matrix, test_peak_correlation_matrix,
                feature_matrix, test_edge_index)

    return test_set


def process_full_batch_data(rna_path, atac_path, device,
                            num_of_cell, num_of_gene, num_of_peak,
                            test_num_of_cell, emb_size,
                            use_highly_variable, cor,
                            use_mask=False, mask_ratio=0.2,
                            use_noise=False, noise_ratio=0.2):
    print("======  start processing data  ======")
    feature_matrix = torch.randn((num_of_peak + num_of_gene, emb_size))

    """
    =====================================================================================
    Generate: scRNA_adata, scATAC_adata
    =====================================================================================
    """

    scRNA_adata = anndata.read_h5ad(rna_path)
    scATAC_adata = anndata.read_h5ad(atac_path)

    # sc.pp.filter_genes(scRNA_adata, min_cells=3)
    # sc.pp.filter_genes(scRNA_adata, max_counts=100)
    # sc.pp.filter_genes(scRNA_adata, max_cells=int(scRNA_adata.shape[0]*0.6))

    # sc.pp.filter_genes(scATAC_adata, min_cells=3)
    # sc.pp.filter_genes(scATAC_adata, max_counts=100)
    # sc.pp.filter_genes(scATAC_adata, max_cells=int(scRNA_adata.shape[0]*0.6))

    print(scRNA_adata)
    print(scATAC_adata)

    if use_highly_variable:
        print("======  using highly_variable  ======")

        scRNA_adata_copy = scRNA_adata.copy()
        sc.pp.normalize_total(scRNA_adata_copy, target_sum=1e4)
        sc.pp.log1p(scRNA_adata_copy)
        sc.pp.highly_variable_genes(scRNA_adata_copy, n_top_genes=num_of_gene)
        scRNA_adata.var['highly_variable'] = scRNA_adata_copy.var['highly_variable']

        index1 = scRNA_adata.var['highly_variable'].values
        scRNA_adata = scRNA_adata[:, index1]

        scATAC_adata_copy = scATAC_adata.copy()
        sc.pp.normalize_total(scATAC_adata_copy, target_sum=1e4)
        sc.pp.log1p(scATAC_adata_copy)
        sc.pp.highly_variable_genes(scATAC_adata_copy, n_top_genes=num_of_peak)
        scATAC_adata.var['highly_variable'] = scATAC_adata_copy.var['highly_variable']

        index2 = scATAC_adata.var['highly_variable'].values
        scATAC_adata = scATAC_adata[:, index2]

    # random mask
    gene_expression = scRNA_adata.X[:num_of_cell, :num_of_gene]
    mask_matrix1 = np.random.choice([0, 1], size=gene_expression.shape, p=[mask_ratio, 1 - mask_ratio])

    peak_expression = scATAC_adata.X[:num_of_cell, :num_of_peak]
    mask_matrix2 = np.random.choice([0, 1], size=peak_expression.shape, p=[mask_ratio, 1 - mask_ratio])

    """
    =====================================================================================
    Generate: X_rna_tensor, X_rna_tensor_normalized, scRNA_mini_batch_anndata
    =====================================================================================
    """
    print("====== scRNA_adata_full_batch  ======")
    scRNA_adata_full_batch = scRNA_adata[:num_of_cell, :]

    X_rna = scRNA_adata_full_batch.X.toarray()[:num_of_cell, :num_of_gene]
    X_rna_tensor_copy = torch.from_numpy(X_rna)
    # X_rna_tensor_copy = torch.tensor(X_rna_tensor_copy, dtype=torch.float32)
    X_rna_tensor_copy = X_rna_tensor_copy.clone().detach().float()

    if use_mask:
        X_rna = scRNA_adata_full_batch.X.toarray()[:num_of_cell, :num_of_gene] * mask_matrix1
        mask_matrix1 = torch.from_numpy(mask_matrix1)
        # mask_matrix1 = torch.tensor(mask_matrix1, dtype=torch.float32)
        mask_matrix1 = mask_matrix1.clone().detach().float()
        mask_matrix1 = mask_matrix1.to(device)
        # print(sum(mask_matrix1))
    else:
        X_rna = scRNA_adata_full_batch.X.toarray()[:num_of_cell, :num_of_gene]


    X_rna_tensor = torch.from_numpy(X_rna)
    # X_rna_tensor = torch.tensor(X_rna_tensor, dtype=torch.float32)
    X_rna_tensor = X_rna_tensor.clone().detach().float()

    if use_noise:
        X_rna_tensor = add_noise(X_rna_tensor, noise_ratio)

    sums_rna = X_rna_tensor.sum(1).unsqueeze(1)
    X_rna_tensor_normalized = X_rna_tensor / sums_rna

    scRNA_mini_batch_anndata = anndata.AnnData(X=scRNA_adata_full_batch.X[:num_of_cell, :num_of_gene].toarray())
    scRNA_mini_batch_anndata.obs['Celltype'] = scRNA_adata_full_batch.obs['cell_type'].values[:num_of_cell]
    print(len(list(set(scRNA_mini_batch_anndata.obs['Celltype']))))

    """
    =====================================================================================
    Generate: X_atac_tensor, X_atac_tensor_normalized, scATAC_mini_batch_anndata
    =====================================================================================
    """
    print("====== scATAC_adata_full_batch  ======")
    scATAC_adata_full_batch = scATAC_adata[:num_of_cell, :]

    X_atac = scATAC_adata_full_batch.X.toarray()[:num_of_cell, :num_of_peak]
    X_atac_tensor_copy = torch.from_numpy(X_atac)
    # X_atac_tensor_copy = torch.tensor(X_atac_tensor_copy, dtype=torch.float32)
    X_atac_tensor_copy = X_atac_tensor_copy.clone().detach().float()

    if use_mask:
        X_atac = scATAC_adata_full_batch.X.toarray()[:num_of_cell, :num_of_peak] * mask_matrix2
        mask_matrix2 = torch.from_numpy(mask_matrix2)
        # mask_matrix2 = torch.tensor(mask_matrix2, dtype=torch.float32)
        mask_matrix2 = mask_matrix2.clone().detach().float()
        mask_matrix2 = mask_matrix2.to(device)
        # print(sum(mask_matrix2))
    else:
        X_atac = scATAC_adata_full_batch.X.toarray()[:num_of_cell, :num_of_peak]

    X_atac_tensor = torch.from_numpy(X_atac)
    # X_atac_tensor = torch.tensor(X_atac_tensor, dtype=torch.float32)
    X_atac_tensor = X_atac_tensor.clone().detach().float()


    # if use_noise:
    #     X_atac_tensor = add_noise(X_atac_tensor, noise_ratio)

    sums_atac = X_atac_tensor.sum(1).unsqueeze(1)
    X_atac_tensor_normalized = X_atac_tensor / sums_atac

    scATAC_mini_batch_anndata = anndata.AnnData(X=scATAC_adata_full_batch.X[:num_of_cell, :num_of_peak].toarray())
    scATAC_mini_batch_anndata.obs['Celltype'] = scATAC_adata_full_batch.obs['cell_type'].values[:num_of_cell]
    print(len(list(set(scATAC_mini_batch_anndata.obs['Celltype']))))

    """
    =====================================================================================
    Generate: edge_index, gene_correlation_matrix, peak_correlation_matrix
    =====================================================================================
    """
    print("====== get cor matrix  ======")
    gene_correlation_matrix, peak_correlation_matrix, edge_index = cal_cor(scRNA_adata_full_batch,
                                                                           scATAC_adata_full_batch,
                                                                           num_of_gene,
                                                                           num_of_peak, cor)

    X_rna_tensor = X_rna_tensor.to(device)
    X_rna_tensor_normalized = X_rna_tensor_normalized.to(device)
    X_atac_tensor = X_atac_tensor.to(device)
    X_atac_tensor_normalized = X_atac_tensor_normalized.to(device)
    gene_correlation_matrix = gene_correlation_matrix.to(device)
    peak_correlation_matrix = peak_correlation_matrix.to(device)
    feature_matrix = feature_matrix.to(device)
    edge_index = edge_index.to(device)
    X_rna_tensor_copy = X_rna_tensor_copy.to(device)
    X_atac_tensor_copy = X_atac_tensor_copy.to(device)

    training_set = (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
                    scRNA_mini_batch_anndata, scATAC_mini_batch_anndata, gene_correlation_matrix,
                    peak_correlation_matrix, feature_matrix, edge_index, X_rna_tensor_copy, X_atac_tensor_copy)

    total_training_set = (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
                    scRNA_mini_batch_anndata, scATAC_mini_batch_anndata, gene_correlation_matrix,
                    peak_correlation_matrix, feature_matrix, edge_index)

    print("======  generate test_set  ======")
    test_set = get_val_data(
        start=num_of_cell,
        end=num_of_cell + test_num_of_cell,
        num_of_gene=num_of_gene,
        num_of_peak=num_of_peak,
        scRNA_adata=scRNA_adata,
        scATAC_adata=scATAC_adata,
        cor=cor,
        feature_matrix=feature_matrix,
        device=device,
    )

    # print(scRNA_mini_batch_anndata.obs['Celltype'])
    # print(scATAC_mini_batch_anndata.obs['Celltype'])

    return training_set, total_training_set, test_set, scRNA_adata, scATAC_adata, mask_matrix1, mask_matrix2


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index()
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

    rna_path = "../data/10x-Multiome-Pbmc10k-RNA.h5ad"
    atac_path = "../data/10x-Multiome-Pbmc10k-ATAC.h5ad"

    training_set, total_train_set, test_set, scRNA_adata, scATAC_adata, mask_matrix1, mask_matrix2 = process_full_batch_data(
        rna_path=rna_path,
        atac_path=atac_path,
        device=device,
        num_of_cell=2000,
        num_of_gene=200,
        num_of_peak=200,
        test_num_of_cell=1000,
        emb_size=512,
        use_highly_variable=True,
        cor='pearson',
        use_mask=True,
        mask_ratio=0.2,
    )

    X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized, \
    scRNA_mini_batch_anndata, scATAC_mini_batch_anndata, gene_correlation_matrix, \
    peak_correlation_matrix, feature_matrix, edge_index, X_rna_tensor_copy, X_atac_tensor_copy = training_set

    (X_rna_test_tensor, X_rna_test_tensor_normalized, X_atac_test_tensor,
     X_atac_test_tensor_normalized, scRNA_test_anndata, scATAC_test_anndata,
     test_gene_correlation_matrix, test_peak_correlation_matrix,
     test_feature_matrix, test_edge_index) = test_set

    print(gene_correlation_matrix)
    print(peak_correlation_matrix)
