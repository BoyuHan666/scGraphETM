import torch
from tqdm import tqdm
import anndata


def generate_feature_matrix2(atac, rna, random_matrix):
    concatenated = torch.cat((atac, rna), dim=0)
    expanded = concatenated.repeat(random_matrix.size(1), 1)
    return expanded.T*random_matrix


def process_mini_batch_data(scRNA_adata, scATAC_adata, device,
                            num_of_cell, test_num_of_cell,
                            num_of_gene, num_of_peak,
                            emb_size):
    print("======  start processing data  ======")
    random_matrix = torch.randn((num_of_peak + num_of_gene, emb_size))

    # process scRNA
    scRNA_adata_train = scRNA_adata[:num_of_cell, :]
    X_rna_train = scRNA_adata_train.X.toarray()[:num_of_cell, :num_of_gene]
    X_rna_train_tensor = torch.from_numpy(X_rna_train)
    X_rna_train_tensor = X_rna_train_tensor.clone().detach().float()

    sums_rna = X_rna_train_tensor.sum(1).unsqueeze(1)
    X_rna_train_tensor_normalized = X_rna_train_tensor / sums_rna

    scRNA_train_anndata = anndata.AnnData(X=scRNA_adata_train.X[:num_of_cell, :num_of_gene].toarray())
    scRNA_train_anndata.obs['Celltype'] = scRNA_adata_train.obs['cell_type'].values[:num_of_cell]

    # process scATAC
    scATAC_adata_train = scATAC_adata[:num_of_cell, :]
    X_atac_train = scATAC_adata_train.X.toarray()[:num_of_cell, :num_of_peak]
    X_atac_train_tensor = torch.from_numpy(X_atac_train)
    X_atac_train_tensor = X_atac_train_tensor.clone().detach().float()

    sums_atac = X_atac_train_tensor.sum(1).unsqueeze(1)
    X_atac_train_tensor_normalized = X_atac_train_tensor / sums_atac

    scATAC_train_anndata = anndata.AnnData(X=scATAC_adata_train.X[:num_of_cell, :num_of_peak].toarray())
    scATAC_train_anndata.obs['Celltype'] = scATAC_adata_train.obs['cell_type'].values[:num_of_cell]

    training_set = []
    for i in tqdm(range(num_of_cell)):
        gene_exp = X_rna_train_tensor[i]
        gene_exp_normalized = X_rna_train_tensor_normalized[i]

        peak_exp = X_atac_train_tensor[i]
        peak_exp_normalized = X_atac_train_tensor_normalized[i]

        feature_matrix = generate_feature_matrix2(peak_exp_normalized, gene_exp_normalized, random_matrix)

        gene_exp = gene_exp.to(device)
        gene_exp_normalized = gene_exp_normalized.to(device)
        peak_exp = peak_exp.to(device)
        peak_exp_normalized = peak_exp_normalized.to(device)
        feature_matrix = feature_matrix.to(device)

        training_batch = (gene_exp, gene_exp_normalized, peak_exp, peak_exp_normalized, feature_matrix)
        training_set.append(training_batch)

    return training_set
