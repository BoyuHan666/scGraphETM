import torch
from torch.utils.data import DataLoader
import anndata as ad
import scanpy as sc


def custom_collate_fn(batch_list):
    rna_data_segment = [item[0] for item in batch_list]
    rna_normalized_data_segment = [item[1] for item in batch_list]
    atac_data_segment = [item[2] for item in batch_list]
    atac_normalized_data_segment = [item[3] for item in batch_list]
    rna = ad.concat(rna_data_segment, merge='same', uns_merge='same')
    rna_normalized = ad.concat(rna_normalized_data_segment, merge='same', uns_merge='same')
    atac = ad.concat(atac_data_segment, merge='same', uns_merge='same')
    atac_normalized = ad.concat(atac_normalized_data_segment, merge='same', uns_merge='same')
    return rna, rna_normalized, atac, atac_normalized


def to_gpu_tensor(adata, device):
    return torch.from_numpy(adata.X.toarray()).to(device)


def normalize(data):
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    return data


def get_highly_variable_index(data):
    if 'highly_variable' not in data.var_names:
        sc.pp.highly_variable_genes(data)
    index = data.var['highly_variable'].values
    return index


class H5AD_Dataset(torch.utils.data.Dataset):
    def __init__(self, rna_path, atac_path, num_of_cell=0, num_of_gene=0, num_of_peak=0):
        rna = ad.read_h5ad(rna_path)
        rna_copy = rna.copy()
        rna_normalized = normalize(rna_copy)
        if num_of_gene == 0:
            self.num_of_gene = rna.n_obs
        else:
            self.num_of_gene = num_of_gene
        self.gene_highly_variable_index = get_highly_variable_index(rna_normalized)

        self.rna = rna[:, self.gene_highly_variable_index].copy()
        self.rna_normalized = rna_normalized[:num_of_cell, self.gene_highly_variable_index].copy()

        self.rna = self.rna[:num_of_cell, :num_of_gene]
        self.rna_normalized = self.rna_normalized[:num_of_cell, :num_of_gene]

        atac = ad.read_h5ad(atac_path)
        atac_copy = atac.copy()
        atac_normalized = normalize(atac_copy)
        if num_of_peak == 0:
            self.num_of_peak = atac.n_obs
        else:
            self.num_of_peak = num_of_peak
        self.peak_highly_variable_index = get_highly_variable_index(atac_normalized)

        self.atac = atac[:, self.peak_highly_variable_index].copy()
        self.atac_normalized = atac_normalized[:, self.peak_highly_variable_index].copy()

        self.atac = self.atac[:num_of_cell, :num_of_peak]
        self.atac_normalized = self.atac_normalized[:num_of_cell, :num_of_peak]

    def __len__(self):
        return self.rna.shape[0]

    def __getitem__(self, index):
        rna_data_segment = self.rna[index]
        rna_normalized_data_segment = self.rna_normalized[index]
        atac_data_segment = self.atac[index]
        atac_normalized_data_segment = self.atac_normalized[index]
        return rna_data_segment, rna_normalized_data_segment, atac_data_segment, atac_normalized_data_segment
