import torch
from torch.utils.data import DataLoader
import anndata as ad
import scanpy as sc
import numpy as np
import random
import correlation

def custom_collate_fn(batch_list):
    # define a custom behavior for how individual samples in your dataset are collated into a batch.
    # A list of 1 cell
    rna_data_segment = [item[0] for item in batch_list]
    rna_normalized_data_segment = [item[1] for item in batch_list]
    atac_data_segment = [item[2] for item in batch_list]
    atac_normalized_data_segment = [item[3] for item in batch_list]

    # mini batch genes and peaks
    num_genes_subsampled = 2000
    num_peaks_subsampled = 2000

    # print("RNA subsample: ", rna_data_segment)
    gene_indices = random.sample(range(rna_data_segment[0].X.shape[1]), num_genes_subsampled)
    rna_subsample =[item[:, gene_indices].copy() for item in  rna_data_segment]
    rna_subsample_norm = [item[:, gene_indices].copy() for item in rna_normalized_data_segment]

    peak_indices = random.sample(range(atac_data_segment[0].X.shape[1]), num_peaks_subsampled)
    atac_subsample = [item[:, peak_indices].copy() for item in atac_data_segment]
    atac_subsample_norm = [item[:, peak_indices].copy() for item in atac_normalized_data_segment]

    # an item is a cell thus concat to make one batch into a single anndata with numerous cells
    rna = ad.concat(rna_subsample, merge='same', uns_merge='same')
    rna_normalized = ad.concat(rna_subsample_norm, merge='same', uns_merge='same')
    atac = ad.concat(atac_subsample, merge='same', uns_merge='same')
    atac_normalized = ad.concat(atac_subsample_norm, merge='same', uns_merge='same')

    # print("=======  generate gene-gene cor_matrix  ======")
    # gene_correlation, gene_cor_matrix = correlation.get_one_modality_cor(
    #     adata=rna, rate1=0.6, rate2=-0.6, dis_rate=1
    # )
    #
    # print("=======  generate peak-peak cor_matrix  ======")
    # peak_correlation, peak_cor_matrix = correlation.get_one_modality_cor(
    #     adata=atac, rate1=0.6, rate2=-0.6, dis_rate=1
    # )

    print("=======  generate peak-gene cor_matrix  ======")
    cor_matrix = correlation.get_two_modality_cor(
        scRNA_adata=rna, scATAC_adata=atac,
        # gene_cor_mat=gene_cor_matrix, peak_cor_mat=peak_cor_matrix
        )
    feature_matrix = torch.randn((num_genes_subsampled + num_peaks_subsampled, 512))
    return rna, rna_normalized, atac, atac_normalized, cor_matrix,feature_matrix


def to_gpu_tensor(adata, device):
    return torch.from_numpy(adata.X.toarray()).to(device)


def normalize(data):
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    return data


def get_highly_variable_index(data):

    # Calculate coefficient of variation (CV)
    newData = data.copy()
    sc.pp.normalize_total(newData, target_sum=1e4)  # Normalize total counts per cell
    sc.pp.log1p(newData)  # Log-transform the data
    sc.pp.highly_variable_genes(newData)  # Identify highly variable genes

    # Get the highly variable genes
    index = newData.var['highly_variable']
    return index


class H5AD_Dataset(torch.utils.data.Dataset):
    def __init__(self, rna_path, atac_path, num_of_cell=2000, num_of_gene=2000, num_of_peak=2000):
        rna = ad.read_h5ad(rna_path)
        rna_copy = rna.copy()
        rna_normalized = rna_copy
        self.rna = rna
        self.rna_normalized = rna_normalized
        self.num_of_gene = num_of_gene
        self.gene_highly_variable_index = get_highly_variable_index(rna_normalized)

        self.rna = self.rna[:, self.gene_highly_variable_index]
        self.rna_normalized = rna_normalized[:, self.gene_highly_variable_index].copy()

        # self.rna = self.rna[:num_of_cell, :num_of_gene]
        # self.rna_normalized = self.rna_normalized[:num_of_cell, :num_of_gene]

        atac = ad.read_h5ad(atac_path)
        atac_copy = atac.copy()
        atac_normalized = atac_copy
        self.atac = atac
        self.atac_normalized = atac_normalized
        self.num_of_peak = num_of_peak
        self.peak_highly_variable_index = get_highly_variable_index(atac_normalized)
        #
        self.atac = self.atac[:, self.peak_highly_variable_index]
        self.atac_normalized = atac_normalized[:, self.peak_highly_variable_index]
        #
        # self.atac = self.atac[:num_of_cell, :num_of_peak]
        # self.atac_normalized = self.atac_normalized[:num_of_cell, :num_of_peak]

    def __len__(self):
        return self.rna.shape[0]

    def __getitem__(self, index):
        # select cell @ index
        rna_sample = self.rna[index]
        rna_normalized_sample = self.rna_normalized[index]
        atac_sample = self.atac[index]
        atac_normalized_sample = self.atac_normalized[index]
        num_sample_gene = self.num_of_gene
        num_sample_peak = self.num_of_peak

        return rna_sample, rna_normalized_sample, atac_sample, atac_normalized_sample,num_sample_gene,num_sample_peak

    def getBatch(self, batchSize):
        # print(self.rna.X.shape[1])
        gene_indices = random.sample(range(self.rna.X.shape[1]), batchSize)
        rna_subsample = self.rna[:2000, gene_indices].copy()
        rna_subsample_norm = self.rna_normalized[:2000, gene_indices].copy()

        # print(self.atac.X.shape[1])
        peak_indices = random.sample(range(self.atac.X.shape[1]), batchSize)
        atac_subsample = self.atac[:2000, peak_indices].copy()
        atac_subsample_norm = self.atac_normalized[:2000, peak_indices].copy()

        # print("=======  generate gene-gene cor_matrix  ======")
        # gene_correlation, gene_cor_matrix = correlation.get_one_modality_cor(
        #     adata=rna_subsample, rate1=0.6, rate2=-0.6, dis_rate=1
        # )
        #
        # print("=======  generate peak-peak cor_matrix  ======")
        # peak_correlation, peak_cor_matrix = correlation.get_one_modality_cor(
        #     adata= atac_subsample, rate1=0.6, rate2=-0.6, dis_rate=1
        # )

        print("=======  generate peak-gene cor_matrix  ======")
        cor_matrix = correlation.get_two_modality_cor(
            scRNA_adata=rna_subsample, scATAC_adata=atac_subsample)
        feature_matrix = torch.randn((batchSize + batchSize, 512))

        return rna_subsample, rna_subsample_norm, atac_subsample, atac_subsample_norm, cor_matrix, feature_matrix

