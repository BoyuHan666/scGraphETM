import gpu_status
import torch
from tqdm import tqdm
import time
import torch.sparse as sp
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch import optim
from torch.nn import functional as F
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader
import multiprocessing
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# 9631 cell × 29095 gene
rna_path = "./data/10x-Multiome-Pbmc10k-RNA.h5ad"
# 9631 cell × 107194 peak
atac_path = "./data/10x-Multiome-Pbmc10k-ATAC.h5ad"

num_cores = multiprocessing.cpu_count()
# num_cores = 20 # max = 40
num_of_gene = 2000
num_of_peak = 2000


# num_of_gene = 0
# num_of_peak = 0


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


class H5AD_Dataset(torch.utils.data.Dataset):
    def __init__(self, rna_path, atac_path, num_of_gene=0, num_of_peak=0):
        self.rna = ad.read_h5ad(rna_path)
        rna = self.rna.copy()
        self.rna_normalized = self.normalize(rna)
        if num_of_gene == 0:
            self.num_of_gene = rna.n_obs
        else:
            self.num_of_gene = num_of_gene
        self.highly_variable_index = self.get_highly_variable_index(self.rna_normalized)

        self.atac = ad.read_h5ad(atac_path)
        atac = self.atac.copy()
        self.atac_normalized = self.normalize(atac)
        if num_of_peak == 0:
            self.num_of_peak = atac.n_obs
        else:
            self.num_of_peak = num_of_peak
        self.highly_variable_index = self.get_highly_variable_index(self.atac_normalized)

    def __len__(self):
        return self.rna.shape[0]

    def __getitem__(self, index):
        rna_data_segment = self.rna[index][:, :self.num_of_gene]
        rna_normalized_data_segment = self.rna_normalized[index][:, :self.num_of_gene]
        atac_data_segment = self.atac[index][:, :self.num_of_peak]
        atac_normalized_data_segment = self.atac_normalized[index][:, :self.num_of_peak]
        return rna_data_segment, rna_normalized_data_segment, atac_data_segment, atac_normalized_data_segment

    @staticmethod
    def normalize(data):
        sc.pp.normalize_total(data, target_sum=1e4)
        sc.pp.log1p(data)
        return data

    @staticmethod
    def get_highly_variable_index(data):
        if 'highly_variable' not in data.var_names:
            sc.pp.highly_variable_genes(data)
        index = data.var['highly_variable'].values
        return index


if __name__ == '__main__':
    # Check GPU availability
    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = gpu_status.get_lowest_usage_gpu_index()
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

    h5ad_dataloader = DataLoader(
        H5AD_Dataset(
            rna_path=rna_path,
            atac_path=atac_path,
            num_of_gene=num_of_gene,
            num_of_peak=num_of_peak
        ),
        batch_size=200,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=num_cores
    )

    start_time = time.time()

    for i, batch in enumerate(tqdm(h5ad_dataloader)):
        scRNA_adata, scRNA_adata_normalized, scATAC_adata, scATAC_adata_normalized = batch

        # TODO








    end_time = time.time()
    elapsed_time = end_time - start_time

    minutes = elapsed_time // 60
    seconds = elapsed_time % 60

    hours = minutes // 60
    minutes = minutes % 60

    print(f"Elapsed time: {int(hours)} hours:{int(minutes)} minutes:{int(seconds)} seconds")
