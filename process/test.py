import anndata
import torch
import time
import pickle
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse import vstack, hstack


def get_sub_graph(sp_matrix, num_gene, num_peak, total_peak):
    peak_peak = sp_matrix[:num_peak, :num_peak]
    peak_gene_down = sp_matrix[total_peak:(total_peak + num_gene), :num_peak]
    peak_gene_up = sp_matrix[:num_peak, total_peak:(total_peak + num_gene)]
    gene_gene = sp_matrix[total_peak:total_peak + num_gene, total_peak:total_peak + num_gene]

    top = hstack([peak_peak, peak_gene_up])
    bottom = hstack([peak_gene_down, gene_gene])

    result = vstack([top, bottom])
    gene_gene = torch.from_numpy(gene_gene.toarray())
    peak_peak = torch.from_numpy(peak_peak.toarray())

    return result, gene_gene, peak_peak


if __name__ == "__main__":
    path = '../data/TF_gene/top1_peak_tf_gene.pickle'
    with open(path, 'rb') as fp:
        gpr = pickle.load(fp)

    peak_exp = anndata.read('../data/10x-Multiome-Pbmc10k-ATAC.h5ad')
    total_peak_num = peak_exp.shape[1]
    print(total_peak_num)

    gene_exp = anndata.read('../data/10x-Multiome-Pbmc10k-RNA.h5ad')
    total_gene_num = gene_exp.shape[1]
    print(total_gene_num)

    sub_graph, gene_gene, peak_peak = get_sub_graph(gpr, total_gene_num, 100000, total_peak_num)
    print(sub_graph.sum())
    print(gene_gene.sum())
    print(peak_peak.sum())
