import anndata
import torch
import time
import pickle
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp


def get_peak_index(path, top=5, threshould=None):
    with open(path, 'rb') as fp:
        gene_peak = pickle.load(fp)

    gene_index_list = []
    peak_index_list = []

    if threshould is None:
        for i, gene in tqdm(enumerate(gene_peak.keys())):
            gene_index_list.append(gene)
            for j, dist in gene_peak[gene][:top]:
                peak_index_list.append(j)
    else:
        for i, gene in tqdm(enumerate(gene_peak.keys())):
            gene_index_list.append(gene)
            for j, dist in gene_peak[gene][:top]:
                if dist < threshould:
                    peak_index_list.append(j)

    gene_index_list = list(set(gene_index_list))
    peak_index_list = list(set(peak_index_list))

    return gene_index_list, peak_index_list

if __name__ == "__main__":

    # scRNA_adata = anndata.read_h5ad("../data/10x-Multiome-Pbmc10k-RNA.h5ad")
    # scATAC_adata = anndata.read_h5ad("../data/10x-Multiome-Pbmc10k-ATAC.h5ad")
    #
    # num_of_gene = scRNA_adata.X.shape[1]
    # num_of_peak = scATAC_adata.X.shape[1]
    #
    # import pickle
    # with open(path, 'rb') as fp:
    #     gpr = pickle.load(fp)
    #
    # print(len(gpr))
    #
    # csr_cor = sp.csr_matrix((num_of_peak+num_of_gene, num_of_peak+num_of_gene), dtype=int)
    #
    # gene_names = np.array(scRNA_adata.var_names)
    # peak_names = np.array(scATAC_adata.var_names)
    #
    # for i, gene in tqdm(enumerate(gpr.keys())):
    #     # try:
    #     #     peak, dist = gpr[gene][0]
    #     #     j = (peak_names == peak).argmax()
    #     #     csr_cor[num_of_peak + i, j] = 1
    #     #     csr_cor[j, num_of_peak + i] = 1
    #     # except:
    #     #     continue
    #     for peak, dist in gpr[gene][:3]:
    #         # if dist < 2000:
    #         j = (peak_names == peak).argmax()
    #         csr_cor[num_of_peak + i, j] = 1
    #         csr_cor[j, num_of_peak + i] = 1
    #
    #
    # sp.save_npz(path, csr_cor)
    #
    # loaded_matrix = sp.load_npz(path)
    # print(loaded_matrix.sum())

    # path = '../data/relation/2000bp_top5peak_gene_relation.npz'
    # loaded_matrix = sp.load_npz(path)
    #
    # path = '../data/TF_gene/2000bp_top5peak_gene.pickle'
    # with open(path, 'wb') as fp:
    #     pickle.dump(loaded_matrix, fp)
    #
    # with open(path, 'rb') as fp:
    #     tf_gene = pickle.load(fp)
    #
    # print(tf_gene.sum())

    path = '../data/relation/gene_peak_index_relation.pickle'
    gene_index_list, peak_index_list = get_peak_index(path, top=5, threshould=2000)

    print(len(gene_index_list))
    print(len(peak_index_list))

    scRNA_adata = anndata.read_h5ad("../data/10x-Multiome-Pbmc10k-RNA.h5ad")
    scRNA_adata = scRNA_adata[:, gene_index_list]

    scATAC_adata = anndata.read_h5ad("../data/10x-Multiome-Pbmc10k-ATAC.h5ad")
    scATAC_adata = scATAC_adata[:, peak_index_list]

    print(scRNA_adata)
    print(scATAC_adata)










