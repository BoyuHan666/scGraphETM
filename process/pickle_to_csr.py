import anndata
import torch
import time
import pickle
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp

path = '../data/relation/top3peak_gene_relation.npz'
# threshold = 2000000000
if __name__ == "__main__":

    scRNA_adata = anndata.read_h5ad("../data/10x-Multiome-Pbmc10k-RNA.h5ad")
    scATAC_adata = anndata.read_h5ad("../data/10x-Multiome-Pbmc10k-ATAC.h5ad")

    num_of_gene = scRNA_adata.X.shape[1]
    num_of_peak = scATAC_adata.X.shape[1]

    with open('../data/relation/gene_peak_relation.pickle', 'rb') as fp:
        gpr = pickle.load(fp)

    print(len(gpr))

    csr_cor = sp.csr_matrix((num_of_peak+num_of_gene, num_of_peak+num_of_gene), dtype=int)

    gene_names = np.array(scRNA_adata.var_names)
    peak_names = np.array(scATAC_adata.var_names)

    for i, gene in tqdm(enumerate(gpr.keys())):
        # try:
        #     peak, dist = gpr[gene][0]
        #     j = (peak_names == peak).argmax()
        #     csr_cor[num_of_peak + i, j] = 1
        #     csr_cor[j, num_of_peak + i] = 1
        # except:
        #     continue
        for peak, dist in gpr[gene][:3]:
            # if dist < 2000:
            j = (peak_names == peak).argmax()
            csr_cor[num_of_peak + i, j] = 1
            csr_cor[j, num_of_peak + i] = 1


    sp.save_npz(path, csr_cor)

    import scipy.sparse as sp
    loaded_matrix = sp.load_npz(path)
    print(loaded_matrix.sum())




