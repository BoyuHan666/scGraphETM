import pandas as pd
from scipy.sparse import dok_matrix
import scanpy as sc
import numpy as np
from tqdm import tqdm
import pickle
import scipy.sparse as sp

adata = sc.read_h5ad('/home/vickarry/projects/ctb-liyue/vickarry/data/10x-Multiome-Pbmc10k-RNA.h5ad')
PBMC_Gene = adata.var['gene_ids'].tolist()
print(len(PBMC_Gene))

df = pd.read_csv('TFtoGene.txt', delimiter='\t', header=None, names=['TF', 'Target_Gene'])
links = df['TF'].tolist()
values = df['Target_Gene'].tolist()


tf = []
gene = []
tf_gene = np.zeros((len(PBMC_Gene), len(PBMC_Gene)))
tf_notfound = []
for gene_index in tqdm(range(len(PBMC_Gene)), desc='Processing genes', unit='gene'):
    target = PBMC_Gene[gene_index]
    # Get rows where the target gene appeared in
    indices = df.index[df['Target_Gene'] == target].tolist()
    for j in range(len(indices)):
        # Get index of TF
        tf = links[indices[j]]
        try:
            tf_index = PBMC_Gene.index(links[indices[j]])
            tf_gene[gene_index][tf_index] = 1
        except ValueError:
            if tf not in tf_notfound:
                tf_notfound.append(tf)
                # print("ValueError: TF: ", tf, " is not in PBMC")
            continue

sparse_matrix = sp.csr_matrix(tf_gene)
print(sparse_matrix)
file_path = 'tf_gene.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(sparse_matrix, file)