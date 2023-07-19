import pandas as pd
from scipy.sparse import dok_matrix
import scanpy as sc


adata = sc.read_h5ad('/home/vickarry/projects/ctb-liyue/vickarry/data/10x-Multiome-Pbmc10k-RNA.h5ad')
PBMC_Gene = adata.var['gene_ids'].tolist()

df = pd.read_csv('TFtoGene.txt', delimiter='\t', header=None, names=['TF', 'Target_Gene'])
links = df['TF']
values = df['Target_Gene']

for i in range(len(values)):
    if values[i] not in PBMC_Gene:
        df.drop(i)
print(df)
# Step 2: Create a dictionary to map links to indices
link_dict = {link: i for i, link in enumerate(set(links))}

# Step 3: Initialize a sparse matrix
num_links = len(link_dict)
sparse_matrix = dok_matrix((num_links, num_links), dtype=int)

# Step 4: Update the sparse matrix with indices from the data
for link, value in zip(links, values):
    link_index = link_dict[link]
    sparse_matrix[link_index, link_index] = value

# Step 5: Optionally convert the sparse matrix to a desired format
# For example, to convert it to a Compressed Sparse Column (CSC) matrix:
csc_matrix = sparse_matrix.tocsc()