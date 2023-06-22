import numpy as np
import scanpy as sc
import anndata
import random
import time

import torch
from etm import ETM
from torch import optim
import helper

num_of_topics = 20
num_of_cell = 4000
test_num_of_cell = 1000
num_of_gene = 2000
batch_size = 400
batch_num = 10

scRNA_adata = anndata.read_h5ad("../data/10x-Multiome-Pbmc10k-RNA.h5ad")
index = scRNA_adata.var['highly_variable'].values
scRNA_adata_highvar = scRNA_adata[:, index]

# get train set with batch
X_train_batch = []
X_train_nor_batch = []
for i in range(batch_num):
    n_cells = scRNA_adata_highvar.shape[0]
    selected_cells = np.random.choice(n_cells, size=batch_size, replace=False)
    scRNA_adata_highvar_mini_batch = scRNA_adata_highvar[selected_cells, :]

    X = scRNA_adata_highvar_mini_batch.X.toarray()[:, :num_of_gene]
    X_tensor = torch.from_numpy(X)
    sums = X_tensor.sum(1).unsqueeze(1)
    X_tensor_normalized = X_tensor / sums

    scRNA_1000_mp_anndata = anndata.AnnData(X=scRNA_adata_highvar_mini_batch.X[:, :num_of_gene].toarray())
    scRNA_1000_mp_anndata.obs['Celltype'] = scRNA_adata_highvar_mini_batch.obs['cell_type']
    num_topics = len(scRNA_1000_mp_anndata.obs['Celltype'].values.unique())
    # print(num_topics)
    X_train_batch.append(X_tensor)
    X_train_nor_batch.append(X_tensor_normalized)

# get test set
X_test = scRNA_adata_highvar.X.toarray()[num_of_cell:num_of_cell+test_num_of_cell, :num_of_gene]
X_test_tensor = torch.from_numpy(X_test)
sums = X_test_tensor.sum(1).unsqueeze(1)
X_test_tensor_normalized = X_test_tensor / sums

scRNA_test_anndata = anndata.AnnData(X=scRNA_adata_highvar.X[num_of_cell:num_of_cell+test_num_of_cell, :num_of_gene].toarray())
scRNA_test_anndata.obs['Celltype'] = scRNA_adata_highvar.obs['cell_type'].values[num_of_cell:num_of_cell+test_num_of_cell]
num_test_topics = len(scRNA_test_anndata.obs['Celltype'].values.unique())
print(num_test_topics)

model = ETM(
    num_topics=num_of_topics,
    vocab_size=num_of_gene,
    t_hidden_size=256,
    rho_size=256,
    theta_act='relu'
)

st = time.time()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1.2e-6)
print(X_train_batch[0].shape)
# model, perf = helper.train_scETM(model, optimizer, X_tensor, X_tensor_normalized, scRNA_1000_mp_anndata, niter=100)
model, perf = helper.train_batch_scETM(model, optimizer, X_train_batch, X_train_nor_batch, X_test_tensor_normalized, scRNA_test_anndata, niter=100)
# model, optimizer, X_train_batch, X_train_nor_batch, X_test_nor, test_adata, niter=1000
ed = time.time()
print(ed-st)