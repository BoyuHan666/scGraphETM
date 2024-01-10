import torch
from torch_geometric.nn import Node2Vec
import anndata as ad
import helper2


index_path = '../data/relation/highly_gene_peak_index_relation.pickle'
gene_index_list, peak_index_list = helper2.get_peak_index(index_path, top=5, threshould=None)
num_of_cell = 9631

# rna
data_path_rna = '../data/10x-Multiome-Pbmc10k-RNA.h5ad'
adata_rna = ad.read_h5ad(data_path_rna)
total_gene = adata_rna.X.shape[1]
adata_rna = adata_rna[:num_of_cell, gene_index_list]
X_rna = adata_rna.X.toarray()
X_rna_tensor = torch.from_numpy(X_rna)
X_rna_tensor = X_rna_tensor.to(torch.float32)
sums_rna = X_rna_tensor.sum(1).unsqueeze(1)
X_rna_tensor_normalized = X_rna_tensor / sums_rna
X_rna = X_rna_tensor_normalized

# atac
data_path_atac = '../data/10x-Multiome-Pbmc10k-ATAC.h5ad'
adata_atac = ad.read_h5ad(data_path_atac)
total_peak = adata_atac.X.shape[1]
adata_atac = adata_atac[:num_of_cell, peak_index_list]
X_atac = adata_atac.X.toarray()
X_atac_tensor = torch.from_numpy(X_atac)
X_atac_tensor = X_atac_tensor.to(torch.float32)
sums_atac = X_atac_tensor.sum(1).unsqueeze(1)
X_atac_tensor_normalized = X_atac_tensor / sums_atac
y = adata_atac.obs['cell_type'].values

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
mtx_path = '../data/gene_peak/top5peak_gene.pickle'
result, edge_index = helper2.get_sub_graph_by_index(
    path=mtx_path,
    gene_index_list=gene_index_list,
    peak_index_list=peak_index_list,
    total_peak=total_peak
)
edge_index = edge_index.to(device)

# edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
num_nodes = edge_index.max().item() + 1
embedding_dim = 128
model = Node2Vec(edge_index, embedding_dim, walk_length=20, context_size=10, walks_per_node=10)
model = model.to(device)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loader = model.loader(batch_size=128, shuffle=True, num_workers=0)  # Set num_workers to 0
for epoch in range(200):
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

node_embeddings = model.embedding.weight.cpu().detach().numpy()
print(node_embeddings.shape)

