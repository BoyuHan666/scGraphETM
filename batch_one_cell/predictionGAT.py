import scanpy as sc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from anndata import read_h5ad
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from torch.nn import functional as F
import os
from tqdm import tqdm

import helper2
import model2

index_path = '../data/relation/highly_gene_peak_index_relation.pickle'
gene_index_list, peak_index_list = helper2.get_peak_index(index_path, top=5, threshould=None)
num_of_cell = 9631
# rna
data_path_rna = '../data/10x-Multiome-Pbmc10k-RNA.h5ad'
adata_rna = read_h5ad(data_path_rna)
adata_rna = adata_rna[:num_of_cell, gene_index_list]
X_rna = adata_rna.X.toarray()
X_rna_tensor = torch.from_numpy(X_rna)
X_rna_tensor = X_rna_tensor.to(torch.float32)
sums_rna = X_rna_tensor.sum(1).unsqueeze(1)
X_rna_tensor_normalized = X_rna_tensor / sums_rna
X_rna = X_rna_tensor_normalized

# atac
data_path_atac = '../data/10x-Multiome-Pbmc10k-ATAC.h5ad'
adata_atac = read_h5ad(data_path_atac)
adata_atac = adata_atac[:num_of_cell, peak_index_list]
X_atac = adata_atac.X.toarray()
X_atac_tensor = torch.from_numpy(X_atac)
X_atac_tensor = X_atac_tensor.to(torch.float32)
sums_atac = X_atac_tensor.sum(1).unsqueeze(1)
X_atac_tensor_normalized = X_atac_tensor / sums_atac
y = adata_atac.obs['cell_type'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_rna_train, X_rna_test, y_rna_train, y_rna_test = train_test_split(X_rna_tensor_normalized, y_encoded, test_size=0.2, random_state=42)
X_atac_train, X_atac_test, y_atac_train, y_atac_test = train_test_split(X_atac_tensor_normalized, y_encoded, test_size=0.2, random_state=42)


class MultiModalVAE_GNN_MLP(nn.Module):
    def __init__(self, num_modality_rna, num_modality_atac, emb_size, num_topics, hidden_dim, out_dim, beta, gnn_params):
        super(MultiModalVAE_GNN_MLP, self).__init__()

        self.vae_rna = model2.VAE(num_modality_rna, emb_size, num_topics)
        self.vae_atac = model2.VAE(num_modality_atac, emb_size, num_topics)

        self.gnn = model2.GNN(**gnn_params)

        # combined_input_dim = num_topics + emb_size*(num_modality_rna+num_modality_atac)
        combined_input_dim = num_topics + emb_size
        self.mlp = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 64)
        )
        self.classifier = nn.Linear(64, out_dim)
        self.beta = beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)

    def forward(self, x_rna, x_atac, edge_index, random_matrix):
        mu_rna, logvar_rna, _ = self.vae_rna(x_rna)
        mu_atac, logvar_atac, _ = self.vae_atac(x_atac)

        theta_rna = self.reparameterize(mu_rna, logvar_rna)
        theta_atac = self.reparameterize(mu_atac, logvar_atac)

        feature_matrix = generate_feature_matrix(x_rna, x_atac, emb_size, random_matrix, device)
        gene_peak_embedding = self.gnn(feature_matrix, edge_index)
        # gene_peak_embedding = gene_peak_embedding.flatten(start_dim=0).unsqueeze(0)
        gene_peak_embedding = gene_peak_embedding.max(dim=0)[0].unsqueeze(0)

        theta = theta_rna * (1 - self.beta) + theta_atac * self.beta

        out = torch.cat([theta, gene_peak_embedding], dim=1)
        out = self.mlp(out)
        return self.classifier(out)


def generate_feature_matrix(gene_exp_normalized, peak_exp_normalized, emb_size, random_matrix, device):
    concatenated = torch.cat((peak_exp_normalized.T, gene_exp_normalized.T), dim=0)
    org_feature_matrix = random_matrix.to(device) * concatenated.repeat(1, emb_size).to(device)
    non_zero_mask = concatenated != 0
    concatenated[non_zero_mask] = 1
    binary_feature_matrix = random_matrix.to(device) * concatenated.repeat(1, emb_size).to(device)

    # feature_matrix = random_matrix.to(device) + concatenated.repeat(1, emb_size).to(device)
    feature_matrix = random_matrix.to(device) + org_feature_matrix
    # feature_matrix = random_matrix.to(device) + binary_feature_matrix
    return feature_matrix


batch_size = 16
train_dataset = TensorDataset(X_rna_train, X_atac_train, torch.LongTensor(y_rna_train))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

emb_size = 256
num_topics = 20
hidden_dim = 128
beta = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_matrix = torch.randn((X_rna_train.shape[1] + X_atac_train.shape[1], emb_size)).to(device)

# edge_index
mtx_path = '../data/gene_peak/top5peak_gene.pickle'
result, edge_index = helper2.get_sub_graph_by_index(
    path=mtx_path,
    gene_index_list=gene_index_list,
    peak_index_list=peak_index_list,
    total_peak=X_atac_train.shape[1],
)
edge_index = edge_index.to(device)

gnn_params = {
    "in_channels": emb_size,
    "hidden_channels": emb_size * 2,
    "out_channels": emb_size,
    "num_heads": 1,
    "device": device,
    "dropout": 0,
    "conv_model": "GATv2"
}

model = MultiModalVAE_GNN_MLP(
    num_modality_rna=X_rna_train.shape[1],
    num_modality_atac=X_atac_train.shape[1],
    emb_size=emb_size,
    num_topics=num_topics,
    hidden_dim=hidden_dim,
    out_dim=len(label_encoder.classes_),
    beta=beta,
    gnn_params=gnn_params
)

model.to(device)

checkpoint_path = f"./model_params/best_model_{emb_size}.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.vae_rna.load_state_dict(checkpoint['encoder1'])
    model.vae_atac.load_state_dict(checkpoint['encoder2'])
    model.gnn.load_state_dict(checkpoint['gnn'])
else:
    print(f"Warning: Checkpoint not found at {checkpoint_path}. Skipping parameter loading.")


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

num_epochs = 20
losses = []
accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_X_rna, batch_X_atac, batch_y in tqdm(train_loader):
        batch_X_rna, batch_X_atac, batch_y = batch_X_rna.to(device), batch_X_atac.to(device), batch_y.to(device)
        optimizer.zero_grad()

        batch_outputs = []
        for i in range(len(batch_X_rna)):
            single_output = model(batch_X_rna[i].unsqueeze(0), batch_X_atac[i].unsqueeze(0), edge_index, random_matrix)
            batch_outputs.append(single_output)

        outputs = torch.cat(batch_outputs, dim=0)

        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    train_accuracy = correct / total
    avg_loss = epoch_loss / len(train_loader)

    losses.append(avg_loss)
    accuracies.append(train_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy * 100:.2f}%")

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(losses, label='Training Loss')
# plt.title('Epoch vs Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(accuracies, label='Training Accuracy', color='orange')
# plt.title('Epoch vs Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.tight_layout()
# plt.show()

model.eval()
correct = 0
total = len(X_rna_test)

with torch.no_grad():
    for i in range(total):
        single_X_rna = X_rna_test[i].unsqueeze(0).to(device)
        single_X_atac = X_atac_test[i].unsqueeze(0).to(device)
        single_y = torch.LongTensor([y_rna_test[i]]).to(device)

        y_pred = model(single_X_rna, single_X_atac, edge_index, random_matrix)
        _, predicted = y_pred.max(1)

        correct += (predicted == single_y).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
