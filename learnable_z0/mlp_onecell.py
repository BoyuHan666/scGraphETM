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

import helper2

index_path = '../data/relation/2000highly_gene_peak_index_relation.pickle'
gene_index_list, peak_index_list = helper2.get_peak_index(index_path, top=5, threshould=None)

# rna
data_path_rna = '../data/10x-Multiome-Pbmc10k-RNA.h5ad'
adata_rna = read_h5ad(data_path_rna)
adata_rna = adata_rna[:, gene_index_list]
X_rna = adata_rna.X.toarray()
X_rna_tensor = torch.from_numpy(X_rna)
X_rna_tensor = X_rna_tensor.to(torch.float32)
sums_rna = X_rna_tensor.sum(1).unsqueeze(1)
X_rna_tensor_normalized = X_rna_tensor / sums_rna
X_rna = X_rna_tensor_normalized

# atac
data_path_atac = '../data/10x-Multiome-Pbmc10k-ATAC.h5ad'
adata_atac = read_h5ad(data_path_atac)
adata_atac = adata_atac[:, gene_index_list]
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


class MultiModalMLP(nn.Module):
    def __init__(self, input_dim_rna, input_dim_atac, hidden_dim, output_dim, beta):
        super(MultiModalMLP, self).__init__()
        self.mlp_rna = nn.Sequential(
            nn.Linear(input_dim_rna, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 64)
        )

        self.mlp_atac = nn.Sequential(
            nn.Linear(input_dim_atac, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 64)
        )

        self.classifier = nn.Linear(64, output_dim)
        self.beta = beta

    def forward(self, x_rna, x_atac):
        out_rna = self.mlp_rna(x_rna)
        out_atac = self.mlp_atac(x_atac)
        out = out_rna * (1-self.beta) + out_atac * self.beta
        return self.classifier(out)


batch_size = 32
train_dataset = TensorDataset(X_rna_train, X_atac_train, torch.LongTensor(y_rna_train))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model = MultiModalMLP(input_dim_rna=X_rna_train.shape[1],
                      input_dim_atac=X_atac_train.shape[1],
                      hidden_dim=128,
                      output_dim=len(label_encoder.classes_),
                      beta=0.8)

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

    for batch_X_rna, batch_X_atac, batch_y in train_loader:
        optimizer.zero_grad()

        batch_outputs = []
        for i in range(len(batch_X_rna)):
            single_output = model(batch_X_rna[i].unsqueeze(0), batch_X_atac[i].unsqueeze(0))
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

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Training Loss')
plt.title('Epoch vs Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Training Accuracy', color='orange')
plt.title('Epoch vs Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

model.eval()
with torch.no_grad():
    y_pred = model(X_rna_test, X_atac_test)
    _, predicted = y_pred.max(1)
    accuracy = (predicted == torch.LongTensor(y_rna_test)).sum().item() / len(y_rna_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")