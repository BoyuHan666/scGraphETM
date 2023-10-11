import scanpy as sc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import anndata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


import helper2

gene_adata = anndata.read('../../../data/10x-Multiome-Pbmc10k-RNA.h5ad')
peak_adata = anndata.read('../../../data/10x-Multiome-Pbmc10k-ATAC.h5ad')

index_path = '../../data/highly_gene_peak_index_relation.pickle'
gene_index_list, peak_index_list = helper2.get_peak_index(index_path, top=5, threshould=None)
gene_adata = gene_adata[:, gene_index_list]
peak_adata = peak_adata[: peak_index_list[0]]
gene_num = gene_adata.shape[1]
peak_num = peak_adata.shape[1]

X_rna = gene_adata.X.toarray()
X_rna_tensor = torch.from_numpy(X_rna)
X_rna_tensor = X_rna_tensor.to(torch.float32)

sums_rna = X_rna_tensor.sum(1).unsqueeze(1)
X_rna_tensor_normalized = X_rna_tensor / sums_rna


X_peak = peak_adata.X.toarray()
X_peak_tensor = torch.from_numpy(X_peak)
X_peak_tensor = X_peak_tensor.to(torch.float32)

sums_peak = X_peak_tensor.sum(1).unsqueeze(1)
X_peak_tensor_normalized = X_peak_tensor / sums_peak

# Batch
data = "RNA"
act = "RELU"

X = X_rna_tensor_normalized
# X = torch.hstack((X_rna_tensor_normalized, X_peak_tensor_normalized))
# X = X_peak_tensor_normalized
y = peak_adata.obs['cell_type'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

batch_size = 16
train_dataset = TensorDataset(X_train, torch.LongTensor(y_train))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        # self.relu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


input_dim = X_train.shape[1]
hidden_dim = 128
output_dim = len(label_encoder.classes_)

model = MLP(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

num_epochs = 1000
path = './mlp_plot/'
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        print(predicted)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    train_accuracy = correct / total
    acc_list.append(train_accuracy)
    avg_loss = epoch_loss / len(train_loader)
    loss_list.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy * 100:.2f}%")

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    _, predicted = y_pred.max(1)
    accuracy = (predicted == torch.LongTensor(y_test)).sum().item() / len(y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_list, label='Training Loss')
plt.title('Epoch vs Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_list, label='Training Accuracy', color='orange')
plt.title('Epoch vs Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(path + f'{data}_{act}_{epoch}.png')
