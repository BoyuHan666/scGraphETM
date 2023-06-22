import numpy as np
import scanpy as sc
import anndata
import random

import torch
from etm import ETM
from torch import optim
from torch.nn import functional as F

import os
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt


def evaluate_ari(cell_embed, adata):
    adata.obsm['cell_embed'] = cell_embed
    sc.pp.neighbors(adata, use_rep="cell_embed", n_neighbors=30)
    sc.tl.louvain(adata, resolution=0.15)
    ari = adjusted_rand_score(adata.obs['Celltype'], adata.obs['louvain'])
    return ari


# train the VAE for one epoch
def train_scETM_helper(model, optimizer, X_tensor, X_tensor_normalized):
    # initialize the model and loss
    model.train()
    optimizer.zero_grad()
    model.zero_grad()

    # forward and backward pass
    nll, kl_theta = model(X_tensor, X_tensor_normalized)
    loss = nll + kl_theta
    loss.backward()  # backprop gradients w.r.t. negative ELBO

    # clip gradients to 2.0 if it gets too large
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

    # update model to minimize negative ELBO
    optimizer.step()

    return torch.sum(loss).item()


def get_theta(model, input_x):
    model.eval()
    with torch.no_grad():
        q_theta = model.q_theta(input_x)
        mu_theta = model.mu_q_theta(q_theta)
        theta = F.softmax(mu_theta, dim=-1)
        return theta


def train_scETM(model, optimizer, X_tensor, X_tensor_normalized, X_test_nor, test_adata, niter=1000):
    perf = np.ndarray(shape=(niter, 3), dtype='float')
    for i in range(niter):
        nelbo = train_scETM_helper(model, optimizer, X_tensor, X_tensor_normalized)
        if i % 10 == 0:
            with torch.no_grad():
                theta = get_theta(model, X_test_nor)
                perf[i, 0] = i
                perf[i, 1] = nelbo
                perf[i, 2] = evaluate_ari(theta, test_adata)
                print('Iter: {} ..  NELBO: {:.4f} .. ARI: {:.4f}'.format(i, perf[i, 1], perf[i, 2]))
    return model, perf


def train_batch_scETM(model, optimizer, X_train_batch, X_train_nor_batch, X_test_nor, test_adata, niter=1000):
    perf = np.ndarray(shape=(niter, 3), dtype='float')
    for i in range(niter):
        for X_tensor, X_tensor_normalized in zip(X_train_batch, X_train_nor_batch):
            nelbo = train_scETM_helper(model, optimizer, X_tensor, X_tensor_normalized)
        if i % 10 == 0:
            with torch.no_grad():
                theta = get_theta(model, X_test_nor)
                perf[i, 0] = i
                perf[i, 1] = nelbo
                perf[i, 2] = evaluate_ari(theta, test_adata)
                print('Iter: {} ..  NELBO: {:.4f} .. ARI: {:.4f}'.format(i, perf[i, 1], perf[i, 2]))
    return model, perf
