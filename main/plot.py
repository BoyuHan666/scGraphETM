import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
==============================================================================
Plotting
==============================================================================
"""


def monitor_perf(perf, ari_perf, ari_freq, objective, path):
    niter = []
    ari_niter = []
    mse = []
    ari = []
    ari1 = []
    ari2 = []
    for i in range(len(perf)):
        niter.append(perf[i][0])
        mse.append(perf[i][1])
    for i in range(len(ari_perf)):
        ari_niter.append(ari_perf[i][0] * ari_freq)
        ari.append(ari_perf[i][1])
        ari1.append(ari_perf[i][2])
        ari2.append(ari_perf[i][3])
    if objective == "both":
        plt.plot(ari_niter, ari, label="GNN_scETM", color='red')
        plt.plot(ari_niter, ari1, label="scETM_RNA", color='black')
        plt.plot(ari_niter, ari2, label="scETM_ATAC", color='blue')
        plt.legend()
        plt.title("ARI comparison on GCN_scETM and scETM")
        plt.xlabel("iter")
        plt.ylabel("ARI")
        plt.ylim(0, 1)
        # plt.show()
        plt.savefig(path + 'BOTH_ARI.png')
    if objective == "NELBO":
        plt.plot(niter, mse)
        plt.xlabel("iter")
        plt.ylabel("NELBO")
        # plt.show()
        plt.savefig(path + 'NELBO.png')


def generate_cluster_plot(theta, sc_adata, plot_path_rel):
    adata = ad.AnnData(np.array(theta.detach().cpu()))
    labels = sc_adata.obs['cell_type']
    adata.obs["Cell_Type"] = pd.Categorical(list(labels))
    sc.tl.tsne(adata, use_rep='X')
    # sc.pl.tsne(adata, color=adata.obs)
    fig1 = sc.pl.tsne(adata, color="Cell_Type", show=False, return_fig=True)
    fig1.savefig(plot_path_rel + 'tsne.png')

    sc.pp.neighbors(adata)
    sc.tl.louvain(adata, resolution=0.5)
    sc.tl.umap(adata, spread=1.0, min_dist=0.4)

    fig2 = sc.pl.umap(adata, color="louvain", title="louvain_0.5", show=False, return_fig=True)
    fig2.savefig(plot_path_rel + 'louvain.png')

    fig3 = sc.pl.umap(adata, color="Cell_Type", title="Cell_Type", show=False, return_fig=True)
    fig3.savefig(plot_path_rel + 'umap.png')
