import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from seaborn import heatmap, lineplot, clustermap


def generate_clustermap(theta, sc_adata, plot_path_rel):
    adata = ad.AnnData(np.array(theta.detach().cpu()))
    labels = sc_adata.obs['Celltype']
    adata.obs["Celltype"] = pd.Categorical(list(labels))
    sc.tl.tsne(adata, use_rep='X')
    # sc.pl.tsne(adata, color=adata.obs)
    fig1 = sc.pl.tsne(adata, color="Celltype", show=False, return_fig=True)
    fig1.savefig(plot_path_rel + 'tsne.png')

    sc.pp.neighbors(adata)
    sc.tl.louvain(adata, resolution=0.5)
    sc.tl.umap(adata, spread=1.0, min_dist=0.4)

    fig2 = sc.pl.umap(adata, color="louvain", title="louvain_0.5", show=False, return_fig=True)
    fig2.savefig(plot_path_rel + 'louvain.png')

    fig3 = sc.pl.umap(adata, color="Celltype", title="Celltype", show=False, return_fig=True)
    fig3.savefig(plot_path_rel + 'umap.png')


def generate_gene_heatmap(num_of_topics, sc_adata, beta, num_of_modality, path):
    K = 5
    top_features = np.zeros((K * num_of_topics, num_of_topics))
    select = []
    genes = list(sc_adata.var["gene_name"][:num_of_modality])

    beta_T = beta.detach().cpu().numpy()
    # print(beta_T.shape)
    top5 = []
    for i in range(num_of_topics):
        top5.append(np.flip(np.concatenate((np.array(beta_T)[i, :].argsort()[-5:], select)), axis=0))

    # print(np.array(top5).shape)
    geneNames = []
    count = 0
    for i in range(num_of_topics):
        for j in range(K):
            top_features[count][i] = np.array(beta_T)[i][int(top5[i][j])]
            geneNames.append(genes[int(top5[i][j])])
            count += 1

    plt.figure(figsize=(8, 16))
    hmp = heatmap(top_features, cmap='RdBu_r', vmax=0.03, center=0,
                  xticklabels=[item for item in range(0, num_of_topics)], yticklabels=geneNames)
    plt.savefig(path + 'gene_heatmap.png')


def generate_cell_heatmap(sc_adata, theta, num_of_cell, path):
    cell_type = sc_adata.obs["cell_type"][:num_of_cell]
    lut = dict(zip(cell_type.unique(), ['red',
                                        '#00FF00',
                                        '#0000FF',
                                        '#FFFF00',
                                        '#FF00FF',
                                        '#00FFFF',
                                        '#FFA500',
                                        '#800080',
                                        '#FFC0CB',
                                        '#FF69B4',
                                        '#00FF7F',
                                        '#FFD700',
                                        '#1E90FF',
                                        '#2F4F4F',
                                        '#808000',
                                        '#FF8C00',
                                        '#8B0000',
                                        '#4B0082',
                                        '#2E8B57',
                                        '#FF1493',
                                        '#6B8E23',
                                        '#48D1CC',
                                        '#B22222',
                                        '#DC143C',
                                        '#008080']))
    row_colors = cell_type.map(lut)
    data = [row_colors[i] for i in range(len(row_colors))]
    row_colors = pd.Series(data)
    theta_T = (theta.detach().cpu().numpy())
    clustermap(pd.DataFrame(theta_T), center=0, cmap="RdBu_r", row_colors=row_colors)

    from matplotlib.patches import Patch

    handles = [Patch(facecolor=lut[name]) for name in lut]
    plt.legend(handles, lut, title='Cell Types',
               bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.savefig(path + 'cell_heatmap.png')
