import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from seaborn import heatmap, lineplot, clustermap
import networkx as nx
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

def plot_imputation (recon, true, name, path, recon_type, pear, spear):

    recon_indices = np.arange(len(recon))
    recon_values = np.array(recon)
    true_indices = np.arange(len(true))
    true_values = np.array(true)

    ##### count
    np.sum(true_values==1)/len(true_values)# 53% true value is 1
    np.sum(recon_values<=1)/len(recon_values)# 97% recon value is less than 1


    ##### plot histogram
    fig1, ax1 = plt.subplots(2,2,figsize=(15, 10))
    # plot recon values
    ax1[0,0].hist(recon_values, bins=100,density=False)
    ax1[0,0].set(xlabel=f'recon {recon_type}', ylabel='frequency')
    ax1[0,0].set_title(f'recon {recon_type} '+name)

    # plot a subset
    recon_sub=recon_values[recon_values<5]
    ax1[0,1].hist(recon_sub, bins=100,density=False)
    ax1[0,1].set(xlabel=f'recon {recon_type} subset', ylabel='frequency')
    ax1[0,1].set_title(f'recon {recon_type}_{name}')

    # plot true values
    ax1[1,0].hist(true_values, bins=100,density=False)
    ax1[1,0].set(xlabel=f'True {recon_type}', ylabel='frequency')
    ax1[1,0].set_title(f'True {recon_type}_{name}')

    # plot subset values
    true_sub=true_values[true_values<10]
    ax1[1,1].hist(true_sub, bins=100,density=False)
    ax1[1,1].set(xlabel=f'True {recon_type}', ylabel='frequency')
    ax1[1,1].set_title(f'True {recon_type}_{name}')

    fig1.savefig(f'{path}{name}_{recon_type}_hist.png')
    fig1.clf()


    ##### plot scatter plot
    fig2, ax2 = plt.subplots(1,2,figsize=(15, 5))

    ## plot all values
    recon_sub=recon_values[true_indices.tolist()]
    v_max=np.max([np.max(true_values),np.max(recon_sub)])

    ax2[0].scatter(true_values, recon_sub, s=0.2)
    ax2[0].set_xlim([0, v_max])
    ax2[0].set_ylim([0, v_max])
    ax2[0].set(xlabel='True', ylabel='recon')
    ax2[0].plot([0,v_max],[0,v_max], 'blue', linestyle=":")
    ax2[0].set_title(f'Imputed vs True {recon_type}_{name}')

    ### plot a subset
    ind1=true_values<500
    ind=ind1
    recon_sub_sub=recon_sub[ind]
    true_sub=true_values[ind]

    v_max=np.max([np.max(recon_sub_sub),np.max(true_sub)])

    ax2[1].scatter(true_sub, recon_sub_sub, s=0.2)
    ax2[1].set_xlim([0, v_max])
    ax2[1].set_ylim([0, v_max])
    ax2[1].set(xlabel='True', ylabel='recon')
    ax2[1].plot([0,v_max],[0,v_max], 'blue', linestyle=":", label=f'Pearson:{pear}\n Spearman:{spear}')
    ax2[1].legend()
    fig2.savefig(f'{path}{name}_{recon_type}_scatter.png')
    ax2[0].set_title(f'Zoomed Imputed vs True {recon_type}_{name}')
    fig2.clf()

    plt.close()


def plot_half_graph(plot_path, title, left, left_label, right, right_label):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(left, label=left_label)
    plt.title('Epoch vs '+left_label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # title = f'{emb1}_{emb2}_{pretrain}'
    plt.subplot(1, 2, 2)
    plt.plot(right, label=right_label)
    # plt.plot([], label=f"Testing Accuracy: {round(accuracy, 2)}")
    plt.title('Epoch vs '+right_label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.text(0.8, 0.6, f'Test accuracy: {round(accuracy,4)}', {'color': 'C0', 'fontsize': 13})
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path + f'{title}.png')
    plt.clf()


def plot_graph(plot_path, title, list, data_label, ylim = True):
    """
    :param plot_path: where to save the plot
    :param title: title of the plot in the format 'xxx' vs 'yyy'
    :param list: data being ploted
    :param data_label:
    :param ylim:
    :return:
    """
    plt.plot(list, label=data_label)
    words = title.split(' ')
    plt.xlabel(words[0])
    plt.ylabel(words[2])
    plt.title(title)

    # Adding a legend
    plt.legend()
    if ylim:
        plt.ylim(0, 1)
    plt.savefig(plot_path + title + '.png')
    plt.clf()


def plot_graph2(plot_path, title, list1, data_label1, list2, data_label2, ylim = True):
    """
    :param plot_path: where to save the plot
    :param title: title of the plot in the format 'xxx' vs 'yyy'
    :param list: data being ploted
    :param data_label:
    :param ylim:
    :return: NONE
    """
    plt.plot(list1, label=data_label1)
    plt.plot(list2, label=data_label2)
    words = title.split('_')
    plt.xlabel('Epoch')
    plt.ylabel(words[-1])
    plt.title(title)

    # Adding a legend
    plt.legend()
    if ylim:
        plt.ylim(0, 1)
    plt.savefig(plot_path + title + '.png')
    plt.clf()
