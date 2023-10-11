import pandas as pd
import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from seaborn import heatmap, lineplot, clustermap
import networkx as nx
from tqdm import tqdm


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
