import argparse
import logging
import os
import shutil
import sys
import tarfile
import traceback
import urllib
import warnings
from typing import Dict, Optional
from typing import Iterable, Literal
import matplotlib.pyplot as plt
import wandb

from rich import print

# ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('genegeneformer')[0]
# os.chdir(ROOT_PATH)

# sys.path.append(ROOT_PATH)
exc_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(exc_dir)

print(f'Initialized root_path to {exc_dir}')

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scib
import seaborn as sns
import torch
from anndata import AnnData
from tqdm import tqdm as tqdm_base
import pickle

logger = logging.getLogger('batch_integration')
sns.set()

available_datasets = {"2.1.0": ["pbmc8k", "pbmc4k", "t_3k", "t_4k", "neuron_9k"]}

group_to_url_skeleton = {"2.1.0": "http://cf.10xgenomics.com/samples/cell-exp/{}/{}/{}_{}_gene_bc_matrices.tar.gz"}

group_to_filename_skeleton = {"2.1.0": "{}_gene_bc_matrices.tar.gz"}

dataset_to_group = {
    dataset_name: group
    for group, list_datasets in available_datasets.items()
    for dataset_name in list_datasets
}

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')


# ! From SCVI

def track(
        sequence: Iterable,
        description: str = "Working...",
        disable: bool = False,
        style: Literal["rich", "tqdm"] = None,
        **kwargs,
):
    """From SCVI
    """
    if style is None:
        style = 'tqdm'
    if style not in ["rich", "tqdm"]:
        raise ValueError("style must be one of ['rich', 'tqdm']")
    if disable:
        return sequence
    if style == "tqdm":
        # fixes repeated pbar in jupyter
        # see https://github.com/tqdm/tqdm/issues/375
        if hasattr(tqdm_base, "_instances"):
            for instance in list(tqdm_base._instances):
                tqdm_base._decr_instances(instance)
        return tqdm_base(sequence, desc=description, file=sys.stdout, **kwargs)


def _download(url: Optional[str], save_path: str, filename: str):
    """From SCVI: Writes data from url to file."""
    if os.path.exists(os.path.join(save_path, filename)):
        logger.info(f"File {os.path.join(save_path, filename)} already downloaded")
        return
    elif url is None:
        logger.info(
            f"No backup URL provided for missing file {os.path.join(save_path, filename)}"
        )
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Magic Browser"})
    try:
        r = urllib.request.urlopen(req)
        if r.getheader("Content-Length") is None:
            raise FileNotFoundError(
                f"Found file with no content at {url}. "
                "This is possibly a directory rather than a file path."
            )
    except urllib.error.HTTPError as exc:
        if exc.code == "404":
            raise FileNotFoundError(f"Could not find file at {url}") from exc
        raise exc
    logger.info("Downloading file at %s" % os.path.join(save_path, filename))

    def read_iter(file, block_size=1000):
        """Iterates through file.

        Given a file 'file', returns an iterator that returns bytes of
        size 'blocksize' from the file, using read().
        """
        while True:
            block = file.read(block_size)
            if not block:
                break
            yield block

    # Create the path to save the data
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    block_size = 1000

    filesize = int(r.getheader("Content-Length"))
    filesize = np.rint(filesize / block_size)
    with open(os.path.join(save_path, filename), "wb") as f:
        iterator = read_iter(r, block_size=block_size)
        for data in track(
                iterator, style="tqdm", total=filesize, description="Downloading..."
        ):
            f.write(data)


def load_scib_reproducibility_dataset(h5ad_file, cell_type_name="cell_type"):
    # Download the file
    adata = anndata.read_h5ad(h5ad_file)
    # Process cell data
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.filter_genes(adata, min_counts=1)

    # Group Obs data
    # adata.obs["batch"] = adata.obs["batch"].astype(np.int64)
    labels = adata.obs[cell_type_name]
    cell_types = np.array(labels.values.categories)
    type2id = {t: i for i, t in enumerate(cell_types)}

    adata.obs["labels"] = np.vectorize(type2id.get)(labels)
    adata.uns["cell_types"] = cell_types
    adata.obs["str_labels"] = labels
    #adata.var["n_counts"] = np.squeeze(np.asarray(np.sum(adata.X, axis=0)))
    return adata

def dataset_10x(
        dataset_name: Optional[str] = None,
        filename: Optional[str] = None,
        save_path: str = "data/10X",
        url: str = None,
        return_filtered: bool = True,
        remove_extracted_data: bool = False,
        **scanpy_read_10x_kwargs,
) -> anndata.AnnData:
    """Loads a file from `10x <http://cf.10xgenomics.com/>`_ website.
    """
    return _load_dataset_10x(
        dataset_name=dataset_name,
        filename=filename,
        save_path=save_path,
        url=url,
        return_filtered=return_filtered,
        remove_extracted_data=remove_extracted_data,
        **scanpy_read_10x_kwargs,
    )


def _load_dataset_10x(
        dataset_name: str = None,
        filename: str = None,
        save_path: str = "data/10X",
        url: str = None,
        return_filtered: bool = True,
        remove_extracted_data: bool = False,
        **scanpy_read_10x_kwargs,
):
    # ! From SCVI
    try:
        import scanpy
    except ImportError as err:
        raise ImportError("Please install scanpy -- `pip install scanpy`") from err

    # form data url and filename unless manual override
    if dataset_name is not None:
        if url is not None:
            warnings.warn(
                "dataset_name provided, manual url is disregarded.",
                UserWarning,
            )
        if filename is not None:
            warnings.warn(
                "dataset_name provided, manual filename is disregarded.",
                UserWarning,
            )
        group = dataset_to_group[dataset_name]
        url_skeleton = group_to_url_skeleton[group]

        filter_type = "filtered" if return_filtered else "raw"
        url = url_skeleton.format(group, dataset_name, dataset_name, filter_type)
        filename_skeleton = group_to_filename_skeleton[group]
        filename = filename_skeleton.format(filter_type)
        save_path = os.path.join(save_path, dataset_name)
    elif filename is not None and url is not None:
        logger.info("Loading 10X dataset with custom url and filename")
    elif filename is not None and url is None:
        logger.info("Loading local 10X dataset with custom filename")
    else:
        logger.info("Loading extracted local 10X dataset with custom filename")
    _download(url, save_path=save_path, filename=filename)
    file_path = os.path.join(save_path, filename)

    # untar
    download_is_targz = url[-7:] == ".tar.gz"
    was_extracted = False
    if download_is_targz is True:
        if not os.path.exists(file_path[:-7]):  # nothing extracted yet
            if tarfile.is_tarfile(file_path):
                logger.info("Extracting tar file")
                tar = tarfile.open(file_path, "r:gz")
                tar.extractall(path=save_path)
                was_extracted = True
                tar.close()
        path_to_data_folder, suffix = _find_path_to_mtx(save_path)
        adata = scanpy.read_10x_mtx(path_to_data_folder, **scanpy_read_10x_kwargs)
        if was_extracted and remove_extracted_data:
            folders_in_save_path = path_to_data_folder[len(save_path) + 1:].split("/")
            extracted_folder_path = save_path + "/" + folders_in_save_path[0]
            logger.info(f"Removing extracted data at {extracted_folder_path}")
            shutil.rmtree(extracted_folder_path)
    else:
        adata = scanpy.read_10x_h5(file_path, **scanpy_read_10x_kwargs)

    adata.var_names_make_unique()
    scanpy.pp.filter_cells(adata, min_counts=1)
    scanpy.pp.filter_genes(adata, min_counts=1)

    return adata



def _load_pbmc_dataset(
        save_path: str = "data/",
        remove_extracted_data: bool = True,
) -> anndata.AnnData:
    # ! From SCVI
    urls = [
        "https://github.com/YosefLab/scVI-data/raw/master/gene_info.csv",
        "https://github.com/YosefLab/scVI-data/raw/master/pbmc_metadata.pickle",
    ]
    save_fns = ["gene_info_pbmc.csv", "pbmc_metadata.pickle"]

    for i in range(len(urls)):
        _download(urls[i], save_path, save_fns[i])

    de_metadata = pd.read_csv(os.path.join(save_path, "gene_info_pbmc.csv"), sep=",")
    pbmc_metadata = pd.read_pickle(os.path.join(save_path, "pbmc_metadata.pickle"))
    pbmc8k = _load_dataset_10x(
        "pbmc8k",
        save_path=save_path,
        var_names="gene_ids",
        remove_extracted_data=remove_extracted_data,
    )
    pbmc4k = _load_dataset_10x(
        "pbmc4k",
        save_path=save_path,
        var_names="gene_ids",
        remove_extracted_data=remove_extracted_data,
    )
    barcodes = np.concatenate((pbmc8k.obs_names, pbmc4k.obs_names))

    adata = pbmc8k.concatenate(pbmc4k)
    adata.obs_names = barcodes

    dict_barcodes = dict(zip(barcodes, np.arange(len(barcodes))))
    subset_cells = []
    barcodes_metadata = pbmc_metadata["barcodes"].index.values.ravel().astype(str)
    for barcode in barcodes_metadata:
        if (
                barcode in dict_barcodes
        ):  # barcodes with end -11 filtered on 10X website (49 cells)
            subset_cells += [dict_barcodes[barcode]]
    adata = adata[np.asarray(subset_cells), :].copy()
    idx_metadata = np.asarray(
        [not barcode.endswith("11") for barcode in barcodes_metadata], dtype=bool
    )
    genes_to_keep = list(
        de_metadata["ENSG"].values
    )  # only keep the genes for which we have de data
    difference = list(
        set(genes_to_keep).difference(set(adata.var_names))
    )  # Non empty only for unit tests
    for gene in difference:
        genes_to_keep.remove(gene)

    adata = adata[:, genes_to_keep].copy()
    design = pbmc_metadata["design"][idx_metadata]
    raw_qc = pbmc_metadata["raw_qc"][idx_metadata]
    normalized_qc = pbmc_metadata["normalized_qc"][idx_metadata]

    design.index = adata.obs_names
    raw_qc.index = adata.obs_names
    normalized_qc.index = adata.obs_names
    adata.obs["batch"] = adata.obs["batch"].astype(np.int64)
    adata.obsm["design"] = design
    adata.obsm["raw_qc"] = raw_qc
    adata.obsm["normalized_qc"] = normalized_qc

    adata.obsm["qc_pc"] = pbmc_metadata["qc_pc"][idx_metadata]
    labels = pbmc_metadata["clusters"][idx_metadata]
    cell_types = pbmc_metadata["list_clusters"]
    adata.obs["labels"] = labels
    adata.uns["cell_types"] = cell_types
    adata.obs["str_labels"] = [cell_types[i] for i in labels]

    adata.var["n_counts"] = np.squeeze(np.asarray(np.sum(adata.X, axis=0)))

    return adata


def _find_path_to_mtx(save_path: str):
    # ! From SCVI
    for root, _, files in os.walk(save_path):
        # do not consider hidden files
        files = [f for f in files if not f[0] == "."]
        contains_mat = [
            filename == "matrix.mtx" or filename == "matrix.mtx.gz"
            for filename in files
        ]
        contains_mat = np.asarray(contains_mat).any()
        if contains_mat:
            is_tar = files[0][-3:] == ".gz"
            suffix = ".gz" if is_tar else ""
            return root, suffix
    raise FileNotFoundError("No matrix.mtx(.gz) found in path (%s)." % save_path)


def eval_scib_metrics(
        adata: AnnData,
        batch_key: str,
        label_key: str,
        notes: Optional[str] = None,
) -> Dict:
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed="cell_emb",
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )

    if notes is not None:
        logger.info(f"{notes}")

    logger.info(f"{results}")

    result_dict = results[0].to_dict()
    logger.info(
        "Biological Conservation Metrics: \n"
        f"ASW (cell-type): {result_dict['ASW_label']:.4f}, graph cLISI: {result_dict['cLISI']:.4f}, "
        f"isolated label silhouette: {result_dict['isolated_label_silhouette']:.4f}, \n"
        "Batch Effect Removal Metrics: \n"
        f"PCR_batch: {result_dict['PCR_batch']:.4f}, ASW (batch): {result_dict['ASW_label/batch']:.4f}, "
        f"graph connectivity: {result_dict['graph_conn']:.4f}, graph iLISI: {result_dict['iLISI']:.4f}"
    )

    result_dict["avg_bio"] = np.mean(
        [result_dict["NMI_cluster/label"], result_dict["ARI_cluster/label"], result_dict["ASW_label"], ])
    result_dict["avg_batch"] = np.mean([result_dict["ASW_label/batch"], result_dict["graph_conn"]])
    result_dict['overall'] = 0.6 * result_dict["avg_bio"] + 0.4 * result_dict["avg_batch"]
    # remove nan value in result_dict
    result_dict = {k: round(v, 4) for k, v in result_dict.items() if not np.isnan(v)}

    return result_dict


def eval_testdata(
        adata_t: AnnData,
        batch_key: str,
        cell_type_key: str,
        cell_emb: str,
        path: str
) -> Optional[Dict]:

    results = {}
    try:
        results = eval_scib_metrics(adata_t, batch_key, cell_type_key)
    except Exception as e:
        traceback.print_exc()
        logger.error(e)

    sc.pp.neighbors(adata_t, use_rep=cell_emb)
    sc.tl.umap(adata_t, min_dist=0.3)  # ! UMAP for Embedding Visualization
    fig = sc.pl.umap(
        adata_t,
        color=batch_key,
        # title=f"batch, avg_batch = {results.get('avg_batch', 0.0):.4f}",
        # frameon=False,
        return_fig=True,
        show=False,
    )

    fig_fig = fig.get_figure()
    fig_fig.set_size_inches(12, 8)
    fig_fig.savefig(f'./figures/batch_umap_{path}.png', bbox_inches='tight')
    # results["batch_umap"] = fig

    sc.pp.neighbors(adata_t, use_rep=cell_emb)
    sc.tl.umap(adata_t, min_dist=0.3)
    fig = sc.pl.umap(
        adata_t,
        color=cell_type_key,
        # title=f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
        # frameon=False,
        return_fig=True,
        show=False,
    )

    fig_fig = fig.get_figure()
    fig_fig.set_size_inches(12, 8)
    fig_fig.savefig(f'./figures/celltype_umap_{path}.png', bbox_inches='tight')
    # results["celltype_umap"] = fig
    return results

