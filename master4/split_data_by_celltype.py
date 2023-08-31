import anndata


def split_anndata_by_cell_type(adata_rna: anndata.AnnData, adata_atac: anndata.AnnData):
    """
    Split the given AnnData objects based on the 'cell_type' in the .obs dataframe.

    Parameters:
    - adata_rna: AnnData object containing scRNA-seq data
    - adata_atac: AnnData object containing scATAC-seq data

    Returns:
    - A 2D list where the first dimension represents different cell types and the second dimension contains
      the split scRNA-seq and scATAC-seq data for that cell type.
    """
    # Ensure both AnnData objects have the same cell types in the same order
    assert all(adata_rna.obs['cell_type'] == adata_atac.obs['cell_type']), "Mismatch in cell types between the two AnnData objects"

    unique_cell_types = adata_rna.obs['cell_type'].unique()
    result = []

    for cell_type in unique_cell_types:
        rna_sub = adata_rna[adata_rna.obs['cell_type'] == cell_type, :]
        atac_sub = adata_atac[adata_atac.obs['cell_type'] == cell_type, :]
        result.append([rna_sub, atac_sub, rna_sub.n_obs])

    return result, unique_cell_types


if __name__ == "__main__":
    rna = anndata.read_h5ad('../data/10x-Multiome-Pbmc10k-RNA.h5ad')
    atac = anndata.read_h5ad('../data/10x-Multiome-Pbmc10k-ATAC.h5ad')

    result, cell_type = split_anndata_by_cell_type(rna, atac)
    for i in range(len(result)):
        print(cell_type[i])
        print(result[i][0])
        print(result[i][1])
        print(result[i][2])
        print("="*30)

