import numpy as np
import torch
from scipy.sparse import csr_matrix


def get_pos(adata):
    pos_dic = {}
    num_of_modality = len(adata.var_names)
    for i in range(num_of_modality):
        names = adata.var_names[i]
        chrom = adata.var["chrom"][i]
        chromStart = adata.var["chromStart"][i]
        chromEnd = adata.var["chromEnd"][i]
        pos_dic[names] = [chrom, chromStart, chromEnd]

    return pos_dic


def get_one_modality_cor(adata, rate1=0.6, rate2=-0.6, dis_rate=1):
    expression = adata.X.toarray()
    correlation_matrix = np.corrcoef(expression, rowvar=False)
    correlation_matrix_cleaned = np.nan_to_num(correlation_matrix, nan=0, posinf=1, neginf=-1)
    pos_dic = get_pos(adata)
    num_of_modality = len(adata.var_names)

    modality_cor_mat = torch.zeros(num_of_modality, num_of_modality)

    for i in range(len(list(pos_dic.keys()))):
        for j in range(len(list(pos_dic.keys()))):
            modality_cor = correlation_matrix_cleaned[i, j]
            if modality_cor > rate1 and modality_cor != dis_rate:
                modality_cor_mat[i, j] = 1
            if modality_cor < rate2:
                modality_cor_mat[i, j] = 1

    return correlation_matrix_cleaned, modality_cor_mat


def get_two_modality_cor(scRNA_adata, scATAC_adata, gene_cor_mat, peak_cor_mat):
    gene_pos_dic = get_pos(scRNA_adata)
    peak_pos_dic = get_pos(scATAC_adata)
    num_of_gene = len(scRNA_adata.var_names)
    num_of_peak = len(scATAC_adata.var_names)
    cor_mat = torch.zeros(num_of_peak + num_of_gene, num_of_peak + num_of_gene)

    for i, gene in enumerate(list(gene_pos_dic.keys())):
        for j, peak in enumerate(list(peak_pos_dic.keys())):
            gene_chrom = gene_pos_dic[gene][0]
            gene_start = gene_pos_dic[gene][1]
            # gene_end = gene_pos_dic[gene][2]

            peak_chrom = peak_pos_dic[peak][0]
            peak_start = peak_pos_dic[peak][1]
            # peak_end = peak_pos_dic[peak][2]

            if gene_chrom == peak_chrom and abs(gene_start - peak_start) <= 2000:
                cor_mat[num_of_peak + i, j] = 1
                cor_mat[j, num_of_peak + i] = 1

    for i in range(len(list(gene_pos_dic.keys()))):
        for j in range(len(list(gene_pos_dic.keys()))):
            cor_mat[num_of_peak + i, num_of_peak + j] = gene_cor_mat[i, j]

    for i in range(len(list(peak_pos_dic.keys()))):
        for j in range(len(list(peak_pos_dic.keys()))):
            cor_mat[i, j] = peak_cor_mat[i, j]

    return cor_mat


def convert_to_edge_index(cor_matrix, device):
    sparse_cor_mat = csr_matrix(cor_matrix)
    rows, cols = sparse_cor_mat.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long, device=device)

    return edge_index
