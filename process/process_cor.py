import anndata
import torch
import time
# import numpy as np
# import scipy.sparse as sp
from tqdm import tqdm

import heapq
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pickle

if __name__ == "__main__":

    scRNA_adata = anndata.read_h5ad("../data/10x-Multiome-Pbmc10k-RNA.h5ad")
    scATAC_adata = anndata.read_h5ad("../data/10x-Multiome-Pbmc10k-ATAC.h5ad")

    num_of_gene = scRNA_adata.X.shape[1]
    num_of_peak = scATAC_adata.X.shape[1]

    # num_of_gene = 1000
    # cor_mat = torch.zeros(num_of_peak + num_of_gene, num_of_peak + num_of_gene)

    gene_pos_dic = {}
    for i in range(num_of_gene):
        gene_names = scRNA_adata.var_names[i]
        chrom = scRNA_adata.var["chrom"][i]
        chromStart = scRNA_adata.var["chromStart"][i]
        chromEnd = scRNA_adata.var["chromEnd"][i]
        gene_pos_dic[gene_names] = [chrom, chromStart, chromEnd]

    print(gene_pos_dic)

    peak_pos_dic = {}
    for i in range(num_of_peak):
        peak_names = scATAC_adata.var_names[i]
        chrom = scATAC_adata.var["chrom"][i]
        chromStart = scATAC_adata.var["chromStart"][i]
        chromEnd = scATAC_adata.var["chromEnd"][i]
        peak_pos_dic[peak_names] = [chrom, chromStart, chromEnd]

    start = time.time()
    gene_peak_relation = {}


    for i, gene in tqdm(enumerate(gene_pos_dic.keys())):
        gene_chrom, gene_start, gene_end = gene_pos_dic[gene]
        dist_peak_list = []
        for j, peak in enumerate(peak_pos_dic.keys()):
            peak_chrom, peak_start, peak_end = peak_pos_dic[peak]
            if gene_chrom == peak_chrom:
                dist = min(abs(peak_start - gene_start), abs(peak_end - gene_end))
                # Push item onto heap, then pop and return smallest item
                if len(dist_peak_list) < 5:
                    heapq.heappush(dist_peak_list, (-dist, peak))
                else:
                    heapq.heappushpop(dist_peak_list, (-dist, peak))

        # Get nearest peaks and reverse the negative distance
        nearest_peaks = sorted([(peak, -dist) for dist, peak in dist_peak_list], key=lambda x: x[1])
        gene_peak_relation[gene] = nearest_peaks

    # def process_gene(gene):
    #     gene_chrom, gene_start, gene_end = gene_pos_dic[gene]
    #     dist_peak_list = []
    #     for j, peak in enumerate(peak_pos_dic.keys()):
    #         peak_chrom, peak_start, peak_end = peak_pos_dic[peak]
    #         if gene_chrom == peak_chrom:
    #             dist = min(abs(peak_start - gene_start), abs(peak_end - gene_end))
    #             if len(dist_peak_list) < 5:
    #                 heapq.heappush(dist_peak_list, (-dist, peak))
    #             else:
    #                 heapq.heappushpop(dist_peak_list, (-dist, peak))
    #
    #     nearest_peaks = sorted([(peak, -dist) for dist, peak in dist_peak_list], key=lambda x: x[1])
    #     return gene, nearest_peaks
    #
    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     future_to_gene = {executor.submit(process_gene, gene): gene for gene in gene_pos_dic.keys()}
    #     for future in concurrent.futures.as_completed(future_to_gene):
    #         gene, nearest_peaks = future.result()
    #         gene_peak_relation[gene] = nearest_peaks



    end = time.time()
    print(end - start)
    # print(gene_peak_relation)

    with open('../data/relation/gene_peak_relation.pickle', 'wb') as fp:
        pickle.dump(gene_peak_relation, fp)

    with open('../data/relation/gene_peak_relation.pickle', 'rb') as fp:
        gpr = pickle.load(fp)

    print(len(gpr))


