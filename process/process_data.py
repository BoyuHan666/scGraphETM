import anndata
import pandas as pd
import numpy as np
#
# path = "../data/BMMC.h5ad"
# adata = anndata.read_h5ad(path)
# print(adata)
#
# # 切分成两个anndata对象
# adata_GEX = adata[:, adata.var['feature_types'] == 'GEX']
# adata_ATAC = adata[:, adata.var['feature_types'] == 'ATAC']
#
# # 保存两个anndata对象
# adata_GEX.write('../data/BMMC_rna.h5ad')
# adata_ATAC.write('../data/BMMC_atac.h5ad')

# BMMC_rna = anndata.read_h5ad('../data/BMMC_rna_loci.h5ad')
#
# mask = np.logical_or(
#     BMMC_rna.var['chromosome_name'].isna(),
#     np.logical_or(
#         BMMC_rna.var['start_position'].isna(),
#         BMMC_rna.var['end_position'].isna()
#     )
# )
#
# mask = np.logical_or(
#     mask,
#     np.logical_or(
#         BMMC_rna.var['chromosome_name'].isna(),
#         np.logical_or(
#             np.isinf(BMMC_rna.var['start_position']),
#             np.isinf(BMMC_rna.var['end_position'])
#         )
#     )
# )
#
# # 使用这个mask筛选anndata
# BMMC_rna = BMMC_rna[:, ~mask]
#
# if 'chromosome_name' in BMMC_rna.var.columns:
#     # 将'chromosome_name'列重命名为'chrom'
#     BMMC_rna.var['chrom'] = BMMC_rna.var['chromosome_name']
#     BMMC_rna.var.drop('chromosome_name', axis=1, inplace=True)
# else:
#     print("Column 'chromosome_name' not found in var.")
#
# if 'start_position' in BMMC_rna.var.columns:
#     BMMC_rna.var['chromStart'] = BMMC_rna.var['start_position'].astype(int)
#     BMMC_rna.var.drop('start_position', axis=1, inplace=True)
# else:
#     print("Column 'start_position' not found in var.")
#
# if 'end_position' in BMMC_rna.var.columns:
#     BMMC_rna.var['chromEnd'] = BMMC_rna.var['end_position'].astype(int)
#     BMMC_rna.var.drop('end_position', axis=1, inplace=True)
# else:
#     print("Column 'end_position' not found in var.")
#
# print(BMMC_rna)
# BMMC_rna.write('../data/BMMC_rna.h5ad')

# BMMC_atac = anndata.read_h5ad('../data/BMMC_atac.h5ad')
# split_names = BMMC_atac.var.index.str.split('-', expand=True)
#
# chrom = []
# chromStrat = []
# chromEnd = []
# for i in range(len(split_names)):
#     chrom.append(split_names[i][0].replace('chr', ''))
#     chromStrat.append(int(split_names[i][1]))
#     chromEnd.append(int(split_names[i][2]))
#
# chrom = pd.Series(chrom, index=list(BMMC_atac.var_names))
# chromStrat = pd.Series(chromStrat, index=list(BMMC_atac.var_names))
# chromEnd = pd.Series(chromEnd, index=list(BMMC_atac.var_names))
#
# BMMC_atac.var['chrom'] = chrom
# BMMC_atac.var['chromStart'] = chromStrat
# BMMC_atac.var['chromEnd'] = chromEnd
#
# BMMC_atac.write('../data/BMMC_atac.h5ad')

# rna = anndata.read_h5ad('../data/BMMC_rna.h5ad')
# print(rna)
# chrom_rna = sorted(list(set(rna.var['chrom'])))
#
# atac = anndata.read_h5ad('../data/BMMC_atac.h5ad')
# print(atac)
# chrom_atac = sorted(list(set(atac.var['chrom'])))
#
# print(chrom_rna)
# print(chrom_atac)
#
# not_match = []
# for i in chrom_atac:
#     if i not in chrom_rna:
#         not_match.append(i)
# print(not_match)
#
# for i in chrom_rna:
#     if i not in chrom_atac:
#         not_match.append(i)
# print(not_match)

# chrom_to_remove = ['GL000195.1', 'GL000205.2', 'KI270713.1', 'KI270726.1', 'Y', 'KI270711.1', 'MT', 'GL000219.1']
#
# # 创建两个数据集的'chrom'的集合
# adata1_chroms = set(rna.var['chrom'].unique())
# adata2_chroms = set(atac.var['chrom'].unique())
#
# # 找出两个数据集中共有的'chrom'，并去除我们不需要的'chrom'
# shared_chroms = adata1_chroms.intersection(adata2_chroms) - set(chrom_to_remove)
#
# # 创建用于筛选的mask
# adata1_mask = rna.var['chrom'].isin(shared_chroms)
# adata2_mask = atac.var['chrom'].isin(shared_chroms)
#
# # 应用mask
# adata1_filtered = rna[:, adata1_mask]
# adata2_filtered = atac[:, adata2_mask]
#
# print(adata1_filtered)
# print(adata2_filtered)
#
# chrom_rna2 = sorted(list(set(adata1_filtered.var['chrom'])))
# chrom_atac2 = sorted(list(set(adata2_filtered.var['chrom'])))
#
# print(chrom_rna2)
# print(chrom_atac2)
#
# adata1_filtered.write("../data/BMMC_rna_filtered.h5ad")
# adata2_filtered.write("../data/BMMC_atac_filtered.h5ad")

rna_filtered = anndata.read_h5ad('../data/BMMC_rna_filtered.h5ad')
print(rna_filtered)
# print(rna_filtered.var['chrom'])

atac_filtered = anndata.read_h5ad('../data/BMMC_atac_filtered.h5ad')
print(atac_filtered)
# print(atac_filtered.var['chrom'])



