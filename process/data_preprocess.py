import anndata as ad
from scipy.sparse import csc_matrix, coo_matrix
import numpy as np
import pandas as pd
# ################################################################
#
# # Gene name
# f = open('../data/GSE117089_RAW/RNA_mouse_kidney_gene.txt')
# line = f.readline()
# Gene_name = []
# Gene_id  = []
# Gene_type = []
# while line:
#     line = f.readline()
#     if line=='':
#         break
#
#     Gene_id.append(line.split(',')[0])
#     Gene_type.append(line.split(',')[1])
#     Gene_name.append(line.split(',')[2][:-1])
#
# f.close()
#
# # Count matrix
# f = open('../data/GSE117089_RAW/RNA_mouse_kidney_gene_count.txt')
# line = f.readline()
# line = f.readline()
# num_obs = int(line.split(' ')[1])
# num_var = int(line.split(' ')[0])
# row = []
# col = []
# value = []
# while line:
#     line = f.readline()
#     if line == '':
#         break
#     row.append(int(line.split(' ')[1]))
#     col.append(int(line.split(' ')[0]))
#     value.append(int(line.split(' ')[-1][:-1]))
# row = np.array(row)
# col = np.array(col)
# value = np.array(value)
#
# X = coo_matrix((value, (row-1,col-1)), shape=(num_obs, num_var))
#
# # Cell information
# cell_infor = pd.read_csv('../data/GSE117089_RAW/RNA_mouse_kidney_cell.txt')
# cell_name = list(cell_infor.cell_name)
# # print(cell_name)
# Flag = []
# for name in cell_name:
#     if isinstance(name, str):
#         Flag.append(True)
#     else:
#         Flag.append(False)
#
# cell_infor = cell_infor[Flag]
# X = coo_matrix(X.todense()[Flag])
# adata = ad.AnnData(X=X.tocsc(), obs=cell_infor, var=pd.DataFrame(index=Gene_id))
# adata.var['Gene_name'] = pd.Series(Gene_name, index=list(adata.var_names))
# adata.var['Gene_type'] = pd.Series(Gene_type, index=list(adata.var_names))
# adata.write("../data/Mouse_kidney_rna.h5ad")
# del X
#
# #########################################################################################################################
# # The above is for Gex
# # The following is for ATAC
# #
# # Peak name
# f = open('../data/GSE117089_RAW/ATAC_mouse_kidney_peak.txt')
# line = f.readline()
# Peak_name = []
# while line:
#     line = f.readline()
#     if line=='':
#         break
#
#     Peak_name.append(line.split(',')[1])
#
# f.close()
#
# # Cell information
# cell_infor = pd.read_csv('../data/GSE117089_RAW/ATAC_mouse_kidney_cell.txt')
#
# ## ATAC count matrix
# f = open('../data/GSE117089_RAW/ATAC_mouse_kidney_peak_count.txt')
# line = f.readline()
# line = f.readline()
# num_obs = int(line.split(' ')[1])
# num_var = int(line.split(' ')[0])
# row = []
# col = []
# value = []
# while line:
#     line = f.readline()
#     if line == '':
#         break
#     row.append(int(line.split(' ')[1]))
#     col.append(int(line.split(' ')[0]))
#     value.append(int(np.ceil(float(line.split(' ')[-1][:-1]))))
# row = np.array(row)
# col = np.array(col)
# value = np.array(value)
#
# X = coo_matrix((value, (row-1,col-1)), shape=(num_obs, num_var))
# atac_adata = ad.AnnData(X=X.tocsc(), obs=cell_infor, var=pd.DataFrame(index=Peak_name))
# atac_adata.obs['cell_name'] = adata.obs['cell_name']
# split_names = atac_adata.var.index.str.split('-', expand=True)
# chrom = []
# chromStrat = []
# chromEnd = []
# for i in range(len(split_names)):
#     chrom.append(split_names[i][0].replace('chr', ''))
#     chromStrat.append(int(split_names[i][1]))
#     chromEnd.append(int(split_names[i][2]))
#
# chrom = pd.Series(chrom, index=list(atac_adata.var_names))
# chromStrat = pd.Series(chromStrat, index=list(atac_adata.var_names))
# chromEnd = pd.Series(chromEnd, index=list(atac_adata.var_names))
#
# atac_adata.var['chrom'] = chrom
# atac_adata.var['chromStart'] = chromStrat
# atac_adata.var['chromEnd'] = chromEnd
# atac_adata.write("../data/Mouse_kidney_atac.h5ad")
#
# ################################################################################
#
#
# gex_data = ad.read_h5ad("../data/Mouse_kidney_rna.h5ad")
# atac_data = ad.read_h5ad("../data/Mouse_kidney_atac.h5ad")
#
# cell_code_from_GEX = list(gex_data.obs['sample'])
# cell_code_from_ATAC = list(atac_data.obs['sample'])
#
# Index_gex = []
# Index_atac = []
#
# for i in range(len(cell_code_from_GEX)):
#     cell = cell_code_from_GEX[i]
#     if cell in cell_code_from_ATAC:
#         Index_gex.append(i)
#         Index_atac.append(cell_code_from_ATAC.index(cell))
#
# gex_data = gex_data[Index_gex]
# atac_data = atac_data[Index_atac]
#
# gex_data.write("../data/Mouse_kidney_rna.h5ad")
# atac_data.write("../data/Mouse_kidney_atac.h5ad")

# ##########################################################
# gex_data = ad.read_h5ad("../data/Mouse_kidney_rna.h5ad")
# print(gex_data)
# atac_data = ad.read_h5ad("../data/Mouse_kidney_atac.h5ad")

# print(list(gex_data.var_names))

# gene_id_list = list(gex_data.var_names)
# gene_id_list_new = []
# for gene_id in gene_id_list:
#     gene_id_new = gene_id.split('.')[0]
#     gene_id_list_new.append(gene_id_new)

# # 读取txt文件
# with open('/Users/alain/Desktop/gene_id_list.txt', 'r') as f:
#     txt_gene_ids = f.read().splitlines()
# print(len(txt_gene_ids))
# # 读取csv文件
# df = pd.read_csv('/Users/alain/Desktop/mouse_gene_location.csv')
#
# # 以txt文件中的顺序，重排csv文件中的数据
# df.set_index('ensembl_gene_id', inplace=True)
# df = df.reindex(txt_gene_ids)
#
# # 将缺失的gene id的gene location设为NA
# df['chromosome_name'].fillna('NA', inplace=True)
# df['start_position'].fillna('NA', inplace=True)
# df['end_position'].fillna('NA', inplace=True)
#
# # 重置index
# df.reset_index(inplace=True)
#
# # 保存到新的csv文件
# df.to_csv('/Users/alain/Desktop/mouse_gene_location2.csv', index=False)

# rna = ad.read_h5ad("../data/Mouse_kidney_rna.h5ad")
# print(rna.var)
#
# df = pd.read_csv('../data/GSE117089_RAW/mouse_gene_location.csv')
# print(len(list(df['chromosome_name'])))
#
# chrom = []
# chromStrat = []
# chromEnd = []
# for i in range(49584):
#     chrom.append(df['chromosome_name'][i])
#     chromStrat.append(df['start_position'][i])
#     chromEnd.append(df['end_position'][i])
#
# chrom = pd.Series(chrom, index=list(rna.var_names))
# chromStrat = pd.Series(chromStrat, index=list(rna.var_names))
# chromEnd = pd.Series(chromEnd, index=list(rna.var_names))
#
# rna.var['chrom'] = chrom
# rna.var['chromStart'] = chromStrat
# rna.var['chromEnd'] = chromEnd
#
# rna.write('../data/Mouse_kidney_rna.h5ad')

# data = ad.read_h5ad("../data/Mouse_kidney_rna.h5ad")
# print(data.var)
#
# mask = np.logical_or(
#     data.var['chrom'].isna(),
#     np.logical_or(
#         data.var['chromStart'].isna(),
#         data.var['chromEnd'].isna()
#     )
# )
#
# mask = np.logical_or(
#     mask,
#     np.logical_or(
#         data.var['chrom'].isna(),
#         np.logical_or(
#             np.isinf(data.var['chromStart']),
#             np.isinf(data.var['chromEnd'])
#         )
#     )
# )
#
# # 使用这个mask筛选anndata
# data = data[:, ~mask]
#
# if 'chromosome_name' in data.var.columns:
#     # 将'chromosome_name'列重命名为'chrom'
#     data.var['chrom'] = data.var['chromosome_name']
#     data.var.drop('chromosome_name', axis=1, inplace=True)
# else:
#     print("Column 'chromosome_name' not found in var.")
#
# if 'start_position' in data.var.columns:
#     data.var['chromStart'] = data.var['start_position'].astype(int)
#     data.var.drop('start_position', axis=1, inplace=True)
# else:
#     print("Column 'start_position' not found in var.")
#
# if 'end_position' in data.var.columns:
#     data.var['chromEnd'] = data.var['end_position'].astype(int)
#     data.var.drop('end_position', axis=1, inplace=True)
# else:
#     print("Column 'end_position' not found in var.")
#
# print(data)
# data.write('../data/Mouse_kidney_rna.h5ad')
#
# rna1 = ad.read_h5ad("../data/Mouse_kidney_rna.h5ad")
# chrom_rna = sorted(list(set(rna1.var['chrom'])))
#
# atac1 = ad.read_h5ad("../data/Mouse_kidney_atac.h5ad")
# chrom_atac = sorted(list(set(atac1.var['chrom'])))
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
#
# # 创建两个数据集的'chrom'的集合
# adata1_chroms = set(rna1.var['chrom'].unique())
# adata2_chroms = set(atac1.var['chrom'].unique())
#
# # 找出两个数据集中共有的'chrom'，并去除我们不需要的'chrom'
# shared_chroms = adata1_chroms.intersection(adata2_chroms) - set(not_match)
#
# # 创建用于筛选的mask
# adata1_mask = rna1.var['chrom'].isin(shared_chroms)
# adata2_mask = atac1.var['chrom'].isin(shared_chroms)
#
# # 应用mask
# adata1_filtered = rna1[:, adata1_mask]
# adata2_filtered = atac1[:, adata2_mask]
#
# print(adata1_filtered)
# print(adata2_filtered)
#
# chrom_rna2 = sorted(list(set(adata1_filtered.var['chrom'])))
# chrom_atac2 = sorted(list(set(adata2_filtered.var['chrom'])))
#
# print(chrom_rna2)
# print(chrom_atac2)
# adata1_filtered.obs['cell_type'] = adata1_filtered.obs['cell_name']
# adata2_filtered.obs['cell_type'] = adata2_filtered.obs['cell_name']
#
# adata1_filtered.write("../data/Mouse_kidney_rna_filtered.h5ad")
# adata2_filtered.write("../data/Mouse_kidney_atac_filtered.h5ad")

# rna_filtered = ad.read_h5ad('../data/Mouse_kidney_rna_filtered.h5ad')
# print(rna_filtered.obs["cell_type"])
#
atac_filtered = ad.read_h5ad('../data/Mouse_kidney_atac_filtered.h5ad')
print(atac_filtered.X)













