import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
import time

import select_gpu
import full_batch
import helper2
import model2
import view_result

#
# # Assuming you have two matrices called matrix1 and matrix2
# matrix1 = np.array([[1, 2, 3],
#                     [4, 5, 6],
#                     [7, 8, 9]])
# matrix2 = np.array([[2, 4, 6],
#                     [8, 10, 12],
#                     [14, 16, 18],
#                     [14, 16, 18]])
#
# # Calculate the correlation matrix
# correlation_matrix = np.zeros((matrix1.shape[0], matrix2.shape[0]))
#
# for i in range(matrix1.shape[0]):
#     for j in range(matrix2.shape[0]):
#         coef = np.corrcoef(matrix1[i], matrix2[j], rowvar=False)[0][1]
#         correlation_matrix[i,j] = coef
#

# Print the correlation matrix
# print(correlation_matrix)

if __name__ == "__main__":

    rna_path = "../data/10x-Multiome-Pbmc10k-RNA.h5ad"
    atac_path = "../data/10x-Multiome-Pbmc10k-ATAC.h5ad"

    num_of_cell = 6000
    num_of_gene = 2000
    num_of_peak = 2000
    title = 'Full_' + str(num_of_peak)
    test_num_of_cell = 2000
    emb_size = 512
    emb_size2 = 512
    num_of_topic = 40
    gnn_conv = 'GATv2'
    num_epochs = 2000
    ari_freq = 40
    plot_path_rel = "./plot/"
    metric = 'theta'  # mu or theta
    lr = 0.001
    use_mlp = False
    use_mask_train = False
    use_mask_reconstruct = False  # False: one side mask for reconstructing the masked expressions
    mask_ratio = 0.2

    if torch.cuda.is_available():
        print("=======  GPU device found  =======")
        selected_gpu = select_gpu.get_lowest_usage_gpu_index()
        torch.cuda.set_device(selected_gpu)
        device = torch.device("cuda:{}".format(selected_gpu))
    else:
        device = torch.device("cpu")
        print("=======  No GPU found  =======")

    training_set, total_training_set, test_set, scRNA_adata, scATAC_adata, mask_matrix1, mask_matrix2 = full_batch.process_full_batch_data(
        rna_path=rna_path,
        atac_path=atac_path,
        device=device,
        num_of_cell=num_of_cell,
        num_of_gene=num_of_gene,
        num_of_peak=num_of_peak,
        test_num_of_cell=test_num_of_cell,
        emb_size=emb_size,
        use_highly_variable=True,
        cor='pearson',
        use_mask=use_mask_train,
        mask_ratio=mask_ratio
    )
