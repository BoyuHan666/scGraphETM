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


def pretrain_graph(model_tuple, optimizer1, train_set, ari_freq, niter, use_mlp, use_mask, mask1, mask2):

    gnn, mlp_dec1, mlp_dec2 = model_tuple

    for i in range(niter):
        kl_weight = helper2.calc_weight(i, niter, 0, 1 / 3, 1e-2, 4)

        (X_rna_tensor, X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized,
         scRNA_mini_batch_anndata, scATAC_mini_batch_anndata, gene_correlation_matrix,
         peak_correlation_matrix, feature_matrix, edge_index, X_rna_tensor_copy, X_atac_tensor_copy) = train_set

        NELBO, gnn = helper2.graph_train_one_epoch(
            gnn, mlp_dec1, mlp_dec2, optimizer1, X_rna_tensor,
            X_rna_tensor_normalized, X_atac_tensor, X_atac_tensor_normalized, feature_matrix,
            edge_index, gene_correlation_matrix, peak_correlation_matrix, kl_weight, use_mlp,
            use_mask, mask1, mask2, X_rna_tensor_copy, X_atac_tensor_copy
        )

        if i % ari_freq == 0:
            print('====  Iter: {},  NELBO: {:.4f}  ====\n'.format(i, NELBO))

    return gnn, mlp_dec1, mlp_dec2


if __name__ == "__main__":

    rna_path = "../data/10x-Multiome-Pbmc10k-RNA.h5ad"
    atac_path = "../data/10x-Multiome-Pbmc10k-ATAC.h5ad"
    num_of_cell = 2000
    num_of_gene = 2000
    num_of_peak = 2000
    test_num_of_cell = 2000
    emb_size = 512
    emb_size2 = 512
    num_of_topic = 60
    gnn_conv = 'GATv2'
    num_epochs = 2000
    ari_freq = 100
    plot_path_rel = "./plot/"
    metric = 'theta'  # mu or theta
    lr = 0.001
    use_mlp = False
    use_mask_train = False
    use_mask_reconstruct = False  # False: one side mask for reconstructing the masked expressions
    mask_ratio = 0.2
    use_noise = False
    noise_ratio = 0.2

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
        mask_ratio=mask_ratio,
        use_noise=use_noise,
        noise_ratio=noise_ratio
    )

    gnn = model2.GNN(emb_size, emb_size * 2, emb_size, 1, device, 0, gnn_conv).to(device)
    mlp_dec1 = model2.MLP(emb_size, emb_size2 * 2, num_of_cell).to(device)
    mlp_dec2 = model2.MLP(emb_size, emb_size2 * 2, num_of_cell).to(device)

    parameters = [
                  {'params': gnn.parameters()},
                  {'params': mlp_dec1.parameters()},
                  {'params': mlp_dec2.parameters()},
                  ]

    optimizer_adamW = optim.AdamW(parameters, lr=lr, weight_decay=1.2e-6)
    optimizer_adam = optim.Adam(parameters, lr=lr, weight_decay=1.2e-6)

    model_tuple = (gnn, mlp_dec1, mlp_dec2)
    for model in model_tuple:
        print(model)

    print(f"=========  start training {num_of_topic} =========")
    st = time.time()
    gnn, mlp_dec1, mlp_dec2 = pretrain_graph(
        model_tuple=model_tuple,
        optimizer1=optimizer_adam,
        train_set=training_set,
        ari_freq=ari_freq,
        niter=num_epochs,
        use_mlp=use_mlp,
        use_mask=use_mask_reconstruct,
        mask1=mask_matrix1,
        mask2=mask_matrix2,
    )
    ed = time.time()
