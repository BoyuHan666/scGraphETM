import torch
from scipy import stats
from torch.nn import functional as F
from sklearn.metrics import adjusted_rand_score
from torch.autograd import Variable

import anndata as ad
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

"""
==============================================================================
New Helper
==============================================================================
"""


def evaluate_ari(cell_embed, adata):
    """
        This function is used to evaluate ARI using the lower-dimensional embedding
        cell_embed of the single-cell data
        :param cell_embed: a NxK single-cell embedding generated from NMF or scETM
        :param adata: single-cell AnnData data object (default to to mp_anndata)
        :return: ARI score of the clustering results produced by Louvain
    """
    adata.obsm['cell_embed'] = cell_embed
    sc.pp.neighbors(adata, use_rep="cell_embed", n_neighbors=30)
    sc.tl.louvain(adata, resolution=0.15)
    ari = adjusted_rand_score(adata.obs['Celltype'], adata.obs['louvain'])
    return ari


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul_(std).add_(mu)


def split_tensor(tensor, num_rows):
    """
    split (gene+peak) x emb tensor to gene x emb tensor and peak x emb tensor
    """
    if num_rows >= tensor.shape[0]:
        raise ValueError("num_rows should be less than tensor's number of rows")

    top_matrix = tensor[:num_rows, :]
    bottom_matrix = tensor[num_rows:, :]

    return top_matrix, bottom_matrix


def calc_weight(
        epoch: int,
        n_epochs: int,
        cutoff_ratio: float = 0.,
        warmup_ratio: float = 1 / 3,
        min_weight: float = 0.,
        max_weight: float = 1e-7
) -> float:
    """Calculates weights.
    Args:
        epoch: current epoch.
        n_epochs: the total number of epochs to train the model.
        cutoff_ratio: ratio of cutoff epochs (set weight to zero) and
            n_epochs.
        warmup_ratio: ratio of warmup epochs and n_epochs.
        min_weight: minimum weight.
        max_weight: maximum weight.
    Returns:
        The current weight of the KL term.
    """

    fully_warmup_epoch = n_epochs * warmup_ratio

    if epoch < n_epochs * cutoff_ratio:
        return 0.
    if warmup_ratio:
        return max(min(1., epoch / fully_warmup_epoch) * max_weight, min_weight)
    else:
        return max_weight


def train_one_epoch(encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2, optimizer,
                    RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized,
                    feature_matrix, edge_index, gene_gene, peak_peak, kl_weight, use_mlp):
    encoder1.train()
    encoder2.train()
    gnn.train()
    mlp1.train()
    mlp2.train()
    decoder1.train()
    decoder2.train()

    optimizer.zero_grad()

    encoder1.zero_grad()
    encoder2.zero_grad()
    gnn.zero_grad()
    mlp1.zero_grad()
    mlp2.zero_grad()
    decoder1.zero_grad()
    decoder2.zero_grad()

    mu1, log_sigma1, kl_theta1 = encoder1(RNA_tensor_normalized)
    mu2, log_sigma2, kl_theta2 = encoder2(ATAC_tensor_normalized)

    z1 = reparameterize(mu1, log_sigma1)
    theta1 = F.softmax(z1, dim=-1)

    z2 = reparameterize(mu2, log_sigma2)
    theta2 = F.softmax(z2, dim=-1)

    emb = gnn(feature_matrix, edge_index)
    new_emb = mlp1(emb)
    # num_of_peak x emb, num_of_gene x emb
    # eta, rho = split_tensor(emb, ATAC_tensor_normalized.shape[1])
    eta, rho = split_tensor(new_emb, ATAC_tensor_normalized.shape[1])

    pred_gene_gene = torch.mm(rho, rho.t())
    nan_mask = torch.isnan(pred_gene_gene)
    pred_gene_gene[nan_mask] = 1
    pred_gene_gene = torch.log(pred_gene_gene + 1e-6)
    nan_mask = torch.isnan(pred_gene_gene)
    pred_gene_gene[nan_mask] = 1

    pred_peak_peak = torch.mm(eta, eta.t())
    nan_mask = torch.isnan(pred_peak_peak)
    pred_peak_peak[nan_mask] = 1
    pred_peak_peak = torch.log(pred_peak_peak + 1e-6)
    nan_mask = torch.isnan(pred_peak_peak)
    pred_peak_peak[nan_mask] = 1

    if use_mlp:
        # print("using mlp")
        rho = mlp1(rho)
        eta = mlp2(eta)
    pred_RNA_tensor = decoder1(theta1, rho)
    pred_ATAC_tensor = decoder2(theta2, eta)

    """
    modify loss here
    """

    recon_loss1 = -(pred_RNA_tensor * RNA_tensor).sum(-1)
    recon_loss2 = -(pred_ATAC_tensor * ATAC_tensor).sum(-1)

    recon_loss3 = -(pred_gene_gene * gene_gene).sum(-1)
    recon_loss4 = -(pred_peak_peak * peak_peak).sum(-1)

    # print(f"recon_loss3: {recon_loss3.mean()}")
    # print(f"recon_loss4: {recon_loss4.mean()}")

    recon_loss = (recon_loss1 + recon_loss2).mean()
    # recon_loss += (recon_loss3 + recon_loss4).mean()
    kl_loss = kl_theta1 + kl_theta2

    # loss = recon_loss + kl_loss * kl_weight
    loss = recon_loss + kl_loss
    loss.backward()

    clamp_num = 2.0
    torch.nn.utils.clip_grad_norm_(encoder1.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(encoder2.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(gnn.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(mlp1.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(mlp2.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(decoder1.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(decoder2.parameters(), clamp_num)

    optimizer.step()

    return torch.sum(loss).item()


# get sample encoding theta from the trained encoder network
def get_theta_GNN(encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2, RNA_tensor_normalized,
                  ATAC_tensor_normalized, metric):
    encoder1.eval()
    encoder2.eval()
    gnn.eval()
    mlp1.eval()
    mlp2.eval()
    decoder1.eval()
    decoder2.eval()

    with torch.no_grad():
        mu1, log_sigma1, kl_theta1 = encoder1(RNA_tensor_normalized)
        mu2, log_sigma2, kl_theta2 = encoder2(ATAC_tensor_normalized)

        z1 = reparameterize(mu1, log_sigma1)
        theta1 = F.softmax(z1, dim=-1)

        z2 = reparameterize(mu2, log_sigma2)
        theta2 = F.softmax(z2, dim=-1)

        """
        change metric here, theta or mu
        """
        rate1 = 0.5
        rate2 = 0.5

        theta = None
        if metric == 'theta':
            theta = rate1 * theta1 + rate2 * theta2
        if metric == 'mu':
            theta = rate1 * mu1 + rate2 * mu2

        return theta, theta1, theta2


def get_beta_GNN(encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
                 RNA_tensor_normalized, ATAC_tensor_normalized,
                 feature_matrix, edge_index, use_mlp):
    encoder1.eval()
    encoder2.eval()
    gnn.eval()
    mlp1.eval()
    mlp2.eval()
    decoder1.eval()
    decoder2.eval()

    with torch.no_grad():
        mu1, log_sigma1, kl_theta1 = encoder1(RNA_tensor_normalized)
        mu2, log_sigma2, kl_theta2 = encoder2(ATAC_tensor_normalized)

        z1 = reparameterize(mu1, log_sigma1)
        theta1 = F.softmax(z1, dim=-1)

        z2 = reparameterize(mu2, log_sigma2)
        theta2 = F.softmax(z2, dim=-1)

        emb = gnn(feature_matrix, edge_index)
        new_emb = mlp1(emb)
        # num_of_peak x emb, num_of_gene x emb
        # eta, rho = split_tensor(emb, ATAC_tensor_normalized.shape[1])
        eta, rho = split_tensor(new_emb, ATAC_tensor_normalized.shape[1])

        pred_gene_gene = torch.mm(rho, rho.t())
        pred_gene_gene = torch.log(pred_gene_gene + 1e-6)

        pred_peak_peak = torch.mm(eta, eta.t())
        pred_peak_peak = torch.log(pred_peak_peak + 1e-6)
        if use_mlp:
            rho = mlp1(rho)
            eta = mlp2(eta)
        pred_RNA_tensor = decoder1(theta1, rho)
        pred_ATAC_tensor = decoder2(theta2, eta)

        beta1 = decoder1.get_beta()
        beta2 = decoder2.get_beta()

        return beta1, beta2


def train(encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
          optimizer, RNA_tensor, RNA_tensor_normalized,
          ATAC_tensor, ATAC_tensor_normalized, feature_matrix, edge_index,
          gene_gene, peak_peak, adata, adata2, metric, ari_freq, niter=100):
    perf = np.ndarray(shape=(niter, 2), dtype='float')
    num_of_ari = int(niter / ari_freq)
    ari_perf = np.ndarray(shape=(num_of_ari, 4), dtype='float')
    best_ari = 0
    best_theta = None
    best_beta_gene = None
    best_beta_peak = None
    # WRITE YOUR CODE HERE
    for i in range(niter):
        kl_weight = calc_weight(i, niter, 0, 1 / 3, 1e-2, 4)
        NELBO = train_one_epoch(encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2, optimizer, RNA_tensor,
                                RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized,
                                feature_matrix, edge_index, gene_gene, peak_peak, kl_weight)

        theta, theta_gene, theta_peak = get_theta_GNN(encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
                                                      RNA_tensor_normalized, ATAC_tensor_normalized, metric)
        beta_gene, beta_peak = get_beta_GNN(encoder1, encoder2, gnn, mlp1, mlp2, decoder1, decoder2,
                                            RNA_tensor_normalized, ATAC_tensor_normalized,
                                            feature_matrix, edge_index)
        perf[i, 0] = i
        perf[i, 1] = NELBO
        if i % ari_freq == 0:
            ari = evaluate_ari(theta.to('cpu'), adata)
            ari1 = evaluate_ari(theta_gene.to('cpu'), adata)
            ari2 = evaluate_ari(theta_peak.to('cpu'), adata)
            idx = (int)(i / ari_freq)
            ari_perf[idx, 0] = idx
            ari_perf[idx, 1] = ari
            ari_perf[idx, 2] = ari1
            ari_perf[idx, 3] = ari2
            print("iter: " + str(i) + " ari: " + str(ari))

            if best_ari < ari:
                best_ari = ari
                best_theta = theta
                best_beta_gene = beta_gene
                best_beta_peak = beta_peak
        else:
            if i % 100 == 0:
                print("iter: " + str(i))

    return encoder1, encoder2, gnn, decoder1, decoder2, perf, ari_perf, \
           best_ari, best_theta, best_beta_gene, best_beta_peak


def monitor_perf(perf, ari_perf, ari_freq, objective, path):
    niter = []
    ari_niter = []
    mse = []
    ari = []
    ari1 = []
    ari2 = []
    for i in range(len(perf)):
        niter.append(perf[i][0])
        mse.append(perf[i][1])
    for i in range(len(ari_perf)):
        ari_niter.append(ari_perf[i][0] * ari_freq)
        ari.append(ari_perf[i][1])
        ari1.append(ari_perf[i][2])
        ari2.append(ari_perf[i][3])
    if objective == "both":
        print(path + 'BOTH_ARI.png')
        plt.plot(ari_niter, ari, label="scGraphETM", color='red')
        # plt.plot(ari_niter, ari1, label="scETM_RNA", color='black')
        # plt.plot(ari_niter, ari2, label="scETM_ATAC", color='blue')
        plt.legend()
        # plt.title("ARI comparison on scGraphETM and scETM")
        plt.title("ARI of scGraphETM")
        plt.xlabel("iter")
        plt.ylabel("ARI")
        plt.ylim(0, 1)
        # plt.show()
        plt.savefig(path + 'BOTH_ARI.png')
    if objective == "NELBO":
        plt.plot(niter, mse)
        plt.xlabel("iter")
        plt.ylabel("NELBO")
        # plt.show()
        plt.savefig(path + 'NELBO.png')
    if objective == "ARI":
        plt.plot(ari_niter, ari)
        plt.xlabel("iter")
        plt.ylabel("ARI")
        plt.show()
    if objective == "ARI1":
        plt.plot(ari_niter, ari1)
        plt.xlabel("iter")
        plt.ylabel("ARI1")
        plt.show()
    if objective == "ARI2":
        plt.plot(ari_niter, ari2)
        plt.xlabel("iter")
        plt.ylabel("ARI2")
        plt.show()
    return


def process_data(rna_path, atac_path, device, num_of_cell,
                 num_of_gene, num_of_peak, emb_size, cor):
    scRNA_adata = ad.read_h5ad(rna_path)
    scATAC_adata = ad.read_h5ad(atac_path)

    print(scRNA_adata)
    print(scATAC_adata)

    scRNA_adata_copy = scRNA_adata.copy()
    sc.pp.normalize_total(scRNA_adata, target_sum=1e4)
    sc.pp.log1p(scRNA_adata)

    scATAC_adata_copy = scATAC_adata.copy()
    sc.pp.normalize_total(scATAC_adata, target_sum=1e4)
    sc.pp.log1p(scATAC_adata)
    sc.pp.highly_variable_genes(scATAC_adata)

    print(sum(scRNA_adata.var["highly_variable"]))
    print(sum(scATAC_adata.var["highly_variable"]))

    index1 = scRNA_adata.var['highly_variable'].values
    scRNA_adata = scRNA_adata[:, index1].copy()

    index2 = scATAC_adata.var['highly_variable'].values
    scATAC_adata = scATAC_adata[:, index2].copy()

    scRNA_1000 = scRNA_adata_copy[:, index1]
    scRNA_1000 = scRNA_1000.X[:num_of_cell, :num_of_gene].toarray()
    scRNA_1000_tensor = torch.from_numpy(scRNA_1000).to(device)

    scRNA_1000_tensor_normalized = (torch.from_numpy(scRNA_adata_copy.X[:num_of_cell, :num_of_gene].toarray())).to(
        device)
    scRNA_1000_mp_anndata = ad.AnnData(X=scRNA_adata_copy.X[:num_of_cell, :num_of_gene].toarray())
    scRNA_1000_mp_anndata.obs['Celltype'] = scRNA_adata_copy.obs['cell_type'].values[:num_of_cell]
    num_topics = len(scRNA_1000_mp_anndata.obs['Celltype'].values.unique())
    print(num_topics)

    scATAC_1000 = scATAC_adata_copy[:, index2]
    scATAC_1000 = scATAC_1000.X[:num_of_cell, :num_of_peak].toarray()
    scATAC_1000_tensor = torch.from_numpy(scATAC_1000).to(device)

    scATAC_1000_tensor_normalized = (torch.from_numpy(scATAC_adata_copy.X[:num_of_cell, :num_of_peak].toarray())).to(
        device)
    scATAC_1000_mp_anndata = ad.AnnData(X=scATAC_adata_copy.X[:num_of_cell, :num_of_peak].toarray())
    scATAC_1000_mp_anndata.obs['Celltype'] = scATAC_adata_copy.obs['cell_type'].values[:num_of_cell]
    num_topics2 = len(scATAC_1000_mp_anndata.obs['Celltype'].values.unique())
    print(num_topics2)

    """# **Generate peak-gene correlation graph**"""

    print("----------------------")
    print("Compute gene-gene correlation")
    gene_expression = scRNA_adata.X[:num_of_cell, :num_of_gene].toarray()
    if cor == 'pearson':
        correlation_matrix = np.corrcoef(gene_expression + 1e-6, rowvar=False)
    if cor == 'spearman':
        correlation_matrix = stats.spearmanr(gene_expression + 1e-6).correlation
    correlation_matrix_cleaned = np.nan_to_num(correlation_matrix, nan=0, posinf=1, neginf=-1)

    print(correlation_matrix_cleaned.shape)
    print(np.max(correlation_matrix_cleaned), np.min(correlation_matrix_cleaned))
    print(sum(correlation_matrix_cleaned))

    print("----------------------")
    print("Compute peak-peak correlation")
    peak_expression = scATAC_adata.X[:num_of_cell, :num_of_peak].toarray()
    if cor == 'pearson':
        correlation_matrix2 = np.corrcoef(peak_expression + 1e-6, rowvar=False)
    if cor == 'spearman':
        correlation_matrix2 = stats.spearmanr(peak_expression + 1e-6).correlation
    correlation_matrix2_cleaned = np.nan_to_num(correlation_matrix2, nan=0, posinf=1, neginf=-1)

    print(correlation_matrix2_cleaned.shape)
    print(np.max(correlation_matrix2_cleaned), np.min(correlation_matrix2_cleaned))
    print(sum(correlation_matrix2_cleaned))

    gene_pos_dic = {}
    for i in range(num_of_gene):
        gene_names = scRNA_adata.var_names[i]
        chrom = scRNA_adata.var["chrom"][i]
        chromStart = scRNA_adata.var["chromStart"][i]
        chromEnd = scRNA_adata.var["chromEnd"][i]
        gene_pos_dic[gene_names] = [chrom, chromStart, chromEnd]

    print(len(gene_pos_dic))

    peak_pos_dic = {}
    for i in range(num_of_peak):
        peak_names = scATAC_adata.var_names[i]
        chrom = scATAC_adata.var["chrom"][i]
        chromStart = scATAC_adata.var["chromStart"][i]
        chromEnd = scATAC_adata.var["chromEnd"][i]
        peak_pos_dic[peak_names] = [chrom, chromStart, chromEnd]

    print(len(peak_pos_dic))

    print("----------------------")
    print("Compute correlation matrix")
    cor_mat = torch.zeros(num_of_peak + num_of_gene, num_of_peak + num_of_gene)
    print(cor_mat.shape)
    for i, gene in enumerate(list(gene_pos_dic.keys())):
        for j, peak in enumerate(list(peak_pos_dic.keys())):
            gene_chrom = gene_pos_dic[gene][0]
            gene_start = gene_pos_dic[gene][1]
            gene_end = gene_pos_dic[gene][2]

            peak_chrom = peak_pos_dic[peak][0]
            peak_start = peak_pos_dic[peak][1]
            peak_end = peak_pos_dic[peak][2]

            if gene_chrom == peak_chrom and abs(gene_start - peak_start) <= 2000:
                cor_mat[num_of_peak + i, j] = 1
                cor_mat[j, num_of_peak + i] = 1

    for i, gene in enumerate(list(gene_pos_dic.keys())):
        for j, gene in enumerate(list(gene_pos_dic.keys())):
            gen_cor = correlation_matrix_cleaned[i, j]
            if gen_cor > 0.6:
                cor_mat[num_of_peak + i, num_of_peak + j] = 1

    for i, peak in enumerate(list(peak_pos_dic.keys())):
        for j, peak in enumerate(list(peak_pos_dic.keys())):
            peak_cor = correlation_matrix2_cleaned[i, j]
            if peak_cor > 0.6:
                cor_mat[i, j] = 1
    print("finish cor_mat")
    torch.sum(cor_mat)

    from scipy.sparse import csr_matrix

    sparse_cor_mat = csr_matrix(cor_mat.cpu())
    # print(sparse_cor_mat)
    rows, cols = sparse_cor_mat.nonzero()
    edge_index = torch.tensor([rows, cols], dtype=torch.long).to(device)

    feature_matrix = torch.randn((num_of_peak + num_of_gene, emb_size)).to(device)
    correlation_matrix_cleaned = torch.tensor(correlation_matrix_cleaned, dtype=torch.float32).to(device)
    correlation_matrix2_cleaned = torch.tensor(correlation_matrix2_cleaned, dtype=torch.float32).to(device)

    return feature_matrix, edge_index, \
           scRNA_adata, scATAC_adata, \
           scRNA_1000_tensor, scRNA_1000_tensor_normalized, \
           scATAC_1000_tensor, scATAC_1000_tensor_normalized, \
           scRNA_1000_mp_anndata, scATAC_1000_mp_anndata, \
           correlation_matrix_cleaned, correlation_matrix2_cleaned


def prior_expert(size, use_cuda=False):
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


def experts(mu, logsigma, eps=1e-8):
    var = torch.exp(2 * logsigma) + eps
    T = 1. / (var + eps)
    pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
    pd_var = 1. / torch.sum(T, dim=0)
    pd_logsigma = 0.5 * torch.log(pd_var + eps)
    return pd_mu, pd_logsigma


def train_one_epoch_pog(encoder1, encoder2, gnn, mlp1, mlp2, pog_decoder, optimizer,
                        RNA_tensor, RNA_tensor_normalized, ATAC_tensor, ATAC_tensor_normalized,
                        feature_matrix, edge_index, gene_gene, peak_peak, kl_weight, use_mlp):
    encoder1.train()
    encoder2.train()
    gnn.train()
    mlp1.train()
    mlp2.train()
    pog_decoder.train()

    optimizer.zero_grad()

    encoder1.zero_grad()
    encoder2.zero_grad()
    gnn.zero_grad()
    mlp1.zero_grad()
    mlp2.zero_grad()
    pog_decoder.zero_grad()

    mu1, log_sigma1, kl_theta1 = encoder1(RNA_tensor_normalized)
    mu2, log_sigma2, kl_theta2 = encoder2(ATAC_tensor_normalized)
    mu_prior, logsigma_prior = prior_expert((1, RNA_tensor_normalized.shape[0], mu1.shape[1]), use_cuda=True)

    # Mu = torch.cat((mu_prior, mu1.unsqueeze(0), mu2.unsqueeze(0)), dim=0)
    # Log_sigma = torch.cat((logsigma_prior, log_sigma1.unsqueeze(0), log_sigma2.unsqueeze(0)), dim=0)

    Mu = torch.cat((mu1.unsqueeze(0), mu2.unsqueeze(0)), dim=0)
    Log_sigma = torch.cat((log_sigma1.unsqueeze(0), log_sigma2.unsqueeze(0)), dim=0)

    mu, log_sigma = experts(Mu, Log_sigma)

    Theta = F.softmax(reparameterize(mu, log_sigma), dim=-1)

    emb = gnn(feature_matrix, edge_index)
    new_emb = mlp1(emb)
    # num_of_peak x emb, num_of_gene x emb
    # eta, rho = split_tensor(emb, ATAC_tensor_normalized.shape[1])
    eta, rho = split_tensor(new_emb, ATAC_tensor_normalized.shape[1])

    pred_gene_gene = torch.mm(rho, rho.t())
    nan_mask = torch.isnan(pred_gene_gene)
    pred_gene_gene[nan_mask] = 1
    pred_gene_gene = torch.log(pred_gene_gene + 1e-6)
    nan_mask = torch.isnan(pred_gene_gene)
    pred_gene_gene[nan_mask] = 1

    pred_peak_peak = torch.mm(eta, eta.t())
    nan_mask = torch.isnan(pred_peak_peak)
    pred_peak_peak[nan_mask] = 1
    pred_peak_peak = torch.log(pred_peak_peak + 1e-6)
    nan_mask = torch.isnan(pred_peak_peak)
    pred_peak_peak[nan_mask] = 1

    if use_mlp:
        rho = mlp1(rho)
        eta = mlp2(eta)
    pred_RNA_tensor, pred_ATAC_tensor = pog_decoder(Theta, rho, eta)

    """
    modify loss here
    """

    recon_loss1 = -(pred_RNA_tensor * RNA_tensor).sum(-1)
    recon_loss2 = -(pred_ATAC_tensor * ATAC_tensor).sum(-1)

    recon_loss3 = -(pred_gene_gene * gene_gene).sum(-1)
    recon_loss4 = -(pred_peak_peak * peak_peak).sum(-1)

    # print(f"recon_loss3: {recon_loss3.mean()}")
    # print(f"recon_loss4: {recon_loss4.mean()}")

    recon_loss = (recon_loss1 + recon_loss2).mean()
    # recon_loss += (recon_loss3 + recon_loss4).mean()
    kl_loss = kl_theta1 + kl_theta2

    # loss = recon_loss + kl_loss * kl_weight
    loss = recon_loss + kl_loss
    loss.backward()

    clamp_num = 2.0
    torch.nn.utils.clip_grad_norm_(encoder1.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(encoder2.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(gnn.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(mlp1.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(mlp2.parameters(), clamp_num)
    torch.nn.utils.clip_grad_norm_(pog_decoder.parameters(), clamp_num)

    optimizer.step()

    return torch.sum(loss).item()


def get_theta_GNN_pog(encoder1, encoder2, gnn, mlp1, mlp2, pog_decoder, RNA_tensor_normalized,
                      ATAC_tensor_normalized, metric):
    encoder1.eval()
    encoder2.eval()
    gnn.eval()
    mlp1.eval()
    mlp2.eval()
    pog_decoder.eval()

    with torch.no_grad():
        mu1, log_sigma1, kl_theta1 = encoder1(RNA_tensor_normalized)
        mu2, log_sigma2, kl_theta2 = encoder2(ATAC_tensor_normalized)

        mu_prior, logsigma_prior = prior_expert((1, RNA_tensor_normalized.shape[0], mu1.shape[1]), use_cuda=True)

        # Mu = torch.cat((mu_prior, mu1.unsqueeze(0), mu2.unsqueeze(0)), dim=0)
        # Log_sigma = torch.cat((logsigma_prior, log_sigma1.unsqueeze(0), log_sigma2.unsqueeze(0)), dim=0)

        Mu = torch.cat((mu1.unsqueeze(0), mu2.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((log_sigma1.unsqueeze(0), log_sigma2.unsqueeze(0)), dim=0)

        mu, log_sigma = experts(Mu, Log_sigma)

        Theta = F.softmax(reparameterize(mu, log_sigma), dim=-1)

        """
        change metric here, theta or mu
        """
        rate1 = 0.5
        rate2 = 0.5

        theta = None
        if metric == 'theta':
            theta = Theta
        if metric == 'mu':
            theta = mu

        return theta


def get_beta_GNN_pog(encoder1, encoder2, gnn, mlp1, mlp2, pog_decoder,
                     RNA_tensor_normalized, ATAC_tensor_normalized,
                     feature_matrix, edge_index, use_mlp):
    encoder1.eval()
    encoder2.eval()
    gnn.eval()
    mlp1.eval()
    mlp2.eval()
    pog_decoder.eval()

    with torch.no_grad():
        mu1, log_sigma1, kl_theta1 = encoder1(RNA_tensor_normalized)
        mu2, log_sigma2, kl_theta2 = encoder2(ATAC_tensor_normalized)

        mu_prior, logsigma_prior = prior_expert((1, RNA_tensor_normalized.shape[0], mu1.shape[1]), use_cuda=True)

        # Mu = torch.cat((mu_prior, mu1.unsqueeze(0), mu2.unsqueeze(0)), dim=0)
        # Log_sigma = torch.cat((logsigma_prior, log_sigma1.unsqueeze(0), log_sigma2.unsqueeze(0)), dim=0)

        Mu = torch.cat((mu1.unsqueeze(0), mu2.unsqueeze(0)), dim=0)
        Log_sigma = torch.cat((log_sigma1.unsqueeze(0), log_sigma2.unsqueeze(0)), dim=0)

        mu, log_sigma = experts(Mu, Log_sigma)

        Theta = F.softmax(reparameterize(mu, log_sigma), dim=-1)

        emb = gnn(feature_matrix, edge_index)
        new_emb = mlp1(emb)
        # num_of_peak x emb, num_of_gene x emb
        # eta, rho = split_tensor(emb, ATAC_tensor_normalized.shape[1])
        eta, rho = split_tensor(new_emb, ATAC_tensor_normalized.shape[1])

        pred_gene_gene = torch.mm(rho, rho.t())
        pred_gene_gene = torch.log(pred_gene_gene + 1e-6)

        pred_peak_peak = torch.mm(eta, eta.t())
        pred_peak_peak = torch.log(pred_peak_peak + 1e-6)

        if use_mlp:
            rho = mlp1(rho)
            eta = mlp2(eta)

        pred_RNA_tensor = pog_decoder(Theta, rho, eta)

        beta1, beta2 = pog_decoder.get_beta()

        return beta1, beta2
