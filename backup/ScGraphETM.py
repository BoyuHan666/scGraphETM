import torch
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable


def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def reparameterize(mu, log_sigma):
    std = torch.exp(log_sigma)
    eps = torch.randn_like(std)
    return eps * std + mu


def get_kl(mu, log_sigma):
    log_sigma = 2 * log_sigma
    return -0.5 * (1 + log_sigma - mu.pow(2) - log_sigma.exp()).sum(-1)


# two GNN to embed the gene x gene matrix and peak x peak matrix
def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).
    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


def experts(mu, logsigma, eps=1e-8):
    var = torch.exp(2 * logsigma) + eps
    # precision of i-th Gaussian expert at point x
    T = 1. / (var + eps)
    pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
    pd_var = 1. / torch.sum(T, dim=0)
    pd_logsigma = 0.5 * torch.log(pd_var + eps)
    return pd_mu, pd_logsigma


class ScGraphETM1(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder_mod1, decoder_mod2, gnn_mod1, gnn_mod2):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder_mod1 = decoder_mod1
        self.decoder_mod2 = decoder_mod2
        self.gnn_mod1 = gnn_mod1
        self.gnn_mod2 = gnn_mod2
        parameters = [{'params': encoder_mod1.parameters()},
                      {'params': encoder_mod2.parameters()},
                      {'params': decoder_mod1.parameters()},
                      {'params': decoder_mod2.parameters()},
                      {'params': gnn_mod1.parameters()},
                      {'params': gnn_mod2.parameters()}
                      ]
        self.optimizer = optim.Adam(parameters, lr=0.001)

    def train(self, X_mod1, X_mod2, A_mod1, edge_index1, A_mod2, edge_index2):
        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder_mod1, True)
        toogle_grad(self.decoder_mod2, True)
        toogle_grad(self.gnn_mod1, True)
        toogle_grad(self.gnn_mod2, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder_mod1.train()
        self.decoder_mod2.train()
        self.gnn_mod1.train()
        self.gnn_mod2.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1 = self.encoder_mod1(X_mod1)
        mu_mod2, log_sigma_mod2 = self.encoder_mod2(X_mod2)
        # mod_num x emb
        mod1_emb = self.gnn_mod1(A_mod1, edge_index1)
        mod2_emb = self.gnn_mod2(A_mod2, edge_index2)

        rho1 = mod1_emb.T
        rho2 = mod2_emb.T

        z1 = reparameterize(mu_mod1, log_sigma_mod1)
        theta_mod1 = F.softmax(z1, dim=-1)

        z2 = reparameterize(mu_mod2, log_sigma_mod2)
        theta_mod2 = F.softmax(z2, dim=-1)

        # cell x mod_num
        X_mod1_hat = self.decoder_mod1(theta_mod1, rho1)
        X_mod2_hat = self.decoder_mod2(theta_mod2, rho2)

        # mod_num x mod_num
        A_mod1_hat = F.sigmoid(torch.mm(mod1_emb, mod1_emb.T))
        A_mod2_hat = F.sigmoid(torch.mm(mod2_emb, mod2_emb.T))

        nll1_mod1 = (-X_mod1_hat * X_mod1).sum(-1).mean()
        nll1_mod2 = (-X_mod2_hat * X_mod2).sum(-1).mean()

        nll2_mod1 = (-A_mod1_hat * A_mod1).sum(-1).mean()
        nll2_mod2 = (-A_mod2_hat * A_mod2).sum(-1).mean()

        kl_mod1 = get_kl(mu_mod1, log_sigma_mod1).mean()
        kl_mod2 = get_kl(mu_mod2, log_sigma_mod2).mean()

        loss = nll1_mod1 + nll1_mod2 + nll2_mod1 + nll2_mod2 + kl_mod1 + kl_mod2
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.gnn_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.gnn_mod2.parameters(), 50)

        self.optimizer.step()
        return loss.item(), nll1_mod1.item(), nll1_mod2.item(), nll2_mod1.item(), nll2_mod2.item(), kl_mod1.item(), kl_mod2.item()

    def get_pog_embed(self, X_mod1, X_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder_mod1.eval()
        self.decoder_mod2.eval()
        self.gnn_mod1.eval()
        self.gnn_mod2.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(X_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(X_mod2)
            mu_prior, logsigma_prior = prior_expert((1, X_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = experts(Mu, Log_sigma)

        return mu, mu_mod1, mu_mod2

    def get_theta(self, X_mod1, X_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder_mod1.eval()
        self.decoder_mod2.eval()
        self.gnn_mod1.eval()
        self.gnn_mod2.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(X_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(X_mod2)

            z_d = reparameterize(mu_mod1, log_sigma_mod1)
            theta_mod1 = F.softmax(z_d, dim=-1)

            z_j = reparameterize(mu_mod2, log_sigma_mod2)
            theta_mod2 = F.softmax(z_j, dim=-1)

            theta_mu_mod1 = F.softmax(mu_mod1, dim=-1)
            theta_mu_mod2 = F.softmax(mu_mod2, dim=-1)

            theta = 0.5 * (theta_mod1 + theta_mod2)
            theta_mu = 0.5 * (theta_mu_mod1 + theta_mu_mod2)

        return theta, theta_mu

    def get_pog_theta(self, X_mod1, X_mod2):
        self.encoder_mod1.eval()
        self.encoder_mod2.eval()
        self.decoder_mod1.eval()
        self.decoder_mod2.eval()
        self.gnn_mod1.eval()
        self.gnn_mod2.eval()

        with torch.no_grad():
            mu_mod1, log_sigma_mod1 = self.encoder_mod1(X_mod1)
            mu_mod2, log_sigma_mod2 = self.encoder_mod2(X_mod2)
            mu_prior, logsigma_prior = prior_expert((1, X_mod1.shape[0], mu_mod1.shape[1]), use_cuda=False)

            Mu = torch.cat((mu_prior, mu_mod1.unsqueeze(0), mu_mod2.unsqueeze(0)), dim=0)
            Log_sigma = torch.cat((logsigma_prior, log_sigma_mod1.unsqueeze(0), log_sigma_mod2.unsqueeze(0)), dim=0)

            mu, log_sigma = experts(Mu, Log_sigma)

            z_d = reparameterize(mu, log_sigma)
            theta = F.softmax(z_d, dim=-1)

            theta_mu = F.softmax(mu, dim=-1)

        return theta, theta_mu


# one GNN to embed the whole peak+gene x peak+gene matrix
class ScGraphETM2(object):
    def __init__(self, encoder_mod1, encoder_mod2, decoder_mod1, decoder_mod2, gnn):
        self.encoder_mod1 = encoder_mod1
        self.encoder_mod2 = encoder_mod2
        self.decoder_mod1 = decoder_mod1
        self.decoder_mod2 = decoder_mod2
        self.gnn = gnn
        parameters = [{'params': encoder_mod1.parameters()},
                      {'params': encoder_mod2.parameters()},
                      {'params': decoder_mod1.parameters()},
                      {'params': decoder_mod2.parameters()},
                      {'params': gnn.parameters()}
                      ]
        self.optimizer = optim.Adam(parameters, lr=0.001)

    def train(self, X_mod1, X_mod2, A, edge_index):
        toogle_grad(self.encoder_mod1, True)
        toogle_grad(self.encoder_mod2, True)
        toogle_grad(self.decoder_mod1, True)
        toogle_grad(self.decoder_mod2, True)
        toogle_grad(self.gnn, True)

        self.encoder_mod1.train()
        self.encoder_mod2.train()
        self.decoder_mod1.train()
        self.decoder_mod2.train()
        self.gnn.train()

        self.optimizer.zero_grad()

        mu_mod1, log_sigma_mod1 = self.encoder_mod1(X_mod1)
        mu_mod2, log_sigma_mod2 = self.encoder_mod2(X_mod2)
        # mod_num x emb
        emb = self.gnn(A, edge_index)

        rho = emb.T

        z1 = reparameterize(mu_mod1, log_sigma_mod1)
        theta_mod1 = F.softmax(z1, dim=-1)

        z2 = reparameterize(mu_mod2, log_sigma_mod2)
        theta_mod2 = F.softmax(z2, dim=-1)

        # cell x mod_num
        X_mod1_hat = self.decoder_mod1(theta_mod1, rho)
        X_mod2_hat = self.decoder_mod1(theta_mod2, rho)

        # mod_num x mod_num
        A_hat = F.sigmoid(torch.mm(emb, emb.T))

        nll1_mod1 = (-X_mod1_hat * X_mod1).sum(-1).mean()
        nll1_mod2 = (-X_mod2_hat * X_mod2).sum(-1).mean()

        nll2 = (-A_hat * A).sum(-1).mean()

        kl_mod1 = get_kl(mu_mod1, log_sigma_mod1).mean()
        kl_mod2 = get_kl(mu_mod2, log_sigma_mod2).mean()

        loss = nll1_mod1 + nll1_mod2 + nll2 + kl_mod1 + kl_mod2
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.encoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder_mod1.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder_mod2.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), 50)

        self.optimizer.step()
        return loss.item(), nll1_mod1.item(), nll1_mod2.item(), nll2.item(), kl_mod1.item(), kl_mod2.item()
