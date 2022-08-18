import torch
import numpy as np
import torch.nn as nn
import torch.functional as F


class GaussianVariational(nn.Module):
    def __init__(self, mu, rho):
        super().__init__()

        self.mu = mu
        self.rho = rho
        self.register_buffer('eps_w', torch.Tensor(self.mu.shape).cuda())
        self.sigma = None
        self.w = None
        self.pi = np.pi  # 3.1415
        # self.normal = torch.distributions.Normal(0, 1)

    def sample(self):
        # 重参数技巧，使其可导
        self.eps_w.data.normal_()  #
        self.sigma = torch.log1p(torch.exp(self.rho))
        # print(self.mu)
        # print(self.sigma)
        # print(self.eps_w)
        self.w = self.mu + self.sigma * self.eps_w
        return self.w

    def log_posterior(self, w=None):

        assert (self.w is not None)
        if w is None:
            w = self.w

        log_sqrt2pi = np.log(np.sqrt(2 * self.pi))  # 0.918，就是常数项
        # log_posteriors = -log_sqrt2pi - torch.log(self.sigma) - (((w - self.mu) ** 2) / (2 * self.sigma ** 2)) - 0.5
        log_posteriors = -log_sqrt2pi - torch.log(self.sigma + 1e-10) - (((w - self.mu) ** 2) / (2 * self.sigma ** 2)) - 0.5
        posteriors = torch.exp(log_posteriors)
        # return posteriors, log_posteriors     # ,
        return log_posteriors.sum()


class ScaleMixturePrior(nn.Module):
    def __init__(self,
                 pi,
                 sigma1,
                 sigma2,
                 dist=None):
        super().__init__()

        if (dist is None):
            self.pi = pi
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            self.dist1 = torch.distributions.Normal(0.3, sigma1)
            self.dist2 = torch.distributions.Normal(0.1, sigma2)

        if (dist is not None):
            self.pi = 1
            self.dist1 = dist
            self.dist2 = None

    def log_prior(self, w):
        log_prior_pdf = torch.tensor(0.).cuda()
        _prior_pdf = torch.ones(14, 8).cuda()
        n = 2
        start = 0
        for i in range(4):
            end = start + n
            temp_weights = w[start:end]
            temp_prob_n1 = torch.exp(self.dist1.log_prob(temp_weights))
            temp_prob_n2 = torch.exp(self.dist2.log_prob(temp_weights))
            temp_prior_pdf = (self.pi * temp_prob_n1 + (1 - self.pi) * temp_prob_n2)
            temp_log_prior_pdf = (torch.log(temp_prior_pdf + 1e-10) - 0.5).sum()

            _prior_pdf[start:end] = torch.log(temp_prior_pdf + 1e-10)

            log_prior_pdf += temp_log_prior_pdf
            start = end
            n += 1

        # =====================================================
        # prob_n1 = torch.exp(self.dist1.log_prob(w))
        #
        # if self.dist2 is not None:
        #     prob_n2 = torch.exp(self.dist2.log_prob(w))
        # if self.dist2 is None:
        #     prob_n2 = 0
        #
        # prior_pdf = (self.pi * prob_n1 + (1 - self.pi) * prob_n2)
        # ======================================================
        # return _prior_pdf    #, log_prior_pdf            # (torch.log(prior_pdf) - 0.5).sum()
        return log_prior_pdf


