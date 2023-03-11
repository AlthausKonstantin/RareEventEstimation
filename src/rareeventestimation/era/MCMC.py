import numpy as np
import scipy as sp

from scipy.stats import norm

from rareeventestimation.era.EMvMFNM import (
    EMvMFNM,
    lognakagamipdf,
    logvMFpdf,
    logsumexp,
)
from rareeventestimation.era.EMvMFNM import vMFNM_sample
import sys


class MCMC:
    def __init__(
        self,
        uk,
        gk_current,
        wnork,
        nchain,
        burn,
        limit_state_fun_current,
        sigma,
        xi=0,
        gk_fine=[],
        limit_state_fun_fine=id,
        dim_last=None,
        mcmc_type="aCS",
        k_init=1,
    ):
        self.uk = uk
        self.gk_current = gk_current
        self.gk_fine = gk_fine

        if len(gk_fine) == 0:
            self.gk_fine = self.gk_current

        self.wnork = wnork
        self.limit_state_fun_current = limit_state_fun_current
        self.limit_state_fun_fine = limit_state_fun_fine
        self.nchain = nchain
        self.sigma = sigma
        self.xi = xi

        self.dim = np.size(uk, 1)
        self.dim_last = dim_last
        self.nsamlev = np.size(uk, 0)
        self.p = self.nchain / self.nsamlev
        self.burn = burn
        self.lenchain = int(self.nsamlev / self.nchain)
        self.alphak = np.zeros([self.nchain])

        self.sigmafk = None

        self.mcmc_type = mcmc_type
        self.k_init = k_init
        self.k_final = k_init

        self.mu_vMFNM = np.zeros(self.dim)
        self.alpha_em = 1

    def generate_seeds(self):
        ind = np.random.choice(range(self.nsamlev), self.nchain, True, self.wnork)
        uk0 = self.uk[ind, :]
        gk0_current = self.gk_current[ind]
        gk0_fine = self.gk_fine[ind]

        return [gk0_current, gk0_fine, uk0]

    def calculate_sigmaf(self, opc):
        if opc == "a":  # 1a. sigma = ones(n,1)
            sigmaf = 1
        elif opc == "b":  # 1b. sigma = sigma_hat (sample standard deviations)
            muf = np.mean(np.repmat(self.wnork, self.dim, 1) * self.uk, 1)
            sigmaf = np.zeros((1, self.dim))
            for k in range(self.nsamlev):
                sigmaf = sigmaf + self.wnork[k] * (self.uk[k, :] - muf) ** 2
        else:
            raise RuntimeError("Choose a or b")

        return sigmaf

    def mcmc_procedure_tempering(self):
        if self.mcmc_type == "aCS":
            self.mcmc_procedure_tempering_acs()
        elif self.mcmc_type == "vMFNM":
            self.mcmc_procedure_tempering_vmfnm()
        else:
            print("Invalid MCMC type")
            sys.exit()

    def mcmc_procedure_tempering_acs(self):
        # parameters for adaptive MCMC
        opc = "a"
        adapchains = int(np.ceil(100 * self.nchain / self.nsamlev))
        adapflag = 1
        lam = 0.6

        [gk0, gk0_fine, uk0] = MCMC.generate_seeds(self)
        sigmaf = MCMC.calculate_sigmaf(self, opc)

        # compute parameter rho
        self.sigmafk = min(lam * sigmaf, 1)
        rhok = np.sqrt(1 - self.sigmafk**2)
        counta = 0
        count = 0

        # initialize chain acceptance rate
        self.gk_current = np.zeros(
            [self.nsamlev + self.burn]
        )  # delete previous samples
        self.uk = np.zeros(
            [self.nsamlev + self.burn, self.dim]
        )  # delete previous samples

        for k in range(self.nchain):
            # set seed for chain
            u0 = uk0[k, :]
            g0 = gk0[k]
            for i in range(self.lenchain + self.burn):
                if i == self.burn:
                    count = count - self.burn

                # get candidate sample from conditional normal distribution
                ucand = np.random.normal(loc=rhok * u0, scale=self.sigmafk)

                # Evaluate limit-state function
                gcand = self.limit_state_fun_current(ucand)

                # compute acceptance probability
                acceptance_rate = norm.cdf(-gcand / self.sigma) / norm.cdf(
                    -g0 / self.sigma
                )
                alpha = min(1, acceptance_rate)
                self.alphak[k] = self.alphak[k] + alpha / (self.lenchain + self.burn)

                # check if sample is accepted
                uhelp = sp.stats.uniform.rvs()
                if uhelp <= alpha:
                    self.uk[count, :] = ucand
                    self.gk_current[count] = gcand
                    u0 = ucand
                    g0 = gcand
                else:
                    self.uk[count, :] = u0
                    self.gk_current[count] = g0

                count = count + 1

            # adapt the chain correlation
            if adapflag == 1:
                # check whether to adapt now
                if (k + 1) % adapchains == 0:
                    # mean acceptance rate of last adap_chains
                    alpha_mu = np.mean(self.alphak[k - adapchains + 1 : k + 1])
                    counta = counta + 1
                    gamma = counta ** (-0.5)
                    lam = min(np.exp(np.log(lam) + gamma * (alpha_mu - 0.44)), 1)

                    # compute parameter rho
                    self.sigmafk = min(lam * sigmaf, 1)
                    rhok = np.sqrt(1 - self.sigmafk**2)

        self.uk = self.uk[: self.nsamlev, :]
        self.gk_current = self.gk_current[: self.nsamlev]

    def mcmc_procedure_tempering_vmfnm(self):
        [gk0, gk0_next, uk0] = MCMC.generate_seeds(self)

        count = 0

        [mu_vMFNM, kappa, m, omega, alpha_em] = EMvMFNM(
            self.uk.T, self.wnork, self.k_init
        )
        mu_vMFNM = mu_vMFNM.T

        self.gk_current = np.zeros(
            [self.nsamlev + self.burn]
        )  # delete previous samples
        self.uk = np.zeros(
            [self.nsamlev + self.burn, self.dim]
        )  # delete previous samples

        ucandidates = vMFNM_sample(
            mu_vMFNM,
            kappa,
            omega,
            m,
            alpha_em,
            self.nchain * (self.lenchain + self.burn),
        )
        z = 0

        for k in range(self.nchain):
            # set seed for chain
            u0 = uk0[k, :]
            g0 = gk0[k]

            for i in range(self.lenchain + self.burn):
                if i == self.burn:
                    count = count - self.burn

                # get candidate sample from conditional normal distribution
                ucand = ucandidates[z, :]

                # Evaluate limit-state function
                gcand = self.limit_state_fun_current(ucand)

                acceptance_rate = (
                    sp.stats.norm.cdf(-gcand / self.sigma)
                    / sp.stats.norm.cdf(-g0 / self.sigma)
                    * np.exp(
                        -likelihood_ratio_log(
                            u0.reshape((1, self.dim)),
                            mu_vMFNM,
                            kappa,
                            omega,
                            m,
                            alpha_em,
                        )[:]
                        + likelihood_ratio_log(
                            ucand.reshape((1, self.dim)),
                            mu_vMFNM,
                            kappa,
                            omega,
                            m,
                            alpha_em,
                        )[:]
                    )
                )

                alpha = min(1, acceptance_rate)
                self.alphak[k] = self.alphak[k] + alpha / (self.lenchain + self.burn)

                # check if sample is accepted
                uhelp = sp.stats.uniform.rvs()
                if uhelp <= alpha:
                    self.uk[count, :] = ucand
                    self.gk_current[count] = gcand
                    u0 = ucand
                    g0 = gcand
                else:
                    self.uk[count, :] = u0
                    self.gk_current[count] = g0

                count = count + 1
                z = z + 1
        self.k_final = len(alpha_em)
        self.uk = self.uk[: self.nsamlev, :]
        self.gk_current = self.gk_current[: self.nsamlev]
        self.mu_vMFNM = mu_vMFNM
        self.alpha_em = alpha_em


def likelihood_ratio_log(X, mu, kappa, omega, m, alpha):
    k = len(alpha)
    [N, dim] = np.shape(X)
    R = np.sqrt(np.sum(X * X, axis=1)).reshape(-1, 1)
    if k == 1:
        # log pdf of vMF distribution
        logpdf_vMF = logvMFpdf((X / R).T, mu.T, kappa).T
        # log pdf of Nakagami distribution
        logpdf_N = lognakagamipdf(R, m, omega)
        # log pdf of weighted combined distribution
        h_log = logpdf_vMF + logpdf_N
    else:
        logpdf_vMF = np.zeros([N, k])
        logpdf_N = np.zeros([N, k])
        h_log = np.zeros([N, k])

        # log pdf of distributions in the mixture
        for p in range(k):
            # log pdf of vMF distribution
            logpdf_vMF[:, p] = logvMFpdf((X / R).T, mu[p, :].T, kappa[p]).squeeze()
            # log pdf of Nakagami distribution
            logpdf_N[:, p] = lognakagamipdf(R, m[:, p], omega[:, p]).squeeze()
            # log pdf of weighted combined distribution
            h_log[:, p] = logpdf_vMF[:, p] + logpdf_N[:, p] + np.log(alpha[p])

        # mixture log pdf
        h_log = logsumexp(h_log, 1)

    # unit hypersphere uniform log pdf
    A = np.log(dim) + np.log(np.pi ** (dim / 2)) - sp.special.gammaln(dim / 2 + 1)
    f_u = -A

    # chi log pdf
    f_chi = (
        np.log(2) * (1 - dim / 2)
        + np.log(R) * (dim - 1)
        - 0.5 * R**2
        - sp.special.gammaln(dim / 2)
    )

    # logpdf of the standard distribution (uniform combined with chi distribution)
    f_log = f_u + f_chi

    W_log = f_log - h_log

    return W_log
