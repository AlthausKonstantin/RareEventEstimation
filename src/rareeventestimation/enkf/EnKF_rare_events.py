import numpy as np
from scipy.optimize import fminbound
from scipy.stats import multivariate_normal
from rareeventestimation.era.EMGM import EMGM, GM_sample, h_calc
from rareeventestimation.era.EMvMFNM import EMvMFNM, vMFNM_sample
from rareeventestimation.era.CEIS_vMFNM import likelihood_ratio_log
from scipy.spatial.distance import pdist, squareform


class EnKF_rare_events:
    def __init__(
        self,
        g,
        y_data,
        niter_enkf,
        ensemble_size,
        parameter_dim,
        tar_cov,
        space_dim,
        local_covariances,
        g_original,
        k_init,
        mixture_model,
        gamma_weights=1,
        tar_cov_pf=0.05,
    ):
        """Define the parameters that a relevant for the EnKF"""
        self.lsf = g
        self.y_data = y_data
        self.data_dim = len(y_data)

        self.niter_enKF = niter_enkf
        self.ensemble_size = ensemble_size
        self.parameter_dim = parameter_dim

        self.tar_cov = tar_cov
        self.tar_cov_pf = tar_cov_pf
        self.space_dim = space_dim
        self.local_covariances = local_covariances

        self.iter = 0
        self.pf = 0
        self.pf_form_mean = 0
        self.pf_form_smallest_distance = 0

        self.number_fun_eval = 0
        self.computational_cost = 0
        self.rel_error = list()
        self.delta_sigma = 0

        self.uk = np.random.normal(0, 1, (self.ensemble_size, self.parameter_dim))
        self.uk_initial = self.uk.copy()
        self.uk_save = list()
        self.uk_save.append(self.uk)
        self.sigma_save = list()

        # initial samples
        self.uk_form_mean = np.zeros(self.parameter_dim)
        self.uk_form_smallest_distance = np.zeros(self.parameter_dim)
        self.gk = np.zeros((self.ensemble_size, self.data_dim))

        self.posterior_mean = np.zeros((self.niter_enKF, parameter_dim))
        self.posterior_lsf_mean = np.zeros((self.niter_enKF, self.data_dim))
        self.sigmak_list = np.zeros(self.niter_enKF)

        """parameters for specific EnkF method"""
        self.ensemble_mean = []
        self.lsf_mean = []

        """parameters for specific EnkF method"""
        self.gamma_weights = gamma_weights

        self.g_original = g_original
        self.k_init = k_init

        self.tempering_finished = False
        self.number_tempering = 0

        self.mixture_model = mixture_model
        self.uk_fitted = np.zeros((self.ensemble_size, self.parameter_dim))
        self.gk_fitted = np.zeros(self.ensemble_size)

        self.mean_k = np.zeros((self.niter_enKF + 1, self.parameter_dim))
        self.mean_k[0, :] = np.mean(self.uk, axis=0)

    def perform_iteration(self):
        EnKF_rare_events.evaluate_lsf(self)

        if self.local_covariances:
            EnKF_rare_events.iteration_local_covariances(self)
        else:
            EnKF_rare_events.iteration_global_covariances(self)

    def iteration_global_covariances(self):
        """Iteration"""
        while self.iter < self.niter_enKF:
            EnKF_rare_events.perform_tempering(self)

            EnKF_rare_events.evaluate_lsf(self)

            EnKF_rare_events.update_posterior_and_error(self)

            self.iter = self.iter + 1

            if self.tempering_finished:
                break

    def iteration_local_covariances(self):
        """Iteration"""
        while self.iter < self.niter_enKF:
            EnKF_rare_events.perform_tempering_local(self)

            EnKF_rare_events.evaluate_lsf(self)

            EnKF_rare_events.update_posterior_and_error(self)

            self.iter = self.iter + 1

            if self.tempering_finished:
                break

    def perform_tempering(self):
        self.number_tempering = self.number_tempering + 1

        """Update limit state function values and calculate new sigma"""
        EnKF_rare_events.update_g_and_sigma(self)

        # print("perform Tempering; current temperature", self.sigmak_list[self.iter])

        "Calculate the ensemble and network mean"
        self.ensemble_mean = np.mean(self.uk, 0).reshape((1, self.parameter_dim))
        self.lsf_mean = np.mean(self.gk, 0).reshape((1, self.data_dim))

        "Calculate the matrices C_up and C_pp"
        c_up = EnKF_rare_events.calculate_c_up(
            self, self.uk, self.gk, self.ensemble_mean, self.lsf_mean
        )
        c_pp = EnKF_rare_events.calculate_c_pp(self, self.gk, self.lsf_mean)

        "sample the noise for the data"
        noise = np.random.normal(
            0, np.sqrt(1 / self.delta_sigma), (self.data_dim, self.ensemble_size)
        )

        if self.data_dim == 1:
            y_noisy = self.y_data + noise.T
            self.uk = (
                self.uk
                + np.outer(
                    c_up, 1 / (c_pp + 1 / self.delta_sigma) * (y_noisy - self.gk)
                ).T
            )
        else:
            "Update the parameter vector u"
            for j in range(0, self.ensemble_size):
                y_noisy = self.y_data + noise[:, j]
                m = np.linalg.solve(
                    c_pp + 1 / self.delta_sigma * np.identity(self.data_dim),
                    y_noisy - self.gk[j],
                )

                self.uk[j] = self.uk[j] + np.matmul(c_up, m).flatten()

        self.mean_k[self.iter + 1, :] = np.mean(self.uk, axis=0)

        if self.iter % 20 == 0:
            self.uk_save.append(self.uk)

    def perform_tempering_local(self):
        self.number_tempering = self.number_tempering + 1

        "Update limit state function values and calculate new sigma"
        EnKF_rare_events.update_g_and_sigma(self)

        # print("perform Tempering; current temperature", self.sigmak_list[self.iter])

        "Update the parameter vector u"
        w_mat = EnKF_rare_events.calculate_weights(self, self.uk)

        weighted_mean = np.matmul(w_mat, self.uk)
        weighted_lsf_mean = np.matmul(w_mat, self.gk)

        noise = np.random.normal(
            0, np.sqrt(1 / self.delta_sigma), (self.data_dim, self.ensemble_size)
        )

        if self.data_dim == 1:
            c_pp_loc = np.zeros((self.ensemble_size, self.data_dim))
            c_up_loc = np.zeros((self.ensemble_size, self.parameter_dim))
            for j in range(self.ensemble_size):
                diff = self.gk - weighted_lsf_mean[j]
                c_pp_loc[j, :] = np.matmul(w_mat[j, :], np.square(diff))
                c_up_loc[j, :] = np.matmul(
                    w_mat[j, :], (self.uk - weighted_mean[j]) * diff
                )
            y_noisy = self.y_data + noise.T
            self.uk = self.uk + np.multiply(
                c_up_loc, 1 / (c_pp_loc + 1 / self.delta_sigma) * (y_noisy - self.gk)
            )

        else:
            uk_new = np.zeros(self.uk.shape)
            "Update the parameter vector u"
            for j in range(self.ensemble_size):
                c_pp_loc = np.zeros((self.data_dim, self.data_dim))
                c_up_loc = np.zeros((self.parameter_dim, self.data_dim))
                for i in range(self.ensemble_size):
                    c_pp_loc = c_pp_loc + w_mat[j, i] * np.outer(
                        self.gk[i] - weighted_lsf_mean[j],
                        self.gk[i] - weighted_lsf_mean[j],
                    )
                    c_up_loc = c_up_loc + w_mat[j, i] * np.outer(
                        self.uk[i] - weighted_mean[j], self.gk[i] - weighted_lsf_mean[j]
                    )
                y_noisy = self.y_data + noise[:, j]
                m = np.linalg.solve(
                    c_pp_loc + 1 / self.delta_sigma * np.identity(self.data_dim),
                    y_noisy - self.gk[j],
                )

                uk_new[j] = self.uk[j] + np.matmul(c_up_loc, m).flatten()
            self.uk = uk_new

    def calculate_c_up(self, u, g, ensemble_mean, lsf_mean):
        """
        The function calculate_c_up() calculates the matrix C_up from the paper

        :return:
            c_up: covariance matirx
        """

        if self.data_dim == 1:
            c_up = np.mean(np.multiply(g - lsf_mean, u - ensemble_mean), axis=0)
            c_up = c_up.reshape((self.parameter_dim, 1))
        else:
            c_up = np.zeros((np.size(u, 1), np.size(g, 1)))
            ensemble_size = np.size(g, 0)

            for i in range(0, ensemble_size):
                c_up = c_up + np.matmul(
                    (u[i] - ensemble_mean).transpose(), (g[i] - lsf_mean)
                )

            c_up = c_up / ensemble_size
        return c_up

    def calculate_c_pp(self, g, lsf_mean):
        """
        The function calculate_c_pp() calculates the matrix C_pp from the paper

        :return:
            c_pp: covariance matirx
        """
        if self.data_dim == 1:
            c_pp = np.mean(np.square(g - lsf_mean))
        else:
            c_pp = np.zeros((np.size(g, 1), np.size(g, 1)))
            ensemble_size = np.size(g, 0)

            for i in range(0, ensemble_size):
                c_pp = c_pp + np.matmul(
                    (g[i] - lsf_mean).transpose(), (g[i] - lsf_mean)
                )

            c_pp = c_pp / ensemble_size
        return c_pp

    def calculate_weights(self, u):
        Y = squareform(pdist(u, "euclidean"))
        w_ij = np.exp(-1 / 2 / self.gamma_weights * Y**2)
        w_ij = (w_ij / np.sum(w_ij, 1)).transpose()

        return w_ij

    def update_g_and_sigma(self):
        "Calculate the temperature of the next step"
        if self.iter == 0:
            gmu = np.mean(self.gk)

            def func_0(x, y_data, gk, tar_cov):
                w = np.exp(-1 / x / 2 * np.linalg.norm(y_data - gk, axis=1) ** 2)
                return (np.std(w) / np.mean(w) - tar_cov) ** 2

            self.sigmak_list[self.iter] = fminbound(
                func_0,
                0,
                10 * gmu,
                args=(self.y_data, self.gk, self.tar_cov),
                xtol=1e-20,
            )

            self.delta_sigma = 1 / self.sigmak_list[self.iter]

        else:

            def func(x, sigma, y_data, gk, tar_cov):
                w = np.exp(
                    -(1 / x - 1 / sigma) / 2 * np.linalg.norm(y_data - gk, axis=1) ** 2
                )
                return (np.std(w) / np.mean(w) - tar_cov) ** 2

            self.sigmak_list[self.iter] = fminbound(
                func=func,
                x1=0,
                x2=self.sigmak_list[self.iter - 1],
                args=(
                    self.sigmak_list[self.iter - 1],
                    self.y_data,
                    self.gk,
                    self.tar_cov,
                ),
                xtol=1e-20,
            )

            self.delta_sigma = (
                1 / self.sigmak_list[self.iter] - 1 / self.sigmak_list[self.iter - 1]
            )

        if self.iter % 20 == 0:
            self.sigma_save.append(self.sigmak_list[self.iter])

    def update_posterior_and_error(self):
        self.posterior_mean = np.mean(self.uk, 0)
        self.posterior_lsf_mean = self.lsf(self.posterior_mean)

        self.rel_error.append(
            np.linalg.norm(self.y_data - self.gk) / self.ensemble_size
        )

        if np.min(self.gk) == 0:
            wk_opt = np.zeros((self.ensemble_size, 1))
            wk_opt[self.gk == 0] = np.exp(
                1
                / self.sigmak_list[self.iter]
                / 2
                * np.linalg.norm(self.y_data - self.gk[self.gk == 0], axis=0) ** 2
            )
            cov_opt = np.std(wk_opt) / np.mean(wk_opt)
        else:
            cov_opt = np.inf

        if cov_opt < self.tar_cov or self.sigmak_list[self.iter] < 1e-18:
            self.tempering_finished = True

    def evaluate_lsf(self):
        """Calculate the limit state function values with respect to the current particles"""
        self.gk = self.lsf(self.uk)[..., None]

        "Update the computational costs and number of function evaluations"
        self.computational_cost = self.computational_cost + self.ensemble_size
        self.number_fun_eval = self.number_fun_eval + self.ensemble_size

    def show_computational_cost(self):
        print("Computational Cost:", self.computational_cost)
        print("Number of EnKF runs:", self.iter)
        print("Number of function evaluations:", self.number_fun_eval)

    def calculate_failure_probability(self):
        if self.mixture_model == "GM":
            EnKF_rare_events.calculate_failure_probability_GM(self)
        elif self.mixture_model == "vMFNM":
            EnKF_rare_events.calculate_failure_probability_vMFNM(self)
        else:
            print("Error! Specified mixture model is invalid")

    def calculate_failure_probability_GM(self):
        W = np.ones(self.ensemble_size)

        [mu, si, pi] = EMGM(self.uk.T, W, self.k_init)

        self.uk_fitted = GM_sample(mu.T, si, pi, self.ensemble_size)

        for j in range(self.ensemble_size):
            self.gk_fitted[j] = self.g_original(self.uk_fitted[j])

        "Update the computational costs and number of function evaluations"
        self.computational_cost = self.computational_cost + self.ensemble_size
        self.number_fun_eval = self.number_fun_eval + self.ensemble_size

        wk = (
            (self.gk_fitted < 0)
            * multivariate_normal.pdf(
                self.uk_fitted,
                mean=np.zeros(self.parameter_dim),
                cov=np.eye(self.parameter_dim),
            )
            / h_calc(self.uk_fitted, mu.T, si, pi)
        )

        self.pf = np.mean(wk)
        if self.pf > 0.01:
            print("Outlier")

        # print("Probability of failure estimate:", self.pf)
        # print("Number of tempering steps:", self.iter)

        return wk, mu, si, pi

    def calculate_failure_probability_vMFNM(self):
        W = np.ones(self.ensemble_size)
        [mu, kappa, m, omega, alpha] = EMvMFNM(self.uk.T, W, self.k_init)
        self.uk_fitted = vMFNM_sample(mu.T, kappa, omega, m, alpha, self.ensemble_size)

        for j in range(self.ensemble_size):
            self.gk_fitted[j] = self.g_original(self.uk_fitted[j])

        "Update the computational costs and number of function evaluations"
        self.computational_cost = self.computational_cost + self.ensemble_size
        self.number_fun_eval = self.number_fun_eval + self.ensemble_size

        wk = (self.gk_fitted < 0) * np.exp(
            likelihood_ratio_log(self.uk_fitted, mu.T, kappa, omega, m, alpha)
        ).flatten()

        self.pf = np.mean(wk)

        # print("Probability of failure estimate:", self.pf)
        # print("Number of tempering steps:", self.iter)
