from numpy import (
    argmax,
    moveaxis,
    nan,
    ndarray,
    ones,
    zeros,
    matmul,
    eye,
    average,
    zeros_like,
    exp,
    log,
)
from numpy.linalg import norm
from numpy.random import default_rng
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from scipy.linalg import sqrtm
from rareeventestimation.era.EMGM import EMGM
from rareeventestimation.era.EMvMFNM import (
    EMvMFNM,
    lognakagamipdf,
    logvMFpdf,
    vMFNM_sample,
)


class MixtureModel:
    """Parent for different mixture models."""

    def __init__(self, num_comps: int) -> None:
        """Initialize MixtureModel object.

        Args:
            num_comps (int): Number of components in the mixture.
            seed (int, optional): Seed for sampling. Defaults to None.
        """
        self.num_comps = num_comps
        self.fitted = False


class GaussianMixture(MixtureModel):
    """Fit, sample, evaluate GM model."""

    def __init__(self, num_comps: int, seed=None) -> None:
        """Initialize GaussianMixture object.

        Args:
            num_comps (int): Number of Gaussians in the mixture.
            seed (int, optional): Seed for sampling. Defaults to None.
        """
        super().__init__(num_comps)

    def fit(self, sample: ndarray, weights_sample: ndarray = None) -> None:
        """Fit GM model with EM algorithm.

        Args:
            sample (ndarray): Array of shape (N,d) with sample to fit.
            weights_sample (ndarray,optional): Array of shape (N,) if samples should be weighted non uniformly. Use [1/N,...,1/N] as a default.
        """
        if weights_sample is None:
            weights_sample = ones(sample.shape[0]) / sample.shape[0]
        means, covs, self.weights = EMGM(sample.T, weights_sample, self.num_comps)
        self.means = means.T  # (num_comp,d)
        self.covs = moveaxis(covs, -1, 0)  # new shape (num_comp,d,d)
        self.fitted = True
        self.d = self.means.shape[1]
        # EMGM might change number of components
        self.num_comps = self.means.shape[0]

    def sample(self, N: int, rng=default_rng()) -> ndarray:
        """Sample from fitted GM distribution

        Args:
            N (int): Number of samples.

        Returns:
            ndarray: (N,d) array with samples.
        """
        idx = rng.choice(self.num_comps, size=N, p=self.weights)
        sample = rng.multivariate_normal(zeros(self.d), eye(self.d), N)
        cov_roots = self.get_cov_roots()
        for i, (m, c_root) in enumerate(zip(self.means, cov_roots)):
            sample[idx == i] = matmul(sample[idx == i], c_root) + m
        return sample

    def logpdf(self, sample: ndarray) -> ndarray:
        """Evaluate logpdf of GM in sample

        Args:
            sample (ndarray): Sample points in array of shape (N,d)

        Returns:
            ndarray: Evaluations in array of sahpe (N,)
        """
        assert self.fitted, "Fit GM before accessing parameters!"
        logpdf_evals_comp = zeros((sample.shape[0], self.num_comps))
        for i, (m, c) in enumerate(zip(self.means, self.covs)):
            logpdf_evals_comp[..., i] = multivariate_normal.logpdf(
                sample, mean=m, cov=c
            )
        return logsumexp(logpdf_evals_comp, axis=1, b=self.weights)

    def pdf(self, sample: ndarray) -> ndarray:
        """Evaluate pdf of GM in sample

        Args:
            sample (ndarray): Sample points in array of shape (N,d)

        Returns:
            ndarray: Evaluations in array of sahpe (N,)
        """
        assert self.fitted, "Fit GM before accessing parameters!"
        pdf_evals_comp = zeros((sample.shape[0], self.num_comps))
        for i, (m, c) in enumerate(zip(self.means, self.covs)):
            pdf_evals_comp[..., i] = multivariate_normal.pdf(sample, mean=m, cov=c)
        return average(pdf_evals_comp, axis=1, weights=self.weights)

    def predict(self, sample: ndarray) -> ndarray:
        """Predict to which respective component the sample entries belong.

        Args:
            sample (ndarray): Sample points in array of shape (N,d)

        Returns:
            ndarray: Predicted components (N,)
        """
        assert self.fitted, "Fit GM before accessing parameters!"
        logpdf_evals_comp = zeros((sample.shape[0], self.num_comps))
        for i, (m, c) in enumerate(zip(self.means, self.covs)):
            logpdf_evals_comp[..., i] = multivariate_normal.logpdf(
                sample, mean=m, cov=c
            )
        return argmax(logpdf_evals_comp, axis=1)

    def get_cov_roots(self) -> ndarray:
        """Get roots of covariance matrices.

        Returns:
            ndarray: Array of shape (d,d,num_comps) with roots of covariance matrices
        """
        assert self.fitted, "Fit GM before accessing parameters!"
        if hasattr(self, "cov_roots"):
            return self.cov_roots
        else:
            self.cov_roots = zeros_like(self.covs)
            for i, c in enumerate(self.covs):
                self.cov_roots[i, ...] = sqrtm(c)
            return self.cov_roots


class VMFNMixture(MixtureModel):
    """Fit, sample, evaluate von Mises Fisher Nakagami model."""

    def __init__(self, num_comps: int) -> None:
        """Initialize von Mises Fisher Nakagami Mixture object.

        Args:
            num_comps (int): Number of vMFN distributions in the mixture.
        """
        super().__init__(num_comps)

    def fit(self, sample: ndarray, weights_sample: ndarray = None) -> None:
        """Fit vMFN model with EM algorithm.

        Args:
            sample (ndarray): Array of shape (N,d) with sample to fit.
            weights_sample (ndarray,optional): Array of shape (N,) if samples should be weighted non uniformly. Use [1/N,...,1/N] as a default.
        """
        if weights_sample is None:
            weights_sample = ones(sample.shape[0]) / sample.shape[0]
        self.mu, self.kappa, self.m, self.omega, self.alpha = EMvMFNM(
            sample.T, weights_sample, self.num_comps
        )
        self.fitted = True
        self.num_comps = len(self.alpha)

    def sample(self, N: int, rng=default_rng()) -> ndarray:
        """Sample from fitted vMFNM distribution

        Args:
            N (int): Number of samples.

        Returns:
            ndarray: (N,d) array with samples.
        """
        assert self.fitted, "Fit vMFNM before accessing parameters!"
        return vMFNM_sample(
            self.mu.T, self.kappa, self.omega, self.m, self.alpha, N, rng=rng
        )

    def logpdf(self, sample: ndarray) -> ndarray:
        """Evaluate logpdf of vMFNM in sample

        Args:
            sample (ndarray): Sample points in array of shape (N,d)

        Returns:
            ndarray: Evaluations in array of sahpe (N,)
        """
        assert self.fitted, "Fit vMFNM before accessing parameters!"
        logpdf_evals_comp = zeros((sample.shape[0], self.num_comps)) * nan
        for i in range(self.num_comps):
            sample_r = norm(sample, ord=2, axis=1)
            sample_norm = sample / sample_r[:, None]
            vmf = logvMFpdf(sample_norm.T, self.mu[..., i], self.kappa[..., i])
            nk = lognakagamipdf(sample_r, self.m[..., i], self.omega[..., i])
            logpdf_evals_comp[:, i] = vmf + nk - (sample.shape[-1] - 1) * log(sample_r)
        return logsumexp(logpdf_evals_comp, axis=1, b=self.alpha)

    def pdf(self, sample: ndarray) -> ndarray:
        """Evaluate pdf of vMFNM in sample

        Args:
            sample (ndarray): Sample points in array of shape (N,d)

        Returns:
            ndarray: Evaluations in array of sahpe (N,)
        """
        assert self.fitted, "Fit vMFNM before accessing parameters!"
        return exp(self.logpdf(sample))

    def predict(self, sample: ndarray) -> ndarray:
        """Predict to which respective component the sample entries belong.

        Args:
            sample (ndarray): Sample points in array of shape (N,d)

        Returns:
            ndarray: Predicted components (N,)
        """
        assert self.fitted, "Fit vMFNM before accessing parameters!"
        logpdf_evals_comp = zeros((self.num_comps, sample.shape[0])) * nan
        for i in range(self.num_comps):
            logpdf_evals_comp[i, ...] = logvMFpdf(
                sample.T, self.mu[..., i], self.kappa[i]
            ) + lognakagamipdf(sample.T, self.m[..., i], self.omega[..., i])
        return argmax(logpdf_evals_comp, axis=0)
