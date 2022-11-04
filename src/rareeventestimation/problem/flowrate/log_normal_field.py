

from numbers import Real
from rareeventestimation.problem.diffusion import EigFcnKL
from numpy import ndarray, sum, sqrt, zeros, exp
from numpy.random import default_rng, Generator


class LogNormalField:
    """Implementation of a log-normal-process on [0,1] with exponential covariance.
    Can approximately evaluated by its Karhunen-Loéve Expansion.
    """
    mean_gaussian = 0.0
    variance_gaussian = 1.0
    correlation_length = 0.1
    kle_terms = 10

    _rng = None
    _eigenvalues = None
    _eigenfunctions = None

    def __init__(self,
                 mean_gaussian: Real,
                 variance_gaussian: Real,
                 correlation_length: Real,
                 kle_terms: int,
                 rng: Generator = None) -> None:
        """Initialize KLE parameters and compute KLE expansion.

        Args:
            mean_gaussian (Real): Mean of the underlying Gaussian process.
            variance_gaussian (Real): Variance of the underlying Gaussian process.
            correlation_length (Real): Correlation length of exponential covariance.
            kle_terms (int): Number of terms in the KL expansion.
        """
        self.mean_gaussian = mean_gaussian
        self.variance_gaussian = variance_gaussian
        self.correlation_length = correlation_length
        self.kle_terms = kle_terms
        self._eigenvalues, self._eigenfunctions = EigFcnKL(
            correlation_length, kle_terms)
        if rng is None:
            self._rng = default_rng()
        else:
            self._rng = rng

    def __call__(self, x: ndarray) -> ndarray:
        """Sample the lognormal field in `x`.

        Args:
            x (ndarray): Array of shape (N,) of points to evaluate random field.

        Returns:
            ndarray: Evaluation of field, has shape (N,)
        """
        # evaluate kle
        eigenfunction_eval = zeros((len(x), self.kle_terms))
        for i in range(self.kle_terms):
            eigenfunction_eval[..., i] = self._eigenfunctions(x, i+1)
        kle_coeff = sqrt(self._eigenvalues) * \
            self._rng.standard_normal(self.kle_terms)
        kle = sum(eigenfunction_eval*kle_coeff, axis=1)
        return exp(self.mean_gaussian + self.variance_gaussian * kle)
