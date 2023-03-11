"""Custom functions of existing functions used in package."""
from numpy import (
    average,
    exp,
    log,
    ndarray,
    pi,
    zeros,
    eye,
    isfinite,
    amax,
    sqrt,
    inf,
    ones_like,
    isnan,
    argmax,
    zeros_like,
    nan,
    amin,
    arange,
    polyfit,
    seterr,
)
from numpy.linalg import norm
from scipy.stats import multivariate_normal
from scipy.special import gammaln


def importance_sampling(
    target_pdf_evals: ndarray, aux_pdf_evals: ndarray, lsf_evals: ndarray, logpdf=False
) -> float:
    """Importance sampling acc. to Wagner2021 eq. (2.2)

    Args:
        target_pdf_evals (ndarray): 1-d array  with evaluations of the target density function.
        aux_pdf_evals (ndarray): 1-d array  with evaluations of the auxilliary density function.
        lsf_evals (ndarray): 1-d array  with evaluations of the limit state function.
        logpdf(bool, optional): Indicated of target_pdf_evals and aux_pdf_evals are evaluations of a logpdf instead of a pdf

    Returns:
        float: probability of failure
    """
    if logpdf:
        w = target_pdf_evals - aux_pdf_evals
        return average(exp(w) * (lsf_evals <= 0))
    else:
        w = target_pdf_evals / aux_pdf_evals
        return average(w * (lsf_evals <= 0))


def radial_gaussian_logpdf(sample: ndarray) -> ndarray:
    """Taken from CEIS-VMFNM, likelihood_ratio_log."""
    dim = sample.shape[-1]
    R = norm(sample, axis=1, ord=2)
    # unit hypersphere uniform log pdf
    A = log(dim) + log(pi ** (dim / 2)) - gammaln(dim / 2 + 1)
    f_u = -A

    # chi log pdf
    f_chi = (
        log(2) * (1 - dim / 2) + log(R) * (dim - 1) - 0.5 * R**2 - gammaln(dim / 2)
    )

    # logpdf of the standard distribution (uniform combined with chi distribution)
    rad_gauss_log = gaussian_logpdf(sample) + log(norm(sample, axis=1, ord=2))
    print(norm(rad_gauss_log - f_u - f_chi))
    return f_u + f_chi


def gaussian_logpdf(sample: ndarray) -> ndarray:
    """Evaluate logpdf of multivariate standard normal distribuution in sample."""
    d = sample.shape[-1]
    return multivariate_normal.logpdf(sample, mean=zeros(d), cov=eye(d))


def my_log_cvar(log_samples: ndarray, multiplier=None) -> float:
    """Compute coefficient of variation of `exp(log_samples)* multiplier`.

    Args:
        log_samples (ndarray): Samples in array of shape (J,)
        multiplier (ndarray, optional): See above. Defaults to None. Then we use
            an array of ones.

    Returns:
        float: coefficient of variation of `exp(log_samples)* multiplier`.
    """
    msk = isfinite(log_samples)
    log_samples = log_samples[msk]
    if multiplier is None:
        multiplier = ones_like(log_samples)
    log_samples_max = amax(log_samples)
    log_samples = log_samples - log_samples_max
    tmp = average(exp(log_samples) * multiplier)
    m = exp(log_samples_max) * tmp
    s = exp(log_samples_max) * sqrt(average((exp(log_samples) * multiplier - tmp) ** 2))
    if m == 0:
        return inf
    out = s / m
    if isnan(out):
        return inf
    else:
        return out


def my_softmax(weights: ndarray) -> ndarray:
    """Custom softmax function to deal with degenerate weights.

    Args:
        weights (ndarray): log weights in array of shape (J,)

    Returns:
        ndarray: Normalized weights in array of shape (J,)
    """
    old = seterr(under="ignore")  # temporarily ignore underflow (will be cast to 0)
    # Ignore nonfinite values
    weights = weights.squeeze()
    msk = isfinite(weights)
    weights = weights[msk]

    # Do the softmax trick
    idx = argmax(weights)
    w_max = weights[idx]
    tmp = exp(weights - w_max)
    w_out = zeros_like(msk, dtype=weights.dtype)
    if any(tmp > 0.0):
        w_out[msk] = tmp  # correct shape
        w_out = w_out / sum(w_out)
    else:
        # No information in exponential weights => return normalized weights
        w_min = amin(weights)
        if w_min < 0:
            weights = weights + w_min  # Make weights positive
        total = sum(weights)
        if total == 0:
            w_out = w_out + 1 / len(
                w_out
            )  # No information in weights => return uniform weights
        else:
            w_out[msk] = weights
            w_out = w_out / total
    seterr(**old)  # old error behavior
    return w_out


def get_slope(y: ndarray) -> float:
    """Compute slope of points (i, y[i]).

    Args:
        y (ndarray): y-values of data to fit.

    Returns:
        float: Slope of linear fitting.
    """
    x = arange(len(y))
    msk = isfinite(y)
    if not all(msk):
        return nan
    slope = nan
    try:
        slope = polyfit(x[msk], y[msk], deg=1, full=True)[0][0]
    except Exception:
        return nan
    return slope
