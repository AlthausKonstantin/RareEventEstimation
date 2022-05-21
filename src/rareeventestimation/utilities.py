    
import os
from numpy import average, exp, log, ndarray, pi, zeros, eye, isfinite,  amax, sqrt, inf, ones_like, isnan, argmax, zeros_like, diff, nan, trace, amin, arange, polyfit, seterr
from numpy.linalg import norm
from scipy.stats import multivariate_normal
from scipy.special import gammaln

def importance_sampling(target_pdf_evals: ndarray, aux_pdf_evals: ndarray, lsf_evals: ndarray, logpdf=False) -> float:
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
        w  = target_pdf_evals - aux_pdf_evals
        return average(exp(w)*(lsf_evals <= 0))
    else:
        w = target_pdf_evals / aux_pdf_evals
        return average(w*(lsf_evals <= 0))




def radial_gaussian_logpdf(sample:ndarray) -> ndarray:
        """Taken from CEIS-VMFNM, likelihood_ratio_log."""
        dim = sample.shape[-1]
        R = norm(sample, axis=1, ord=2)
        # unit hypersphere uniform log pdf
        A = log(dim) + log(pi ** (dim / 2)) - gammaln(dim / 2 + 1)
        f_u = -A

        # chi log pdf
        f_chi = log(2) * (1 - dim / 2) + log(R) * (dim - 1) - 0.5 * R ** 2 - gammaln(dim / 2)

        # logpdf of the standard distribution (uniform combined with chi distribution)
        rad_gauss_log = gaussian_logpdf(sample) + log(norm(sample, axis=1, ord=2))
        print(norm(rad_gauss_log -f_u - f_chi))
        return f_u + f_chi
    
def gaussian_logpdf(sample:ndarray) -> ndarray:
    """Evaluate logpdf of multivariate standard normal distribuution in sample."""
    d = sample.shape[-1]
    return multivariate_normal.logpdf(sample, mean=zeros(d), cov=eye(d))

    
def my_log_cvar(log_samples, multiplier=None):
    msk = isfinite(log_samples)
    log_samples = log_samples[msk]
    if multiplier is None:
        multiplier = ones_like(log_samples)
    log_samples_max = amax(log_samples)
    log_samples = log_samples - log_samples_max
    tmp = average(exp(log_samples) * multiplier)
    m = exp(log_samples_max) * tmp
    s = exp(log_samples_max) * sqrt(average((exp(log_samples)* multiplier-tmp)**2))
    if m == 0:
        return inf
    out = s/m
    if isnan(out):
        return inf
    else:
        return out

    
def my_softmax(weights):
    old = seterr(under="ignore") # temporarily ignore underflow (will be cast to 0)
    # Ignore nonfinite values
    weights = weights.squeeze()
    msk = isfinite(weights)
    weights = weights[msk]
    
    # Do the softmax trick
    idx = argmax(weights)
    w_max = weights[idx]
    tmp = exp(weights-w_max)
    w_out = zeros_like(msk, dtype=weights.dtype)
    if any(tmp>0.0):
        w_out[msk] = tmp # correct shape
        w_out = w_out/sum(w_out)
    else:
        # No information in exponential weights => return normalized weights
        w_min = amin(weights)
        if w_min < 0: 
            weights = weights + w_min # Make weights positive
        total = sum(weights)
        if total == 0: 
            w_out =  w_out + 1/len(w_out) # No information in weights => return uniform weights
        else:
            w_out[msk] = weights
            w_out =  w_out /total
    seterr(**old) # old error behavior
    return w_out


def get_slope(y):
    x= arange(len(y))
    msk = isfinite(y)
    if not all(msk):
        return  nan
    slope=nan
    try:
        slope = polyfit(x[msk],y[msk],deg=1, full=True)[0][0]
    except Exception:
        return nan
    
    return slope
            
        
def compute_delta_normalized(means, covs, alpha):
    d = means.shape[1]
    expected_diff_means = average(sum(diff(means, axis=0)**2,axis=1),axis=0)
    expected_trace = average(trace(covs,axis1=1,axis2=2))
    normed_delta_mean = expected_diff_means/(d*(1-alpha**2) + (1-alpha**2)*expected_trace)
    
    expected_diff_covs = average(sum(diff(covs, axis=0)**2, axis=(1,2)),axis=0)
    expected_cov_norm = average(sum(covs**2,axis=(1,2)),axis=0)
    expected_diag_norm = average(trace(covs**2, axis1=1, axis2=2),axis=0)
    normed_cov_mean = expected_diff_covs / (d**2 + expected_cov_norm + expected_diag_norm)
    normed_cov_mean /= 1-alpha**2
    return (normed_delta_mean + normed_cov_mean)/2
    
      