import numpy as np
from scipy.stats import multivariate_normal

"""
---------------------------------------------------------------------------
Perform soft EM algorithm for fitting the Gaussian mixture model
---------------------------------------------------------------------------
Created by:
Sebastian Geyer (s.geyer@tum.de),
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Matthias Willer

Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2019-02
---------------------------------------------------------------------------
Input:
* X   : data matrix (dimensions x Number of samples)
* W   : vector of likelihood ratios for weighted samples
* nGM : number of Gaussians in the Mixture
---------------------------------------------------------------------------
Output:
* mu : [npi x d]-array of means of Gaussians in the Mixture
* si : [d x d x npi]-array of cov-matrices of Gaussians in the Mixture
* pi : [npi]-array of weights of Gaussians in the Mixture (sum(Pi) = 1)
---------------------------------------------------------------------------
Based on:
1. "EM Demystified: An Expectation-Maximization Tutorial"
   Yihua Chen and Maya R. Gupta
   University of Washington, Dep. of EE (Feb. 2010)
---------------------------------------------------------------------------
"""


def EMGM(X, W, nGM):
    # reshaping just to be sure
    W = W.reshape(-1, 1)

    # initialization
    R = initialization(X, nGM)

    tol = 1e-5
    maxiter = 500
    llh = np.full([maxiter], -np.inf)
    converged = False
    t = 0

    # soft EM algorithm
    while (not converged) and (t + 1 < maxiter):
        t = t + 1
        label = np.argmax(R, axis=1)
        u = np.unique(label)  # non-empty components
        if np.size(R, axis=1) != np.size(u, axis=0):
            R = R[:, u]  # remove empty components

        [mu, si, pi] = maximization(X, W, R)
        [R, llh[t]] = expectation(X, W, mu, si, pi)

        if t > 1:
            diff = llh[t] - llh[t - 1]
            eps = abs(diff)
            converged = eps < tol * abs(llh[t])

    if converged:
        pass
    # print('Converged in', t,'steps.')
    else:
        pass
    # print('Not converged in ', maxiter, ' steps.')

    return mu, si, pi


# ===========================================================================
# =============================AUX FUNCTIONS=================================
# ===========================================================================
# --------------------------------------------------------------------------
# Initialization
# --------------------------------------------------------------------------
def initialization(X, nGM):
    # Random initialization
    n = np.size(X, axis=1)
    idx = np.random.choice(range(n), nGM)
    m = X[:, idx]
    label = np.argmax(
        np.matmul(m.T, X) - np.sum(m * m, axis=0).reshape(-1, 1) / 2, axis=0
    )
    u = np.unique(label)
    while nGM != len(u):
        idx = np.random.choice(range(n), nGM)
        m = X[:, idx]
        label = np.argmax(
            np.matmul(m.T, X) - np.sum(m * m, axis=0).reshape(-1, 1) / 2, axis=0
        )
        u = np.unique(label)

    R = np.zeros([n, nGM], dtype=int)
    for i in range(n):
        R[i, label[i]] = 1

    return R


# ===========================================================================
# --------------------------------------------------------------------------
# Expectation
# --------------------------------------------------------------------------
def expectation(X, W, mu, si, pi):
    n = np.size(X, axis=1)
    k = np.size(mu, axis=1)

    logpdf = np.zeros([n, k])
    for i in range(k):
        logpdf[:, i] = loggausspdf(X, mu[:, i], si[:, :, i])

    logpdf = logpdf + np.log(pi)
    T = logsumexp(logpdf, 1)
    llh = np.sum(W * T) / np.sum(W)
    logR = logpdf - T
    R = np.exp(logR)

    return R, llh


# ===========================================================================
# --------------------------------------------------------------------------
# Maximization
# --------------------------------------------------------------------------
def maximization(X, W, R):
    R = W * R
    d = np.size(X, axis=0)
    k = np.size(R, axis=1)

    nk = np.sum(R, axis=0)
    if any(nk == 0):  # prevent division by zero
        nk += 1e-6

    w = nk / np.sum(W)
    mu = np.matmul(X, R) / nk.reshape(1, -1)

    Sigma = np.zeros([d, d, k])
    sqrtR = np.sqrt(R)
    for i in range(k):
        Xo = X - mu[:, i].reshape(-1, 1)
        Xo = Xo * sqrtR[:, i].reshape(1, -1)
        Sigma[:, :, i] = np.matmul(Xo, Xo.T) / nk[i]
        Sigma[:, :, i] = Sigma[:, :, i] + np.eye(d) * (
            1e-6
        )  # add a prior for numerical stability

    return mu, Sigma, w


# ===========================================================================
# --------------------------------------------------------------------------
# Returns the log of the gaussian pdf
# --------------------------------------------------------------------------
def loggausspdf(X, mu, Sigma):
    d = np.size(X, axis=0)
    X = X - mu.reshape(-1, 1)
    U = np.linalg.cholesky(Sigma).T.conj()
    Q = np.linalg.solve(U.T, X)
    q = np.sum(Q * Q, axis=0)  # quadratic term (M distance)
    # normalization constant
    c = d * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(U)))
    y = -(c + q) / 2

    return y


# ===========================================================================
# --------------------------------------------------------------------------
# Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
#   By default dim = 0 (columns).
# Written by Michael Chen (sth4nth@gmail.com).
# --------------------------------------------------------------------------
def logsumexp(x, dim=0):
    # subtract the largest in each column
    y = np.max(x, axis=dim).reshape(-1, 1)
    x = x - y
    s = y + np.log(np.sum(np.exp(x), axis=dim)).reshape(-1, 1)
    # if a bug occurs here, maybe find a better translation from matlab:
    i = np.where(np.invert(np.isfinite(y).squeeze()))
    s[i] = y[i]

    return s


# ===========================================================================
# --------------------------------------------------------------------------
# Translation of the Matlab-function "dummyvar()" to Python
# --------------------------------------------------------------------------
def dummyvar(idx):
    n = np.max(idx) + 1
    d = np.zeros([len(idx), n], int)
    for i in range(len(idx)):
        d[i, idx[i]] = 1

    return d


# ===========================================================================
# =============================AUX FUNCTIONS=================================
# ===========================================================================
def GM_sample(mu, si, pi, N):
    # ---------------------------------------------------------------------------
    # Algorithm to draw samples from a Gaussian-Mixture (GM) distribution
    # ---------------------------------------------------------------------------
    # Input:
    # * mu : [npi x d]-array of means of Gaussians in the Mixture
    # * Si : [d x d x npi]-array of cov-matrices of Gaussians in the Mixture
    # * Pi : [npi]-array of weights of Gaussians in the Mixture (sum(Pi) = 1)
    # * N  : number of samples to draw from the GM distribution
    # ---------------------------------------------------------------------------
    # Output:
    # * X  : samples from the GM distribution
    # ---------------------------------------------------------------------------
    if np.size(mu, axis=0) == 1:
        mu = mu.squeeze()
        si = si.squeeze()
        X = multivariate_normal.rvs(mean=mu, cov=si, size=N)
    else:
        # Determine number of samples from each distribution
        z = np.round(pi * N)
        if np.sum(z) != N:
            dif = np.sum(z) - N
            ind = np.argmax(z)
            z[ind] = z[ind] - dif

        z = z.astype(int)  # integer conversion

        # Generate samples
        d = np.size(mu, axis=1)
        X = np.zeros([N, d])
        ind = 0
        for p in range(len(pi)):
            X[ind : ind + z[p], :] = multivariate_normal.rvs(
                mean=mu[p, :], cov=si[:, :, p], size=z[p]
            ).reshape(-1, d)
            ind = ind + z[p]

    return X


# ===========================================================================
def h_calc(X, mu, si, Pi):
    # ---------------------------------------------------------------------------
    # Basic algorithm to calculate h for the likelihood ratio
    # ---------------------------------------------------------------------------
    # Input:
    # * X  : input samples
    # * mu : [npi x d]-array of means of Gaussians in the Mixture
    # * Si : [d x d x npi]-array of cov-matrices of Gaussians in the Mixture
    # * Pi : [npi]-array of weights of Gaussians in the Mixture (sum(Pi) = 1)
    # ---------------------------------------------------------------------------
    # Output:
    # * h  : parameters h (IS density)
    # ---------------------------------------------------------------------------
    N = len(X)
    k_tmp = len(Pi)
    if k_tmp == 1:
        mu = mu.squeeze()
        si = si.squeeze()
        h = multivariate_normal.pdf(X, mu, si)
    else:
        h_pre = np.zeros((N, k_tmp))
        for q in range(k_tmp):
            h_pre[:, q] = Pi[q] * multivariate_normal.pdf(X, mu[q, :], si[:, :, q])

        h = np.sum(h_pre, axis=1)

    return h
