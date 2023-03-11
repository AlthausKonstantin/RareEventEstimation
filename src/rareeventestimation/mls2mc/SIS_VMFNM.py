import numpy as np
from scipy.stats import norm
from scipy.optimize import fminbound
from rareeventestimation.era.MCMC import MCMC

"""
---------------------------------------------------------------------------
Multilevel Sequential² Monte Carlo for rare event estimation
---------------------------------------------------------------------------
Created by:
Max Ehre (max.ehre@tum.de)
Iason Papaioannou (iason.papaioannou@tum.de)
Fabian Wagner (fabian.wagner@ma.tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-05
---------------------------------------------------------------------------
Comments:
* The SIS-method in combination with the adaptive conditional MH sampler
  (aCS) performs well in high-dimensions. For low-dimensional problems, the
  Gaussian mixture proposal should be chosen over aCS.
* The way the initial standard deviation is computed can be changed in line 79.
  By default we use option 'a' (it is equal to one).
  In option 'b', it is computed from the seeds.
---------------------------------------------------------------------------
Input:
* n      : number of samples per level
* p      : n/number of chains per level
* g_fun  : limit state function
* distr  : Nataf distribution object or
           marginal distribution object of the input variables
* burn   : burn-in period
* tar_cov: target coefficient of variation of the weights
* tau_lu : bound for decision problem bridging vs. tempering
* tau_min: target COV for finishing bridging
* h_seq  : sequence of mesh sizes
---------------------------------------------------------------------------
Output:
* pf        : probability of failure
* l_tot     : total number of levels
* samples_u : object with the samples in the standard normal space
* samples_x : object with the samples in the original space
---------------------------------------------------------------------------
Based on:
1. "Sequential importance sampling for structural reliability analysis"
   Papaioannou et al.
   Structural Safety 62 (2016) 66-75
2. "Multilevel Sequential² Monte Carlo for Bayesian Inverse Problems"
   Jonas Latz, Iason Papaioannou, Elisabeth Ullmann
---------------------------------------------------------------------------
"""


def sis(
    n,
    p,
    g,
    burn,
    tar_cov,
    dim,
    mcmc_type="vMFNM",
    k_init=1,
    initial_sample=None,
    seed=None,
    verbose=False,
):
    if seed is not None:
        np.random.seed(seed=seed)
    # %% Initialization of variables and storage
    max_it = 100  # estimated number of iterations
    samples_u = list()  # space for samples in U-space

    # Properties of SIS
    nsamlev = n  # number of samples
    nchain = int(nsamlev * p)  # number Markov chains

    # initialize samples
    if initial_sample is None:
        uk = np.random.normal(0, 1, (nsamlev, dim))  # initial samples
    else:
        uk = initial_sample
    gk = np.zeros([nsamlev])  # space for evaluations of g
    accrate = list()  # space for acceptance rate
    sk = list()  # space for expected weights
    sigmak = np.zeros([max_it + 1])  # space for sigmak
    sigmak[0] = np.infty  # first entry in 'sigmak' is infinity

    number_fun_eval = 0
    computational_cost = 0

    # %% Step 1
    # Perform the first Monte Carlo simulation
    gk = g(uk)

    number_fun_eval = number_fun_eval + nsamlev
    computational_cost = computational_cost + nsamlev
    # save samples
    samples_u.append(uk.T)

    # set initial subset and failure level
    gmu = np.mean(gk)

    # parameter for the number of level and tempering updates
    number_tempering = 0

    # %% Iteration
    m = 0
    for m in range(max_it):
        # Perform Tempering
        number_tempering = number_tempering + 1
        if verbose:
            print("Tempering Update")

        # %% Step 2 and 3: compute sigma and weights
        wk = 0
        if m == 0:

            def func(x):
                w = norm.cdf(-gk / x)
                return (np.std(w) / np.mean(w) - tar_cov) ** 2

            sigma2 = fminbound(func, 0, 10.0 * gmu, xtol=1e-15)
            sigmak[m + 1] = sigma2
            wk = norm.cdf(-gk / sigmak[m + 1])

        if m > 0:

            def func(x):
                w = norm.cdf(-gk / x) / norm.cdf(-gk / sigmak[m])
                return abs(np.std(w) / np.mean(w) - tar_cov) ** 2

            sigma2 = fminbound(func, 0, sigmak[m], xtol=1e-15)
            sigmak[m + 1] = sigma2
            wk = norm.cdf(-gk / sigmak[m + 1]) / norm.cdf(-gk / sigmak[m])

        # %% Step 4: compute estimate of expected w
        sk.append(np.mean(wk))

        # Exit algorithm if no convergence is achieved
        if sk[-1] == 0:
            break
        wnork = wk / sk[-1] / nsamlev  # compute normalized weights

        def limit_state_fun(u):
            return g(u)

        dist = MCMC(
            uk=uk,
            gk_current=gk,
            wnork=wnork,
            nchain=nchain,
            burn=burn,
            limit_state_fun_current=limit_state_fun,
            sigma=sigmak[m + 1],
            mcmc_type=mcmc_type,
            k_init=k_init,
        )
        dist.mcmc_procedure_tempering()
        uk = dist.uk
        gk = dist.gk_current

        number_fun_eval = number_fun_eval + nchain * (burn + nsamlev / nchain)
        computational_cost = computational_cost + nchain * (burn + nsamlev / nchain)
        # save samples
        samples_u.append(uk.T)

        # compute mean acceptance rate of all chains in level m
        accrate.append(np.mean(dist.alphak))

        if sigmak[m + 1] == 0:
            cov_sl = np.nan
        elif not (any(gk < 0)):
            cov_sl = np.nan
        else:
            cov_sl = np.std((gk < 0) / norm.cdf(-gk / sigmak[m + 1])) / np.mean(
                (gk < 0) / norm.cdf(-gk / sigmak[m + 1])
            )

        if verbose:
            print("\nCOV_Sl =", cov_sl)
            print(
                "\t*aCS sigma =",
                dist.sigmafk,
                "\t*aCS accrate =",
                accrate[len(accrate) - 1],
            )

        if cov_sl < tar_cov:
            break

    # %% Calculation of the Probability of failure
    const = np.prod(sk)
    tmp1 = gk < 0
    tmp2 = -gk / sigmak[m + 1]
    tmp3 = norm.cdf(tmp2)
    tmp4 = tmp1 / tmp3
    pf = np.mean(tmp4) * const

    return (
        pf,
        number_tempering,
        sigmak[0 : m + 2],
        number_fun_eval,
        computational_cost,
        accrate[0 : len(accrate) - 1],
        uk[None, ...],
        gk,
    )
