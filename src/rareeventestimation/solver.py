"""Definition of the solvers for rare event estimation."""
import traceback
from copy import deepcopy
from dataclasses import dataclass
from distutils.log import warn
from logging import info
from numbers import Real
from sys import float_info
from typing import Optional
from prettytable import PrettyTable
from numpy import (
    amax,
    arange,
    arctan,
    array,
    average,
    concatenate,
    cov,
    exp,
    insert,
    isfinite,
    log,
    log1p,
    maximum,
    minimum,
    nan,
    ndarray,
    pi,
    prod,
    sqrt,
    tanh,
    tril,
    sum,
    zeros,
    ones,
    mean,
)
from numpy.random import default_rng, seed

from scipy.optimize import minimize_scalar, root
from scipy.stats import multivariate_normal, variation
from scipy.stats import norm as univariat_normal

from rareeventestimation.mixturemodel import MixtureModel, VMFNMixture
from rareeventestimation.problem.problem import NormalProblem, Problem
from rareeventestimation.solution import Solution
from rareeventestimation.utilities import (
    gaussian_logpdf,
    get_slope,
    importance_sampling,
    my_log_cvar,
    my_softmax,
)

from rareeventestimation.enkf.EnKF_rare_events import EnKF_rare_events
from rareeventestimation.era.ERANataf import ERANataf
from rareeventestimation.problem.problem import Vectorizer
from rareeventestimation.sis.SIS_aCS import SIS_aCS
from rareeventestimation.sis.SIS_GM import SIS_GM
from rareeventestimation.mls2mc.SIS_VMFNM import sis


@dataclass
class CBREECache:
    """A cache to hold all information of the current CBREE step.

    J is the number of samples.
    d is the number of dimensions.

    Attributes:
        ensemble (ndarray): Current sample. Shape is (J,d)
        lsf_evals (ndarray): LSF evaluated in current sample. Shape is (N,)
        e_fun_evals (ndarray): Energy function of the problem evaluated in current sample. Shape is (N,)
        weighted_mean (ndarray): Weighted mean of the ensemble. Shape is (d,)
        weighted_cov (ndarray): Weighted covariance of the ensemble. Shape is (d,d)
        sigma (float): Smoothing parameter of the indicator function.
        beta (float): Temperature, controls skewness of weights
        t_step (float): Stepsize of exponential Euler method. Corresponds to gridsize -log(t_step).
        cvar_is_weights (float): Coefficient of variation of the importance sampling weights.
        mixture_model (MixtureModel): Model used to for fitting.
        converged (bool): If the method is converged.
        iteration (int): Number of iterations.
        slope_cvar (float): Slope of cvar_is_weights over the last iterations.
        sfp_slope (float): Slope of sfp over the last iterations.
        estimate (float): Importance sampling estimate.
        estimate_uniform_avg (float): Uniform average over the last importance sampling estimates.
        estimate_sqrt_avg (float):  Square root weighted average over the last importance sampling estimates.
        estimate_cvar_avg (float):  cvar_is_weights weighted average over the last importance sampling estimates.
        sfp (float): Share of failure particles.
        ess (float): Effective sample size based on the weights.
        num_lsf_evals (int): Number of calls to the limit state functions so far.
        msg (str): Message for this step
    """

    ensemble: ndarray
    lsf_evals: ndarray
    e_fun_evals: ndarray
    weighted_mean: ndarray = None
    weighted_cov: ndarray = None
    sigma: float = 1
    beta: float = 1
    t_step: float = 0.5
    cvar_is_weights: float = nan
    mixture_model: MixtureModel = MixtureModel(1)
    converged: bool = False
    iteration: int = 0
    slope_cvar: float = nan
    sfp_slope: float = nan
    estimate: float = 0.0
    estimate_uniform_avg: float = 0.0
    estimate_sqrt_avg: float = 0.0
    estimate_cvar_avg: float = 0.0
    sfp: float = 0.0
    ess: float = nan
    num_lsf_evals: int = 0
    msg: str = ""


def flatten_cache_list(cache_list: list, attrs=None) -> dict:
    """Transform list of CBREECaches in dict with list of specified attributes.

    Args:
        cache_list (list): List of CBREECache objects-
        attrs (list, optional): List of attributes to extract.
            Defaults to None and all attributes are returned.

    Returns:
        dict: (attr, list)
    """
    if attrs is None:
        attrs = vars(cache_list[0]).keys()
    out = dict.fromkeys(attrs)
    for a in attrs:
        v = [getattr(c, a) for c in cache_list if getattr(c, a) is not None]
        out[a] = array(v).squeeze()
    return out


class Solver:
    """
    Methods to estimate the rare event probability of a `Problem`.
    """

    def __init__(self) -> None:
        pass

    def solve(self, prob: Problem):
        """To be implemented by different child classes."""
        raise NotImplementedError

    def set_options(self, options: dict, in_situ=True):
        """Reset all members in vars(self) by values specified in options.

        Args:
            options (dict): Pairs of ("member_name", value)
            in_situ (bool, optional): Return deep copy of object if False
        """
        if in_situ:
            for k, v in options.items():
                if k in vars(self):
                    setattr(self, k, v)
        else:
            copy_of_me = deepcopy(self)
            copy_of_me.set_options(options)
            return copy_of_me


class CBREE(Solver):
    """
    Instances of this class hold all the options for consensus based sampling.

    Set solver options in the constructor.
    Use method solve to apply consensus based sampling to a problem.
    """

    def __init__(
        self,
        stepsize_tolerance=0.5,
        t_step=-1,
        num_steps=100,
        tgt_fun="algebraic",
        observation_window=5,
        seed=None,
        sigma_adaptivity="cvar",
        beta_adaptivity=True,
        cvar_tgt=2,
        sfp_tgt=1 / 3,
        ess_tgt=1 / 2,
        lip_sigma=1,
        divergence_check=True,
        convergence_check=True,
        mixture_model="GM",
        resample=False,
        verbose=False,
        save_history=False,
        return_other=False,
        return_caches=False,
        name=None,
        callback=None,
    ) -> None:
        """
        The constructor can handle the following keyword arguments.

        Args:
            stepsize_tolerance (float, optional): Tolerance for updating `t_step`. Defaults to 0.5.
            t_step (float, optional): Set to a positive number to set a constant
            stepssize. Otherwise choose stepsize adaptively. Defaults to -1.
            num_steps (int, optional): Maximal number of steps. Defaults to 100.
            tgt_fun (str, optional): Which smoothing function should be used.
                Can be one of:
                    * "algebraic"
                    * "tanh"
                    * "arctan"
                    * "erf"
                    * "relu"
                    * "sigmoid"
                Defaults to "algebraic" if not provided or not set to one of the above.
            observation_window (int|Real, optional): How many steps should be considered for the divergence check
            and the average failure estimates. Defaults to 5.
            seed (int, optional): Seed for the random number generator. Defaults to `None`.
            sigma_adaptivity (str|Real, optional): Method of updating `sigma`.
                Can be one of:
                    * "cvar": Update `sigma` based of coefficient of variation of importance sampling weights.
                    * "sfp": Update `sigma` based on share of failure particles.
                    *  positive number: `sigma` is constant to this value.
                Defaults to "cvar".
            beta_adaptivity (bool|Real, optional): Whether to update the temperature `beta`
                Can be one of:
                    * `True`
                    * positive number: Temperature is constant to this value.
                Defaults to `True`
            cvar_tgt (Real, optional): Stop if coefficient of variation is smaller than this and `sigma_adaptivity=="cvar"`.
                Is also used for updating `sigma` if `sigma_adaptivity=="cvar"`.
                Defaults to 2.
            sfp_tgt (Real, optional): Stop if share of failure particles is greater than this and `sigma_adaptivity=="sfp"`.
                Is also used for updating `sigma` if `sigma_adaptivity=="sfp"` or
                if sigma_adaptivity=="cvar"` and share of failure particles is smaller `sfp_tgt`
                Defaults to 1/3.
            ess_tgt (Real, optional): Target for relative effective sample size used to update temperature `beta`. Defaults to 1/2.
            lip_sigma (Real, optional): Limits increase of `sigma` by `-lip_sigma*log(t_step)`. Defaults to 1.
            divergence_check (bool, optional): Whether to check for divergence by looking at the last `observation_window` iterations. Defaults to `True`.
            convergence_check (bool, optional): Whether to check for convergence by looking at the last ensemble. Defaults to `True`.
            mixture_model (str, optional): Which model is used to for resampling. Defaults to "GM".
                Can be one of:
                    * "GM"
                    * "vMFNM"
            resample (bool, optional): Whether to resample each iteration using the density specified by `mixture_model`. Defaults to `False`.
            verbose (bool, optional): Whether to print information during solving. Defaults to `False`.
            save_history (bool, optional): Whether to return information of all iterations. Defaults to `False`.
            return_other (bool, optional): Whether to return additional information as a dictionary in the attribute `other` of the solution.
                Defaults to `False`.
                The dictionary contains among other the keys the averaged estimates based on the last `observation_window` previous iterations:
                    * "Average Estimate": Uniform average of the `observation_window` last estimates. (best)
                    * "Root Weighted Average Estimate": Weighted average of the `observation_window` last estimates. Weights decay proportional to `sqrt(final_iteration - iteration)`. (good default choice)
                    * "VAR Weighted Average Estimate": Weighted Average Estimate": Weighted average of the `observation_window` last estimates. Weights are proportional to the inverse of the estimates coefficient of variation. (unstable)
            return_caches (bool, optional): Whether to return a list of the caches. Defaults to `False`.
                Mainly for debugging and more intricate testing.
            name (str, optional): Name of the solver. Defaults to "CBREE".
            callback (Callable, optional): Apply this function to the current `CBREECache` after each iteration. Defaults to None.
        """
        super().__init__()
        if t_step > 0:
            self.stepsize_adaptivity = False
            self.t_step = t_step
        else:
            self.stepsize_adaptivity = True
        print(f"adaptive: {self.stepsize_adaptivity}")

        if isinstance(sigma_adaptivity, str):
            assert sigma_adaptivity in ["cvar", "sfp"]
            self.sigma = 0  # initial sigma
            self.sigma_adaptivity = sigma_adaptivity
        else:
            assert (
                sigma_adaptivity > 0
            ), f"sigma_adaptivity must be either `cvar`, `sfp` or `True`. I got '{sigma_adaptivity}'"
            self.sigma = sigma_adaptivity  # constant sigma
            self.sigma_adaptivity = "constant"

        if beta_adaptivity is True:
            self.beta = 1  # initial beta, will be changed
            self.beta_adaptivity = beta_adaptivity
        else:
            assert (
                beta_adaptivity > 0
            ), f"beta_adaptivity must be either a positive number or `True`. I got '{beta_adaptivity}'"
            self.beta = beta_adaptivity  # constant beta
            self.beta_adaptivity = False

        self.num_comps = 1  # currently only 1
        self.stepsize_tolerance = stepsize_tolerance
        self.num_steps = num_steps
        self.tgt_fun = tgt_fun
        self.observation_window = observation_window
        self.seed = seed
        self.rng = default_rng(self.seed)
        self.cvar_tgt = cvar_tgt
        self.sfp_tgt = sfp_tgt
        self.ess_tgt = ess_tgt
        self.lip_sigma = lip_sigma
        self.divergence_check = divergence_check
        self.convergence_check = convergence_check
        self.mixture_model = mixture_model
        self.resample = resample
        self.verbose = verbose
        self.save_history = save_history
        self.return_other = return_other
        self.return_caches = return_caches
        self.name = name
        self.callback = callback

    def __str__(self) -> str:
        """Return given name of method."""
        if self.name is not None:
            return self.name
        if self.num_comps == 1:
            return "CBREE"
        else:
            return f"CBREE ({self.cluster_model})"

    def __log_tgt_fun(
        self, lsf_evals: ndarray, e_fun_evals: ndarray, sigma: Real, method="sigmoid"
    ):
        """Compute log of current target density.

        Args:
            lsf_evals (ndarray): LSF evaluated in current sample. Shape is (N,)
            e_fun_evals (ndarray): Energy function of the problem evaluated in current sample. Shape is (N,)
            sigma (Real): Smoothing parameter of the indicator function.
            method (str, optional): Which smoothing function should be used.
                Can be one of:
                    * "algebraic"
                    * "tanh"
                    * "arctan"
                    * "erf"
                    * "relu"
                    * "sigmoid"
                Defaults to "sigmoid" if not provided or not set to one of the above.

        Returns:
            ndarray: log of current target density evaluated in current sample.
        """
        if method == "tanh":
            # 1/2 * (1+tanh(-sigma*lsf)) * e^(-e_fun)
            return -log(2) + log1p(tanh(-sigma * lsf_evals)) - e_fun_evals
        if method == "arctan":
            # 1/2 *(1+2/pi*arctan(-sigma*lsf)) * e^(-e_fun)
            return -log(2) + log1p(2 / pi * arctan(-sigma * lsf_evals)) - e_fun_evals
        if method == "algebraic":
            # 1/2 (1-sigma*lsf/sqrt(sigma**2lsf**2+1)) * e^(-e_fun)
            return (
                -log(2)
                + +log1p(-sigma * lsf_evals / sqrt(1 + sigma**2 * lsf_evals**2))
                - e_fun_evals
            )
        if method == "erf":
            # 1/2(1+erf(-sigma*lsf)) +e^(-e_fun)
            return univariat_normal.logcdf(-sigma * lsf_evals) - e_fun_evals
        if method == "relu":
            return -sigma * maximum(0, lsf_evals) ** 3 - e_fun_evals
        else:  # sigmoid
            # 1/(1+e^sigma*lsf) *e^(-e_fun)
            return -log1p(exp(sigma * lsf_evals)) - e_fun_evals

    def solve(self, prob: Problem) -> Solution:
        """Estimate the rare event of `prob`.

        Args:
            prob (Problem): Description of the rare event problem

        Returns:
            Solution: Estimate of probability of failure and other results from computation.
        """

        # Define counting lsf function, save energy function
        def my_lsf(x):
            my_lsf.counter += prod(x.shape[:-1])
            return prob.lsf(x)

        my_lsf.counter = 0
        self.lsf = my_lsf
        self.e_fun = prob.e_fun

        # initialize cache
        cache = CBREECache(
            prob.sample,
            self.lsf(prob.sample),
            self.e_fun(prob.sample),
            sigma=self.sigma,
            beta=self.beta,
            num_lsf_evals=self.lsf.counter,
        )
        self.__compute_weights(cache)
        if self.stepsize_adaptivity:
            self.__compute_initial_stepsize(cache)
        else:
            cache.t_step = self.t_step
        cache_list = [cache]

        # maybe print some info
        if self.verbose:
            cols = ["Iteration", "Sigma", "Beta", "Stepsize", "CVAR", "SFP", "Comment"]
            col_width = amax([len(s) for s in cols])
            table = PrettyTable(
                cols, float_format=".5", max_width=col_width, min_width=col_width
            )

        # start iteration
        msg = "Success"
        while (
            not cache_list[-1].converged and cache_list[-1].iteration <= self.num_steps
        ):
            # set stepsize for next two iterations
            if (
                self.stepsize_adaptivity
                and len(cache_list) > 1
                and cache_list[-1].iteration % 2 == 0
            ):
                self.__update_stepsize(cache_list)
            # set sigma and beta
            self.__update_beta_and_sigma(cache_list[-1])

            # perform step
            try:
                new_cache = self.__perfrom_step(cache_list[-1])
                cache_list.append(new_cache)
            except Exception as e:
                msg = str(e)
                if not self.verbose:
                    warn(str(e))
                break
            # maybe prune list
            if not self.save_history and len(cache_list) > self.observation_window:
                cache_list.pop(0)

            # check for convergence
            self.__convergence_check(cache_list)

            if self.callback is not None:
                try:
                    cache_list[-1] = self.callback(cache_list[-1], self)
                except Exception as e:
                    msg = str(e)
                    if not self.verbose:
                        warn(str(e))
                    break
            # maybe print info about this iteration
            if self.verbose:
                table.add_row(
                    [
                        cache_list[-1].iteration,
                        cache_list[-1].sigma,
                        cache_list[-1].beta,
                        log(1 / cache_list[-1].t_step),
                        cache_list[-1].cvar_is_weights,
                        cache_list[-1].sfp,
                        cache_list[-2].msg,
                    ]
                )
                print(
                    table.get_string(
                        start=len(table.rows) - 1,
                        end=len(table.rows),
                        header=cache_list[-1].iteration == 1,
                        border=False,
                    )
                )
        # Set message
        if not cache_list[-1].converged and cache_list[-1].iteration > self.num_steps:
            msg = "Not Converged."

        # importance sampling
        for c in cache_list:
            self.__importance_sampling(c)
        self.__compute_weighted_estimates(cache_list)

        # build solution
        other = {}
        other["Average Estimate"] = cache_list[-1].estimate_uniform_avg
        other["Root Weighted Average Estimate"] = cache_list[-1].estimate_sqrt_avg
        other["VAR Weighted Average Estimate"] = cache_list[-1].estimate_cvar_avg
        other["CVAR"] = cache_list[-1].cvar_is_weights
        other["SFP"] = cache_list[-1].sfp
        tmp = flatten_cache_list(cache_list)
        # correct sigma beta and t_step to match thesis notation
        tmp["sigma"] = insert(tmp["sigma"], 0, 0)
        tmp["beta"] = insert(tmp["beta"], 0, nan)
        tmp["t_step"] = insert(tmp["t_step"], 0, nan)

        if self.return_other:
            other = other | tmp
        if self.return_caches:
            other["cache_list"] = cache_list
        return Solution(
            tmp["ensemble"],
            tmp["beta"],
            tmp["lsf_evals"],
            tmp["estimate"],
            self.lsf.counter,
            msg,
            num_steps=cache_list[-1].iteration,
            other=other,
        )

    def __update_beta_and_sigma(self, cache: CBREECache) -> None:
        """Update sigma, beta and ess in cache.

        Args:
            cache (CBREECache): Updated cache.
        """
        if self.sigma_adaptivity == "cvar":
            self.__update_sigma_cvar(cache)
        if self.sigma_adaptivity == "sfp":
            self.__update_sigma_sfp(cache)

        if self.beta_adaptivity:
            self.__update_beta(cache)

        log_tgt_evals = self.__log_tgt_fun(
            cache.lsf_evals, cache.e_fun_evals, cache.sigma, method=self.tgt_fun
        )
        weights_ensemble = my_softmax(log_tgt_evals * cache.beta).squeeze()
        cache.ess = sum(weights_ensemble) ** 2 / sum(weights_ensemble**2)

    def __update_sigma_cvar(self, cache: CBREECache) -> None:
        """Update `sigma` based on `cvar_tgt`.

        Use share of failure particles for update if not enough particles are in failure domain.

        Args:
            cache (CBREECache): Cache to be updated.
        """
        try:
            if (sum(cache.lsf_evals <= 0) / len(cache.lsf_evals)) < self.sfp_tgt:
                self.__update_sigma_sfp(cache)
            else:
                # Define objective function
                def obj_fun(sigma):
                    log_approx_evals = self.__log_tgt_fun(
                        cache.lsf_evals, 0.0, sigma, method=self.tgt_fun
                    )
                    log_approx_evals -= self.__log_tgt_fun(
                        cache.lsf_evals, 0.0, cache.sigma, method=self.tgt_fun
                    )
                    return (my_log_cvar(log_approx_evals) - self.cvar_tgt) ** 2

                # Minimize objective function
                opt_sol = minimize_scalar(
                    obj_fun,
                    bounds=[
                        cache.sigma,
                        cache.sigma + self.lip_sigma * log(1 / cache.t_step),
                    ],
                    method="Bounded",
                )
                cache.sigma = opt_sol.x
        except Exception as e:
            msg = f"Failed to update sigma: {str(e)}"
            if self.verbose:
                cache.msg += msg
            else:
                info(f"Iteration {cache.iteration}: {msg}")

    def __update_sigma_sfp(self, cache: CBREECache) -> None:
        """Update `sigma` based on `sfp_tgt`.

        Args:
            cache (CBREECache): Cache to be updated.
        """
        current = sum(cache.lsf_evals <= 0) / len(cache.lsf_evals)
        delta_max = self.lip_sigma * log(1 / cache.t_step)
        if self.sfp_tgt > float_info.epsilon:
            delta = (
                sqrt(maximum(0, self.sfp_tgt - current))
                * delta_max
                / sqrt(self.sfp_tgt)
            )
        else:
            delta = 0
        cache.sigma += delta

    def __update_beta(self, cache: CBREECache) -> None:
        """Compute beta for current ensemble.

        Args:
            cache (CBREECache): Cache to be updated.

        """

        # Define objective function
        def obj_fun(temperature):
            log_tgt_evals = self.__log_tgt_fun(
                cache.lsf_evals, cache.e_fun_evals, cache.sigma, method=self.tgt_fun
            )
            weights = my_softmax(log_tgt_evals * temperature)
            val = sum(weights) ** 2 / sum(weights**2) - self.ess_tgt * len(weights)
            return val

        # Find root of objective function
        try:
            sol = root(obj_fun, cache.beta)
            if sol.x.item() <= 0:
                new_beta = cache.beta
            else:
                new_beta = sol.x.item()
            cache.beta = new_beta
        except Exception as e:
            msg = f"Failed to update beta: {str(e)}"
            if self.verbose:
                cache.msg += msg
            else:
                info(f"Iteration {cache.iteration}: {msg}")

    def __compute_weights(
        self, cache: CBREECache, return_weights=False
    ) -> Optional[ndarray]:
        """Compute weighted mean and covariance.

        Args:
            cache (CBREECache): Cache to be updated.
            return_weights (bool, optional): Whether to return the normalized weights. Defaults to False.

        Raises:
            ValueError: Raise error if weighted covariance contains nonfinite elements.

        Returns:
            Optional[ndarray]: Normalized weights.
        """
        log_tgt_evals = self.__log_tgt_fun(
            cache.lsf_evals, cache.e_fun_evals, cache.sigma, method=self.tgt_fun
        )
        weights_ensemble = my_softmax(log_tgt_evals * cache.beta).squeeze()
        cache.weighted_mean = average(cache.ensemble, axis=0, weights=weights_ensemble)
        cache.weighted_cov = cov(
            cache.ensemble, aweights=weights_ensemble, ddof=0, rowvar=False
        )
        if not isfinite(cache.weighted_cov).all():
            raise ValueError("Weighted covariance contains non-finite elements.")
        if return_weights:
            return weights_ensemble

    def __perfrom_step(self, cache: CBREECache) -> CBREECache:
        """Perform a consensus based sampling step.

        If `resample` is True and the mixture model is "vMFNM", fit a
        von Mises Fisher Nakagami mixture to the ensemble and resample the ensemble from this model.

        Args:
            cache (CBREECache): Cache with current ensemble.

        Returns:
            CBREECache: Cache with new ensemble.
        """
        if self.resample and self.mixture_model == "vMFNM":
            m_noise = (1 - cache.t_step) * cache.weighted_mean
            c_noise = (1 - cache.t_step**2) * (1 + cache.beta) * cache.weighted_cov
            ensemble_new = cache.t_step * cache.ensemble + self.rng.multivariate_normal(
                m_noise, c_noise, cache.ensemble.shape[0]
            )
            model = VMFNMixture(1)
            model.fit(ensemble_new)
            ensemble_new = model.sample(cache.ensemble.shape[0], rng=self.rng)
            log_pdf_evals = model.logpdf(ensemble_new)
        else:
            m_noise = (1 - cache.t_step) * cache.weighted_mean
            c_noise = (1 - cache.t_step**2) * (1 + cache.beta) * cache.weighted_cov
            ensemble_new = cache.t_step * cache.ensemble + self.rng.multivariate_normal(
                m_noise, c_noise, cache.ensemble.shape[0]
            )
            m_new = average(ensemble_new, axis=0)
            c_new = cov(ensemble_new, ddof=1, rowvar=False)
            log_pdf_evals = multivariate_normal.logpdf(
                ensemble_new, mean=m_new, cov=c_new
            )
            model = MixtureModel(1)

        cache_new = CBREECache(
            ensemble_new,
            self.lsf(ensemble_new),
            self.e_fun(ensemble_new),
            mixture_model=model,
        )
        cache_new.cvar_is_weights = my_log_cvar(
            -cache_new.e_fun_evals - log_pdf_evals,
            multiplier=(cache_new.lsf_evals <= 0),
        )
        cache_new.iteration = cache.iteration + 1
        cache_new.converged = cache.converged
        cache_new.sigma = cache.sigma
        cache_new.beta = cache.beta
        cache_new.t_step = cache.t_step
        cache_new.num_lsf_evals = self.lsf.counter
        self.__compute_weights(cache_new)
        return cache_new

    def __convergence_check(self, cache_list: list) -> None:
        """Check for convergence and/or divergence and save result in last cache.

        Args:
            cache_list (list): List of last chaches to sonsider for con- and divergence
        """
        iteration = cache_list[-1].iteration

        # compute and save quantities of interest
        histories = flatten_cache_list(
            cache_list[-self.observation_window :],
            attrs=["ensemble", "lsf_evals", "cvar_is_weights"],
        )
        histories["SFP"] = sum(histories["lsf_evals"] <= 0, axis=1)
        cache_list[-1].sfp = sum(cache_list[-1].lsf_evals <= 0) / len(
            cache_list[-1].lsf_evals
        )
        cache_list[-1].slope_cvar = get_slope(histories["cvar_is_weights"])
        cache_list[-1].sfp_slope = get_slope(histories["SFP"])

        # Check convergence
        if self.sigma_adaptivity == "cvar":
            cache_list[-1].converged = (
                cache_list[-1].cvar_is_weights <= self.cvar_tgt
            ) and self.convergence_check
            if cache_list[-1].converged:
                cache_list[-1].msg += "Converged with given `cvar_tgt`."
            if self.divergence_check and iteration >= self.observation_window:
                sfp_mean = average(
                    [c.sfp for c in cache_list[-self.observation_window :]]
                )
                cache_list[-1].converged = cache_list[-1].converged or (
                    cache_list[-1].slope_cvar > 0.0 and sfp_mean >= self.sfp_tgt
                )
                if cache_list[-1].converged:
                    cache_list[-1].msg += "Converged due to `divergence_check`."

        if self.sigma_adaptivity == "sfp":
            cache_list[-1].converged = (
                cache_list[-1].sfp >= self.sfp_tgt
            ) and self.convergence_check
            if cache_list[-1].converged:
                cache_list[-1].msg += "Converged with given `sfp_tgt`."

    def __update_stepsize(self, cache_list: list) -> None:
        """Compute new stepsize, save it in `cache_list[-1].t_step`.

        Assume that the stepsize `t_step` has been constant for the last two caches.

        Args:
            cache_list (list): cache list of length at least 2.
        """
        # compute higher order approximation of mean and covariance
        ch1 = cache_list[-2]  # 1. stage
        ch2 = cache_list[-1]  # 2. stage
        h = -log(ch1.t_step)  # half stepsize of 2 stage method
        # compute linear combination of means
        phi = (exp(-2 * h) - 1) / (-2 * h)
        bm1 = phi - 2 * (phi - 1) / (-2 * h)
        bm2 = 2 * (phi - 1) / (-2 * h)
        m1 = average(ch1.ensemble, axis=0)
        m2 = average(ch2.ensemble, axis=0)
        m2_hat = 2 * h * (bm1 * m1 + bm2 * m2)

        # compute linear combination of covs
        phi = (exp(-4 * h) - 1) / (-2 * h)
        bc1 = phi - 2 * (phi - 1) / (-2 * h)
        bc2 = 2 * (phi - 1) / (-2 * h)
        c1 = cov(ch1.ensemble, ddof=1, rowvar=False)
        c2 = cov(ch2.ensemble, ddof=1, rowvar=False)
        c2_hat = 2 * h * (bc1 * c1 + bc2 * c2)

        # Compute new stepsize
        x = concatenate((m2[None, ...], tril(c2)))
        x_hat = concatenate((m2_hat[None, ...], tril(c2_hat)))
        delta = x - x_hat
        w = self.stepsize_tolerance + self.stepsize_tolerance * maximum(
            abs(x), abs(x_hat)
        )
        err = sqrt(average((delta / w) ** 2))
        h_min = 1e-5
        h_max = 100
        fac = 0.9
        q = fac * sqrt(1 / err)
        t_new = maximum(exp(-h_max), minimum(exp(-h_min), exp(-q * h)))
        ch2.t_step = t_new

    def __compute_initial_stepsize(self, cache: CBREECache) -> None:
        """Compute initial stepsize `cache.t_step`

        Args:
            cache (CBREECache): Cache to be updated.
        """
        # first guess
        m0 = average(cache.ensemble, axis=0)
        c0 = cov(cache.ensemble, rowvar=False, ddof=1)
        x0 = concatenate((m0[None, ...], tril(c0)))  # initial value
        w = self.stepsize_tolerance + self.stepsize_tolerance * abs(x0)
        d0 = sqrt(average((x0 / w) ** 2))
        f0_m = -m0 + cache.weighted_mean
        f0_c = -2 * c0 + 2 * cache.weighted_cov
        f0 = concatenate((f0_m[None, ...], tril(f0_c)))  # initial rhs-eval
        d1 = sqrt(average((f0 / w) ** 2))
        h0 = 0.01 * d0 / d1
        cache.t_step = exp(-h0)  # first guess

        # Euler step
        self.__update_beta_and_sigma(cache)
        cache_new = self.__perfrom_step(cache)

        # Approximate second derivative
        m2 = average(cache_new.ensemble, axis=0)
        c2 = cov(cache_new.ensemble, rowvar=False, ddof=1)
        f2_m = -m2 + cache_new.weighted_mean
        f2_c = -2 * c2 + 2 * cache_new.weighted_cov
        f2 = concatenate((f2_m[None, ...], tril(f2_c)))
        d2 = sqrt(average(((f0 - f2 / w)) ** 2)) / h0

        # second guess
        h1 = sqrt(0.01 / maximum(d1, d2))
        cache.t_step = exp(-maximum(100 * h0, h1))
        cache.num_lsf_evals = cache_new.num_lsf_evals

    def __importance_sampling(self, cache: CBREECache) -> None:
        """Compute failure probability and save it in `cache.estimate`.

        Args:
            cache (CBREECache): Cache to be updated.
        """
        if cache.mixture_model.fitted:
            aux_logpdf = cache.mixture_model.logpdf(cache.ensemble)
        else:
            if self.mixture_model == "vMFNM":
                cache.mixture_model = VMFNMixture(1)
                cache.mixture_model.fit(cache.ensemble)
                cache.ensemble = cache.mixture_model.sample(
                    cache.ensemble.shape[0], rng=self.rng
                )
                cache.lsf_evals = self.lsf(cache.ensemble)
                cache.num_lsf_evals = self.lsf.counter
                aux_logpdf = cache.mixture_model.logpdf(cache.ensemble)
            else:
                aux_logpdf = multivariate_normal.logpdf(
                    cache.ensemble,
                    mean=average(cache.ensemble, axis=0),
                    cov=cov(cache.ensemble, rowvar=False, ddof=1),
                )
        tgt_logpdf = gaussian_logpdf(cache.ensemble)
        cache.estimate = importance_sampling(
            tgt_logpdf, aux_logpdf, cache.lsf_evals, logpdf=True
        )

    def __compute_weighted_estimates(self, cache_list: list) -> None:
        """Compute weighted failure probability estimates.

        Args:
            cache_list (list): Use estimates of caches in this list.
                Save results in `cache_list[-1]`.
        """
        k = minimum(len(cache_list), self.observation_window)
        sqrt_w = sqrt(arange(k))
        cvar_w = 1 / array([c.cvar_is_weights**2 for c in cache_list[-k:]])
        cvar_w[~isfinite(cvar_w)] = 0.0
        pfs = [c.estimate for c in cache_list[-k:]]
        cache_list[-1].estimate_uniform_avg = average(pfs)
        cache_list[-1].estimate_cvar_avg = (
            average(pfs, weights=cvar_w) if sum(cvar_w) > 0 else 0.0
        )
        cache_list[-1].estimate_sqrt_avg = average(pfs, weights=sqrt_w)

    def solve_from_caches(self, cache_list: list) -> Solution:
        """Simulate `solve` from existing list of caches.

        Can be used to test behavior of different stopping criteria.

        Args:
            cache_list (list): List of precomputed caches.

        Returns:
            Solution: Estimate of probability of failure and other results from computation.
        """
        # initialize
        cache_list_new = [cache_list.pop(0)]
        # maybe print some info
        if self.verbose:
            cols = ["Iteration", "Sigma", "Beta", "Stepsize", "CVAR", "SFP", "Comment"]
            col_width = amax([len(s) for s in cols])
            table = PrettyTable(
                cols, float_format=".5", max_width=col_width, min_width=col_width
            )

        # start iteration
        msg = "Success"
        while (
            not cache_list_new[-1].converged
            and cache_list_new[-1].iteration <= self.num_steps
            and len(cache_list) > 0
        ):
            # perform step
            cache_list_new.append(cache_list.pop(0))

            # maybe prune list
            if not self.save_history and len(cache_list_new) > self.observation_window:
                cache_list_new.pop(0)

            # check for convergence
            self.__convergence_check(cache_list_new)
            if self.callback is not None:
                try:
                    cache_list_new[-1] = self.callback(cache_list_new[-1], self)
                except Exception as e:
                    msg = str(e)
                    if not self.verbose:
                        warn(str(e))
                    break

            # maybe print info about this iteration
            if self.verbose:
                table.add_row(
                    [
                        cache_list_new[-1].iteration,
                        cache_list_new[-1].sigma,
                        cache_list_new[-1].beta,
                        log(1 / cache_list_new[-1].t_step),
                        cache_list_new[-1].cvar_is_weights,
                        cache_list_new[-1].sfp,
                        cache_list_new[-2].msg,
                    ]
                )
                print(
                    table.get_string(
                        start=len(table.rows) - 1,
                        end=len(table.rows),
                        header=cache_list_new[-1].iteration == 1,
                        border=False,
                    )
                )
        # Set message
        if (
            not cache_list_new[-1].converged
            and cache_list_new[-1].iteration > self.num_steps
        ):
            msg = "Not Converged."

        # importance sampling
        for c in cache_list_new:
            self.__importance_sampling(c)
        self.__compute_weighted_estimates(cache_list_new)

        # build solution
        other = {}
        other["Average Estimate"] = cache_list_new[-1].estimate_uniform_avg
        other["Root Weighted Average Estimate"] = cache_list_new[-1].estimate_sqrt_avg
        other["VAR Weighted Average Estimate"] = cache_list_new[-1].estimate_cvar_avg
        other["CVAR"] = cache_list_new[-1].cvar_is_weights
        other["SFP"] = cache_list_new[-1].sfp
        tmp = flatten_cache_list(cache_list_new)
        # correct sigma beta and t_step to match thesis notation
        tmp["sigma"] = insert(tmp["sigma"], 0, 0)
        tmp["beta"] = insert(tmp["beta"], 0, nan)
        tmp["t_step"] = insert(tmp["t_step"], 0, nan)

        if self.return_other:
            other = other | tmp
        if self.return_caches:
            other["cache_list"] = cache_list_new
        return Solution(
            tmp["ensemble"],
            tmp["beta"],
            tmp["lsf_evals"],
            tmp["estimate"],
            cache_list_new[-1].num_lsf_evals,
            msg,
            num_steps=cache_list_new[-1].iteration,
            other=other,
        )


class ENKF(Solver):
    """
    This class is a wrapper for code from Fabian Wagner.
    Please consult the credits section of the package's readme file.
    """

    def __init__(self, **kwargs) -> None:
        """
        The constructor can handle the following keyword arguments.

        Args:
            cvar_tgt:  Stop if coefficient of variation is smaller than this, Defaults to 1.
            mixture_model: Which model is used to for resampling. Defaults to "GM".
                Can be one of:
                    * "GM"
                    * "vMFNM"
            num_steps: Maximal number of steps. Defaults to 100.
            num_comps: Number of components for multimodal problems. Defaults to 1.
            localize: Whether to use local covariances.
            seed: Seed for the random number generator. Defaults to None.
        """
        super().__init__()
        self.cvar_tgt = kwargs.get("cvar_tgt", 1.0)
        self.mixture_model = kwargs.get("mixture_model", "GM")
        assert self.mixture_model in [
            "GM",
            "vMFNM",
        ], "Importance sampling distribution must be 'GM' or 'vMFNM'."
        self.num_steps = kwargs.get("num_steps", 100)
        self.num_comps = kwargs.get("num_comps", 1)
        self.localize = kwargs.get("localize", False)
        self.seed = kwargs.get("seed", None)

    def __str__(self) -> str:
        """Return abbreviated name of method."""
        return f"EnKF ({self.mixture_model})"

    def solve(self, prob: Problem):
        """Estimate the rare event of `prob`.

        Args:
            prob (Problem): Description of the rare event problem

        Returns:
            Solution: Estimate of probability of failure and other results from computation.
        """
        try:
            enkf = EnKF_rare_events(
                lambda x: maximum(prob.lsf(x), 0),
                zeros((1, 1)),
                self.num_steps,
                prob.sample.shape[0],
                prob.sample.shape[1],
                self.cvar_tgt,
                0,
                False,
                g_original=prob.lsf,
                k_init=self.num_comps,
                mixture_model=self.mixture_model,
            )
            enkf.uk = prob.sample
            enkf.uk_initial = enkf.uk.copy()
            enkf.uk_save.append(enkf.uk)
            enkf.mean_k[0, :] = mean(enkf.uk, axis=0)

            seed(self.seed)
            enkf.perform_iteration()
            enkf.calculate_failure_probability()
            return Solution(
                array([enkf.uk_fitted]),
                array(enkf.sigma_save),
                array([enkf.gk]),
                array([enkf.pf]),
                enkf.number_fun_eval,
                "Success",
                num_steps=enkf.iter,
                other={"Final Iteration": array([enkf.uk])},
            )
        except Exception as e:
            warn(str(e))
            Solution(
                zeros(0),
                array([nan]),
                zeros((1, 0)) * nan,
                array([nan]),
                nan,
                str(e),
            )


class SIS(Solver):
    """
    This class is a wrapper for code from the Engineering Risk Analysis Group, TU Munich.
    Please consult the credits section of the package's readme file.
    """

    def __init__(self, **kwargs) -> None:
        """
        The constructor can handle the following keyword arguments.

        Args:
            num_chains_wrt_sample: Use `num_chains_wrt_sample`*sample-size number of chains for the MCMC routine.
            burn_in: burn-in period for MCMC routine.
            cvar_tgt (Real, optional): Stop if coefficient of variation is smaller than this. Defaults to 1.
            mixture_model (str, optional): Which method is used to for resampling. Defaults to "GM".
                Can be one of:
                    * "GM"
                    * "aCS"
                    * "vMFNM"
            num_comps: Number of components for multimodal problems. Defaults to 1.
            seed (int, optional): Seed for the random number generator. Defaults to None.
            fixed_initial_sample (bool): Whether to use the initial sample provided by `prob` in `solve`.
                Defaults to True.
            verbose (bool, optional): Whether to print information during solving. Defaults to False.
            return_other (bool, optional): Whether to return the final number of components
                and coefficient of variation of the weights.
        """
        super().__init__()
        self.num_chains_wrt_sample = kwargs.get("num_chains_wrt_sample", 0.1)
        self.burn_in = kwargs.get("burn_in", 0)
        self.cvar_tgt = kwargs.get("cvar_tgt", 1.0)
        self.mixture_model = kwargs.get("mixture_model", "GM")
        assert self.mixture_model in [
            "GM",
            "aCS",
            "vMFNM",
        ], "mixture_model must be either 'GM' or 'aCS'"
        self.num_comps = kwargs.get("num_comps", 1)
        self.seed = kwargs.get("seed", None)
        self.fixed_initial_sample = kwargs.get("fixed_initial_sample", True)
        self.verbose = kwargs.get("verbose", False)
        self.return_other = kwargs.get("return_other", False)

    def __str__(self) -> str:
        """Return abbreviated name of method."""
        return f"SiS ({self.mixture_model})"

    def solve(self, prob: Problem):
        """Estimate the rare event of `prob`.

        Args:
            prob (Problem): Description of the rare event problem

        Returns:
            Solution: Estimate of probability of failure and other results from computation.
        """
        # Check and set up input for SIS subroutine
        assert isinstance(
            prob.eranataf_dist, ERANataf
        ), "For sequential importance sampling, prob needs to have an ERANataf distribution."
        N = prob.sample.shape[0]
        d = prob.sample.shape[1]
        p = self.num_chains_wrt_sample
        my_lsf = Vectorizer(prob.lsf, d)
        if self.fixed_initial_sample:
            initial_sample = deepcopy(prob.sample)
        else:
            initial_sample = None
        # Solve
        try:
            if self.mixture_model == "aCS":
                Pr, l_tot, samplesU, samplesX, COV_Sl = SIS_aCS(
                    N,
                    p,
                    my_lsf,
                    prob.eranataf_dist,
                    self.burn_in,
                    self.cvar_tgt,
                    seed=self.seed,
                    initial_sample=initial_sample,
                    transform_lsf=not isinstance(prob, NormalProblem),
                    verbose=self.verbose,
                )
                k_fin = nan
            elif self.mixture_model == "vMFNM":
                Pr, l_tot, _, _, _, _, samplesX, _ = sis(
                    N,
                    p,
                    my_lsf,
                    self.burn_in,
                    self.cvar_tgt,
                    d,
                    k_init=1,
                    initial_sample=initial_sample,
                    verbose=self.verbose,
                    seed=self.seed,
                )
            else:
                Pr, l_tot, samplesU, samplesX, k_fin, COV_Sl = SIS_GM(
                    N,
                    p,
                    my_lsf,
                    prob.eranataf_dist,
                    self.num_comps,
                    self.burn_in,
                    self.cvar_tgt,
                    seed=self.seed,
                    initial_sample=initial_sample,
                    transform_lsf=not isinstance(prob, NormalProblem),
                    verbose=self.verbose,
                )
        except Exception as e:
            warn(str(e))
            Solution(
                zeros(0, 0),
                array([nan]),
                zeros((1, N)) * nan,
                array([nan]),
                my_lsf.num_evals,
                str(e),
            )

        # Set up Solution object, no history available!
        if self.return_other:
            other = {"k_fin": k_fin, "COV_Sl": COV_Sl}
        else:
            other = None
        return Solution(
            samplesX[-1][None, ...],
            array([nan]),
            zeros((1, N)) * nan,
            array([Pr]),
            my_lsf.num_evals,
            "Success",
            num_steps=l_tot,
            other=other,
        )


class CMC(Solver):
    """Crude Monte Carlo simulation."""

    def __init__(self, num_runs: int, seed: int = None, verbose=True) -> None:
        """
        The constructor can handle the following keyword arguments.

        Args:
            num_runs (int): How often to sample from the problems distribution.
            seed (int, optional): Seed for sampling from the problems distribution. Defaults to None.
            verbose (bool, optional): Whether to print information during solving. Defaults to True.
        """

        super().__init__()
        self.num_runs = num_runs
        self.seed = seed
        self.verbose = verbose
        self.name = "Crude Monte Carlo Simulation"

    def __str__(self) -> str:
        return self.name

    def solve(self, prob: Problem) -> Solution:
        batch_size = 2**10
        rest = self.num_runs % batch_size
        num_batches = self.num_runs // batch_size + (1 if rest != 0 else 0)
        lsf_evals = nan * zeros(self.num_runs)
        final_sample = nan * zeros((self.num_runs, prob.sample.shape[-1]))
        msg = "Success"
        cost = 0
        pf = 0
        cvar = 0
        for i in range(num_batches):
            try:
                if self.verbose:
                    print(f"Batch {i} of {num_batches}.", end="")
                # set seed for this batch
                batch_seed = None
                if self.seed is not None:
                    batch_seed = self.seed + i * self.num_runs
                # compute current batch size
                current_batch_size = (
                    rest if (i == num_batches - 1 and rest != 0) else batch_size
                )
                # evaluate this batch
                prob.set_sample(current_batch_size, seed=batch_seed)
                final_sample[cost : cost + current_batch_size, ...] = prob.sample
                lsf_evals[cost : cost + current_batch_size] = prob.lsf(prob.sample)
                cost += current_batch_size
                pf = sum(lsf_evals[0:cost] <= 0) / cost
                cvar = variation(lsf_evals[0:cost] <= 0, nan_policy="omit")
                if self.verbose:
                    print(f" Current estimate: {pf}, cvar: {cvar}")

            except Exception as e:
                warn(str(e))
                print(traceback.format_exc())
                msg = str(e)
        prob.sample = final_sample

        return Solution(
            final_sample[None, ...],
            nan * zeros(1),
            lsf_evals[None, ...],
            ones(1) * pf,
            cost,
            msg,
            other={"cvar": cvar},
        )
