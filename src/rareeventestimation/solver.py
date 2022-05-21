     
from copy import deepcopy
from dataclasses import dataclass
from sys import float_info

from numpy import (amax, arange, array, average, concatenate, cov, diff, exp,
                   invert, isfinite, isnan, log, log1p, maximum, minimum, nan,
                   ndarray, pi, prod, sqrt, tril, where, sum)
from numpy.linalg import norm
from numpy.random import default_rng
from prettytable import PrettyTable
from scipy.optimize import minimize_scalar, root
from scipy.special import erfc
from scipy.stats import multivariate_normal

from rareeventestimation.mixturemodel import MixtureModel, VMFNMixture
from rareeventestimation.problem import Problem
from rareeventestimation.solution import Solution
from rareeventestimation.utilities import (gaussian_logpdf, get_slope, importance_sampling,
                       my_log_cvar, my_softmax)


@dataclass
class CBSCache:
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
    converged:bool = False
    iteration: int = 0
    delta:float = nan
    delta_slope:float = nan
    slope_cvar: float = nan
    sfp_slope:float = nan
    estimate:float = 0.0
    estimtae_uniform_avg: float = 0.0
    estimate_sqrt_avg:float = 0.0
    estimate_cvar_avg:float = 0.0
    sfp:float=0.0
    ess:float= nan

def flatten_cache_list(cache_list:list, attrs=None)->dict:
    if attrs is None:
        attrs = (vars(cache_list[0]).keys())
    out = dict.fromkeys(attrs)
    for a in attrs:
        v = [getattr(c, a) for c in cache_list if getattr(c, a) is not None]
        out[a] = array(v).squeeze()
    return out


class Solver:
    """
    This class provides methods to estimate the rare event probability of a
    problem given as an instance of the class `Problem`.
    """

    def __init__(self) -> None:
        pass

    def solve(self, prob: Problem):
        """ To be implemented by different child classes."""
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
        
   
class CBS(Solver):
    """
    Instances of this class hold all the options for consensus based sampling.

    Set solver options in the constructor.
    Use method solve to apply conensus based sampling to a problem.
    """

    def __init__(self, **kwargs) -> None:
        """Handle all possible keywords specifying solver options."""
        super().__init__()
        self.t_step = kwargs.get("t_step", 0.1)
        self.adaptive_stepsize = kwargs.get("adaptive_stepsize", True)
        self.tol=kwargs.get("tol", 0.5)
        self.num_steps = kwargs.get("num_steps", 100)
        self.reject = kwargs.get("reject",False)
        self.observation_window = int(kwargs.get("observation_window", 5))
        self.temp = kwargs.get("temp", 1.0)
        self.sigma = kwargs.get("sigma", 1)
        self.tgt_fun = kwargs.get("tgt_fun", "algebraic")
        self.seed = kwargs.get("seed", None)
        self.rng = default_rng(self.seed)
        
        self.sigma_adaptivity_scheme = kwargs.get("sigma_adaptivity_scheme", 1)
        # sigma_adaptivity_scheme==1: CVAR based update and convergence check with self.cvar_tgt
        # sigma_adaptivity_scheme==2: SFP based update and convergence check with self.sfp_tgt
        # sigma_adaptivity_scheme==None: No sigma update and convergence check
        self.sfp_tgt = kwargs.get("sfp_tgt", 0.33)
        self.cvar_tgt = kwargs.get("cvar_tgt",2)
        
        self.beta_adaptivity_scheme = kwargs.get("beta_adaptivity_scheme", 1)
        # beta_adaptivity_scheme==1: ESS based update and distance convergence check with self.ess and self.delta_tgt
        # beta_adaptivity_scheme==None: No beta update and convergence check
        self.ess =kwargs.get("ess",0.5)
        
        self.sigma_inc_max = kwargs.get("sigma_inc_max",10)
        self.divergence_check=kwargs.get("divergence_check", True)
        
    
        self.verbose = kwargs.get("verbose", False)
        self.save_history = kwargs.get("save_history", False)
        
        self.num_comps = kwargs.get("num_comps",1)
        self.mixture_model = kwargs.get("mixture_model", "GM")
        
        self.name = kwargs.get("name", None)
        self.resample= kwargs.get("resample", False)
        self.return_caches = kwargs.get("return_caches", False)
        
    def __str__(self) -> str:
        """Return abbreviated name of method."""
        if self.name is not None:
            return self.name
        if self.num_comps == 1:
            return "CBS"
        else:
            return f"CBS ({self.cluster_model})"

    
    def __log_tgt_fun(self, lsf_evals, e_fun_evals, sigma, method="sigmoid"):
        if method == "tanh":
            return -log1p(exp(2 * sigma * lsf_evals)) - e_fun_evals
        if method == "algebraic":  
            return -log(2) - log(1+sigma**2*lsf_evals**2 + sigma*lsf_evals*sqrt(1+sigma**2*lsf_evals**2)) -e_fun_evals
        if method == "erf":
            #kudos to https://stackoverflow.com/a/60095295
            # and https://math.stackexchange.com/a/3508869
            tmp = where(sigma * lsf_evals > 10, erfc(sigma * lsf_evals), nan)
            msk = isnan(tmp)
            tmp[msk] =-(sigma * lsf_evals[msk])**2 \
                         -log(pi)/2 \
                         -log(sigma * lsf_evals[msk])
            log(tmp, out=tmp, where=invert(msk))
            return -log(4) + tmp - e_fun_evals
        else:
            return -e_fun_evals - log1p(exp(sigma * lsf_evals))

    def solve(self, prob:Problem) ->  Solution:
        
        # Define counting lsf function
        def my_lsf(x):
            my_lsf.counter += prod(x.shape[:-1])
            return prob.lsf(x)
        my_lsf.counter = 0
        self.lsf = my_lsf
        self.e_fun = prob.e_fun
        
        #initialize cache
        cache = CBSCache(prob.sample,  self.lsf(prob.sample),  self.e_fun(prob.sample),
                             sigma=self.sigma, beta=self.temp, t_step=self.t_step)
        self.__compute_weights(cache)
        if self.adaptive_stepsize:
            self.__compute_initial_stepsize(cache)
        cache_list = [cache]
        
        msg = "Success"   
        if self.verbose:
            cols = ['Iteration', 'Sigma', "Beta", "Stepsize", "CVAR", "SFP", "Delta","Converged"]
            col_width = amax([len(s) for s in cols])
            table = PrettyTable(cols, float_format=".5",max_width=col_width, min_width=col_width)
        while not cache_list[-1].converged and cache_list[-1].iteration <= self.num_steps:
            # set stepsize for next two iterations
            if self.adaptive_stepsize and len(cache_list) > 1 and len(cache_list) % 2 == 1:
                self.__update_stepsize(cache_list)
            # set sigma and or beta
            self.__update_beta_and_sigma(cache_list[-1])
            
            # perform step
            try:
                new_cache = self.__perfrom_step(cache_list[-1])
                cache_list.append(new_cache)
            except Exception as e:
                msg = str(e)
            # maybe prune list
            if not self.save_history and len(cache_list) > self.observation_window:
                cache_list.pop(0)
            
            # check for convergence
            self.__convergence_check(cache_list)
            if self.verbose:
                table.add_row([cache_list[-1].iteration, 
                              cache_list[-1].sigma,
                              cache_list[-1].beta,
                              log(1/cache_list[-1].t_step),
                              cache_list[-1].cvar_is_weights,
                              cache_list[-1].sfp,
                              cache_list[-1].delta,
                              cache_list[-1].converged])
                print(table.get_string(start=len(table.rows)-1, 
                                       end=len(table.rows),
                                       header=cache_list[-1].iteration==1,
                                       border=False))
        # Set message
        if not cache_list[-1].converged and cache_list[-1].iteration > self.num_steps:
            msg = "Not Converged."
            
        # importance sampling
        for c in cache_list:
            self.__importance_sampling_cachebased(c)
        self.__compute_weighted_estimates(cache_list)    
            
        # build solution
        other = {}
        other["Average Estimate"] = cache_list[-1].estimtae_uniform_avg
        other["Root Weighted Average Estimate"] = cache_list[-1].estimate_sqrt_avg
        other["VAR Weighted Average Estimate"] = cache_list[-1].estimate_cvar_avg
        other["CVAR"] = cache_list[-1].cvar_is_weights
        other["SFP"] = cache_list[-1].sfp
        tmp = flatten_cache_list(cache_list)
        
        if self.return_caches:
            other = other|tmp
        return Solution(
            tmp["ensemble"],
            tmp["beta"],
            tmp["lsf_evals"],
            tmp["estimate"],
            lambda x: nan,
            self.lsf.counter,
            msg,
            num_steps=cache_list[-1].iteration,
            other=other
        )
        
    def __update_beta_and_sigma(self, cache: CBSCache) -> None:
        if self.sigma_adaptivity_scheme == 1:
            # cvar based update
            self.__update_sigma_cvar(cache)
        if self.sigma_adaptivity_scheme == 2:
            # sfp based sigma update
            self.__update_sigma_sfp_cachebases(cache)
        if self.sigma_adaptivity_scheme is None:
            # constant sigma
            pass
        if self.beta_adaptivity_scheme is not None:
            # ess based update
            self.__update_beta(cache)
        if self.beta_adaptivity_scheme is None:
            # constant beta
            pass
        log_tgt_evals = self.__log_tgt_fun(cache.lsf_evals, cache.e_fun_evals, cache.sigma, method=self.tgt_fun)
        weights_ensemble = my_softmax(log_tgt_evals*cache.beta).squeeze()
        cache.ess = sum(weights_ensemble) ** 2 / sum(weights_ensemble**2) 
     
    def __update_sigma_cvar(self, cache:CBSCache) -> None:
        def obj_fun(sigma):
            log_approx_evals = self.__log_tgt_fun(cache.lsf_evals, 0.0, sigma, method=self.tgt_fun)
            log_approx_evals -= self.__log_tgt_fun(cache.lsf_evals, 0.0, cache.sigma, method=self.tgt_fun)
            return (my_log_cvar(log_approx_evals) -self.cvar_tgt)**2
        try:
            if (sum(cache.lsf_evals <= 0) / len(cache.lsf_evals)) < self.sfp_tgt:
                self.__update_sigma_sfp_cachebases(cache)
            else:
                opt_sol = minimize_scalar(obj_fun, bounds=[cache.sigma, cache.sigma+self.sigma_inc_max*log(1/self.t_step)], method="Bounded")
                cache.sigma = opt_sol.x
        except Exception:
                pass
        
    def __update_sigma_sfp_cachebases(self, cache:CBSCache) -> None:
        current = sum(cache.lsf_evals <=0) / len(cache.lsf_evals)
        delta_max = self.sigma_inc_max*log(1/self.t_step)
        if self.sfp_tgt > float_info.epsilon:
            delta = sqrt(maximum(0,self.sfp_tgt-current)) * delta_max/sqrt(self.sfp_tgt)
        else:
            delta = 0
        new_sigma = cache.sigma + delta

        cache.sigma = new_sigma
    
    def __update_beta(self, cache:CBSCache) -> None:
        """Compute temperature acc. to eq. (4.3) in Carillo2021."""
        # Define objective function
        def obj_fun(temperature):
            log_tgt_evals = self.__log_tgt_fun(cache.lsf_evals,cache.e_fun_evals, cache.sigma, method=self.tgt_fun)
            weights = my_softmax(log_tgt_evals*temperature)
            val = sum(weights)**2 / sum(weights**2) - self.ess*len(weights)
            return val

        # Find root of objective function
        try:
            sol = root(obj_fun, cache.beta)
            if sol.x.item() <= 0:
                new_beta = cache.beta
            else:
                new_beta = sol.x.item()
            cache.beta = new_beta
        except Exception:
            pass
    
    def __compute_weights(self, cache:CBSCache) -> None:
        # Compute weighted mean and covariance
        log_tgt_evals = self.__log_tgt_fun(cache.lsf_evals, cache.e_fun_evals, cache.sigma, method=self.tgt_fun)
        weights_ensemble = my_softmax(log_tgt_evals*cache.beta).squeeze()
        cache.weighted_mean = average(cache.ensemble, axis=0, weights=weights_ensemble)
        cache.weighted_cov = cov(cache.ensemble, aweights=weights_ensemble, ddof=0, rowvar=False)
        if not isfinite(cache.weighted_cov).all():
            raise ValueError("Weighted covariance contains non-finite elements.")

    def __perfrom_step(self, cache:CBSCache) -> CBSCache:
    
        if self.resample and self.mixture_model == "vMFNM":
            m_noise= (1 - cache.t_step) * cache.weighted_mean
            c_noise= (1 - cache.t_step**2) * (1+cache.beta) * cache.weighted_cov
            ensemble_new = cache.t_step * cache.ensemble + self.rng.multivariate_normal(m_noise, c_noise, cache.ensemble.shape[0])
            model= VMFNMixture(1)
            model.fit(ensemble_new)
            ensemble_new = model.sample(cache.ensemble.shape[0], rng=self.rng)
            log_pdf_evals = model.logpdf(ensemble_new)
        else:
            m_noise= (1 - cache.t_step) * cache.weighted_mean
            c_noise= (1 - cache.t_step**2) * (1+cache.beta) * cache.weighted_cov
            ensemble_new = cache.t_step * cache.ensemble + self.rng.multivariate_normal(m_noise, c_noise, cache.ensemble.shape[0])
            m_new = average(ensemble_new, axis=0)
            c_new = cov(ensemble_new, ddof=1, rowvar=False)
            log_pdf_evals = multivariate_normal.logpdf(
                ensemble_new, mean=m_new, cov=c_new)
            model = MixtureModel(1)

        cache_new = CBSCache(ensemble_new, self.lsf(ensemble_new), self.e_fun(ensemble_new), mixture_model=model)
        cache_new.cvar_is_weights = my_log_cvar(-cache_new.e_fun_evals - log_pdf_evals, multiplier=(cache_new.lsf_evals<=0))
        cache_new.iteration = cache.iteration + 1
        cache_new.converged = cache.converged
        cache_new.sigma=cache.sigma
        cache_new.beta = cache.beta
        cache_new.t_step = cache.t_step
        self.__compute_weights(cache_new)
        return cache_new

    def __convergence_check(self, cache_list:list,other=None) -> None:
        iteration = cache_list[-1].iteration

        # compute and save quantities of interest
        histories = flatten_cache_list(cache_list[-self.observation_window:],
                                       attrs=["ensemble", "lsf_evals", "cvar_is_weights"])
        histories["SFP"] = sum(histories["lsf_evals"]<=0, axis = 1)
        means = average(histories["ensemble"], axis=1)
        covs = array([cov(e, rowvar=False, ddof=1) for e in histories["ensemble"]])
        delta_means = norm(diff(means, axis=0), axis=1)
        delta_covs = norm(diff(covs, axis=0), axis=(1,2))
        cache_list[-1].sfp =sum(cache_list[-1].lsf_evals <=0) / len(cache_list[-1].lsf_evals) 
        cache_list[-1].delta = (average(delta_means) + average(delta_covs)) / 2
        cache_list[-1].slope_cvar = get_slope(histories["cvar_is_weights"])
        cache_list[-1].sfp_slope = get_slope(histories["SFP"])
        histories = flatten_cache_list(cache_list[-self.observation_window:],
                                       attrs=["delta"])
        if iteration >= 2 * self.observation_window -1:
            cache_list[-1].delta_slope = get_slope(histories["delta"])
       
        
        # Check  convergence
        if self.sigma_adaptivity_scheme == 1:
            cache_list[-1].converged = cache_list[-1].cvar_is_weights <= self.cvar_tgt

            if self.divergence_check and iteration >= self.observation_window:
                sfp_mean = average([c.sfp for c in cache_list[-self.observation_window:]])
                cache_list[-1].converged = cache_list[-1].converged or (cache_list[-1].slope_cvar > 0.0 and sfp_mean>=self.sfp_tgt)

        if self.sigma_adaptivity_scheme == 2:
            cache_list[-1].converged =cache_list[-1].sfp >= self.sfp_tgt

        if self.sigma_adaptivity_scheme is None:
            cache_list[-1].converged = False
                  
    def __update_stepsize(self, cache_list:list) -> None:
        # compute higher order approximation of mean and covariance
        ch1 = cache_list[-2]
        ch2 = cache_list[-1]
        t_step = ch2.t_step
        h = -log(t_step)
        bm1 = 1 - t_step**2 + 4/h *(1 - t_step)
        bm2 = 4/h *(t_step - 1)
        m2 = average(ch1.ensemble, axis=0)
        m2_hat = t_step**2 * m2 + \
            bm1 * ch1.weighted_mean + \
            bm2 * ch2.weighted_mean
        bc1 = 1 - t_step**4 + 4/h * (1 - t_step**2)
        bc2 = 4/h * (t_step**2 - 1)
        c2 = cov(ch1.ensemble, ddof=1, rowvar=False)
        c2_hat = t_step**4 * c2 + \
            bc1 * ch1.weighted_cov + \
            bc2 * ch2.weighted_cov
            
        # Compute new stepsize 
        x = concatenate((m2[None,...], tril(c2)))
        x_hat = concatenate((m2_hat[None,...], tril(c2_hat)))
        delta = x - x_hat
        tol = self.tol
        w = tol + tol*maximum(abs(x), abs(x_hat))
        err = sqrt(average((delta/w)**2))
        h_min= 1e-5
        h_max= 100
        fac = 0.9
        q = fac * sqrt(1/err)
        t_new = maximum(exp(-h_max), minimum(exp(-h_min), exp(-q*h)))
        if self.reject and q<1:
            ch1.t_step = t_new
            cache_list.pop()
        else:
            ch2.t_step = t_new

    def __compute_initial_stepsize(self, cache:CBSCache) -> None:
        # first guess
        m0 = average(cache.ensemble, axis=0) 
        c0 = cov(cache.ensemble,rowvar=False, ddof=1)
        x0 = concatenate((m0[None,...], tril(c0))) # initial value
        d0 = sqrt(average(x0**2))
        f0_m = -m0 + cache.weighted_mean
        f0_c = -2*c0 + 2*cache.weighted_cov
        f0 = concatenate((f0_m[None,...], tril(f0_c))) # initial rhs-eval
        w = self.tol + self.tol*abs(x0)
        d1 = sqrt(average((f0/w)**2))
        h0 = 0.01*d0/d1
        cache.t_step = exp(-h0) # first guess
        
        # Euler step
        self.__update_beta_and_sigma(cache)
        cache_new = self.__perfrom_step(cache)
        
        # Aaproximate second derivative
        m2 = average(cache_new.ensemble, axis=0) 
        c2= cov(cache_new.ensemble,rowvar=False, ddof=1)
        f2_m = -m2 + cache_new.weighted_mean
        f2_c = -2*c2 + 2*cache_new.weighted_cov
        f2 = concatenate((f2_m[None,...], tril(f2_c)))
        d2 =  sqrt(average((f0 - f2)**2)) / cache.t_step
        
        # second guess
        h1 = sqrt(0.01 / maximum(d1,d2))
        cache.t_step = exp(-maximum(100*h0,h1))
    
    def __importance_sampling_cachebased(self, cache: CBSCache) -> None:
        if cache.mixture_model.fitted:
            aux_logpdf = cache.mixture_model.logpdf(cache.ensemble)
        else:
            if self.mixture_model == "vMFNM":
                cache.mixture_model = VMFNMixture(1)
                cache.mixture_model.fit(cache.ensemble)
                cache.ensemble = cache.mixture_model.sample(cache.ensemble.shape[0], rng=self.rng)
                cache.lsf_evals = self.lsf(cache.ensemble)
                aux_logpdf = cache.mixture_model.logpdf(cache.ensemble)
            else:
                aux_logpdf = multivariate_normal.logpdf(cache.ensemble,
                                                    mean=average(cache.ensemble,axis=0),
                                                    cov=cov(cache.ensemble,rowvar=False, ddof=1))
        tgt_logpdf = gaussian_logpdf(cache.ensemble)
        cache.estimate = importance_sampling(tgt_logpdf, aux_logpdf, cache.lsf_evals, logpdf=True)    

    def __compute_weighted_estimates(self, cache_list) -> None:
        k = minimum(len(cache_list), self.observation_window)
        sqrt_w =  sqrt(arange(k))
        cvar_w = 1 / array([c.cvar_is_weights**2 for c in cache_list[-k:]])
        cvar_w[~isfinite(cvar_w)] = 0.0
        pfs = [c.estimate for c in cache_list[-k:]]
        cache_list[-1].estimtae_uniform_avg = average(pfs)
        cache_list[-1].estimate_cvar_avg = average(pfs, weights=cvar_w)  if sum(cvar_w) > 0  else 0.0
        cache_list[-1].estimate_sqrt_avg = average(pfs, weights=sqrt_w)
        
