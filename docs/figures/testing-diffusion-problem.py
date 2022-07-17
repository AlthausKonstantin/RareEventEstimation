#%%
import rareeventestimation as ree
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rareeventestimation.evaluation.constants import *
import numpy as np
%load_ext autoreload
%autoreload 2
# %%
# Set up solvers
def callback_vmfnm(cache, solver):
    if not cache.converged:
        return cache
    cache.mixture_model = ree.VMFNMixture(1)
    cache.mixture_model.fit(cache.ensemble)
    cache.ensemble = cache.mixture_model.sample(cache.ensemble.shape[0], rng=solver.rng)
    cache.lsf_evals = solver.lsf(cache.ensemble)
    cache.e_fun_evals = solver.e_fun(cache.ensemble)
    log_pdf_evals = cache.mixture_model.logpdf(cache.ensemble)
    cache.cvar_is_weights = ree.my_log_cvar(-cache.e_fun_evals - log_pdf_evals, multiplier=(cache.lsf_evals<=0))
    cache.num_lsf_evals += cache.ensemble.shape[0]
    return cache

prob = ree.diffusion_problem
prob.set_sample(1000, seed=1000)
cbree_resample_always = ree.CBREE(resample=True,
                                  mixture_model="vMFNM",
                                  name="CBREE (always resample)",
                                  stepsize_tolerance=.5,
                                  tgt_fun="algebraic",
                                  cvar_tgt=5,
                                  verbose=True,
                                  divergence_check=False,
                                  seed=1)
#%%
sol1 = cbree_resample_always.solve(prob)
sol2 = cbree_resample_is.solve(prob)
# %%
