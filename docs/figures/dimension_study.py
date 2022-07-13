#%% 
from re import sub
from sympy import solve
import rareeventestimation as ree
import numpy as np
import scipy as sp
import pandas as pd
from rareeventestimation.evaluation.constants import *
import plotly.express as px
import plotly.graph_objects as go

# %% set up
dim_list = np.arange(10, 160, 10)
J=5000
num_runs=50

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

kwargs_dict = {
    "CBREE (GM)": dict(divergence_check=False),
    "CBREE (vMFNM)": dict(callback=callback_vmfnm, 
                          observation_window=6),
    "CBREE (vMFNM, resampled)": dict(resample=True,
                                     mixture_model="vMFNM",
                                     divergence_check=False)
}
solver_list = [
    ree.CBREE(name=solver_name, num_steps=50, **kwargs) 
    for solver_name, kwargs in kwargs_dict.items()
]

problem_list = [
    ree.make_fujita_rackwitz(d).\
        set_sample(J, seed=J)
    for d in dim_list
]

#%%Solve
problem_list.reverse()
for prob in problem_list:
    for solver in solver_list:
        if solver.name == "CBREE (vMFNM)":
            # obs window does not make sense if not all caches can be used for
            # importance sampling: here only the last cache has a fitted vmfnm mixture 
            # due to the callback!
            ree.do_multiple_solves(
                prob,
                solver,
                num_runs=num_runs,
                dir = "./data",
                prefix=sub(r"\W+", "_",f"{prob.name} {solver.name}".lower()),
                other_list=["Average Estimate",
                            "Root Weighted Average Estimate",
                            "VAR Weighted Average Estimate",
                            "CVAR"],
                addtnl_cols=kwargs_dict[solver.name])
        else:
            ree.study_cbree_observation_window(
                prob,
                solver,
                num_runs=num_runs,
                dir = "./data",
                prefix=sub(r"\W+", "_",f"{prob.name} {solver.name}".lower()),
                observation_window_range=[2,4,6,8,10,12],
                other_list=["Average Estimate",
                            "Root Weighted Average Estimate",
                            "VAR Weighted Average Estimate",
                            "CVAR"],
                addtnl_cols=kwargs_dict[solver.name])
# %%
