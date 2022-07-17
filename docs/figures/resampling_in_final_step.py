#%% 
from glob import glob
from os import listdir, path
from unittest.mock import call
import scipy as sp
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.config import config
from sklearn import ensemble; config.update("jax_enable_x64", True)
import rareeventestimation as ree
import numpy as np
import scipy as sp
import sympy as smp
import pandas as pd
from rareeventestimation.evaluation.constants import *
import plotly.express as px
import plotly.graph_objects as go
%load_ext autoreload
%autoreload 2
# %% Choose a toy example
prob = ree.prob_convex
problem_list=[prob]
sigma = 10
tgt_fun = "algebraic"

# # %% Compute Laplace approximation of toy example
# def jlog_tgt_fun(lsf_evals, e_fun_evals, sigma, method="sigmoid"):
#     """Differentiable version of CBREE.__log_tgt_fun."""
#     if method == "tanh":
#         # 1/2 * (1+tanh(-sigma*lsf)) * e^(-e_fun)
#         return -jnp.log(2) + jnp.log1p(jnp.tanh(-sigma*lsf_evals)) - e_fun_evals
#     if method == "arctan":
#         # 1/2 *(1+2/pi*arctan(-sigma*lsf)) * e^(-e_fun)
#         return -jnp.log(2) + jnp.log1p(2/jnp.pi*jnp.arctan(-sigma*lsf_evals)) - e_fun_evals
#     if method == "algebraic":
#         # 1/2 (1-sigma*lsf/sqrt(sigma**2lsf**2+1)) * e^(-e_fun)
#         return -jnp.log(2) + jnp.log1p(-sigma*lsf_evals/jnp.sqrt(1+sigma**2*lsf_evals**2)) - e_fun_evals
#     if method == "erf":
#         # 1/2(1+erf(-sigma*lsf)) +e^(-e_fun)
#         return jsp.univariat_normal.logcdf(-sigma*lsf_evals) - e_fun_evals
#     if method == "relu":
#         return -sigma*jnp.maximum(0,lsf_evals)**3 - e_fun_evals
#     else: # sigmoid
#         # 1/(1+e^sigma*lsf) *e^(-e_fun)
#         return -jnp.log1p(jnp.exp(sigma*lsf_evals)) - e_fun_evals
# jlogpdf = lambda x: -1*np.log(2*np.pi) - (x[...,0]**2+x[...,1]**2)/2
# def jlsf_convex(x): return 0.1*(x[...,0]-x[...,1])**2 - 1/jnp.sqrt(2)*jnp.sum(x,axis=-1)+2.5

# # Find mode of target distribution by optimization
# def obj_fun(x): return -jlog_tgt_fun(jlsf_convex(x),-jlogpdf(x),sigma)
# def obj_grad(x): return jax.grad(obj_fun)(x)
# def obj_hess(x): return jax.hessian(obj_fun)(x)
# def obj_fun_1d(x): return obj_fun(x*np.ones(2))
# sol = sp.optimize.minimize(
#     obj_fun,
#     prob.mpp,
#     jac = obj_grad,
#     hess = obj_hess,
#     tol = 1e-15
# )
# # Check: mode lies on halfline through (1,1)
# sol1d = sp.optimize.minimize_scalar(
#     obj_fun_1d,
#     bracket= np.average(sol.x) + [-0.25,0.25],
#     tol = 1e-15,
#     options= {"maxiter":1e4}
# )
# if sol.fun < sol1d.fun:
#     lpl_mean = np.asarray(sol.x)
# else:
#     lpl_mean = np.asarray(sol1d.x * np.ones(2))
#     print("1d Optimization was better.")
# print(f"Mode is at {lpl_mean}. Gradient here is {obj_grad(lpl_mean)}")
# lpl_cov  = np.asarray(np.linalg.inv(obj_hess(lpl_mean)))


# Test different sampling approaches
# %%1. Compute data or reload it from `./data`
def callback_gm(cache, solver):
    if not cache.converged:
        return cache
    cache.mixture_model = ree.GaussianMixture(1,seed=solver.seed)
    cache.mixture_model.fit(cache.ensemble)
    cache.ensemble = cache.mixture_model.sample(cache.ensemble.shape[0], rng=solver.rng)
    cache.lsf_evals = solver.lsf(cache.ensemble)
    cache.e_fun_evals = solver.e_fun(cache.ensemble)
    log_pdf_evals = cache.mixture_model.logpdf(cache.ensemble)
    cache.cvar_is_weights = ree.my_log_cvar(-cache.e_fun_evals - log_pdf_evals, multiplier=(cache.lsf_evals<=0))
    return cache

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
    return cache

keywords = {
    "callback": [None, callback_gm, callback_vmfnm]
}

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = "object"
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

prod = cartesian_product(*[np.array(v) for v in keywords.values()])
solver_list = []
kwarg_list = []
for col in prod:
    kwargs = dict(zip(keywords.keys(), col))
    solver = ree.CBREE(tgt_fun=tgt_fun,divergence_check=False, observation_window=2, return_other=True, **kwargs)
    solver.name = "CBREE " + str(kwargs)
    solver_list.append(solver)
    kwarg_list.append(kwargs)
    
sample_sizes =[250, 500, 1000, 2000, 3000, 4000, 5000, 6000]
num_runs = 200
other_list=["sigma", "beta", "cvar_is_weights"]
out_dir = "./data"
pattern = f"{prob.name.replace(' ', '_')}*"
total = len(solver_list)*len(sample_sizes)*len(problem_list)
counter = 1
if not (len(glob(path.join(out_dir, pattern+".csv"))) == total or path.exists(path.join(out_dir, "processed_data.pkl"))):
    for s in sample_sizes:
        for prob in problem_list:
            for i, solver in enumerate(solver_list):
                print(f"({counter}/{total}) {prob.name}, {s} Samples, with {solver.name}")
                prob.set_sample(s, seed=s)
                ree.do_multiple_solves(prob,
                                solver,
                                num_runs,
                                dir=out_dir,
                                prefix=f"{prob.name}_".replace(" ", "_"),
                                other_list=other_list,
                                addtnl_cols=kwarg_list[i])
                counter +=1
#%% 2. prepare data
if not path.exists(path.join(out_dir, "processed_data.pkl")):
    df = ree.load_data(out_dir, pattern)
    # Nice solver names
    df.loc[df["callback"].isna(),"Solver"] = "CBREE"
    df.loc[df["callback"].isna(),"callback"] = "None"
    df.loc[df["callback"].str.contains("gm"), "Solver"] = "CBREE (GM)"
    df = df.loc[~df["callback"].str.contains("vmfnm"),:].reset_index()
    df = ree.add_evaluations(df)
    df_agg = ree.aggregate_df(df)
    df_agg.to_pickle(path.join(out_dir, "processed_data.pkl"))
else:
    df_agg = pd.read_pickle(path.join(out_dir, "processed_data.pkl"))
# %% Plot relative error of different sampling methods
figs = ree.make_accuracy_plots(df_agg, layout=MY_LAYOUT, CMAP=CMAP)
fig = figs[0]

fig_name="resampling_in_final_step"
fig.update_yaxes(title_text = "LSF Evaluations")
fig.update_layout(title_text = "", height=800)
fig.write_image(fig_name + ".png",scale=WRITE_SCALE)
fig.show()
# %%
fig_description = f"Solving the {prob.name} with two CBREE methods using  \
$J \\in \\{{{', '.join(map(str, sample_sizes))}\\}}$ particles, \
the stopping criterion $\\Delta_{{\\text{{Target}}}} = {solver.cvar_tgt}$, \
the stepsize tolerance $\\epsilon_{{\\text{{Target}}}} = {solver.stepsize_tolerance}$, \
controlling the increase of $\\sigma$ with $\\text{{Lip}}(\\sigma) = {solver.lip_sigma}$ \
and approximating the indicator function with {INDICATOR_APPROX_LATEX_NAME[solver.tgt_fun]}. \
No divergence check has been performed. \
Each simulation was repeated {num_runs} times. \
While the markers present the empirical means of the visualized quantities, the error bars are drawn from first to the third quartile."
with open(fig_name + "_desc.tex", "w") as file:
    file.write(fig_description)
print(fig_description)


#%% 2. prepare data for effective sample size study
if not  path.exists(path.join(out_dir, "ess_data.pkl")):
    df = ree.load_data(out_dir, pattern)
    # Nice solver names
    df.loc[df["callback"].isna(),"Solver"] = "CBREE"
    df.loc[df["callback"].isna(),"callback"] = "None"
    df.loc[df["callback"].str.contains("gm"), "Solver"] = "CBREE (GM)"
    df = df.loc[~df["callback"].str.contains("vmfnm"),:].reset_index()
    df = ree.add_evaluations(df)
    df["VAR IS Weights"] = (df["Estimate"] * df["cvar_is_weights"] )**2
    df["J_ESS"] = df["VAR IS Weights"] / df["Estimate Variance"]
    df["J_ESS"] = df.apply(lambda x: x["J_ESS"][-1], axis=1)
    df_ess = df[["Problem", "Solver", "Sample Size", "J_ESS"]]
    df_ess.to_pickle(path.join(out_dir, "ess_data.pkl"))
else:
    df_ess = pd.read_pickle(path.join(out_dir, "ess_data.pkl"))

df_ess["J_ESS"] =(df_ess["Sample Size"] -  df_ess.J_ESS) / df_ess["Sample Size"]

# %%
df_ess.sort_values(by="Solver", inplace = True)
fig_hist = px.box(df_ess,
                  x = "Sample Size",
                        y = "J_ESS",
                        color="Solver",
                        points=False,
                        color_discrete_sequence = CMAP,
                        labels={"Solver": "Method"})
df_ess_agg = df_ess.groupby(["Solver","Sample Size"]).mean().reset_index()
# for s in df_ess.Solver.unique():
#     tmp = df_ess_agg[df_ess_agg.Solver == s]
#     fig_hist.add_trace(
#         go.Scatter(
#             x = tmp["Sample Size"].values,
#             y = tmp.J_ESS.values,
#             showlegend=False,
#             mode="markers",
#             marker_symbol=1,
#         )
#     )
fig_hist.update_layout(**MY_LAYOUT)
fig_hist.update_layout(height=800)
fig_hist.update_xaxes(title_text = "Sample Size <i>J</i>")
fig_hist.update_yaxes(title_text = f"Relative Error of ESS(<b><i>r</b></i>) Estimate")
fig_hist.write_image(fig_name + "_boxplot.png",scale=WRITE_SCALE)
fig_hist.show()
# %%
