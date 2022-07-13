#%% 

from numbers import Real
from os import path
import rareeventestimation as ree
import numpy as np
import scipy as sp
import pandas as pd
import plotly.express as px
from rareeventestimation.evaluation.constants import INDICATOR_APPROX_LATEX_NAME, BM_SOLVER_SCATTER_STYLE, MY_LAYOUT, DF_COLUMNS_TO_LATEX, LATEX_TO_HTML, WRITE_SCALE
import plotly.graph_objects as go
from rareeventestimation.evaluation.visualization import add_scatter_to_subplots, sr_to_color_dict
import re
%load_ext autoreload
%autoreload 2

#%% Load data

data_dir ="/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation_data Kopie/cbree_sim/toy_problems"
path_df= path.join(data_dir, "cbree_toy_problems_processed.pkl")
path_df_agg = path.join(data_dir, "cbree_toy_problems_aggregated.pkl")
path_df_agg_all = path.join(data_dir, "cbree_toy_problems_aggregated_all.pkl")
if   (path.exists(path_df) and path.exists(path_df_agg)):
    df = ree.load_data(data_dir, "*")
    df.drop(columns=["index", "Unnamed: 0"], inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    #%% Round parameters
    for col in DF_COLUMNS_TO_LATEX.keys():
        if isinstance(df[col].values[0], float):
            df[col] = df[col].round(5)
    #%%process data: add obs_window and callback to solver name
    def expand_cbree_name(input, columns=[]) -> str:
        for col in columns:
            input.Solver = input.Solver.replace("}", f", '{col}': '{input[col]}'}}")
        return input
    df = df.apply(expand_cbree_name, axis=1, columns= ['observation_window', 'callback'])
    # %% Pretty names
    to_drop = ["mixture_model"] # info is redundant as resample = False and callback exists
    replace_values = {"Method": {"False": "CBREE (GM)", "vMFNM Resample": "CBREE (vMFNM)"}}
    df = df.drop(columns=to_drop) \
        .rename(columns=DF_COLUMNS_TO_LATEX) \
        .replace(replace_values)
    #%%process data: add evaluations
    df_success = ree.add_evaluations(df.copy(),  only_success=True)
    df_all = ree.add_evaluations(df.copy())
    # %% aggregate
    df_agg = ree.aggregate_df(df_success)
    df_agg_all = ree.aggregate_df(df_all)
    for col in DF_COLUMNS_TO_LATEX.values():
        if isinstance(df_agg[col].values[0], float):
            df_agg[col] = df_agg[col].round(5)
    for col in DF_COLUMNS_TO_LATEX.values():
            if isinstance(df_agg_all[col].values[0], float):
                df_agg_all[col] = df_agg_all[col].round(5)

    #%% save
    df_success.to_pickle(path_df)
    df_agg.to_pickle(path_df_agg)
    df_agg_all.to_pickle(path_df_agg_all)
else:
    df_success = pd.read_pickle(path_df)
    df_agg = pd.read_pickle(path_df_agg)
    df_agg_all = pd.read_pickle(path_df_agg_all)
#%%
df_bm, df_bm_agg = ree.get_benchmark_df()
# %% Filter for lowdim stuff
low_dim_probs= ["Convex Problem", "Linear Problem (d=2)", "Fujita Rackwitz (d=2)"]

# %% Make table with used parameters
def my_number_formatter(x:Real) -> str:
    if int(x)==x:
        return str(int(x))
    else:
        return f"{x:.2f}"
    
def list_to_latex(ls:list) -> str:
    if np.all(np.array([isinstance(v, Real) for v in ls])):
        return f"$\\{{ {', '.join(map(my_number_formatter, ls))} \\}}$"
    else:
         return f"{', '.join(ls)}"
     
paras = ["Sample Size"] + list(DF_COLUMNS_TO_LATEX.values())
tbl_params = df_agg.loc[:, tuple(paras)] \
    .replace({"Smoothing Function": INDICATOR_APPROX_LATEX_NAME}) \
    .apply(lambda col: col.sort_values().unique()) \
    .apply(list_to_latex)
tbl_params = tbl_params.to_frame(name="Values")
tbl_params.index.name = "Parameter"
tbl_params.style.to_latex("parameters_used_toy_problems.tex", clines="all;data")
tbl_params

# %% Decide which indicator approximation is better
def decide(grp,par):
    best = grp.sort_values("Relative Root MSE")[par].values[0]
    return(pd.Series([best], index =[f"{par}"]))
df_tgt_fun = df_agg.query("Problem in @low_dim_probs")\
    .loc[:,("Problem", "Sample Size", "$\\Delta_{{\\text{{Target}}}}$", "Method", "Smoothing Function", "Relative Root MSE", "$N_{{ \\text{{obs}} }}$", "$\\epsilon_{{\\text{{Target}}}}$")] \
    .groupby(["Problem", "Sample Size", "$\\Delta_{{\\text{{Target}}}}$", "Method", "$N_{{ \\text{{obs}} }}$", "$\\epsilon_{{\\text{{Target}}}}$"]) \
    .apply(decide, "Smoothing Function") 
tbl = df_tgt_fun.reset_index().value_counts(subset=["Smoothing Function", "Problem"], normalize=False)\
    .to_frame()\
    .unstack(level=1)
tbl.columns = tbl.columns.droplevel(0)
totals = tbl.sum()
tbl = tbl/totals
tbl["Total"] = tbl.mean(numeric_only=True, axis=1)
tbl = tbl.sort_values("Total", ascending=False)
best_approximation = tbl.index.values[0]
tbl = tbl.applymap(lambda x: f"{x*100:.2f}\%")
tbl = tbl.rename(index = INDICATOR_APPROX_LATEX_NAME)
tbl = tbl.rename(columns={c:'\\rotatebox{60}{' + c + '}' for c in tbl.columns})
tbl.style.to_latex("performance_approximations.tex", clines="all;data")
tbl_description = f"Comparing the estimates of $\\textup{{relRootMSE}}(\\hat{{P}}_f)$ for different smoothing functions averaged over  all other parameter choices. The values denote the relative number of cases the corresponding smoothing function performed best for the given problem ({int(totals[0])} per problem)."
with open(f"performance_approximations_desc.tex", "w") as file:
    file.write(tbl_description)
print(tbl_description)
tbl

# %% Decide which stepsize tolerance is better
df_tol = df_agg.query("Problem in @low_dim_probs")\
    .loc[:,("Problem", "Sample Size", "$\\Delta_{{\\text{{Target}}}}$", "Method", "$\\epsilon_{{\\text{{Target}}}}$", "Relative Root MSE", "$N_{{ \\text{{obs}} }}$","Smoothing Function")] \
    .groupby(["Problem", "Sample Size", "$\\Delta_{{\\text{{Target}}}}$", "Method", "$N_{{ \\text{{obs}} }}$", "Smoothing Function"]) \
    .apply(decide,  "$\\epsilon_{{\\text{{Target}}}}$")     
tbl = df_tol.reset_index().value_counts(subset=["$\\epsilon_{{\\text{{Target}}}}$", "Problem"], normalize=False)\
    .unstack(level=1)
totals = tbl.sum()
tbl = tbl/totals
tbl["Total"] = tbl.mean(numeric_only=True, axis=1)
tbl = tbl.sort_values("Total", ascending=False)
best_tolerance = tbl.index.values[0]
tbl = tbl.applymap(lambda x: f"{x*100:.2f}\%")
tbl = tbl.rename(columns={c:'\\rotatebox{60}{' + c + '}' for c in tbl.columns}).\
    rename(index={idx:str(idx) for idx in tbl.index })
tbl.style.to_latex("performance_stepsize_tolerance.tex", clines="all;data")
tbl_description = f"Comparing the estimates of $\\textup{{relRootMSE}}(\\hat{{P}}_f)$ for different values of $\\epsilon_{{\\text{{Target}}}}$ averaged over  all other parameter choices. The values denote the relative number of cases (total {int(totals[0])}) the corresponding value performed best for the given problem."
with open(f"performance_stepsize_tolerance_desc.tex", "w") as file:
    file.write(tbl_description)
print(tbl_description)
tbl

#%% Compare success rates for diverengence check
def vec_to_latex_set(vec:np.ndarray) -> str:
    vec = np.reshape(vec, (-1))
    vec = np.sort(vec)
    if len(vec) == 1:
        return my_number_formatter(vec[0])
    if len(vec) == 2:
        return f"\\{{{my_number_formatter(vec[0])}, {my_number_formatter(vec[-1])}\\}}"
    if len(vec) > 2:
       return f"\\{{{my_number_formatter(vec[0])}, {my_number_formatter(vec[1])}, \\ldots, {my_number_formatter(vec[-1])}\\}}"
    
    
df_rates = df_agg.query("Problem in @low_dim_probs & Method == 'CBREE (GM)'")
df_rates = df_rates[df_rates["$\\epsilon_{{\\text{{Target}}}}$"]==best_tolerance]
df_rates = df_rates[df_rates["Smoothing Function"] == best_approximation]
df_rates = df_rates[df_rates["$\\Delta_{{\\text{{Target}}}}$"]==1]
tbl_success_rates = pd.pivot_table(df_rates, values="Success Rate", index='$N_{{ \\text{{obs}} }}$', columns="Problem", aggfunc=np.mean)
tbl_success_rates = tbl_success_rates.groupby(list(tbl_success_rates))\
    .apply(lambda x: vec_to_latex_set(x.index.values))\
    .to_frame(name='$N_{{ \\text{{obs}} }}$')\
    .reset_index()\
    .set_index('$N_{{ \\text{{obs}} }}$') \
    .applymap(lambda x: f"{x*100:.2f}\%")
tbl_description = f"Comparing the success rates of the CBREE (GM) method for different values of $N_{{ \\text{{obs}} }}$  and $\\Delta_{{\\text{{Target}}}}=1$ averaged over all sample sizes $J =   {vec_to_latex_set(df_rates['Sample Size'].unique())}$. \
The parameter $\\epsilon_{{\\text{{Target}}}} = {best_tolerance}$ \
and the choice of the indicator approximation {INDICATOR_APPROX_LATEX_NAME[best_approximation]} \
are fixed. \
The values denote the relative number of cases the CBREE (GM) method converged successfully for the particular combination of problem and paramter setting (total {6*200})."
with open(f"success_obs_window_desc.tex", "w") as file:
    file.write(tbl_description)
print(tbl_description)
tbl_success_rates.style.to_latex("success_obs_window.tex", clines="all;data")
tbl_success_rates

#%% Compare performance for divergence check
df_rates_all = df_agg_all.query("Problem in @low_dim_probs")
df_rates_all = df_rates_all[df_rates_all["$\\epsilon_{{\\text{{Target}}}}$"]==best_tolerance]
df_rates_all = df_rates_all[df_rates_all["Smoothing Function"] == best_approximation]
df_rates_all = df_rates_all[df_rates_all["$\\Delta_{{\\text{{Target}}}}$"]==1]

tbl_success_err = pd.pivot_table(df_rates, values="Relative Root MSE", index='$N_{{ \\text{{obs}} }}$', columns="Problem", aggfunc=np.mean)

tbl_all_err = pd.pivot_table(df_rates_all, values="Relative Root MSE", index='$N_{{ \\text{{obs}} }}$', columns="Problem", aggfunc=np.mean)

tbl_success_err.loc[0,:] = tbl_all_err.loc[0,:] 
tbl_success_err = tbl_success_err / tbl_success_err.loc[0,:] 
tbl_success_err = tbl_success_err[tbl_success_err.index.isin([2,5,10])]
tbl_success_err = tbl_success_err.applymap(lambda x: f"{x*100:.2f}\%")
tbl_success_err.index = pd.Index(map(my_number_formatter, tbl_success_err.index), name=tbl_success_rates.index.name)
tbl_success_err
tbl_describtion = f"Comparing the emprirical relative root MSE of the CBREE (GM) method for different values of $N_{{ \\text{{obs}} }} > 1$  and $\\Delta_{{\\text{{Target}}}}=1$ to the base case of ommiting the divergence check. \
The parameter $\\epsilon_{{\\text{{Target}}}} = {best_tolerance}$ \
and the choice of the indicator approximation {INDICATOR_APPROX_LATEX_NAME[best_approximation]} \
are fixed. \
For each problem and choice of $N_{{ \\text{{obs}} }} > 1$ as well as for the base case we average the empirical relative root MSE over all sample sizes $J$ before computing the proportion to the base case."
with open(f"performance_obs_window_desc.tex", "w") as file:
    file.write(tbl_describtion)
print(tbl_describtion)
tbl_success_err.style.to_latex("performance_obs_window.tex", clines="all;data")
tbl_success_err

# %% Compare cost for obs_window
tbl_success_cost = pd.pivot_table(df_rates, values="Cost Mean", index='$N_{{ \\text{{obs}} }}$', columns="Problem", aggfunc=np.mean)

tbl_all_cost = pd.pivot_table(df_rates_all, values="Cost Mean", index='$N_{{ \\text{{obs}} }}$', columns="Problem", aggfunc=np.mean)

tbl_success_cost.loc[0,:] = tbl_all_cost.loc[0,:] 
tbl_success_cost = tbl_success_cost / tbl_success_cost.loc[0,:] 
tbl_success_cost = tbl_success_cost[tbl_success_cost.index.isin([2,5,10])]
tbl_success_cost = tbl_success_cost.applymap(lambda x: f"{x*100:.2f}\%")
tbl_success_cost.index = pd.Index(map(my_number_formatter, tbl_success_cost.index), name=tbl_success_cost.index.name)
tbl_description = f"Comparing the emprirical average cost of the CBREE (GM) method for different values of $N_{{ \\text{{obs}} }} > 1$  and $\\Delta_{{\\text{{Target}}}}=1$ to the base case of ommiting the divergence check. \
The parameter $\\epsilon_{{\\text{{Target}}}} = {best_tolerance}$ \
and the choice of the indicator approximation {INDICATOR_APPROX_LATEX_NAME[best_approximation]} \
are fixed. \
For each problem and choice of $N_{{ \\text{{obs}} }} > 1$ as well as for the base case we average the empirical average cost over all sample sizes $J$ before computing the proportion to the base case."
with open(f"cost_obs_window_desc.tex", "w") as file:
    file.write(tbl_description)
print(tbl_description)
tbl_success_cost.style.to_latex("cost_obs_window.tex", clines="all;data")
tbl_success_cost

#%% How often is the divergence check triggered for CBREE (vMFNM)
df_vmfnm = df_success.query("Problem in @low_dim_probs & Method == 'CBREE (vMFNM)'")
df_vmfnm = df_vmfnm[df_vmfnm["$\\epsilon_{{\\text{{Target}}}}$"]==best_tolerance]
df_vmfnm = df_vmfnm[df_vmfnm["Smoothing Function"] == best_approximation]
df_vmfnm = df_vmfnm[df_vmfnm["$N_{{ \\text{{obs}} }}$"].isin([0,2,5,10])]
df_vmfnm = df_vmfnm[["Problem", "Solver", "Sample Size", "Steps", "$N_{{ \\text{{obs}} }}$", "$\\Delta_{{\\text{{Target}}}}$"]]
df_vmfnm = df_vmfnm.groupby(["Problem", "Sample Size", "$\\Delta_{{\\text{{Target}}}}$"])\
    .apply(lambda grp: len(grp["Steps"].unique()))
#%% make plots

fig_list= []
for prob in df_agg.Problem.unique():
    this_df = df_agg.query("Problem == @prob")
    this_df = this_df[this_df["$\\epsilon_{{\\text{{Target}}}}$"]==best_tolerance]
    this_df = this_df[this_df["Smoothing Function"] == best_approximation]
    this_df = this_df[this_df['$N_{{ \\text{{obs}} }}$'].isin([2,5,10])]
    cmap = sr_to_color_dict(this_df["$\\Delta_{{\\text{{Target}}}}$"])
    this_df["cvar_tgt_str"] = this_df["$\\Delta_{{\\text{{Target}}}}$"].apply(str)
    this_df = this_df.sort_values(["$\\Delta_{{\\text{{Target}}}}$", "$N_{{ \\text{{obs}} }}$"])
    
    
    
    fig = px.line(
        this_df,
        x = "Relative Root MSE",
        y="Cost Mean",
        facet_col="Method",
        facet_row="$N_{{ \\text{{obs}} }}$",
        color_discrete_map=cmap,
        color="cvar_tgt_str",
        log_x=True,
        log_y=True,
        markers=True,
        labels=LATEX_TO_HTML | {"cvar_tgt_str": LATEX_TO_HTML[DF_COLUMNS_TO_LATEX["cvar_tgt"]]},
        
    )
    # add benchmark
    this_df_bm = df_bm_agg.query("Problem == @prob & cvar_tgt == 1")
    num_rows = len(this_df["$N_{{ \\text{{obs}} }}$"].unique())
    num_cols = len(this_df["Method"].unique())
    print(prob)
    print(this_df_bm.Solver.unique())
    for bm_solver in this_df_bm.Solver.unique():
        dat =this_df_bm.query("Solver == @bm_solver")
        trace_dict = {
            "x" : dat["Relative Root MSE"],
            "y" : dat["Cost Mean"],
            "legendgrouptitle_text": "Benchmark Methods",
            "name": bm_solver,
            "legendgroup": "group",
            "mode": "markers+lines",
            "opacity": 0.8
        }
        trace_dict = trace_dict | BM_SOLVER_SCATTER_STYLE[bm_solver]
        fig = add_scatter_to_subplots(fig, num_rows, num_cols, **trace_dict)
    # 
    fig.update_layout(**MY_LAYOUT)
    fig.update_layout(**{"width":700,
    "height":900})
    fig.for_each_annotation(
        lambda a: a.update(yshift =  -10 if a.text.startswith("Method") else 0))
    fig.show()
    fig.write_image(f"{prob} stopping criterion.png".replace(" ", "_").lower(), scale=WRITE_SCALE)
# %%
fig_description = f"Solving low dimensional toy problems with the CBREE methods using  \
different parameters. \
We vary the stopping criterion $\\Delta_{{\\text{{Target}}}}$ (color), \
the divergence criterion $N_\\text{{obs}}$ (row) and \
the importance sampling density $\\mu^N$ (column). \
The parameter $\\epsilon_{{\\text{{Target}}}} = {best_tolerance}$ \
and the choice of the indicator approximation {INDICATOR_APPROX_LATEX_NAME[best_approximation]} \
are fixed. \
Furthermore we plot also the performance of the benchmark methods EnKF\
(with different importance sampling densities)\
and SiS (with different MCMC sampling methods). \
We used the sample sizes $J \\in {vec_to_latex_set(df_agg['Sample Size'].unique())}$.\
Each marker represents the empirical estimates based the successful portion of $200$ simulations."
with open(f"lowdim_stopping criterion_desc.tex".replace(" ", "_").lower(), "w") as file:
    file.write(fig_description)
print(fig_description)

# %%
def get_convergence_rate(grp):
    grp = grp.sort_values("Relative Root MSE",ascending=True)
    xx = grp["Relative Root MSE"].values
    yy = np.log10(grp["Cost Mean"].values)
    line = np.polyfit(xx, yy, 1)
    return(pd.Series([line[-1]], ["Convergence Rate With Respect to Cost"]))
# %%
