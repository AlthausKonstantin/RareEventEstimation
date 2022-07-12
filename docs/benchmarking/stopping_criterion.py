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

data_dir ="/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation_data/cbree_sim/toy_problems"
path_df= path.join(data_dir, "cbree_toy_problems_processed.pkl")
path_df_agg = path.join(data_dir, "cbree_toy_problems_aggregated.pkl")
if  (path.exists(path_df) and path.exists(path_df_agg)):
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
    df = ree.add_evaluations(df)
    # %% aggregate
    df_agg = ree.aggregate_df(df)
    for col in DF_COLUMNS_TO_LATEX.values():
        if isinstance(df_agg[col].values[0], float):
            df_agg[col] = df_agg[col].round(5)
    # %% remove obs window 0
    df = df[df['$N_{{ \\text{{obs}} }}$']>0].reset_index()
    df_agg = df_agg[df_agg['$N_{{ \\text{{obs}} }}$']>0].reset_index()
    
    #%% save
    df.to_pickle(path_df)
    df_agg.to_pickle(path_df_agg)
else:
    df = pd.read_pickle(path_df)
    df_agg = pd.read_pickle(path_df_agg)
df_bm, df_bm_agg = ree.get_benchmark_df()

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
df_tgt_fun = df_agg.loc[:,("Problem", "Sample Size", "$\\Delta_{{\\text{{Target}}}}$", "Method", "Smoothing Function", "Relative Root MSE", "$N_{{ \\text{{obs}} }}$", "$\\epsilon_{{\\text{{Target}}}}$")] \
    .groupby(["Problem", "Sample Size", "$\\Delta_{{\\text{{Target}}}}$", "Method", "$N_{{ \\text{{obs}} }}$", "$\\epsilon_{{\\text{{Target}}}}$"]) \
    .apply(decide, "Smoothing Function") 
tbl = df_tgt_fun.reset_index().value_counts(subset=["Smoothing Function", "Problem"], normalize=False)\
    .to_frame()\
    .unstack(level=1)
tbl.columns = tbl.columns.droplevel(0)
totals = tbl.sum()
tbl = tbl/totals
tbl["Total"] = tbl.mean(numeric_only=True, axis=1)
tbl = tbl.applymap(lambda x: f"{x*100:.2f}\%")
tbl = tbl.rename(index = INDICATOR_APPROX_LATEX_NAME)
tbl = tbl.rename(columns={c:'\\rotatebox{60}{' + c + '}' for c in tbl.columns})
tbl.style.to_latex("performance_approximations.tex", clines="all;data")
tbl_description = f"Comparing the estimates of $\\textup{{relRootMSE}}(\\hat{{P}}_f)$ for different smoothing functions averaged over  all other parameter choices. The values denote the relative number of cases (total {int(totals[0])}) the corresponding smoothing function performed best for the given problem."
with open(f"performance_approximations_desc.tex", "w") as file:
    file.write(tbl_description)
print(tbl_description)
tbl

# %% Decide which stepsize tolerance is better
df_tol = df_agg.loc[:,("Problem", "Sample Size", "$\\Delta_{{\\text{{Target}}}}$", "Method", "$\\epsilon_{{\\text{{Target}}}}$", "Relative Root MSE", "$N_{{ \\text{{obs}} }}$","Smoothing Function")] \
    .groupby(["Problem", "Sample Size", "$\\Delta_{{\\text{{Target}}}}$", "Method", "$N_{{ \\text{{obs}} }}$", "Smoothing Function"]) \
    .apply(decide,  "$\\epsilon_{{\\text{{Target}}}}$")     
tbl = df_tol.reset_index().value_counts(subset=["$\\epsilon_{{\\text{{Target}}}}$", "Problem"], normalize=False)\
    .unstack(level=1)
totals = tbl.sum()
tbl = tbl/totals
tbl["Total"] = tbl.mean(numeric_only=True, axis=1)
tbl = tbl.applymap(lambda x: f"{x*100:.2f}\%")
tbl = tbl.rename(columns={c:'\\rotatebox{60}{' + c + '}' for c in tbl.columns}).\
    rename(index={idx:str(idx) for idx in tbl.index })
tbl.style.to_latex("performance_stepsize_tolerance.tex", clines="all;data")
tbl_description = f"Comparing the estimates of $\\textup{{relRootMSE}}(\\hat{{P}}_f)$ for different values of $\\epsilon_{{\\text{{Target}}}}$ averaged over  all other parameter choices. The values denote the relative number of cases (total {int(totals[0])}) the corresponding value performed best for the given problem."
with open(f"performance_stepsize_tolerance_desc.tex", "w") as file:
    file.write(tbl_description)
print(tbl_description)
tbl


#%% make plots
fig_list= []
for prob in ["Convex Problem", "Linear Problem (d=2)", "Fujita Rackwitz (d=2)", "Linear Problem (d=50)"]:# df_agg["Problem"].unique():
    this_df = df_agg.query("Problem == @prob")
    this_df = this_df[this_df["$\\epsilon_{{\\text{{Target}}}}$"]==best_tolerance]
    this_df = this_df[this_df["Smoothing Function"] == best_approximation]
    this_df = this_df[this_df['$N_{{ \\text{{obs}} }}$'].isin([2,5,10])]
    cmap = sr_to_color_dict(this_df["$\\Delta_{{\\text{{Target}}}}$"])
    this_df["cvar_tgt_str"] = this_df["$\\Delta_{{\\text{{Target}}}}$"].apply(str)
    this_df = this_df.sort_values(["$\\Delta_{{\\text{{Target}}}}$", "$N_{{ \\text{{obs}} }}$"])
    
    this_df_bm = df_bm.query("Problem == @prob & cvar_tgt == 1")
    
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
    
    num_rows = len(this_df["$N_{{ \\text{{obs}} }}$"].unique())
    num_cols = len(this_df["Method"].unique())
    for bm_solver in this_df_bm.Solver.unique():
        dat =this_df_bm.query("Solver == @bm_solver")
        # nice name
        solver_name = re.sub(r"\W+", " ",bm_solver)
        solver_name = re.findall(r"(\bSiS\b)|(\bEnKF\b)|(\bvMFNM\b)|(\bGM\b)|(\baCS\b)", solver_name)
        solver_name = f"{''.join(solver_name[0])} ({''.join(solver_name[1])})"
        trace_dict = {
            "x" : dat["Relative Root MSE"],
            "y" : dat["Cost Mean"],
            "legendgrouptitle_text": "Benchmark Methods",
            "name": solver_name,
            "legendgroup": "group",
            "mode": "markers+lines",
            "opacity": 0.8
        }
        trace_dict = trace_dict | BM_SOLVER_SCATTER_STYLE[solver_name]
        fig = add_scatter_to_subplots(fig, num_rows, num_cols, **trace_dict)
    fig.update_layout(**MY_LAYOUT)
    fig.show()
    fig.write_image(f"{prob} stopping criterion.png".replace(" ", "_").lower(), scale=WRITE_SCALE)
    fig_description = f"Solving the {prob} with the CBREE method using  \
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
Each marker represents the point estimates based on 200 simulations."
    with open(f"{prob} stopping criterion_desc.tex".replace(" ", "_").lower(), "w") as file:
        file.write(fig_description)
    print(fig_description)
# %%

# %%
