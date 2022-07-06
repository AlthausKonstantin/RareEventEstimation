#%% 
from glob import glob
from numbers import Real
from os import listdir, path
import scipy as sp
import rareeventestimation as ree
import numpy as np
import scipy as sp
import sympy as smp
import pandas as pd
import plotly.express as px
from rareeventestimation.evaluation.constants import INDICATOR_APPROX_LATEX_NAME
import plotly.graph_objects as go
from rareeventestimation.evaluation.visualization import add_scatter_to_subplots, sr_to_color_dict
%load_ext autoreload
%autoreload 2

#%% Load data
data_dir ="/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation_data/cbree_sim/toy_problems"
df = ree.load_data(data_dir, "*")
df.drop(columns=["index", "Unnamed: 0"], inplace=True)
df.drop_duplicates(inplace=True)
#%%process data: add obs_window and callback to solver name
def expand_cbree_name(input, columns=[]) -> str:
    for col in columns:
        input.Solver = input.Solver.replace("}", f", '{col}': '{input[col]}'}}")
    return input
df = df.apply(expand_cbree_name, axis=1, columns= ['observation_window', 'callback'])
# %% Pretty names
to_drop = ["mixture_model"] # info is redundant as resample = False and callback exists
new_names = {
    "stepsize_tolerance": "$\\epsilon_{{\\text{{Target}}}}$",
    "cvar_tgt": "$\\Delta_{{\\text{{Target}}}}$",
    "lip_sigma": "Lip$(\\sigma)$",
    "tgt_fun": "Smoothing Function",
    "observation_window": "$N_{{ \\text{{obs}} }}$",
    "callback": "IS Density",
}
replace_values = {"IS Density": {"False": "GM", "vMFNM Resample": "vMFNM (Resampled)"}}
df = df.drop(columns=to_drop) \
    .rename(columns=new_names) \
    .replace(replace_values)
#%%process data: add evaluations
df = ree.add_evaluations(df)
out_path = path.join(data_dir, "cbree_toy_problems_processed.pkl")
# 

# %% aggregate
df_agg = ree.aggregate_df(df)
# %% remove obs window 0
df = df[df['$N_{{ \\text{{obs}} }}$']>0].reset_index()
df_agg = df_agg[df_agg['$N_{{ \\text{{obs}} }}$']>0].reset_index()
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
     
paras = ["Sample Size"] + list(new_names.values())
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
    return(pd.Series([best], index =[f"Best {par}"]))
df_tgt_fun = df_agg.loc[:,("Problem", "Sample Size", "$\\Delta_{{\\text{{Target}}}}$", "IS Density", "Smoothing Function", "Relative Root MSE", "$N_{{ \\text{{obs}} }}$")] \
    .groupby(["Problem", "Sample Size", "$\\Delta_{{\\text{{Target}}}}$", "IS Density", "$N_{{ \\text{{obs}} }}$"]) \
    .apply(decide, "Smoothing Function") \
    .loc[:,"Best Smoothing Function"] 
    
tbl = df_tgt_fun.value_counts() / len(df_tgt_fun)
best_approximation = tbl.idxmax()
tbl = tbl.apply(lambda x: f"{x*100:.2f}\%")
tbl = tbl.to_frame(name="Relative Frequency")
tbl.index.name = "Approximation"
tbl = tbl.rename(index = INDICATOR_APPROX_LATEX_NAME)
tbl.style.to_latex("performance_approximations.tex", clines="all;data")
tbl

# %% Decide which stepsize tolerance is better
df_tgt_fun = df_agg.loc[:,("Problem", "Sample Size", "$\\Delta_{{\\text{{Target}}}}$", "IS Density", "$\\epsilon_{{\\text{{Target}}}}$", "Relative Root MSE", "$N_{{ \\text{{obs}} }}$")] \
    .groupby(["Problem", "Sample Size", "$\\Delta_{{\\text{{Target}}}}$", "IS Density", "$N_{{ \\text{{obs}} }}$"]) \
    .apply(decide,  "$\\epsilon_{{\\text{{Target}}}}$") \
    .loc[:,"Best $\\epsilon_{{\\text{{Target}}}}$"] 
    
tbl = df_tgt_fun.value_counts() / len(df_tgt_fun)
best_tolerance = tbl.idxmax()
tbl = tbl.apply(lambda x: f"{x*100:.2f}\%")
tbl = tbl.to_frame(name="Relative Frequency")
tbl.index.name = "$\\epsilon_{{\\text{{Target}}}}$"
tbl.style.to_latex("performance_stepsize_tolerance.tex", clines="all;data")
tbl



#%% make plots
fig_list= []
for prob in df_agg["Problem"].unique():
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
        facet_col="IS Density",
        facet_row="$N_{{ \\text{{obs}} }}$",
        color_discrete_map=cmap,
        color="cvar_tgt_str",
        log_x=True,
        log_y=True,
        markers=True
    )
    
    num_rows = len(this_df["$N_{{ \\text{{obs}} }}$"].unique())
    num_cols = len(this_df["IS Density"].unique())
    trace_dict = {
        "x" : [0.1],
        "y" : [10000],
        "legendgrouptitle_text": "name",
        "marker_color": "blue",
        "mode": "markers" 
    }
    fig = add_scatter_to_subplots(fig, num_rows, num_cols, **trace_dict)
    fig.show()
# %%

# %%
