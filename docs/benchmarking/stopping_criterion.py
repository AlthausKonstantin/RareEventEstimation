#%% 

from numbers import Real
from os import path
import rareeventestimation as ree
import numpy as np
import scipy as sp
import pandas as pd
import plotly.express as px
from rareeventestimation.evaluation.constants import INDICATOR_APPROX_LATEX_NAME, BM_SOLVER_SCATTER_STYLE, MY_LAYOUT, DF_COLUMNS_TO_LATEX, LATEX_TO_HTML, WRITE_SCALE, CMAP
import plotly.graph_objects as go
from rareeventestimation.evaluation.visualization import add_scatter_to_subplots, sr_to_color_dict
from rareeventestimation.evaluation.convergence_analysis import expand_cbree_name, list_to_latex, my_number_formatter, vec_to_latex_set, squeeze_problem_names
import re

%load_ext autoreload
%autoreload 2

#%% Load data

data_dir ="/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation_data Kopie/cbree_sim/toy_problems"
path_df= path.join(data_dir, "cbree_toy_problems_processed.pkl")
path_df_agg = path.join(data_dir, "cbree_toy_problems_aggregated.pkl")
path_df_agg_all = path.join(data_dir, "cbree_toy_problems_aggregated_all.pkl")
if not  (path.exists(path_df) and path.exists(path_df_agg) and path.exists(path_df_agg_all)):
    df = ree.load_data(data_dir, "*")
    df.drop(columns=["index", "Unnamed: 0"], inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    #%% Round parameters
    for col in DF_COLUMNS_TO_LATEX.keys():
        if isinstance(df[col].values[0], float):
            df[col] = df[col].round(5)
    #%%process data: add obs_window and callback to solver name
    
    df = df.apply(expand_cbree_name, axis=1, columns= ['observation_window', 'callback'])
    # %% Pretty names
    to_drop = ["mixture_model"] # info is redundant as resample = False and callback exists
    replace_values = {"Method": {"False": "CBREE", "vMFN Resample": "CBREE (vMFN, resampled)"}}
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
low_dim_probs= ["Convex Problem", "Linear Problem (d=2)", "Fujita Rackwitz Problem (d=2)"]

# %% Make table with used parameters

     
paras = ["Sample Size"] + [v for v in DF_COLUMNS_TO_LATEX.values() if v!= "Method"]
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
df_tgt_fun = df_agg.query("Problem in @low_dim_probs & Method == 'CBREE'")\
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
tbl = tbl.rename(columns={c:squeeze_problem_names(c) for c in tbl.columns})
tbl.style.to_latex("performance_approximations.tex", clines="all;data")
tbl_description = f"Comparing the estimates of $\\textup{{relRootMSE}}(\\hat{{P}}_f)$ for different smoothing functions averaged over  all other parameter choices. The values denote the relative number of cases the corresponding smoothing function performed best for the given problem (in total {int(totals[0])} per problem)."
with open(f"performance_approximations_desc.tex", "w") as file:
    file.write(tbl_description)
print(tbl_description)
tbl

# %% Decide which stepsize tolerance is better
df_tol = df_agg.query("Problem in @low_dim_probs & Method == 'CBREE'")\
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
tbl = tbl.rename(columns={c:squeeze_problem_names(c) for c in tbl.columns}).\
    rename(index={idx:str(idx) for idx in tbl.index })
tbl.style.to_latex("performance_stepsize_tolerance.tex", clines="all;data")
tbl_description = f"Comparing the estimates of $\\textup{{relRootMSE}}(\\hat{{P}}_f)$ for different values of $\\epsilon_{{\\text{{Target}}}}$ averaged over  all other parameter choices. The values denote the relative number of cases (in total {int(totals[0])} per problem) the corresponding value performed best for the given problem."
with open(f"performance_stepsize_tolerance_desc.tex", "w") as file:
    file.write(tbl_description)
print(tbl_description)
tbl

#%% Compare success rates for diverengence check
df_rates = df_agg.query("Problem in @low_dim_probs & Method == 'CBREE'")
df_rates = df_rates[df_rates["$\\epsilon_{{\\text{{Target}}}}$"]==best_tolerance]
df_rates = df_rates[df_rates["Smoothing Function"] == best_approximation]
df_rates = df_rates[df_rates["$\\Delta_{{\\text{{Target}}}}$"]==1]
tbl_success_rates = pd.pivot_table(df_rates, values="Success Rate", index='$N_{{ \\text{{obs}} }}$', columns="Problem", aggfunc=np.mean)
tbl_success_rates = tbl_success_rates.groupby(list(tbl_success_rates))\
    .apply(lambda x: vec_to_latex_set(x.index.values))\
    .to_frame(name='$N_{{ \\text{{obs}} }}$')\
    .reset_index()\
    .set_index('$N_{{ \\text{{obs}} }}$') \
    .applymap(lambda x: f"{x*100:.2f}\%") \
    .rename(index={"0":'No div. check'})
tbl_success_rates.rename(columns = {c: squeeze_problem_names(c) for c in tbl_success_rates.columns}, inplace=True)
tbl_description = f"Comparing the success rates of the CBREE  method for different values of $N_{{ \\text{{obs}} }}$  averaged over all sample sizes $J =   {vec_to_latex_set(df_rates['Sample Size'].unique())}$. \
The parameters $\\Delta_{{\\text{{Target}}}}=1$, \
$\\epsilon_{{\\text{{Target}}}} = {best_tolerance}$ \
and the choice of the indicator approximation {INDICATOR_APPROX_LATEX_NAME[best_approximation]} \
are fixed. \
The values denote the relative number of cases the CBREE method converged successfully for the particular combination of problem and paramter setting (in total {6*200} per setting)."
with open(f"success_obs_window_desc.tex", "w") as file:
    file.write(tbl_description)
print(tbl_description)
tbl_success_rates.style.to_latex("success_obs_window.tex", clines="all;data")
tbl_success_rates


#%% make plot for divergence check with fujita rackwitz
fr_all = df_agg_all.query("Problem == 'Fujita Rackwitz Problem (d=2)' & Method =='CBREE'")
fr_all = fr_all[(fr_all[DF_COLUMNS_TO_LATEX['observation_window']]==0) & \
                (fr_all[DF_COLUMNS_TO_LATEX['stepsize_tolerance']]==best_tolerance) & \
                (fr_all[DF_COLUMNS_TO_LATEX['tgt_fun']]==best_approximation) & \
                (fr_all[DF_COLUMNS_TO_LATEX['cvar_tgt']]==1)]
fr_all = fr_all.assign(Portion="All Simulations")
fr_success = df_agg.query("Problem == 'Fujita Rackwitz Problem (d=2)' & Method =='CBREE'")
fr_success = fr_success[(fr_success[DF_COLUMNS_TO_LATEX['observation_window']]>0) & \
                (fr_success[DF_COLUMNS_TO_LATEX['stepsize_tolerance']]==best_tolerance) & \
                (fr_success[DF_COLUMNS_TO_LATEX['tgt_fun']]==best_approximation) & \
                (fr_success[DF_COLUMNS_TO_LATEX['cvar_tgt']]==1)]
fr_success = fr_success.assign(Portion="Only Successful Simulations")
df_fr = pd.concat([fr_all, fr_success], axis=0)
fig_fr =go.Figure()
n_obs_col_dict = sr_to_color_dict(df_fr[DF_COLUMNS_TO_LATEX['observation_window']])
for n in df_fr[DF_COLUMNS_TO_LATEX['observation_window']].sort_values().unique():
    this_df = df_fr[df_fr[DF_COLUMNS_TO_LATEX['observation_window']] ==n]\
        .sort_values("Sample Size")
    tr = go.Scatter(
        x = this_df["Relative Root MSE"],
        y = this_df["Cost Mean"],
        mode="markers+lines",
        marker_symbol = "circle",
        legendgroup = str(n==0),
        marker_color = CMAP[3] if n==0 else n_obs_col_dict[str(n)],
        legendgrouptitle_text = "All Simulations" if n==0 else "Only Successful Simulations",
        name= "No Divergence Check" if n==0 else f"{LATEX_TO_HTML[DF_COLUMNS_TO_LATEX['observation_window']]} = {int(n)}",
        line_dash = "dash" if n==0 else "solid")
    fig_fr.add_trace(tr)
fig_fr.update_layout(**MY_LAYOUT)
fig_fr.update_layout(height=800)
fig_fr.update_xaxes(title_text="Relative Root MSE", type="log")
fig_fr.update_yaxes(title_text="Cost Mean", type="log",  title_standoff=0)
fig_fr.write_image("divergence_fujita_rackwitz.png", scale=WRITE_SCALE)
fig_description = f"The effect of the divergence check on the Fujita Rackwitz Problem (d=2). \
We show the empirical error and cost estimates based on all 200 simulations (successful or not) \
if the CBREE methods runs with no divergence check. \
The same quantities  based on the successful portion of the 200 simulations \
are plotted for different values of $N_\\text{{obs}}$ \
if the divergence check is active.\
The parameters $\\Delta_{{\\text{{Target}}}}=1$,  $\\epsilon_{{\\text{{Target}}}} = {best_tolerance}$ \
and the choice of the indicator approximation {INDICATOR_APPROX_LATEX_NAME[best_approximation]} \
are fixed."
print(fig_description)
with open("divergence_fujita_rackwitz_desc.tex", "w")  as f:
    f.write(fig_description)
fig_fr.show()
#%% make plots

fig_list= []
for prob in df_agg.Problem.unique():
    this_df = df_agg.query("Problem == @prob & Method=='CBREE'")
    this_df = this_df[this_df["$\\epsilon_{{\\text{{Target}}}}$"]==best_tolerance]
    this_df = this_df[this_df["Smoothing Function"] == best_approximation]
    this_df = this_df[this_df['$N_{{ \\text{{obs}} }}$'].isin([0, 2,5,10])]
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
        hover_name='Success Rate',
        log_x=True,
        log_y=True,
        #hover_name = "Sample Size",
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
    "height":800})
    fig.for_each_annotation(
        lambda a: a.update(text="" if a.text.startswith("Method") else a.text))    
    old_a = LATEX_TO_HTML[DF_COLUMNS_TO_LATEX["observation_window"]] + "=0.0"
    new_a = "No Divergence Check"
    fig.for_each_annotation(
        lambda a: a.update(text = new_a if a.text == old_a else a.text))
    fig.show()
    fig.write_image(f"{prob} stopping criterion.png".replace(" ", "_").lower(), scale=WRITE_SCALE)
    fig_description = f"Solving the {prob} with the CBREE method using  \
different parameters. \
We vary the stopping criterion $\\Delta_{{\\text{{Target}}}}$ (color) and \
the length of the observation window $N_\\text{{obs}}$ (row). \
The parameter $\\epsilon_{{\\text{{Target}}}} = {best_tolerance}$ \
and the choice of the indicator approximation {INDICATOR_APPROX_LATEX_NAME[best_approximation]} \
are fixed. \
Furthermore we plot also the performance of the benchmark methods EnKF \
and SiS. \
We used the sample sizes $J \\in {vec_to_latex_set(df_agg['Sample Size'].unique())}$. \
Each marker represents the empirical estimates based the successful portion of $200$ simulations."
    with open(f"{prob} criterion_desc.tex".replace(" ", "_").lower(), "w") as file:
        file.write(fig_description)
    print(fig_description)
# %%


# %%
def get_convergence_rate(grp):
    grp = grp.sort_values("Relative Root MSE",ascending=True)
    xx = grp["Relative Root MSE"].values
    yy = np.log10(grp["Cost Mean"].values)
    line = np.polyfit(xx, yy, 1)
    return(pd.Series([line[-1]], ["Convergence Rate With Respect to Cost"]))
# %%
