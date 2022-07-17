#%% 
from re import sub
from os import path
import rareeventestimation as ree
import numpy as np
import pandas as pd
from rareeventestimation.evaluation.constants import *
import plotly.express as px
from rareeventestimation.evaluation.constants import INDICATOR_APPROX_LATEX_NAME, BM_SOLVER_SCATTER_STYLE, MY_LAYOUT, DF_COLUMNS_TO_LATEX, LATEX_TO_HTML, WRITE_SCALE
from rareeventestimation.evaluation.convergence_analysis import vec_to_latex_set, expand_cbree_name, list_to_latex
from rareeventestimation.evaluation.visualization import add_scatter_to_subplots, sr_to_color_dict

%load_ext autoreload
%autoreload 2
# %% Load
data_dir ="/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation_data Kopie/cbree_sim/toy_problems/resampled"
df_path =path.join(data_dir, "resampled_toy_problems.pkl")
if  path.exists(df_path):
    df = ree.load_data(data_dir, "*")
    df.drop(columns=["index", "Unnamed: 0", "VAR Weighted Average Estimate","CVAR"], inplace=True, errors="ignore")
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # %% Pretty names
    to_drop = ["callback"] # no callbacks here
    df.rename(columns={"mixture_model": "Method"}, inplace=True)
    replace_values = {"Method": {"GM": "CBREE (GM)", "vMFNM": "CBREE (vMFNM, resampled)"}}
    df = df.drop(columns=to_drop) \
        .rename(columns=DF_COLUMNS_TO_LATEX) \
        .replace(replace_values)
    # melt aggregated estimates into long format
    df = df.rename(columns={"Estimate": "Last Estimate"})\
        .melt(id_vars = [c for c in df.columns if not "Estimate" in c],
              var_name="Averaging Method",
              value_name="Estimate")
    # make solver column with unique names wrt all options.
    df = df.apply(expand_cbree_name, axis=1, columns= [DF_COLUMNS_TO_LATEX['observation_window'], "Averaging Method"])  
    df = ree.add_evaluations(df,  only_success=True)
    # %% aggregate
    df_agg = ree.aggregate_df(df)
    # add dimension of problem
    #%% save
    df_agg.to_pickle(df_path)
else:
   df_agg = pd.read_pickle(df_path)
#%% Load benchmark
df_bm, df_bm_agg = ree.get_benchmark_df()

# %%
for prob in df_agg.Problem.unique():
    this_df = df_agg.query("Problem == @prob & `Averaging Method`=='Average Estimate'")
    this_df = this_df[this_df['$N_{{ \\text{{obs}} }}$'].isin([2,5,10])]
    cmap = sr_to_color_dict(this_df["$\\Delta_{{\\text{{Target}}}}$"])
    this_df["cvar_tgt_str"] = this_df["$\\Delta_{{\\text{{Target}}}}$"].apply(str)
    this_df = this_df.sort_values(["$\\Delta_{{\\text{{Target}}}}$", "$N_{{ \\text{{obs}} }}$", "Sample Size"])
    
    
    
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
        hover_name="Sample Size",
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
    fig.write_image(f"{prob} resampled stopping criterion.png".replace(" ", "_").lower(), scale=WRITE_SCALE)
    fig_description = f"Solving {prob} with the CBREE methods using  \
    different parameters. \
    We vary the stopping criterion $\\Delta_{{\\text{{Target}}}}$ (color), \
    the divergence criterion $N_\\text{{obs}}$ (row) and \
    the method $\\mu^N$ (column). \
    The parameter $\\epsilon_{{\\text{{Target}}}} = {0.5}$ \
    and the choice of the indicator approximation {INDICATOR_APPROX_LATEX_NAME['algebraic']} \
    are fixed. \
    Furthermore we plot also the performance of the benchmark methods EnKF\
    (with different importance sampling densities)\
    and SiS (with different MCMC sampling methods). \
    We used the sample sizes $J \\in {vec_to_latex_set(df_agg['Sample Size'].unique())}$. \
    Each marker represents the empirical estimates based the successful portion of $200$ simulations."
    with open(f"{prob} resampled stopping criterion desc.tex".replace(" ", "_").lower(), "w") as file:
        file.write(fig_description)
    print(fig_description)
# %%
# %% Make table with used parameters

     
paras = ["Sample Size"] + list(DF_COLUMNS_TO_LATEX.values())
tbl_params = df_agg.loc[:, tuple(paras)] \
    .replace({"Smoothing Function": INDICATOR_APPROX_LATEX_NAME}) \
    .apply(lambda col: col.sort_values().unique()) \
    .apply(list_to_latex)
tbl_params = tbl_params.to_frame(name="Values")
tbl_params.index.name = "Parameter"
tbl_params.style.to_latex("parameters_used_toy_problems_highdim.tex", clines="all;data")
tbl_params
# %%
