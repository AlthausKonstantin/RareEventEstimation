# %%
from os import path
import rareeventestimation as ree
import numpy as np
import scipy as sp
import pandas as pd
import plotly.express as px
from rareeventestimation.evaluation.constants import INDICATOR_APPROX_LATEX_NAME, BM_SOLVER_SCATTER_STYLE, MY_LAYOUT, DF_COLUMNS_TO_LATEX, LATEX_TO_HTML, WRITE_SCALE
import plotly.graph_objects as go
from rareeventestimation.evaluation.visualization import add_scatter_to_subplots, sr_to_color_dict
from rareeventestimation.evaluation.convergence_analysis import expand_cbree_name,vec_to_latex_set
import re
%load_ext autoreload
%autoreload 2
# %% Load data
data_dir = "/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation_data Kopie/cbree_sim/diffusion_sim"
path_df = path.join(data_dir, "cbree_diffusion_problem_processed.pkl")
path_df_agg = path.join(data_dir, "cbree_diffusion_problem_aggregated.pkl")
if   (path.exists(path_df) and path.exists(path_df_agg)):
    df = ree.load_data(data_dir, "*vmfnm*")
    df.drop(columns=["index", "Unnamed: 0",  "VAR Weighted Average Estimate","CVAR", "callback"], inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # %% Round parameters
    for col in [c for c in df.columns if c in DF_COLUMNS_TO_LATEX.keys()]:
        if isinstance(df[col].values[0], float):
            df[col] = df[col].round(5)
    # melt aggregated estimates
    df = df.rename(columns={"Estimate": "Last Estimate"})\
        .melt(id_vars = [c for c in df.columns if not "Estimate" in c],
              var_name="Averaging Method",
              value_name="Estimate")
    df = df.apply(expand_cbree_name, axis=1, columns = ["Averaging Method", "observation_window"])
   
    # pretty names
    df = df.rename(columns=DF_COLUMNS_TO_LATEX)
    #%%process data: add evaluations etc
    df = ree.add_evaluations(df)
    df_agg = ree.aggregate_df(df)
    #%% save
    df.to_pickle(path_df)
    df_agg.to_pickle(path_df_agg)
else:
    df = pd.read_pickle(path_df)
    df_agg = pd.read_pickle(path_df_agg)
#%% Load benchmark 
bm_data_dirs = {
    "enkf":"/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation_data Kopie/enkf_sim_diffusion",
    "sis": "/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation_data Kopie/sis_sim_diffusion"
}
bm_df_names ={"df": "benchmark_diffusion_problems_processed.pkl",
              "df_agg": "benchmark_diffusion_problems_aggregated.pkl"}
df_bm, df_bm_agg = ree.get_benchmark_df(data_dirs=bm_data_dirs,
                                        df_names=bm_df_names,
                                        df_dir="/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation_data Kopie",
                                        force_reload=True)


# %% plot error
fig_list= []
for prob in df_agg["Problem"].unique():
    this_df = df_agg.query("Problem == @prob & `Smoothing Function` == 'algebraic'")
    #this_df = this_df[this_df["$\\epsilon_{{\\text{{Target}}}}$"]==best_tolerance]
    #this_df = this_df[this_df["Smoothing Function"] == "algebraic"]
    this_df = this_df[this_df['$N_{{ \\text{{obs}} }}$'].isin([4,8,12])]
   # this_df["cvar_tgt_str"] = this_df["$\\Delta_{{\\text{{Target}}}}$"].apply(str)
    this_df = this_df.sort_values(["$\\Delta_{{\\text{{Target}}}}$", "$N_{{ \\text{{obs}} }}$"])
    
    this_df_bm = df_bm_agg.query("Problem == @prob & cvar_tgt == 1")
    
    fig = px.line(
        this_df,
        x = "Relative Root MSE",
        y="Cost Mean",
        facet_col=r'$\epsilon_{{\text{{Target}}}}$',
        facet_row="$N_{{ \\text{{obs}} }}$",
        color="Averaging Method",
        log_x=True,
        log_y=True,
        markers=True,
        labels=LATEX_TO_HTML | {"cvar_tgt_str": LATEX_TO_HTML[DF_COLUMNS_TO_LATEX["cvar_tgt"]]},
        
    )
    
    num_rows = len(this_df["$N_{{ \\text{{obs}} }}$"].unique())
    num_cols = len(this_df[r'$\epsilon_{{\text{{Target}}}}$'].unique())
    for bm_solver in this_df_bm.Solver.unique():
        dat =this_df_bm.query("Solver == @bm_solver")
        dat = dat.sort_values(["Solver", "Sample Size"])
        trace_dict = {
            "x" : dat["Relative Root MSE"],
            "y" : dat["Cost Mean"],
            "legendgrouptitle_text": "Benchmark Methods",
            "name": bm_solver,
            "legendgroup": "group",
            "mode": "markers+lines",
            "opacity": 0.8,
            "text":dat["Sample Size"],
            "hoverinfo":"text"
        }
        trace_dict = trace_dict  | BM_SOLVER_SCATTER_STYLE[bm_solver]
        fig = add_scatter_to_subplots(fig, num_rows, num_cols, **trace_dict)
    fig.update_layout(**MY_LAYOUT)
    fig.update_layout(height=900)
    fig.for_each_annotation(
        lambda a: a.update(yshift =  -10 if a.text.startswith(LATEX_TO_HTML[DF_COLUMNS_TO_LATEX["stepsize_tolerance"]]) else 0))
    fig.show()
    fig.write_image(f"diffusion problem.png".replace(" ", "_").lower(), scale=WRITE_SCALE)
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1))
    fig.show()
    fig_description = f"Solving the {prob} with the CBREE (vMFNM, resampled) method using  \
different parameters. \
We vary the averaging method (color), \
the stepsize tolerance $\\epsilon_{{\\text{{Target}}}}$ (column) and \
the divergence criterion $N_\\text{{obs}}$ (row). \
The choice of the stopping criterion $\\Delta_{{\\text{{Target}}}} = 2$ and indicator approximation {INDICATOR_APPROX_LATEX_NAME['algebraic']} \
are fixed. \
Furthermore we plot also the performance of the benchmark methods EnKF\
(with different importance sampling densities)\
and SiS (with different MCMC sampling methods). \
We used the sample sizes $J \\in {vec_to_latex_set(df_agg['Sample Size'].unique())}$. \
Each marker represents the empirical estimates based the successful portion of $200$ simulations."
    with open(f"diffusion problem desc.tex".replace(" ", "_").lower(), "w") as file:
        file.write(fig_description)
    print(fig_description)

# %%
