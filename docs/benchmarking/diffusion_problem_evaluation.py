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
import re
%load_ext autoreload
%autoreload 2
# %% Load data
data_dir = "/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation_data Kopie/cbree_sim/diffusion_sim"
path_df = path.join(data_dir, "cbree_diffusion_problem_processed.pkl")
path_df_agg = path.join(data_dir, "cbree_diffusion_problem_aggregated.pkl")
if   not (path.exists(path_df) and path.exists(path_df_agg)):
    df = ree.load_data(data_dir, "*")
    df.drop(columns=["index", "Unnamed: 0"], inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    # %% Round parameters
    for col in DF_COLUMNS_TO_LATEX.keys():
        if isinstance(df[col].values[0], float):
            df[col] = df[col].round(5)
    # expand solver names:
    def expand_cbree_name(input, columns=[]):
        for col in columns:
            input.Solver = input.Solver.replace("}", f", '{col}': '{input[col]}'}}")
        return input
    # style callback column
    df["callback"] = df["callback"].apply(lambda x: "vMFNM" if x else False)
    # no callback iff resample == True
    # Averaged estimates exist iff resample == True
    df["resample"] = df["callback"] == False
    df["averaging"] = df["resample"]
    # melt aggregated estimates
    df_old = df[~df.averaging]
    df_old["Averaging Method"] = "Last Estimate"
    df_melt = df[df.averaging]
    df_melt = df_melt.rename(columns={"Estimate": "Last Estimate"})\
        .melt(id_vars = [c for c in df_melt.columns if not "Estimate" in c],
              var_name="Averaging Method",
              value_name="Estimate")
    df_melt = df_melt.apply(expand_cbree_name, axis=1, columns = ["Averaging Method", "observation_window"])
    df = pd.concat([df_old, df_melt], axis=0, ignore_index=True)
   
    # pretty names
    to_drop = ['Average Estimate', 'Root Weighted Average Estimate',
       'VAR Weighted Average Estimate', 'averaging', "resample"] # info is also available from 'Averaging Method' and 'Method'
    replace_values = {"Method": {False: "CBREE (vMFNM, resampled)", "vMFNM": "CBREE (vMFNM)"}}
    df = df.drop(columns=to_drop) \
        .rename(columns=DF_COLUMNS_TO_LATEX|{"callback": "Method"}) \
        .replace(replace_values)
    #%%process data: add evaluations etc
    df = ree.add_evaluations(df)
    df_agg = ree.aggregate_df(df)
    #%% save
    df.to_pickle(path_df)
    df_agg.to_pickle(path_df_agg)
else:
    df = pd.read_pickle(path_df)
    df_agg = pd.read_pickle(path_df_agg)
bm_data_dirs = {
    "enkf":"/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation_data Kopie/enkf_sim_diffusion",
    "sis": "/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation_data Kopie/sis_sim_diffusion"
}
bm_df_names ={"df": "benchmark_diffusion_problems_processed.pkl",
              "df_agg": "benchmark_diffusion_problems_aggregated.pkl"}
df_bm, df_bm_agg = ree.get_benchmark_df(data_dirs=bm_data_dirs,
                                        df_names=bm_df_names,
                                        df_dir="/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation_data Kopie")


# %% plot error
fig_list= []
for prob in df_agg["Problem"].unique():
    this_df = df_agg.query("Problem == @prob & `Averaging Method`=='Last Estimate' & `Smoothing Function` == 'algebraic'")
    #this_df = this_df[this_df["$\\epsilon_{{\\text{{Target}}}}$"]==best_tolerance]
    #this_df = this_df[this_df["Smoothing Function"] == "algebraic"]
    this_df = this_df[this_df['$N_{{ \\text{{obs}} }}$'].isin([0,2,5,9])]
   # this_df["cvar_tgt_str"] = this_df["$\\Delta_{{\\text{{Target}}}}$"].apply(str)
    this_df = this_df.sort_values(["$\\Delta_{{\\text{{Target}}}}$", "$N_{{ \\text{{obs}} }}$"])
    
    this_df_bm = df_bm.query("Problem == @prob & cvar_tgt == 1")
    
    fig = px.line(
        this_df,
        x = "Relative Root MSE",
        y="Cost Mean",
        facet_col=r'$\epsilon_{{\text{{Target}}}}$',
        facet_row="$N_{{ \\text{{obs}} }}$",
        color="Method",
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
        # nice name
        # solver_name = re.sub(r"\W+", " ",bm_solver)
        # solver_name = re.findall(r"(\bSiS\b)|(\bEnKF\b)|(\bvMFNM\b)|(\bGM\b)|(\baCS\b)", solver_name)
        # solver_name = f"{''.join(solver_name[0])} ({''.join(solver_name[1])})"
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
    fig.show()

# %%
