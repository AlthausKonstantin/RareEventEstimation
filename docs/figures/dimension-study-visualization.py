#%% 
from re import sub
from os import path
import rareeventestimation as ree
import numpy as np
import pandas as pd
from rareeventestimation.evaluation.constants import *
import plotly.express as px
from rareeventestimation.evaluation.constants import INDICATOR_APPROX_LATEX_NAME, BM_SOLVER_SCATTER_STYLE, MY_LAYOUT, DF_COLUMNS_TO_LATEX, LATEX_TO_HTML, WRITE_SCALE
from rareeventestimation.evaluation.convergence_analysis import vec_to_latex_set

%load_ext autoreload
%autoreload 2
# %% Load
data_dir ="./data/"
df_path =path.join(data_dir, "dimension_study_data.pkl")
file_pattern = "fujita_rackwitz_problem*"
if not  path.exists(df_path):
    df = ree.load_data(data_dir, file_pattern)
    df.drop(columns=["index", "Unnamed: 0", "VAR Weighted Average Estimate","CVAR"], inplace=True, errors="ignore")
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.rename(columns={"Solver":"Method"}, inplace=True)
    # CBREE (VMFNM) does not have valid averaged estimates by definition (resample keyword has not been set)
    # it mixes the last vmfnm with preceeding gaussian denisities
    df = df.query("Method != 'CBREE (vMFNM)'")
    # rename methods to comply with thesis notation
    new_method_names = {"CBREE (GM)": "CBREE",
                        "CBREE (vMFNM, resampled)": "CBREE (vMFN)"}
    df = df.replace({"Method": new_method_names})
    # melt aggregated estimates
    df = df.rename(columns={"Estimate": "Last Estimate"})\
        .melt(id_vars = [c for c in df.columns if not "Estimate" in c],
              var_name="Averaging Method",
              value_name="Estimate")
    col_list = ['callback', 'observation_window', 'resample', 'mixture_model','divergence_check', "Averaging Method"]
    # make solver column with unique names wrt all options.
    df["Solver"] = df.apply(lambda row: row["Method"] + f" ({', '.join([str(row[col]) for col in col_list])})", axis=1)    
    df = ree.add_evaluations(df,  only_success=True)
    # %% aggregate
    df_agg = ree.aggregate_df(df)
    # add dimension of problem
    df_agg["Dimension"] = df_agg["Problem"].apply(lambda x: int(sub(r"\D", "", x)))

    #%% save
    df_agg.to_pickle(df_path)
else:
   df_agg = pd.read_pickle(df_path)
#%% Visualize
this_df =df_agg[df_agg["observation_window"].isin([2,6,12])]
this_df = this_df.query("Dimension <=120 & `Sample Size` == 5000")
this_df = this_df.sort_values(["observation_window","Dimension"])
fig = px.line(this_df,
              y = "Relative Root MSE",
                x="Dimension",
                facet_col="Method",
                facet_row="observation_window",
                color="Averaging Method",
                labels={k: LATEX_TO_HTML[DF_COLUMNS_TO_LATEX[k]] for k in ["observation_window"] },
                log_y=True)

fig.update_layout(**MY_LAYOUT)
fig.update_layout(height=800)
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
fig.show()
fig.write_image("dimension_study.png", scale=WRITE_SCALE)
description = f"Solving the Fujita Rackwitz Problem in dimensions $d \\in {vec_to_latex_set(this_df.Dimension.unique())}$ 200 times\
for sample size $J = {this_df['Sample Size'].unique()[0]}$  \
with different values for $N_{{ \\textup{{ obs }} }}$  (row), \
two variants of the CBREE method (column) and different averaging methods of the last \
$N_{{ \\textup{{ obs }} }}$ probability of failure estimates (color). \
Other parameters are fixed. \
Namely, we use the stopping criterion $\\Delta_{{\\text{{Target}}}} = 2$, \
the stepsize tolerance $\\epsilon_{{\\text{{Target}}}} = 0.5$, \
the increase control  of $\\sigma$ with $\\text{{Lip}}(\\sigma) = 1$ \
and approximate the indicator function with {INDICATOR_APPROX_LATEX_NAME['algebraic']}."
with open("dimension_study_desc.tex", "w") as f:
    f.write(description)
print(description)
# %%
