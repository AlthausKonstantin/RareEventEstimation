#%% 
from glob import glob
from os import listdir, path
from re import X
import scipy as sp
import rareeventestimation as ree
from rareeventestimation.evaluation.constants import INDICATOR_APPROX_LATEX_NAME
import numpy as np
import scipy as sp
import sympy as smp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
%load_ext autoreload
%autoreload 2

#%% Preprocess data
data_dir = "./data/cbree_sim/indicator_functions_performance"
df = ree.load_data(data_dir, "*")
df.drop(columns=["index", "Unnamed: 0"], inplace=True)
df.drop_duplicates(inplace=True)
df = df.query("observation_window==0") \
    .reset_index()
df = ree.add_evaluations(df, only_success=True)
# %% count success rates and arrange result in latex table
for tgt in df.cvar_tgt.unique():
    # Count proportion of unsuccessful exit status
    df_success=df.query("cvar_tgt==@tgt")\
        .groupby(["tgt_fun", "Problem"])["Message"].apply(pd.value_counts)
    df_success = pd.DataFrame(df_success)
    df_success = df_success[df_success.index.get_level_values(2)!="Success"]
    df_success["Message"] = df_success["Message"]/200
    df_success.reset_index(inplace=True)
    # Compute order of tgt_funs form best to worst
    lvl_1_order = df.query("cvar_tgt==@tgt") \
        .groupby("tgt_fun") \
        .mean() \
        .loc[:,"Success Rate"] \
        .sort_values(ascending=False) \
        .index \
        .values
    lvl_1 = [idx for idx in lvl_1_order if idx in df_success["tgt_fun"].values]
    # arange
    tbl = pd.pivot_table(df_success,
                        values="Message",
                        columns=["level_2"],
                        index=["tgt_fun", "Problem"],
                        fill_value="0 \%", 
                        aggfunc= lambda x: f"{100*x.values.item():.1f}\\%")
    tbl = tbl.reindex(lvl_1, level=0)
    # style and save
    tbl.columns.name=None
    tbl.index = tbl.index.set_names(names={"tgt_fun": "Approximation"}, )
    tbl = tbl.rename(columns={
            "Not Converged.":"Not Converged",
            "attempt to get argmax of an empty sequence": "No finite weights $\\bm{{w}}^n$",
            "singular matrix":"Singular $c^n$"}, index=INDICATOR_APPROX_LATEX_NAME)  
    tbl.style.to_latex(f"/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Latex/figures/success_rates_tgt_{tgt}.tex",
                       multirow_align="naive",
                       column_format="ccrRP",
                       clines="skip-last;data")
    with open(f"/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Latex/figures/success_rates_tgt_{tgt}_desc.tex", "w") as file:
        file.write(f"Exit Messages of unsuccessful runs with stopping criterion $\\Delta_{{\\text{{Target}}}} = {tgt}$. Values are proportional to 200 sample runs.")
tbl
# %% for different cvar_tgt and success rates (==100%, < 100%) rang tgt_funs by accuracy
for cvar_tgt in df.cvar_tgt.unique():
    df_agg = ree.aggregate_df(df.query(f"cvar_tgt==@cvar_tgt"))
    for op in ["==", "<"]:
        tgt_fun_list = df_agg.groupby("tgt_fun") \
            .mean() \
            .reset_index() \
            .query(f"`Success Rate` {op} 1.0")["tgt_fun"].unique()
        if len(tgt_fun_list) > 0:
            # arrange functions
            df_acc = pd.pivot_table(df_agg.query("tgt_fun in @tgt_fun_list"),
                            values="Relative Root MSE",
                            columns=["tgt_fun"],
                            index=["Problem"]) 
            # order functions
            df_acc = df_acc.reindex(df_acc.mean().sort_values().index, axis=1)
            # style and save
            df_acc = df_acc.rename(columns=INDICATOR_APPROX_LATEX_NAME)
            df_acc.columns.name="Approximation"
            tbl = df_acc.style.format(precision=2)
            tbl.to_latex(f"/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Latex/figures/accuracy_tgt_{cvar_tgt}{'_success_only' if op=='==' else '' }.tex",
                        clines="all;data")
            with open(f"/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Latex/figures/accuracy_tgt_{cvar_tgt}{'_success_only' if op=='==' else '' }_desc.tex", "w") as file:
                file.write(f"Relative root mean squared error of {'successful runs with indicator function approximations that always led to convergence' if op=='==' else 'successful runs with indicator function approximations that have not always converged'} using the stopping criterion $\\Delta_{{\\text{{Target}}}} = {cvar_tgt}$")
        
# %%
