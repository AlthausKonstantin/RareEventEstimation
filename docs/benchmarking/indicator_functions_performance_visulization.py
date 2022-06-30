#%% 
from glob import glob
from os import listdir, path
import scipy as sp
import rareeventestimation as ree
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
if path.exists(path.join(data_dir, "processed_results.pkl")):
    df_agg = pd.read_pickle(path.join(data_dir, "processed_results.pkl"))
else:
    df = ree.load_data(data_dir, "Linear_Problem_(d=2)*")
# %%
