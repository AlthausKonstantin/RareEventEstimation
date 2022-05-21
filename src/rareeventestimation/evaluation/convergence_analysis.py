"""Functions to do an emprical analysis of convergence behaviour.
"""

import tempfile
from numpy import average, sqrt, zeros, var, nan
from scipy.stats import variation
from rareeventestimation.problem import Problem
from rareeventestimation.solver import Solver
from numpy.random import default_rng
import pandas as pd
from os import path
import gc
import hashlib
import time
def do_multiple_solves(prob:Problem, solver:Solver, num_runs:int, dir= ".", prefix="", verbose=True, reset_dict=None, save_other=False, other_list=None, addtnl_cols= None):
    """Solve `prob` with `solver.solve()` for  `num_runs` times.

    Args:
        prob (Problem): Instance of Problem class.
        solver (Solver): Instance of Solver class.
        num_runs (int): Sample size.

    Returns:
        [pandas.DataFrame, list]: information on reasult, list of solution objects
    """
    hash = hashlib.sha1()
    hash.update(str(time.time()).encode('utf-8'))
    file_name = path.join(dir, prefix+hash.hexdigest()[:5]+".csv")
    
    estimtates = zeros(num_runs)
    
    for i in range(num_runs):
        # set up solver

        solver = solver.set_options({"seed":i, "rng": default_rng(i)}, in_situ=False)
        if reset_dict is not None:
            solver = solver.set_options(reset_dict, in_situ=False)
            
        # solve
        solution = solver.solve(prob)
        df = pd.DataFrame(index=[0])
        df["Solver"] = solver.name
        df["Problem"]=prob.name
        df["Seed"]=i
        df["Sample Size"] = prob.sample.shape[0]
        df["Truth"]=prob.prob_fail_true
        df["Estimate"] = solution.prob_fail_hist[-1]
        df["Cost"]=solution.costs
        df["Steps"]=solution.num_steps
        df["Message"]=solution.msg
        if save_other and other_list is None:
            for c in solution.other.keys():
                df[c] = solution.other[c]
        if other_list is not None:
            for c in other_list:
                df[c] = solution.other.get(c, pd.NA)
        if addtnl_cols is not None:
            for k,v in addtnl_cols.items():
                df[k]=v
                
        # save
        df.to_csv(file_name, mode="a", header=not path.exists(file_name))
        
        # talk
        if verbose:
            estimtates[i]  = solution.prob_fail_hist[-1]
            relRootMSE = sqrt(average((estimtates[0:i+1] - prob.prob_fail_true)**2)) / prob.prob_fail_true
            print("Rel. Root MSE after " +  str(i+1) + "/" +str(num_runs) + " runs: " + str(relRootMSE), end="\r" if i < num_runs - 1 else "\n")
        del df
        del solution
        gc.collect()





def add_evaluations(df:pd.DataFrame, only_success=False, remove_outliers=False) -> pd.DataFrame:
    """Add quantities of interest to dataframe.

    Args:
        df (pd.DataFrame): Dataframe with columns constants.DF_INITIAL_COLUMNS.

    Returns:
        df: [description] Dataframe with added columns
    """
    df["Difference"] = df["Truth"] - df["Estimate"]
    df["Relative Error"] = abs(df["Difference"]) / df["Truth"]
    idxs = df.groupby(["Problem", "Solver", "Sample Size"]).indices
    # Mask for successful runs
    msk = df["Message"] == "Success"
    idx_success = df.index[msk]
    # Add MSE et al.
    df.reindex(columns = ["MSE", 
                          "CVAR Estimate"
                          "Relative MSE", 
                          "Root MSE", 
                          ".25 Relative Error",
                          ".50 Relative Error",
                          ".75 Relative Error",
                          "Relative Root MSE", 
                          "Relative Root MSE Variance", 
                          "Estimate Mean", 
                          "Estimate Bias", 
                          "Estimate Variance",
                          "Cost Mean",
                          "Cost Varaince",
                          ".25 Cost",
                          ".50 Cost",
                          ".75 Cost",
                          "Success Rate"])
    for key,idx in idxs.items():
        if only_success:
            idx2 = [i for i in idx if i in idx_success]
        else:
            idx2 = idx
        if remove_outliers:
            p75,p25 = df.loc[idx2, "Estimate"].quantile(q=[0.75, 0.25])
            idx2 = [i for i in idx2 if df.loc[i, "Estimate"] <=p75 + 3*(p75-p25) ]
        df.loc[idx, "Estimate Mean"] = average(df.loc[idx2, "Estimate"])
        df.loc[idx, "Estimate Variance"] = var(df.loc[idx2, "Estimate"])
        df.loc[idx, "Estimate Bias"] = df.loc[idx2, "Estimate Mean"] - df.loc[idx2, "Truth"] 
        df.loc[idx, "MSE"] = average(df.loc[idx2, "Difference"]**2)
        df.loc[idx, ".25 Relative Error"] = (abs(df.loc[idx2, "Difference"]) / df.loc[idx2, "Truth"]).quantile(q=0.25)
        df.loc[idx, ".75 Relative Error"] = (abs(df.loc[idx2, "Difference"]) / df.loc[idx2, "Truth"]).quantile(q=0.75)
        df.loc[idx, ".50 Relative Error"] = (abs(df.loc[idx2, "Difference"]) / df.loc[idx2, "Truth"]).quantile(q=0.5)
        #df.loc[idx, "MSE Average"] = average((df.loc[idx2, "Truth"] - df.loc[idx2, "Average Estimate"])**2)
        df.loc[idx, "Relative MSE"] = df.loc[idx, "MSE"] / df.loc[idx, "Truth"]**2
        df.loc[idx, "Root MSE"] = sqrt(df.loc[idx, "MSE"])
        df.loc[idx, "Relative Root MSE"] = df.loc[idx, "Root MSE"] / df.loc[idx, "Truth"]  
        df.loc[idx, "Relative Root MSE Variance"] = var(abs(df.loc[idx2, "Relative Error"]))
        df.loc[idx, "Cost Mean"] = average(df.loc[idx2, "Cost"])
        df.loc[idx, ".25 Cost"] = df.loc[idx2, "Cost"].quantile(q=0.25)
        df.loc[idx, ".75 Cost"] = df.loc[idx2, "Cost"].quantile(q=0.75)
        df.loc[idx, ".50 Cost"] = df.loc[idx2, "Cost"].quantile(q=0.5)
        df.loc[idx, "Cost Variance"] = var(df.loc[idx2, "Cost"])  
        df.loc[idx, "Success Rate"] = average(df.loc[idx, "Message"]=="Success")
        df.loc[idx, "CVAR Estimate"] = variation(df.loc[idx2, "Estimate"])
        #df.loc[idx, "Success Average"] = average(df.loc[idx2, "MSE"] >= df.loc[idx2, "MSE Average"])
    return df


def aggregate_df(df:pd.DataFrame, cols=None) -> pd.DataFrame:
    """Custom aggregation of df coming from add_evaluations.

    Args:
        df (pd.DataFrame): Dataframe, assumed to come from  add_evaluations.

    Returns:
        pd.DataFrame: Dataframe with multiindex.
    """
    if cols is None:
        cols = ["Problem","Solver","Sample Size"]
    else:
        cols.extend(["Problem","Solver","Sample Size"])
    df = df.groupby(by=cols)
    path_by_multi_index = {
        g: df.get_group(g).loc[:,"Path"].unique().item()
        for g in df.groups.keys()
    }
    df = df.mean(numeric_only=True)
    df["Path"] = nan
    for k,v in path_by_multi_index.items():
        df.loc[k,"Path"] = v
    return df

