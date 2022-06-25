"""Functions to do an empirical analysis of convergence behavior."""

from copy import deepcopy
import tempfile
from numpy import average, sqrt, zeros, var, nan
from scipy.stats import variation
from rareeventestimation.solution import Solution
from rareeventestimation.problem.problem import Problem
from rareeventestimation.solver import CBREE, Solver
from numpy.random import default_rng
import pandas as pd
from os import path
import gc
import hashlib
import time
from tempfile import NamedTemporaryFile

def do_multiple_solves(prob:Problem,
                       solver:Solver,
                       num_runs:int,
                       dir= ".",
                       prefix="",
                       verbose=True,
                       reset_dict=None,
                       save_other=False,
                       other_list=None,
                       addtnl_cols= None) -> str:
    """Call solver multiple times and save results.

    Args:
        prob (Problem): Instance of Problem class.
        solver (Solver): Instance of Solver class.
        num_runs (int): How many times `solver` will be called.
        dir (str, optional): Where to save the results as a csv file. Defaults to ".".
        prefix (str, optional): Prefix to csv file name. Defaults to "".
        verbose (bool, optional): Whether to print some information during solving. Defaults to True.
        reset_dict (dict, optional): Reset the attributes of `solver` after each run according to this dict. Defaults to None.
        save_other (bool, optional): Whether to save the entries from `other` (attribute of solution object) in the csv file. Defaults to False.
        other_list (_type_, optional): Whether to save these entries from `other` (attribute of solution object) in the csv file.. Defaults to None.
        addtnl_cols (dict, optional): Add columns with key names and fill with values from this dict. Defaults to None.
        
    Returns:
        str: Path to results.
    """
    with NamedTemporaryFile(prefix=prefix, suffix=".csv", delete=False, dir=dir) as f:
        file_name = f.name    
    estimtates = zeros(num_runs)
    write_header = True
    for i in range(num_runs):
        # set up solver

        solver = solver.set_options({"seed":i, "rng": default_rng(i)}, in_situ=False)
        if reset_dict is not None:
            solver = solver.set_options(reset_dict, in_situ=False)
            
        # solve
        try: 
            solution = solver.solve(prob)
        except Exception as e:
            # set up emtpy solution
            solution = Solution(prob.sample[None,...],
                                nan * zeros(1),
                                nan * zeros(prob.sample.shape[0]),
                                zeros(1),
                                0,
                                str(e))
            
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
        if solution.other is not None:
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
        df.to_csv(file_name, mode="a", header=write_header)
        write_header=False
        
        
        # talk
        if verbose:
            estimtates[i]  = solution.prob_fail_hist[-1]
            relRootMSE = sqrt(average((estimtates[0:i+1] - prob.prob_fail_true)**2)) / prob.prob_fail_true
            print("Rel. Root MSE after " +  str(i+1) + "/" +str(num_runs) + " runs: " + str(relRootMSE), end="\r" if i < num_runs - 1 else "\n")
        del df
        del solution
        gc.collect()
        
    return file_name


def study_cbree_observation_window(prob:Problem,
                                   solver:CBREE,
                                   num_runs:int,
                                   dir= ".",
                                   prefix="",
                                   verbose=True,
                                   observation_window_range=range(2,15),
                                   reset_dict=None,
                                   save_other=False,
                                   other_list=None,
                                   addtnl_cols= None) -> str:
    """Call CBREE solver multiple times and save results.
    
    Cache the result of each run and redo the computation with different values for
    `observation window`

    Args:
        prob (Problem): Instance of Problem class.
        solver (Solver): Instance of Solver class.
        num_runs (int): How many times `solver` will be called.
        dir (str, optional): Where to save the results as a csv file. Defaults to ".".
        prefix (str, optional): Prefix to csv file name. Defaults to "".
        verbose (bool, optional): Whether to print some information during solving. Defaults to True.
        observation_window_range (optional): Redo runs with  `observation window` values specified here. Defaults to range(2,15)
        reset_dict (dict, optional): Reset the attributes of `solver` after each run according to this dict. Defaults to None.
        save_other (bool, optional): Whether to save the entries from `other` (attribute of solution object) in the csv file. Defaults to False.
        other_list (_type_, optional): Whether to save these entries from `other` (attribute of solution object) in the csv file.. Defaults to None.
        addtnl_cols (dict, optional): Add columns with key names and fill with values from this dict. Defaults to None.
    """
    with NamedTemporaryFile(prefix=prefix, suffix=".csv", delete=False, dir=dir) as f:
        file_name = f.name   
    
    estimtates = zeros(num_runs)
    write_header=True
    for i in range(num_runs):
        solver = solver.set_options({"seed":i, "rng": default_rng(i), "divergence_check": False, "save_history": True, "return_caches":True}, in_situ=False)
        if reset_dict is not None:
            solver = solver.set_options(reset_dict, in_situ=False)
            
        # solve
        solution = solver.solve(prob)
        cache_list = solution.other["cache_list"]
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
        df["observation_window"]=0
        df.to_csv(file_name, mode="a", header=not path.exists(file_name))
        # Now solve with observation window
        for win_len in observation_window_range:
            # set up solver

            solver = solver.set_options({"seed":i, "rng": default_rng(i), "divergence_check": True, "observation_window":win_len}, in_situ=False)
            if reset_dict is not None:
                solver = solver.set_options(reset_dict, in_situ=False)
                
            # solve
            try: 
                solution = solver.solve_from_caches(deepcopy(cache_list))
            except Exception as e:
                # set up emtpy solution
                solution = Solution(prob.sample[None,...],
                                    nan * zeros(1),
                                    nan * zeros(prob.sample.shape[0]),
                                    zeros(1),
                                    0,
                                    str(e))
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
            df["observation_window"]=win_len       
            # save
            df.to_csv(file_name, mode="a", header=write_header)
            write_header=False
            
        # talk
        if verbose:
            estimtates[i]  = solution.prob_fail_hist[-1]
            relRootMSE = sqrt(average((estimtates[0:i+1] - prob.prob_fail_true)**2)) / prob.prob_fail_true
            print("Rel. Root MSE after " +  str(i+1) + "/" +str(num_runs) + " runs: " + str(relRootMSE), end="\r" if i < num_runs - 1 else "\n")
            
    return file_name
    


def add_evaluations(df:pd.DataFrame, only_success=False, remove_outliers=False) -> pd.DataFrame:
    """Add quantities of interest to df.

    Args:
        df (pd.DataFrame): Dataframe with one result per row.
        only_success (bool, optional): Only use successful runs for aggregated evaluations. Defaults to False.
        remove_outliers (bool, optional) Remove outliers for aggregated evaluations.  Defaults to False.

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
    def my_mean(grp):
        vals=[]
        cols = [c for c in grp.columns if c not in ["Problem", "Solver", "Sample Size"]]
        for c in cols:
            if grp[c].dtype.str == '|O':
                vals.append(grp[c].unique()[0])
            else:
                vals.append(grp[c].mean())
        return pd.Series(vals, index=cols)
    df_agg = df.groupby(cols).apply(my_mean)
    df_agg.reset_index(inplace=True)
    return df

