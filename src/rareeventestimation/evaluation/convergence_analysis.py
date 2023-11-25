"""Functions to do an empirical analysis of convergence behavior."""

from ast import literal_eval
from copy import deepcopy
from glob import glob
from re import sub
from numbers import Real
from numpy import (
    asarray,
    average,
    float64,
    ndarray,
    sqrt,
    zeros,
    var,
    nan,
    stack,
    array,
    unique,
    reshape,
)
from scipy.stats import variation
from rareeventestimation.solution import Solution
from rareeventestimation.problem.problem import Problem
from rareeventestimation.solver import CBREE, Solver, CBREECache
from rareeventestimation.evaluation.constants import DOUBLE_PRECISION
from numpy.random import default_rng
import pandas as pd
from os import path
import gc
from tempfile import NamedTemporaryFile


def do_multiple_solves(
    prob: Problem,
    solver: Solver,
    num_runs: int,
    dir=".",
    prefix="",
    verbose=True,
    reset_dict=None,
    save_other=False,
    other_list=None,
    addtnl_cols=None,
) -> str:
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

        solver = solver.set_options({"seed": i, "rng": default_rng(i)}, in_situ=False)
        if reset_dict is not None:
            solver = solver.set_options(reset_dict, in_situ=False)

        # solve
        try:
            solution = solver.solve(prob)
        except Exception as e:
            # set up emtpy solution
            solution = Solution(
                prob.sample[None, ...],
                nan * zeros(1),
                nan * zeros(prob.sample.shape[0]),
                zeros(1),
                0,
                str(e),
            )

        df = pd.DataFrame(index=[0])
        df["Solver"] = str(solver)
        df["Problem"] = prob.name
        df["Seed"] = i
        df["Sample Size"] = prob.sample.shape[0]
        df["Truth"] = prob.prob_fail_true
        df["Estimate"] = solution.prob_fail_hist[-1]
        df["Cost"] = solution.costs
        df["Steps"] = solution.num_steps
        df["Message"] = solution.msg
        if solution.other is not None:
            if save_other and other_list is None:
                for c in solution.other.keys():
                    df[c] = [solution.other[c].tolist()]
            if other_list is not None:
                for c in other_list:
                    val = solution.other.get(c, asarray([pd.NA]))
                    if hasattr(val, "tolist"):
                        df[c] = val.tolist()
                    if isinstance(val, list):
                        df[c] = val
                    else:
                        df[c] = [val]
                    try:
                        df[c] = df[c].map(list)
                    except TypeError:
                        if not (
                            isinstance(df[c].values[0], Real)
                            or isinstance(df[c].values[0], str)
                        ):
                            print(
                                f"{c} cannot be cast to a list and has type {type(df[c].values[0])}"
                            )
        if addtnl_cols is not None:
            for k, v in addtnl_cols.items():
                df[k] = v
        # save
        df.to_csv(file_name, mode="a", header=write_header, index=False)
        write_header = False

        # talk
        if verbose:
            estimtates[i] = solution.prob_fail_hist[-1]
            relRootMSE = (
                sqrt(average((estimtates[0 : i + 1] - prob.prob_fail_true) ** 2))
                / prob.prob_fail_true
            )
            print(
                "Rel. Root MSE after "
                + str(i + 1)
                + "/"
                + str(num_runs)
                + " runs: "
                + str(relRootMSE),
                end="\r" if i < num_runs - 1 else "\n",
            )
        del df
        del solution
        gc.collect()

    return file_name


def study_cbree_observation_window(
    prob: Problem,
    solver: CBREE,
    num_runs: int,
    dir=".",
    prefix="",
    verbose=True,
    observation_window_range=range(2, 15),
    reset_dict=None,
    solve_from_caches_callbacks=None,
    save_other=False,
    other_list=None,
    addtnl_cols=None,
) -> str:
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
        reset_dict (dict, optional): Reset the attributes of `solver` before each run according to this dict. Defaults to None.
        solve_from_caches_callbacks (dict, optional): Apply callbacks (values of dict) when solving from caches. Add name of callback (keys of dict) as a column to results. Defaults to None.
        save_other (bool, optional): Whether to save the entries from `other` (attribute of solution object) in the csv file. Defaults to False.
        other_list (_type_, optional): Whether to save these entries from `other` (attribute of solution object) in the csv file.. Defaults to None.
        addtnl_cols (dict, optional): Add columns with key names and fill with values from this dict. Defaults to None.
    """
    with NamedTemporaryFile(prefix=prefix, suffix=".csv", delete=False, dir=dir) as f:
        file_name = f.name
    write_header = True
    estimtates = zeros(num_runs)
    # save this, callback changes before solving_from_caches
    original_callback = solver.callback
    for i in range(num_runs):
        solver = solver.set_options(
            {
                "seed": i,
                "rng": default_rng(i),
                "divergence_check": False,
                "save_history": True,
                "return_caches": True,
                "callback": original_callback,
            },
            in_situ=False,
        )
        if reset_dict is not None:
            solver = solver.set_options(reset_dict, in_situ=False)

        # solve
        try:
            solution = solver.solve(prob)
            cache_list = solution.other["cache_list"]
            df = pd.DataFrame(index=[0])
            df["Solver"] = solver.name
            df["Problem"] = prob.name
            df["Seed"] = i
            df["Sample Size"] = prob.sample.shape[0]
            df["Truth"] = prob.prob_fail_true
            df["Estimate"] = solution.prob_fail_hist[-1]
            df["Cost"] = solution.costs
            df["Steps"] = solution.num_steps
            df["Message"] = solution.msg
            if solution.other is not None:
                if save_other and other_list is None:
                    for c in solution.other.keys():
                        df[c] = [solution.other[c].tolist()]
                if other_list is not None:
                    for c in other_list:
                        df[c] = [solution.other.get(c, asarray([pd.NA])).tolist()]
                        try:
                            df[c] = df[c].map(list)
                        except TypeError:
                            if not (
                                isinstance(df[c].values[0], Real)
                                or isinstance(df[c].values[0], str)
                            ):
                                print(
                                    f"{c} cannot be cast to a list and has type {type(df[c].values[0])}"
                                )
            if addtnl_cols is not None:
                for k, v in addtnl_cols.items():
                    df[k] = v
            df["divergence_check"] = False
            df["observation_window"] = 0
            df["callback"] = solver.callback is not None
            df.to_csv(file_name, mode="a", header=write_header)
            write_header = False
            # Now solve with observation window and callbacks
            if solve_from_caches_callbacks is None:
                solve_from_caches_callbacks = {False: None}
            for win_len in observation_window_range:
                for callback_name, callback in solve_from_caches_callbacks.items():
                    # set up solver

                    solver = solver.set_options(
                        {
                            "seed": i,
                            "rng": default_rng(i),
                            "divergence_check": True,
                            "observation_window": win_len,
                        },
                        in_situ=False,
                    )
                    if reset_dict is not None:
                        solver = solver.set_options(reset_dict, in_situ=False)
                    solver.callback = callback
                    # solve
                    try:
                        solution = solver.solve_from_caches(deepcopy(cache_list))
                    except Exception as e:
                        # set up emtpy solution
                        solution = Solution(
                            prob.sample[None, ...],
                            nan * zeros(1),
                            nan * zeros(prob.sample.shape[0]),
                            zeros(1),
                            0,
                            str(e),
                        )
                    df = pd.DataFrame(index=[0])
                    df["Solver"] = solver.name
                    df["Problem"] = prob.name
                    df["Seed"] = i
                    df["Sample Size"] = prob.sample.shape[0]
                    df["Truth"] = prob.prob_fail_true
                    df["Estimate"] = solution.prob_fail_hist[-1]
                    df["Cost"] = solution.costs
                    df["Steps"] = solution.num_steps
                    df["Message"] = solution.msg
                    if solution.other is not None:
                        if save_other and other_list is None:
                            for c in solution.other.keys():
                                df[c] = [solution.other[c].tolist()]
                        if other_list is not None:
                            for c in other_list:
                                df[c] = [
                                    solution.other.get(c, asarray([pd.NA])).tolist()
                                ]
                                try:
                                    df[c] = df[c].map(list)
                                except TypeError:
                                    if not (
                                        isinstance(df[c].values[0], Real)
                                        or isinstance(df[c].values[0], str)
                                    ):
                                        print(
                                            f"{c} cannot be cast to a list and has type {type(df[c].values[0])}"
                                        )
                    if addtnl_cols is not None:
                        for k, v in addtnl_cols.items():
                            df[k] = v
                    df["divergence_check"] = True
                    df["observation_window"] = win_len
                    df["callback"] = callback_name
                    # save
                    df.to_csv(file_name, mode="a", header=False)

            # talk
            if verbose:
                estimtates[i] = solution.prob_fail_hist[-1]
                relRootMSE = (
                    sqrt(average((estimtates[0 : i + 1] - prob.prob_fail_true) ** 2))
                    / prob.prob_fail_true
                )
                print(
                    "Rel. Root MSE after "
                    + str(i + 1)
                    + "/"
                    + str(num_runs)
                    + " runs: "
                    + str(relRootMSE),
                    end="\r" if i < num_runs - 1 else "\n",
                )
        except Exception as e:
            print(e)

    return file_name


def load_data(pth: str, pattern: str, recursive=True, kwargs={}) -> pd.DataFrame:
    """Load files from `do_multiple_solves` and return them as a DataFrame.

    Also works for files created by `study_cbree_observation_window`.
    Assume that solution files are of the csv format.

    Args:
        pth (str): Where to look for the solutions.
        pattern (str): Shell style pattern for the filenames that are considered (excluding extension '.csv')
        recursive (bool, optional): Also look into subdirectories of `dir`. Defaults to True.
        kwargs(dict, optional). Options for pd.read_csv
    Returns:
        pd.DataFrame: All solution files in `dir`.
    """
    files = glob(path.join(pth, pattern + ".csv"), recursive=recursive)
    # Hotfix for mixed up columns :(
    df_list = [pd.read_csv(f, **kwargs) for f in files]
    for i, df in enumerate(df_list):
        if all([c in df.columns for c in ["divergence_check", "observation_window"]]):
            if "True" in df["observation_window"].values:
                df = df.rename(
                    columns={"divergence_check": "1", "observation_window": "2"}
                )
                df = df.rename(
                    columns={"1": "observation_window", "2": "divergence_check"}
                )
                df_list[i] = df.replace(
                    {
                        "divergence_check": {"True": True, "0": False},
                        "observation_window": {"False": 0, "2": 2, "5": 5, "10": 10},
                    }
                )
    df = pd.concat(df_list, ignore_index=True)
    # hotfix for missnamed problems :(
    df = df.replace(
        {
            "Problem": {
                "Fujita Rackwitz (d=2)": "Fujita Rackwitz Problem (d=2)",
                "Fujita Rackwitz (d=50)": "Fujita Rackwitz Problem (d=50)",
            }
        }
    )
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    # Now find all columns containing arrays (as strings) and transform 'em
    c = CBREECache(zeros(0), zeros(0), zeros(0))
    cache_attributes = [a for a in dir(c) if a[0] != "_"]

    def my_eval(cell):
        return asarray(literal_eval(cell.replace("nan", "None")))

    for c in df.columns:
        if c in cache_attributes and isinstance(df[c][0], str) and df[c][0][0] == "[":
            df[c] = df[c].apply(my_eval)
    return df


def add_evaluations(
    df: pd.DataFrame, only_success=False, remove_outliers=False
) -> pd.DataFrame:
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
    df.reindex(
        columns=[
            "MSE",
            "CVAR Estimate" "Relative MSE",
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
            "Success Rate",
        ]
    )
    for key, idx in idxs.items():
        if only_success:
            idx2 = [i for i in idx if i in idx_success]
        else:
            idx2 = idx
        if remove_outliers:
            p75, p25 = df.loc[idx2, "Estimate"].quantile(q=[0.75, 0.25])
            idx2 = [i for i in idx2 if df.loc[i, "Estimate"] <= p75 + 3 * (p75 - p25)]
        df.loc[idx, "Estimate Mean"] = average(df.loc[idx2, "Estimate"])
        df.loc[idx, "Estimate Variance"] = var(df.loc[idx2, "Estimate"])
        df.loc[idx, "Estimate Bias"] = (
            df.loc[idx2, "Estimate Mean"] - df.loc[idx2, "Truth"]
        )
        df.loc[idx, "MSE"] = average(df.loc[idx2, "Difference"] ** 2)
        if average(df.loc[idx2, "Difference"] ** 2) == 0.0:
            print("stop")

        df.loc[idx, ".25 Relative Error"] = (
            abs(df.loc[idx2, "Difference"]) / df.loc[idx2, "Truth"]
        ).quantile(q=0.25)
        df.loc[idx, ".75 Relative Error"] = (
            abs(df.loc[idx2, "Difference"]) / df.loc[idx2, "Truth"]
        ).quantile(q=0.75)
        df.loc[idx, ".50 Relative Error"] = (
            abs(df.loc[idx2, "Difference"]) / df.loc[idx2, "Truth"]
        ).quantile(q=0.5)
        # df.loc[idx, "MSE Average"] = average((df.loc[idx2, "Truth"] - df.loc[idx2, "Average Estimate"])**2)
        df.loc[idx, "Relative MSE"] = df.loc[idx, "MSE"] / df.loc[idx, "Truth"] ** 2
        df.loc[idx, "Root MSE"] = sqrt(df.loc[idx, "MSE"])
        df.loc[idx, "Relative Root MSE"] = (
            df.loc[idx, "Root MSE"] / df.loc[idx, "Truth"]
        )
        df.loc[idx, "Relative Root MSE Variance"] = var(
            abs(df.loc[idx2, "Relative Error"])
        )
        df.loc[idx, "Cost Mean"] = average(df.loc[idx2, "Cost"])
        df.loc[idx, ".25 Cost"] = df.loc[idx2, "Cost"].quantile(q=0.25)
        df.loc[idx, ".75 Cost"] = df.loc[idx2, "Cost"].quantile(q=0.75)
        df.loc[idx, ".50 Cost"] = df.loc[idx2, "Cost"].quantile(q=0.5)
        df.loc[idx, "Cost Variance"] = var(df.loc[idx2, "Cost"])
        df.loc[idx, "Success Rate"] = average(df.loc[idx, "Message"] == "Success")
        df.loc[idx, "CVAR Estimate"] = variation(df.loc[idx2, "Estimate"])
        # df.loc[idx, "Success Average"] = average(df.loc[idx2, "MSE"] >= df.loc[idx2, "MSE Average"])
    return df


def aggregate_df(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Custom aggregation of df coming from add_evaluations.

    Args:
        df (pd.DataFrame): Dataframe, assumed to come from  add_evaluations.

    Returns:
        pd.DataFrame: Dataframe with multiindex.
    """
    if cols is None:
        cols = ["Problem", "Solver", "Sample Size"]
    else:
        cols.extend(["Problem", "Solver", "Sample Size"])

    def my_mean(grp):
        vals = []
        cc = [c for c in grp.columns if c not in ["Problem", "Solver", "Sample Size"]]
        for c in cc:
            if len(grp[c]) == 0:
                vals.append(None)
            else:
                if isinstance(grp[c].values[0], ndarray):
                    arr = stack(grp[c].values)
                    vals.append(average(arr.astype(float64), axis=0))
                elif isinstance(grp[c].values[0], Real):
                    vals.append(grp[c].mean())
                else:
                    vals.append(grp[c].unique()[0])
        # for c in cols:
        #     if grp[c].dtype.str == '|O':
        #         vals.append(grp[c].unique()[0])
        #     else:
        #         vals.append(grp[c].mean())
        return pd.Series(vals, index=cc)

    df_agg = df.groupby(cols).apply(my_mean)
    df_agg.reset_index(inplace=True)
    return df_agg


def get_benchmark_df(
    data_dirs={
        "enkf": "docs/benchmarking/data/enkf_sim",
        "sis": "docs/benchmarking/data/sis_sim",
    },
    df_dir="docs/benchmarking/data",
    df_names={
        "df": "benchmark_toy_problems_processed.json",
        "df_agg": "benchmark_toy_problems_aggregated.json",
    },
    file_regex="*",
    force_reload=False,
    remove_outliers=False,
) -> tuple:
    """Custom function to load benchmark simultions.

    Args:
        data_dirs (dict, optional): Paths to simulations. Defaults to {"enkf":"docs/benchmarking/data/enkf_sim", "sis":"docs/benchmarking/data/sis_sim"}.
        df_dir (str, optional): Look here for jsond results before loading. Defaults to "docs/benchmarking/data".
        force_reload (bool, optional): If true, don't look for json'd results.

    Returns:
        tuple: (dataset, aggregated dataset)
    """
    path_df = path.join(df_dir, df_names["df"])
    path_df_agg = path.join(df_dir, df_names["df_agg"])
    if not (path.exists(path_df) and path.exists(path_df_agg)) or force_reload:
        # load dfs
        df = None
        df_agg = None
        for method, data_dir in data_dirs.items():
            this_df = load_data(data_dir, file_regex)
            this_df.drop(columns=["index", "Unnamed: 0"], inplace=True, errors="ignore")
            this_df.drop_duplicates(inplace=True)
            # cast solver names, add cvat_tgt to name for
            # correct grouping in add_evaluation and aggregate_df
            this_df = this_df.apply(
                force_bm_names, axis=1, name="Solver", specification="mixture_model"
            )
            this_df["Solver"] = this_df.apply(
                lambda x: ", ".join([x.Solver, str(x.cvar_tgt)]), axis=1
            )
            this_df = add_evaluations(this_df, remove_outliers=remove_outliers)
            this_df_agg = aggregate_df(this_df)
            if df is None:
                df = this_df
            else:
                df = pd.concat([df, this_df], ignore_index=True)
            if df_agg is None:
                df_agg = this_df_agg
            else:
                df_agg = pd.concat([df_agg, this_df_agg], ignore_index=True)
        # recast solver nmaes for plotting
        df = df.apply(
            force_bm_names, axis=1, name="Solver", specification="mixture_model"
        )
        df.to_json(path_df, double_precision=DOUBLE_PRECISION)
        df_agg = df_agg.apply(
            force_bm_names, axis=1, name="Solver", specification="mixture_model"
        )
        df_agg.to_json(path_df_agg, double_precision=DOUBLE_PRECISION)
    else:
        df = pd.read_json(path_df)
        df_agg = pd.read_json(path_df_agg)
    return df, df_agg


def force_bm_names(
    row: pd.Series, name="Solver", specification="mixture_model"
) -> pd.Series:
    """Rename bechmark solvers from entries `name` and `specification` in `row`.

    Args:
        row (pd.Series): Row of a benchmark dataframe
        name (str, optional): Look in this column for the base name. Defaults to "Solver".
        specification (str, optional): Look in this column for any specifications. Defaults to "mixture_model".

    Returns:
        pd.Series: `row` with changed entry at `name`.
    """
    old_name = row[name]
    new_name = old_name
    if "enkf" in old_name.lower():
        new_name = "EnKF"
    if "sis" in old_name.lower():
        new_name = "SIS"
    my_specs = {"vmfnm": " (vMFNM)", "gm": " (GM)", "acs": " (aCS)"}
    for pattern, spec in my_specs.items():
        if pattern in row[name].lower() or pattern in row[specification].lower():
            new_name += spec
    row[name] = new_name
    return row


def expand_cbree_name(input: pd.Series, columns=[], pattern="}") -> pd.Series:
    """Custom function adds parameter values to field `Solver`.

    Args:
        input (pd.Series): Row of dataframe. Must contain `Solver` and everything in `columns`
        columns (list, optional):Add these columns and their value to the solver name. Defaults to [].
        pattern (str, optional): New values replace this pattern. Defaults to "}".

    Returns:
        pd.Sereis: input with updated field `Solver`
    """
    for col in columns:
        input.Solver = input.Solver.replace(pattern, f", '{col}': '{input[col]}'}}")
    return input


def my_number_formatter(x: Real) -> str:
    """Custom version of `str(x)`.
    Integers get integer notation, floats are cut after 2 digits
    Args:
        x (Real): Number to format.

    Returns:
        str: Formatted number
    """
    if int(x) == x:
        return str(int(x))
    else:
        return f"{x:.2f}"


def vec_to_latex_set(vec: ndarray) -> str:
    """Make a nice latex set representation of the values in vec.

    Args:
        vec (ndarray): Array with `Real` entries.

    Returns:
        str: Something like "vec[0]" or "\\{{vec[0|, vec[1], ..., vec[-1]\\}}".
    """
    vec = reshape(vec, (-1))
    vec = unique(vec)
    if len(vec) == 1:
        return my_number_formatter(vec[0])
    if len(vec) == 2:
        return f"\\{{{my_number_formatter(vec[0])}, {my_number_formatter(vec[-1])}\\}}"
    if len(vec) > 2:
        return f"\\{{{my_number_formatter(vec[0])}, {my_number_formatter(vec[1])}, \\ldots, {my_number_formatter(vec[-1])}\\}}"


def list_to_latex(ls: list) -> str:
    """Exhaustive version of `vec_to_l  ex_set` with no sorting or ellipsis.

    Args:
        ls (list): Stuff to enumerate

    Returns:
        str: Latex set of all `ls` entries if all entries are `Real`.
            Else Enumeration of all objects in `ls` seperated by a ', '.
    """
    if len(ls) == 0:
        return "$\\emptyset$"
    if len(ls) == 1 and isinstance(ls[0], Real):
        return my_number_formatter(ls[0])
    if len(ls) == 1:
        return f"{ls[0]}"
    if all(array([isinstance(v, Real) for v in ls])):
        return f"$\\{{ {', '.join(map(my_number_formatter, ls))} \\}}$"
    else:
        return f"{', '.join(ls)}"


def squeeze_problem_names(prob_name: str) -> str:
    """Abbreviate the problem names.

    Args:
        prob_name (str): Problem name.

    Returns:
        str: Abbreviated problem name.
    """
    d = sub(r"\D", "", prob_name)
    if prob_name.startswith("Convex"):
        return "CP"
    if prob_name.startswith("Linear"):
        return f"LP (d={d})"
    if prob_name.startswith("Fujita"):
        return f"FRP (d={d})"
    return prob_name
