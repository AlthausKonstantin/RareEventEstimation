
from os import path
import rareeventestimation as ree
import pandas as pd
def get_benchmark_df(data_dirs = {"enkf":"/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation/docs/benchmarking/data/enkf_sim",
                "sis":"/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation/docs/benchmarking/data/sis_sim"}, df_dir = "/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation/docs/benchmarking/data") -> tuple:
    """Custom function to load benchmark simultions.

    Args:
        data_dirs (dict, optional): Paths to simulations. Defaults to {"enkf":"/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation/docs/benchmarking/data/enkf_sim", "sis":"/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation/docs/benchmarking/data/sis_sim"}.
        df_dir (str, optional): Look here for pickled results before loading. Defaults to "/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation/docs/benchmarking/data".

    Returns:
        tuple: (dataset, aggregated dataset)
    """    
    path_df= path.join(df_dir, "benchmark_toy_problems_processed.pkl")
    path_df_agg = path.join(df_dir, "benchmark_toy_problems_aggregated.pkl")
    if not (path.exists(path_df) and path.exists(path_df_agg)):
        # load dfs
        df = None
        df_agg = None
        for method, data_dir in data_dirs.items():
            this_df = ree.load_data(data_dir, "*")
            this_df.drop(columns=["index", "Unnamed: 0"], inplace=True)
            this_df.drop_duplicates(inplace=True)
            this_df = ree.add_evaluations(this_df)
            this_df_agg = ree.aggregate_df(this_df)
            if df is None:
                df = this_df
            else:
                df = pd.concat([df, this_df],ignore_index=True)
            if df_agg is None:
                df_agg = this_df_agg
            else:
                df_agg = pd.concat([df_agg, this_df_agg],ignore_index=True)
        df.to_pickle(path_df)
        df_agg.to_pickle(path_df_agg)
    else:
        df = pd.read_pickle(path_df)
        df_agg = pd.read_pickle(path_df_agg)
    return df, df_agg
    