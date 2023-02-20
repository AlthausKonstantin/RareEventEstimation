import tempfile
import rareeventestimation as ree
import pandas as pd
def test_study_cbree_observation_window():
    for p in ree.problems_lowdim:
        p.set_sample(5000,seed=5000)
        cbree = ree.CBREE()
        tmp_dir = tempfile.TemporaryDirectory()
        num_runs = 10
        observation_window_range=range(2,15)
        file = ree.study_cbree_observation_window(p,
                               cbree,
                               num_runs,
                               dir=tmp_dir.name,
                               save_other=True,
                               observation_window_range=observation_window_range,
                               other_list=["t_step"],
                               addtnl_cols={"col":"val"})
        df = pd.read_csv(file)
        assert len(df.index) == num_runs*(len(observation_window_range)+1) , "Dataframe has the wrong number of rows"
    
    tmp_dir.cleanup()

