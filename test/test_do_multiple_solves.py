import tempfile
import rareeventestimation as ree
import pandas as pd
def test_do_multiple_solves():
    for p in ree.problems_lowdim:
        p.set_sample(5000,seed=5000)
        cbree = ree.CBREE()
        tmp_dir = tempfile.TemporaryDirectory()
        num_runs = 10
        file = ree.do_multiple_solves(p,
                               cbree,
                               num_runs,
                               dir=tmp_dir.name,
                               save_other=True,
                               other_list=["t_step"],
                               addtnl_cols={"col":"val"})
        df = pd.read_csv(file)
        assert len(df.index) == num_runs , "Dataframe has the wrong number of rows"
    
    tmp_dir.cleanup()

