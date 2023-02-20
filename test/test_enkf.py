import rareeventestimation as ree
import numpy as np
def test_enkf():
    for m in["GM", "vMFNM"]:
        for p in ree.problems_lowdim:
            p.set_sample(5000,seed=5000)
            enkf = ree.ENKF(seed=1, is_distribution=m)
            sol = enkf.solve(p)
            assert sol.get_rel_err(p)[-1] < 0.2, f"Error too large for {p.name}"
        for p in [ree.make_fujita_rackwitz(50), ree.make_linear_problem(50)]:
            p.set_sample(5000,seed=5000)
            enkf = ree.ENKF(seed=1, is_distribution=m)
            sol = enkf.solve(p)
            assert sol.get_rel_err(p)[-1] < 0.2, f"Error too large for {p.name}"