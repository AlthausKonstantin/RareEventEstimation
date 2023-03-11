import rareeventestimation as ree
import pytest


def test_enkf_lowdim():
    for m in ["GM", "vMFNM"]:
        for p in ree.problems_lowdim:
            p.set_sample(5000, seed=5000)
            enkf = ree.ENKF(seed=1, is_distribution=m)
            sol = enkf.solve(p)
            assert sol.get_rel_err(p)[-1] < 0.2, f"Error too large for {p.name}"


@pytest.mark.skip(reason="no way of currently testing this")
def test_enkf_highdim():
    for m in ["GM", "vMFNM"]:
        for p in [ree.make_fujita_rackwitz(50), ree.make_linear_problem(50)]:
            p.set_sample(5000, seed=5000)
            enkf = ree.ENKF(seed=1, is_distribution=m)
            sol = enkf.solve(p)
            assert sol.get_rel_err(p)[-1] < 0.2, f"Error too large for {p.name}"
