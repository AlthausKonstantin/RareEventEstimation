import rareeventestimation as ree
import numpy as np
from copy import deepcopy


def test_cbree_solve_from_caches():
    p = ree.prob_convex
    p.set_sample(1000, seed=5000)
    cvar_tgt = 0.5
    cbree = ree.CBREE(
        seed=1,
        divergence_check=False,
        cvar_tgt=cvar_tgt,
        save_history=True,
        return_caches=True,
    )
    sol0 = cbree.solve(p)
    for k in range(2, 15):
        cbree = ree.CBREE(
            seed=1, divergence_check=True, cvar_tgt=cvar_tgt, observation_window=k
        )
        cbree_ref = ree.CBREE(
            seed=1, divergence_check=True, cvar_tgt=cvar_tgt, observation_window=k
        )
        simulation = cbree.solve_from_caches(deepcopy(sol0.other["cache_list"]))
        truth = cbree_ref.solve(p)
        same = (
            np.array_equal(
                simulation.ensemble_hist[-1], truth.ensemble_hist[-1], equal_nan=True
            )
            and np.array_equal(
                simulation.lsf_eval_hist[-1], truth.lsf_eval_hist[-1], equal_nan=True
            )
            and simulation.costs == truth.costs
        )
        assert same, "Method `solve_from_caches` does not work properly."
