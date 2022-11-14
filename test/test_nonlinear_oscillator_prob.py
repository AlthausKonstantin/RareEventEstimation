# %%
import rareeventestimation as ree
import numpy as np


def test_nonlinear_oscillator():
    cbree = ree.SIS(seed=1, cvar_tgt=0.25, mixture_model="GM")
    prob = ree.prob_nonlin_osc
    prob.set_sample(5000, seed=1)
    sol = cbree.solve(prob)
    assert sol.get_rel_err(prob) < 0.1
    return sol


# %%
sol = test_nonlinear_oscillator()

# %%
