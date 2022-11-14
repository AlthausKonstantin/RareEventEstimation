# %%
import rareeventestimation as ree
import numpy as np


def test_flow_rate_problem():
    cbree = ree.CBREE(seed=1, mixture_model="GM", cvar_tgt=0.25, num_steps=2)
    prob = ree.make_flowrate_problem(10, 2**8)
    prob.set_sample(1000, seed=1)
    sol = cbree.solve(prob)
    return sol


# %%
sol = test_flow_rate_problem()

# %%
