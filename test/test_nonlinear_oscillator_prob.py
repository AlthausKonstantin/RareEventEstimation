import rareeventestimation as ree


def test_nonlinear_oscillator():
    cbree = ree.CBREE()
    prob = ree.prob_nonlin_osc
    prob.set_sample(5000, seed=1)
    sol = cbree.solve(prob)
    assert sol.get_rel_err(prob)[-1] < 0.1
