import rareeventestimation as ree


def test_cmc():
    num_runs = 100000
    solver = ree.CMC(num_runs, seed=1, verbose=True)
    sol = solver.solve(ree.prob_convex)
    assert sol.get_rel_err(ree.prob_convex) < 0.01
