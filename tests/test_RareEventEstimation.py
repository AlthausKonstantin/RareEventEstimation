import rareeventestimation as ree
def test_cbs():
    for p in ree.problems_lowdim:
        p.set_sample(5000,seed=5000)
        cbs = ree.CBS(seed=1, divergence_check=False)
        sol = cbs.solve(p)
        assert sol.get_rel_err(p)[-1] < 0.075, f"Error too large for {p.name}"
        
    for p in [ree.make_fujita_rackwitz(50), ree.make_linear_problem(50)]:
        p.set_sample(5000,seed=5000)
        cbs = ree.CBS(seed=1,
                      resample=True,
                      mixture_model="vMFNM",
                      cvar_tgt=3, 
                      sigma_inc_max=2)
        sol = cbs.solve(p)
    assert sol.get_rel_err(p)[-1] < 0.075, f"Error too large for {p.name}"
    
test_cbs()