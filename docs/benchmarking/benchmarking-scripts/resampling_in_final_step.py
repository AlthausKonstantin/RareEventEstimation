import rareeventestimation as ree
import numpy as np

# set up problem and solvers
prob = ree.prob_convex
problem_list=[prob]
sigma = 10
tgt_fun = "algebraic"
def callback_gm(cache, solver):
    """Resample if converged with a gaussian.

    Args:
        cache (_type_): current cache of CBREE
        solver (_type_): current instance solving problem

    Returns: modified cache
    """
    if not cache.converged:
        return cache
    cache.mixture_model = ree.GaussianMixture(1,seed=solver.seed)
    cache.mixture_model.fit(cache.ensemble)
    cache.ensemble = cache.mixture_model.sample(cache.ensemble.shape[0], rng=solver.rng)
    cache.lsf_evals = solver.lsf(cache.ensemble)
    cache.e_fun_evals = solver.e_fun(cache.ensemble)
    log_pdf_evals = cache.mixture_model.logpdf(cache.ensemble)
    cache.cvar_is_weights = ree.my_log_cvar(-cache.e_fun_evals - log_pdf_evals, multiplier=(cache.lsf_evals<=0))
    return cache

def callback_vmfnm(cache, solver):
    """Resample if converged with a vMFN.

    Args:
        cache (_type_): current cache of CBREE
        solver (_type_): current instance solving problem

    Returns: modified cache
    """
    if not cache.converged:
        return cache
    cache.mixture_model = ree.VMFNMixture(1)
    cache.mixture_model.fit(cache.ensemble)
    cache.ensemble = cache.mixture_model.sample(cache.ensemble.shape[0], rng=solver.rng)
    cache.lsf_evals = solver.lsf(cache.ensemble)
    cache.e_fun_evals = solver.e_fun(cache.ensemble)
    log_pdf_evals = cache.mixture_model.logpdf(cache.ensemble)
    cache.cvar_is_weights = ree.my_log_cvar(-cache.e_fun_evals - log_pdf_evals, multiplier=(cache.lsf_evals<=0))
    return cache

keywords = {
    "callback": [None, callback_gm, callback_vmfnm]
}

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = "object"
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

prod = cartesian_product(*[np.array(v) for v in keywords.values()])
solver_list = []
kwarg_list = []
for col in prod:
    kwargs = dict(zip(keywords.keys(), col))
    solver = ree.CBREE(tgt_fun=tgt_fun,divergence_check=False, observation_window=2, return_other=True, **kwargs)
    solver.name = "CBREE " + str(kwargs)
    solver_list.append(solver)
    kwarg_list.append(kwargs)
sample_sizes =[250, 500, 1000, 2000, 3000, 4000, 5000, 6000]
num_runs = 200
other_list=["sigma", "beta", "cvar_is_weights"]
out_dir = "docs/benchmarking/data/cbree_sim/resampling_in_final_step"
pattern = f"{prob.name.replace(' ', '_')}*"
total = len(solver_list)*len(sample_sizes)*len(problem_list)
counter = 1
#solve
def main():
    for s in sample_sizes:
        for prob in problem_list:
            for i, solver in enumerate(solver_list):
                print(f"({counter}/{total}) {prob.name}, {s} Samples, with {solver.name}")
                prob.set_sample(s, seed=s)
                ree.do_multiple_solves(prob,
                                solver,
                                num_runs,
                                dir=out_dir,
                                prefix=f"{prob.name}_".replace(" ", "_"),
                                other_list=other_list,
                                addtnl_cols=kwarg_list[i])
                counter +=1

if __name__ == "__main__":
   main()