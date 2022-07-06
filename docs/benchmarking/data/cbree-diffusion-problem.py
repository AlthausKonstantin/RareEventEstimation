import rareeventestimation as ree
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="/Users/konstantinalthaus/Documents/Master TUM/Masterthesis/Package/rareeventestimation/docs/benchmarking/data/cbree_sim/diffusion_sim")
parser.add_argument("--counter",
                    type=int,
                    default=0)
args = parser.parse_args()


# Set up solvers
def callback_vmfnm(cache, solver):
    if not cache.converged:
        return cache
    cache.mixture_model = ree.VMFNMixture(1)
    cache.mixture_model.fit(cache.ensemble)
    cache.ensemble = cache.mixture_model.sample(cache.ensemble.shape[0], rng=solver.rng)
    cache.lsf_evals = solver.lsf(cache.ensemble)
    cache.e_fun_evals = solver.e_fun(cache.ensemble)
    log_pdf_evals = cache.mixture_model.logpdf(cache.ensemble)
    cache.cvar_is_weights = ree.my_log_cvar(-cache.e_fun_evals - log_pdf_evals, multiplier=(cache.lsf_evals<=0))
    cache.num_lsf_evals += cache.ensemble.shape[0]
    return cache

keywords = {
    "stepsize_tolerance" : [1,2],
    "mixture_model": ["GM"],
    "cvar_tgt":[2],
    "lip_sigma": [1],
    "tgt_fun": ["arctan"],
    "observation_window":[5], 
    "callback":[callback_vmfnm]}

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
    solver = ree.CBREE(**kwargs)
    solver.name = "CBREE " + str(kwargs)
    solver_list.append(solver)
    kwarg_list.append(kwargs)
    

# set up problems
problem_list = [ree.diffusion_problem]
# set up other parameters

sample_sizes =[1000, 2000, 3000, 4000, 5000, 6000]
num_runs = 100




def main():
    total = len(solver_list)*len(sample_sizes)*len(problem_list)
    counter = 1
    for problem in problem_list:
        for i, solver in enumerate(solver_list):
            for s in sample_sizes:
                if counter > args.counter:
                    print(f"({counter}/{total}) {problem.name}, {s} Samples, with {solver.name}")
                    problem.set_sample(s, seed=s)
                    ree.do_multiple_solves(problem,
                                    solver,
                                    num_runs,
                                    dir=args.dir,
                                    prefix=f"{problem.name} {counter} ".replace(" ", "_"),
                                    save_other=False,
                                    addtnl_cols=kwarg_list[i])
                counter += 1

if __name__ == "__main__":
   main()