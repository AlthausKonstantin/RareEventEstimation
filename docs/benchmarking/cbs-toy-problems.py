import rareeventestimation as ree
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Script so useful.')
parser.add_argument("--dir", type=str, default="./")

args = parser.parse_args()


# Set up solvers

keywords = {
    "tol" : [0.75],
    "observation_window": [5],
    "cvar_tgt":[2,5],
    "sigma_inc_max": [2],
}

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

prod = cartesian_product(*[np.array(v) for v in keywords.values()])
solver_list = []
kwarg_list = []
for col in prod:
    kwargs = dict(zip(keywords.keys(), col))
    solver = ree.CBS(**kwargs)
    solver.name = "CBS " + str(kwargs)
    solver_list.append(solver)
    kwarg_list.append(kwargs)
    

# set up problems
dims = range(5,105,5)
problem_list = ree.problems_lowdim + ree.problems_highdim

# set up other parameters

sample_sizes =[1000]
num_runs = 10




def main():
    total = len(solver_list)*len(sample_sizes)*len(problem_list)
    counter = 1
    for problem in problem_list:
        for s in sample_sizes:
            for i, solver in enumerate(solver_list):
                if counter > 0:
                    if problem in ree.problems_highdim:
                        solver = solver.set_options(dict(resample=True,
                                                    mixture_model="vMFNM"),
                                                    in_situ=False)
                    print(f"({counter}/{total}) {problem.name}, {s} Samples, with {solver.name}")
                    problem.set_sample(s, seed=s)
                    ree.do_multiple_solves(problem,
                                    solver,
                                    num_runs,
                                    dir=args.dir,
                                    prefix=f"{problem.name} {s}".replace(" ", "_"),
                                    save_other=True,
                                    addtnl_cols=kwarg_list[i])
                counter += 1

if __name__ == "__main__":
   main()