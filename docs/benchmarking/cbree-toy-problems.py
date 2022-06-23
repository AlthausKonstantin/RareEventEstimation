import rareeventestimation as ree
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="/cbree_sim/toy_problems")
parser.add_argument("--counter",
                    type=int,
                    default=0)
args = parser.parse_args()


# Set up solvers

keywords = {
    "stepsize_tolerance" : [0.1, 0.5, 0.75, 1],
    "mixture_model": ["GM", "vMFNM"],
    "cvar_tgt":[0.5,1,2,3,5,7,10],
    "lip_sigma": [1, 2,10],
    "tgt_fun": ["algebraic","tanh","arctan","erf","relu","sigmoid"]}

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
dims = [50]
problem_list = ree.problems_lowdim + [ree.make_fujita_rackwitz(d) for d in dims] + [ree.make_linear_problem(d) for d in dims]

# set up other parameters

sample_sizes =[1000, 2000, 3000, 4000, 5000, 6000]
num_runs = 200




def main():
    total = len(solver_list)*len(sample_sizes)*len(problem_list)
    counter = 1
    for problem in problem_list:
        for i, solver in enumerate(solver_list):
            for s in sample_sizes:
                if counter > args.counter:
                    print(f"({counter}/{total}) {problem.name}, {s} Samples, with {solver.name}")
                    problem.set_sample(s, seed=s)
                    ree.study_cbree_observation_window(problem,
                                    solver,
                                    num_runs,
                                    dir=args.dir,
                                    prefix=f"{problem.name} {counter} ".replace(" ", "_"),
                                    save_other=False,
                                    addtnl_cols=kwarg_list[i])
                counter += 1

if __name__ == "__main__":
   main()