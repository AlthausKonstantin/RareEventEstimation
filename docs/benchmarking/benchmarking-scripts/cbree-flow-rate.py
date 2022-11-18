import rareeventestimation as ree
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir", type=str, default="./docs/benchmarking/data/cbree_sim/cbree_sim_flowrate")
parser.add_argument("--start",
                    type=int,
                    default=0)
parser.add_argument("--stop",
                    type=int,
                    default=np.inf
                    )
args = parser.parse_args()


keywords = {
    "stepsize_tolerance": [1],
    "mixture_model": ["GM"],
    "cvar_tgt": [1, 2, 5],
    "lip_sigma": [1],
    "tgt_fun": ["algebraic"]}


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = "object"
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


prod = cartesian_product(*[np.array(v) for v in keywords.values()])
solver_list = []
kwarg_list = []
for col in prod:
    kwargs = dict(zip(keywords.keys(), col))
    solver = ree.CBREE(**kwargs, resample=True)
    solver.name = "CBREE " + str(kwargs)
    solver_list.append(solver)
    kwarg_list.append(kwargs)


# set up problems
problem_list = [ree.make_flowrate_problem(10, 2**6)]
# set up other parameters

sample_sizes = [1000, 2000, 3000, 4000, 5000, 6000]
num_runs = 100

other = ["Average Estimate", "Root Weighted Average Estimate",
         "VAR Weighted Average Estimate", "CVAR"]


def main():
    total = len(solver_list)*len(sample_sizes)*len(problem_list)
    counter = 1
    for problem in problem_list:
        for i, solver in enumerate(solver_list):
            for s in sample_sizes:
                if counter >= args.start and counter <= args.stop:
                    print(
                        f"({counter}/{total}) {problem.name}, {s} Samples, with {solver.name}")
                    problem.set_sample(s, seed=s)
                    ree.study_cbree_observation_window(problem,
                                                       solver,
                                                       num_runs,
                                                       dir=args.dir,
                                                       prefix=f"{problem.name} gm {counter} ".replace(
                                                           " ", "_"),
                                                       save_other=False,
                                                       other_list=other,
                                                       addtnl_cols=kwarg_list[i])
                counter += 1


if __name__ == "__main__":
    main()
