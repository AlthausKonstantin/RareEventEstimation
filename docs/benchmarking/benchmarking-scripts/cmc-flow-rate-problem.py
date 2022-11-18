import rareeventestimation as ree
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str,
                    default="./docs/benchmarking/data/cmc-flow-rate")
args = parser.parse_args()
# Set up solvers
# num_runs ≥ (1-pf)/(pf*cvar_tgt^2)=(1-1e-4)/(1e-4 *0.05^2)~4*1e6
num_runs = 1e7
solver_list = [ree.CMC(num_runs, seed=1, verbose=True)]

# set up problems
problem_list = [ree.make_flowrate_problem(10, 2**6)]


def main():
    total = len(solver_list)*len(problem_list)
    counter = 1
    for problem in problem_list:
        for i, solver in enumerate(solver_list):
            print(
                f"({counter}/{total}) {problem.name}, {num_runs} Samples, with {solver.name}")
            ree.do_multiple_solves(problem,
                                   solver,
                                   1,
                                   dir=args.dir,
                                   prefix=f"{problem.name}".replace(
                                       " ", "_"),
                                   save_other=False)
            counter += 1


if __name__ == "__main__":
    main()
