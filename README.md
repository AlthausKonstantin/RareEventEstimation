# rareeventestimation

[![coverage](https://raw.githubusercontent.com/AlthausKonstantin/rareeventestimation/gh-pages/_static/images/coverage_badge.svg)](https://github.com/AlthausKonstantin/rareeventestimation/actions/workflows/run_tests.yaml)
[![test](https://github.com/AlthausKonstantin/rareeventestimation/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/AlthausKonstantin/rareeventestimation/actions/workflows/run_tests.yaml)
[![build documentation](https://github.com/AlthausKonstantin/rareeventestimation/actions/workflows/build_documentation.yaml/badge.svg)](https://github.com/AlthausKonstantin/rareeventestimation/actions/workflows/build_documentation.yaml)
[![publish package](https://github.com/AlthausKonstantin/rareeventestimation/actions/workflows/publish.yaml/badge.svg)](https://github.com/AlthausKonstantin/rareeventestimation/actions/workflows/publish.yaml)
[![PyPI version](https://badge.fury.io/py/rareeventestimation.svg)](https://badge.fury.io/py/rareeventestimation)

Estimate rare events with consensus-based sampling and other methods.
## Installation
```bash
$ pip install rareeventestimation
```

## Introduction and Usage
In reliability analysis, it is often crucial to estimate the probability of failure of a system.  
For a $d$-dimensional system the notion of failure is encoded in a limit state function (LSF) $G:\mathbb{R}^d \rightarrow \mathbb{R}$, which maps safe states to positive values and failure states to values of at most $0$.  
If the states follow some distribution that has with respect to the Lebesgue measure on $\mathbb{R}^d$ the density $\pi$,
then the probability of failure $P_f$ is defined as the probability mass of all failure states.
Namely, $P_f =\int_{G\leq 0}\pi(x)dx$.  
By construction the probability of failure is small and additionally, the evaluation of the limit state function is often computationally expensive.
This package contains algorithm for the numerical estimation of $P_f$.
Here is an example with the standard assumption $\pi \sim \mathcal{N}(0,I_d)$:

```python
import rareeventestimation as ree
from numpy import ndarray, sum, sqrt

# define problem with normal distribution:
# lsf does not need to be vectorized
def my_lsf(x: ndarray) -> ndarray:
    """Affine limit state function
    
    Args:
    x: Numpy array. Last axis is the problems dimension.

    Returns:
    Evaluation of LSF in `x` as a numpy array of shape `x.shape[0:-2]`.
    """
    return -sum(x, axis=-1) / sqrt(x.shape[-1]) + 3.5
prob_dim = 4 # specify or write down the problem dimension
sample_size = 2000 # specify size of initial sample
problem = ree.NormalProblem(my_lsf, prob_dim, sample_size)

# initialize solver with all options, here CBREE with default values:
solver = ree.CBREE()

# estimate failure probability:
solution = solver.solve(problem)
print(solution.prob_fail_hist[-1]) # print estimate of last (=best) iteration
``` 

## Features
-  Implementation of the Consensus-Based Rare Event Estimation
- Interface and implementation for other rare event estimations methods:
  + Sequential Importance Sampling [[1]](#Papaioannou2016),
  + Ensemble Kalman Filter for Rare Event Estimation [[3]](#Wagner2022),
  + Multilevel Sequential² Monte Carlo for Rare Event Estimation  [[2]](#Wagner2020), 
- Numerical studies of all methods 

## About
This package was developed during my master's thesis and the publication of a paper about a new method for rare event estimation (the CBREE method).
Therefore this package contains multiple notebooks which produce the figures used in those two publications.  

If you want to reproduce the figures of the paper **K. Althaus I. Papaioannou, and E. Ullmann, Consensus-based rare event estimation, (2023), https://doi.org/10.48550/arXiv.2304.09077**,
use the notebooks in `docs/figures_paper`.


## Content
Here is an incomplete overview of the folder structure of the package.  
* If you are here to reproduce figures from the literature, the tree will point you to the right directory.
If you run a notebook in one of the `figures_*` folders,
the underlying precomputed data is downloaded from [archive.org](https://archive.org/details/konstantinalthaus-rareeventestimation-data).
The data is encoded as `.json` files and you can have look at those online before loading them.  
Of course you can also compute the data yourself using the
the scripts in `docs/benchmarking-scripts`.
These scripts populate the the empty `docs/data` directory.
Then you can compile the data in `docs/data` with the notebooks (uncomment the respective code block) to produce the datasets that are also available online.

* Furthermore, you see in the tree below several folders with code I  have not written myself.
These folders contain various benchmark methods.
The result of these benchmarks  can be found in the `figures_*` folders and are interpreted in the literature.
```
├── dist
├── docs
│   ├── benchmarking
│   │   ├── benchmarking-scripts # scripts to fill data folder
│   │   └── data # empty
│   ├── figures_paper # make figures from paper
│   └── figures_thesis # make figures from thesis
├── src
│   └── rareeventestimation
│       ├── enkf # enkf solver, cf. credits
│       ├── era # subroutines, cf. credits
│       ├── evaluation # code for figures
│       ├── mls2mc # mls2mc solver, cf. credits
│       ├── problem # code to define problems
│       └── sis # sis solver, cf. credits
└── test 
```



## Credits
This package contains code I have not written myself.  
* __Code from the Engineering Risk Analysis Group, Technical University of Munich__  
All files in `rareeventestimation/src/era` and `rareeventestimation/src/sis` are
written by the [ERA Group](https://www.cee.ed.tum.de/era/era-group/) and
licensed under the MIT license. I have added minor changes.  
* __Code from Dr. Fabian Wagner__  
All files in `rareeventestimation/src/enkf`, `rareeventestimation/src/mls2mc` and `src/rareeventestimation/problem/diffusion.py` are written by [Dr. Fabian Wagner](https://www-m2.ma.tum.de/bin/view/Allgemeines/FabianWagner). I have added minor changes.  
* __Stackoverflow__  
I have used several snippets from the almighty community on [stackoverflow](https://stackoverflow.com).
* `rareeventestimation` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

## License
`rareeventestimation` was created by Konstantin Althaus. It is licensed under the terms of the MIT license.


## References
<a id="Papaioannou2016" href="#features">[1]</a> 
I. Papaioannou, C. Papadimitriou, and D. Straub, Sequential importance sampling for structural reliability analysis, Structural Safety, 62 (2016), pp. 66–75, https://doi.org/10.1016/j.strusafe.2016.06.002.

<a id="Wagner2020" href="#features">[2]</a> 
F. Wagner, J. Latz, I. Papaioannou, and E. Ullmann, Multilevel Sequential Importance Sampling for Rare Event Estimation, SIAM Journal on Scientific Computing, 49 (2020), pp. A2062-A2087, https://doi.org/10.1137/19M1289601

<a id="Wagner2022" href="#features">[3]</a> 
F. Wagner, I. Papaioannou, and E. Ullmann, The ensemble kalman ﬁlter for rare event estimation, SIAM/ASA Journal on Uncertainty Quantiﬁcation, 10 (2022), pp. 317–349, https://doi.org/10.1137/21m1404119.

