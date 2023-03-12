# read version from installed package
from importlib.metadata import version

# The following tells flake9 to ignore the imported but not used error
# flake8: noqa: F401
__version__ = version("rareeventestimation")
from rareeventestimation.solver import CBREE, ENKF, SIS, CBREECache, CMC
from rareeventestimation.problem.problem import Problem, NormalProblem
from rareeventestimation.problem.toy_problems import (
    make_fujita_rackwitz,
    make_linear_problem,
    prob_convex,
    problems_highdim,
    problems_lowdim,
    prob_nonlin_osc,
)
from rareeventestimation.problem.diffusion import diffusion_problem
from rareeventestimation.solution import Solution
from rareeventestimation.mixturemodel import GaussianMixture, VMFNMixture
from rareeventestimation.sis.SIS_GM import SIS_GM
from rareeventestimation.sis.SIS_aCS import SIS_aCS
from rareeventestimation.evaluation.convergence_analysis import (
    do_multiple_solves,
    study_cbree_observation_window,
    load_data,
    add_evaluations,
    aggregate_df,
    get_benchmark_df,
    force_bm_names,
    expand_cbree_name,
    my_number_formatter,
    vec_to_latex_set,
    list_to_latex,
    squeeze_problem_names,
)
from rareeventestimation.evaluation.visualization import (
    make_mse_plots,
    make_rel_error_plot,
    make_accuracy_plots,
    make_efficiency_plot,
    update_axes_format,
    plot_iteration,
    add_scatter_to_subplots,
    plot_cbree_parameters,
    sr_to_color_dict,
    get_color,
    get_continuous_color,
)
from rareeventestimation.utilities import (
    importance_sampling,
    radial_gaussian_logpdf,
    gaussian_logpdf,
    my_log_cvar,
    my_softmax,
    get_slope,
)
from rareeventestimation.problem.flowrate_problem import (
    make_flowrate_problem,
    DiffusionPDE,
    LogNormalField,
)
