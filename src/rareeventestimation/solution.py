
from typing import Callable
from numpy import arange, array, ndarray, zeros, nan, zeros_like,maximum, minimum
from rareeventestimation.problem import Problem


class Solution:
    """Organize the history of the method `Solver.solve()`."""

    def __init__(self, ensemble_hist: ndarray, temp_hist: ndarray, lsf_eval_hist: ndarray, prob_fail_hist: ndarray, tgt_fun: Callable, costs: int, msg: str, num_steps=None, other=None):
        """Construct instance of Solution.

        Args:
            ensemble_hist (ndarray): 3-D array of shape (num_steps, N, D).
            temp_hist (ndarray): 1-D array of shape (num_steps)
            lsf_eval_hist (ndarray): 2-D array of shape (num_steps, N)
            prob_fail_hist (ndarray): 1-D array of shape (num_steps)
            tgt_fun (Callable): Function used for reweighting samples.
            costs (int): Measure of computational costs for obtaining solution.
            msg (str): Exit message. Either 'Success' or the measssage of a caught exception
        """
        self.ensemble_hist = ensemble_hist
        self.temp_hist = temp_hist
        self.lsf_eval_hist = lsf_eval_hist
        self.prob_fail_hist = prob_fail_hist
        self.tgt_fun = tgt_fun
        self.costs = costs
        self.msg = msg
        (num_steps_shape, N, d) = ensemble_hist.shape
        if num_steps is None:
            self.num_steps = num_steps_shape
        else:
            # if only last iteration is given, num_steps can be provided explicitly
            self.num_steps = num_steps
        self.N = N
        self.d = d
        self.other = other



    def __compute_rel_error(self, prob: Problem):
        """
        Compute and save relative error of estimated probability of failure.

        Args:
            prob (Problem): Instance of Problem class.

        Notes:
            Prints warning if `prob.prob_fail_true` is `None`.
        """
        assert prob.prob_fail_true is not None, "Cannot not compute relative error as `Problem.prob_fail_true` is not set."
        self.prob_fail_r_err = abs(
            self.prob_fail_hist - prob.prob_fail_true) / prob.prob_fail_true
    

    def get_rel_err(self, prob=None):
        assert prob is not None, "Need problem to compute rel. error. Please provide the problem!"
        self.__compute_rel_error(prob)
        return self.prob_fail_r_err

