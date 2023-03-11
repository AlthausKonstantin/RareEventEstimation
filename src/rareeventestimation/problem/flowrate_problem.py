from numpy.random import default_rng, Generator
from numpy import ndarray, sum, sqrt, zeros, exp, linspace, ones, array
from rareeventestimation.problem.diffusion import EigFcnKL
import skfem as fem
from typing import Callable
from skfem.helpers import dot, grad
from rareeventestimation.problem.problem import Vectorizer, NormalProblem
from numbers import Real


def make_flowrate_problem(
    d: int,
    num_gridpoints: int,
    mean_gaussian=0.1,
    variance_gaussian=0.04,
    correlation_length=0.3,
    dirichlet_conditions=[1, 0],
) -> NormalProblem:
    """Implement Example 5.3 in
    Wagner, Fabian / Latz, Jonas / Papaioannou, Iason / Ullmann, Elisabeth:
    Error analysis for probabilities of rare events with approximate models 2020
    """
    random_field = LogNormalField(
        mean_gaussian, variance_gaussian, correlation_length, d
    )

    def lsf(sample: ndarray) -> Real:
        diffuison_pde = DiffusionPDE(
            array(dirichlet_conditions),
            lambda x: random_field(x, noise=sample),
            lambda x: 0.0,
            num_gridpoints,
        )
        diffuison_pde.solve()
        # compute flow-rate in x=1
        flow = -random_field(ones(1), noise=sample) * diffuison_pde.solution_grad[-1]
        return 1.7 - flow.item()

    return NormalProblem(
        Vectorizer(lsf, d),
        d,
        1,
        prob_fail_true=3.026 * 1e-4,
        name=f"Flow-rate Problem (d={d})",
    )


class DiffusionPDE:
    """Define and numerically solve a  poisson problem on [0,1] with space dependent diffusion coefficient."""

    dirichlet_conditons = None
    diffusion_term = None
    rhs = None
    num_gridpoints = None
    solution = None
    solution_grad = None

    def __init__(
        self,
        dirichlet_conditons: ndarray,
        diffusion_term: Callable,
        rhs: Callable,
        num_gridpoints: int,
    ) -> None:
        """Set up diffusion problem.

        d/dx (diffusion_term(x)*d/dx u(x)) = rhs(x) on (0,1) and
        u(0) = dirichlet_conditions[0],
        u(1) = dirichlet_conditions[1]


        Args:
            dirichlet_conditons (ndarray): see above.
            diffusion_term (Callable): Scalar to scalar function that can handle vector valued input.
            rhs (Callable):  Scalar to scalar function that can handle vector valued input.
            num_gridpoints (int): Approximate u on linspace(0,1,num_gridpoints).

        """

        self.dirichlet_conditions = dirichlet_conditons
        self.diffusion_term = diffusion_term
        self.rhs = rhs
        self.num_gridpoints = num_gridpoints

    def solve(self) -> None:
        """Solve diffusion problem.

        Write solution and gradient to `self.solution` and `self.solution_grad`.
        """
        m = fem.MeshLine(linspace(0, 1, self.num_gridpoints))
        e = fem.ElementLineP1()
        basis = fem.Basis(m, e)
        # bilinear form

        @fem.BilinearForm
        def a(u, v, param):
            return self.diffusion_term(param.x[0]) * dot(grad(u), grad(v))

        # linear form

        @fem.LinearForm()
        def rhs(v, param):
            return self.rhs(param.x[0]) * v

        # boundary conditions

        def dirichlet(x):
            return (1 - x[0]) * self.dirichlet_conditions[0] + x[
                0
            ] * self.dirichlet_conditions[1]

        # assemble
        A = a.assemble(basis)
        b = rhs.assemble(basis)
        u = basis.project(dirichlet)
        u = fem.solve(*fem.condense(A, b, x=u, D=basis.get_dofs()))
        # project onto linear fems to get`num_gridpoints` point evaluations of `u`
        self.solution = basis.with_element(fem.ElementLineP1()).project(
            basis.interpolate(u)
        )
        self.solution_grad = basis.with_element(fem.ElementLineP1()).project(
            basis.interpolate(u).grad[0]
        )


class LogNormalField:
    """Implementation of a log-normal-process on [0,1] with exponential covariance.
    Can approximately evaluated by its Karhunen-Loéve Expansion.
    """

    mean_gaussian = 0.0
    variance_gaussian = 1.0
    correlation_length = 0.1
    kle_terms = 10

    _rng = None
    _eigenvalues = None
    _eigenfunctions = None

    def __init__(
        self,
        mean_gaussian: Real,
        variance_gaussian: Real,
        correlation_length: Real,
        kle_terms: int,
        rng: Generator = None,
    ) -> None:
        """Initialize KLE parameters and compute KLE expansion.

        Args:
            mean_gaussian (Real): Mean of the underlying Gaussian process.
            variance_gaussian (Real): Variance of the underlying Gaussian process.
            correlation_length (Real): Correlation length of exponential covariance.
            kle_terms (int): Number of terms in the KL expansion.
        """
        self.mean_gaussian = mean_gaussian
        self.variance_gaussian = variance_gaussian
        self.correlation_length = correlation_length
        self.kle_terms = kle_terms
        self._eigenvalues, self._eigenfunctions = EigFcnKL(
            correlation_length, kle_terms
        )
        if rng is None:
            self._rng = default_rng()
        else:
            self._rng = rng

    def __call__(self, x: ndarray, noise: ndarray = None) -> ndarray:
        """Sample the lognormal field in `x`.

        Args:
            x (ndarray): Array of shape (N,) of points to evaluate random field.
            noise (ndarray): Array of shape (kle_termns,) with a Gaussian sample.

        Returns:
            ndarray: Evaluation of field, has shape (N,)
        """
        # evaluate kle
        if noise is None:
            noise = self._rng.standard_normal(self.kle_terms)
        eigenfunction_eval = zeros(x.shape + (self.kle_terms,))
        for i in range(self.kle_terms):
            eigenfunction_eval[..., i] = self._eigenfunctions(x, i + 1)
        kle_coeff = sqrt(self._eigenvalues) * noise
        kle = sum(eigenfunction_eval * kle_coeff, axis=-1)
        return exp(self.mean_gaussian + sqrt(self.variance_gaussian) * kle)
