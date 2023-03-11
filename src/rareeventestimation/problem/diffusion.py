import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy.matlib
from rareeventestimation.problem.problem import NormalProblem, Vectorizer

# utilities


def bisectionmethod(f, a, b, tol, nmax):
    k = 0
    # this modification is necessary to get the first root right
    if a == 0:
        a = 10 ** (-16)

    # fprintf('current interval: [%d,%d]\n', a, b)
    if f(a) * f(b) > 0:
        pole = (b + a) / 2

        if f(a) * f(pole - 10 ** (-14)) < 0:
            b = pole - 10 ** (-14)
        else:
            a = pole + 10 ** (-14)

        # fprintf('current interval: [%d,%d]\n', a, b)

    while (b - a >= tol) and (k <= nmax):
        mid = (b + a) / 2

        if f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid

        # fprintf('current interval: [%d,%d]\n', a, b)

        k = k + 1

    return a, b


def newtonmethod(f, df, x0, tol, nmax):
    x = x0 - (f(x0) / df(x0))
    ex = [abs(x - x0)]

    k = 1
    while (ex[k - 1] >= tol) and (k <= nmax):
        xold = x
        x = xold - (f(xold) / df(xold))
        ex.append(abs(x - xold))
        k = k + 1

    return x


def eval_quad(nele, L, xGL):
    ind = np.linspace(1, nele, nele)
    XQ = L / nele / 2 * np.matlib.repmat(
        xGL.T, 1, len(ind)
    ) + L / nele / 2 * np.matlib.repmat((2 * ind - 1), len(xGL), 1)
    XQ = XQ.T
    xq = XQ.reshape(len(ind) * np.size(xGL), 1)

    return xq


def EigFcnKL(correlationlength, numberroots):
    """
    first numberroots roots of tan(w) + 2 * correlationlength * w - correlationlength ^ 2 * w ^ 2 * tan(w)
    !!!!! note: this function may fail in some special situations(occurs rarely,
    e.g.correlationlength = 0.01 root  # 31)

    """

    def g_odd(y):
        return 1 / correlationlength - y * np.tan(y / 2)

    def dg_odd(y):
        return -np.tan(y / 2) - y * 1 / 2 * (1 + np.tan(y / 2) ** 2)

    def g_even(y):
        return 1 / correlationlength * np.tan(y / 2) + y

    def dg_even(y):
        return 1 / correlationlength * 1 / 2 * (1 + np.tan(y / 2) ** 2) + 1

    roots = np.zeros(numberroots)
    for i in range(1, numberroots + 1):
        if i % 2 == 1:
            # observation, that each zero lies between two consecutive zeros of the sin
            a = (i - 0.99999) * np.pi
            b = i * np.pi

            # Prelocation of the zero by application of the bisection method
            a = bisectionmethod(g_odd, a, b, 10 ** (-15), 10000)[0]

            # using left value of the determined interval as starting point for a
            # application of the Newton method
            x = a
            # Newton method
            x = newtonmethod(g_odd, dg_odd, x, 10 ** (-15), 10000)
            roots[i - 1] = x
        else:
            # observation, that each zero lies between two consecutive zeros of the sin
            a = (i - 0.99999) * np.pi
            b = i * np.pi

            # Prelocation of the zero by application of the bisection method
            a = bisectionmethod(g_even, a, b, 10 ** (-15), 10000)[0]

            # using left value of the determined interval as starting point for a
            # application of the Newton method
            x = a
            # Newton method
            x = newtonmethod(g_even, dg_even, x, 10 ** (-15), 10000)
            roots[i - 1] = x

    # Calculate the eigenvectors
    eigenvalues = 2 * correlationlength / (1 + roots**2 * correlationlength**2)
    alpha_eig = 1 / (np.sqrt(1 / 2 + np.sin(roots) / (2 * roots)))

    def eigenfunction(z, k):
        if k % 2 == 1:
            return alpha_eig[k - 1] * np.cos(roots[k - 1] * (z - 1 / 2))
        else:
            return alpha_eig[k - 1] * np.sin(roots[k - 1] * (z - 1 / 2))

    return eigenvalues, eigenfunction


def diffusion_equation_solve(h, theta, mu, sigma, function_values):
    # mesh width and number dof( = number of edges resp.of faces)
    n_vertices = np.int(h**-1 + 1)
    n_ele = n_vertices - 1

    # Quadrature weights
    wGL = np.array([5 / 9, 8 / 9, 5 / 9])

    # Calculate truncated KL
    kl = np.sum(function_values * theta, 1)
    kl = np.exp(mu + sigma * kl)

    # Calculate A on the grid points
    a_grid = kl

    #  Assembling of mass matrix A
    A = sparse.lil_matrix((n_vertices, n_vertices))
    Kei = 1 / h / 2 * a_grid
    Ke = np.sum(
        np.reshape((Kei * np.matlib.repmat(wGL.T, 1, n_ele)).T, (n_ele, 3)).T, 0
    )
    Ke0 = np.hstack((Ke, [0]))
    Ke00 = np.hstack(([0], Ke))
    diagonals = (Ke0, -Ke, -Ke)

    A = A + sparse.diags(diagonals, [0, -1, 1])
    A = A + sparse.diags(Ke00, 0)

    # A[n_vertices-1, n_vertices-2] = A[n_vertices-1, n_vertices-2] - 1
    # A[n_vertices-1, n_vertices-1] = A[n_vertices-1, n_vertices-1] + 1

    A = A[1:n_vertices, 1:n_vertices]

    b = h * np.ones(n_vertices - 1)
    b[n_vertices - 2] = h * 1 / 2
    u = spsolve(A, b)

    u_h = np.zeros(n_vertices)
    u_h[1:n_vertices] = u

    return u_h


def truncated_kl_ofa(theta, mu, sigma, functionvalues):
    kl = np.sum(functionvalues * theta, 1)

    kl = np.exp(mu + sigma * kl)

    return kl


# define problem
parameter_dim = 150  # number of dimensions
d_seq = [parameter_dim]

# definition of the sequence of admissible step sizes
first_step_size = 1 / 4
number_of_step_sizes = 8
exponents = np.linspace(0, number_of_step_sizes - 1, number_of_step_sizes)
h_seq = first_step_size * 1 / 2**exponents

h_seq = [h_seq[number_of_step_sizes - 1]]

correlation_length = 0.01
number_roots = parameter_dim
[eigenvalues, eigenfunction] = EigFcnKL(correlation_length, parameter_dim)

mu_a = 1
sigma_a = 0.1
u_max = 0.535

sigma = np.sqrt(np.log((sigma_a**2 / mu_a**2) + 1))
mu = np.log(mu_a) - sigma**2 / 2

# Calculate the values of the eigenfunction at the nodes for the sequence of mesh sizes
function_values_list = list()
for i in range(len(h_seq)):
    h = h_seq[i]
    dim = d_seq[i]
    n_vertices = int(h**-1 + 1)
    n_ele = n_vertices - 1
    nodes = np.linspace(0, 1, n_vertices)
    L = 1

    # Quadrature nodes
    xGL = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)]).reshape((1, 3))
    wGL = np.array([5 / 9, 8 / 9, 5 / 9])

    xq = eval_quad(n_ele, L, xGL)

    # KL-series
    function_value = np.zeros((len(xq), dim))
    for j in range(dim):
        function_value[:, j] = eigenfunction(xq[:, 0], j + 1)

    function_values_list.append(np.sqrt(eigenvalues[0:dim]) * function_value)

step_size = h_seq[0]
function_values = function_values_list[0]


def u(theta):
    u_h = diffusion_equation_solve(step_size, theta, mu, sigma, function_values)

    return u_h[len(u_h) - 1]


# definition of the limit state function 'g'; failure occurs if g(theta, h)<=0
def g(theta):
    if len(theta) != np.size(function_values, 1):
        print("Error: Dimensions misfit")
    return u_max - u(theta)


lsf = Vectorizer(g, parameter_dim)
diffusion_problem = NormalProblem(
    lsf,
    parameter_dim,
    1,
    name=f"Diffusion Problem (d={parameter_dim})",
    prob_fail_true=1.682 * 1e-4,
)
