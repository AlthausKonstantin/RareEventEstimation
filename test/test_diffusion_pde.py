# %%
import plotly.graph_objects as go
import rareeventestimation as ree
import numpy as np


# model problem: d/dx(exp(sin(10 x)) d/dx(x^2(x-1)) = 2 e^sin(10 x) (3 x + 5 (-2 + 3 x) x cos(10 x) - 1)
def u_true(x): return x**2*(x-1)
# -1 accounts for sign change in integration by parts
def diffusion_term(x): return -1 * np.exp(np.sin(10*x))


def rhs(x):
    return 2 * np.exp(np.sin(10 * x))*(3 * x + 5 * (-2 + 3 * x) * x * np.cos(10 * x) - 1)


dirichlet_hom = np.array([0, 0])
dirichlet_inhom = np.array([1, 0.5])
n = 100


def test_DiffusionPDE():
    problem = ree.DiffusionPDE(dirichlet_hom, diffusion_term, rhs, n)
    problem.solve()
    assert np.amax(
        np.abs(u_true(np.linspace(0, 1, n)) - problem.solution)) < 1e-3
    return problem


def test_dirichlet_conditions():
    problem = ree.DiffusionPDE(dirichlet_inhom, diffusion_term, lambda x: 0, n)
    problem.solve()
    assert np.amax(
        np.abs(dirichlet_inhom - problem.solution[[0, -1]])) < 1e-10
    return problem
