# %%
import plotly.graph_objects as go
import rareeventestimation as ree
import numpy as np
rng = np.random.default_rng(123)
random_field = ree.LogNormalField(mean_gaussian=0.0, variance_gaussian=np.log(
    1.01), correlation_length=0.01, kle_terms=150, rng=rng)
dirichlet_conditions = np.zeros(2)
def rhs(x): return 1.0
def diffusion_coefficient(x): return random_field(x)


pde_problem = ree.DiffusionPDE(dirichlet_conditions,
                               diffusion_coefficient,
                               rhs,
                               2**8)

p = ree.diffusion_problem
p.set_sample(10)
solver = ree.CBREE()
solution = solver.solve(p)
f = go.Figure()
f.add_trace(go.Scatter(
    y=np.load("u_h.npy")
))
# %%
