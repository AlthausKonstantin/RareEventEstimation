#%%
import rareeventestimation as ree
import numpy as np
import plotly.graph_objects as go
import scipy as sp
%load_ext autoreload
%autoreload 2
x_range = np.arange(-5, 5, 0.01)

cbree = ree.CBREE()


fun_dict = {
    "Sigmoid": "sigmoid",
    "Algebraic": "algebraic",
    "Tanh": "tanh",
    "Arctan": "arctan",
    "erf": "erf",
    "relu": "ReLU"
}
s1 =1
s2 =5
sigmas= {"dot":1,
         "dash":1000}
# Plot the smoothed indicator functions
fig = go.Figure()
for line_style, s in sigmas.items():
    for label, method in fun_dict.items():
        fig.add_trace(
            go.Scatter(
                x = x_range,
                y = np.exp(cbree._CBREE__log_tgt_fun(x_range, 0, s,method=method)),
                name = label + f" (\u03C3 = {s})",
                line = dict(dash=line_style)
            )
        )
fig.update_layout(xaxis_title = "x",
                  yaxis_title = "I(x,\u03C3)",
                  title = f"Approximations of the indicator function for \u03C3 = {s1} and {s2}")
fig.write_image("indicator_function_approximations.png")
fig
# %%
# Plot a section of the target density
N = 100000
d= 1
fig2 = go.Figure()
for line_style, s in sigmas.items():
    for label, method in fun_dict.items():
        linear_problem = ree.make_linear_problem(d)
        def lsf(x): return linear_problem.lsf(x)
        def target_density(x): return np.exp(cbree._CBREE__log_tgt_fun(lsf(x), 0.5*np.sum(x**2, axis=1), s,method=method))
        sample = np.random.multivariate_normal(np.zeros(d), np.eye(d), N)
        z = np.average(
            target_density(sample) \
                /sp.stats.multivariate_normal.pdf(sample,
                                                  mean = np.zeros(d),
                                                  cov = np.ones(d)))
        xx = np.ones(d)[None,...] * x_range[..., None]
        fig2.add_trace(
            go.Scatter(
                x = x_range,
                y = target_density(xx) /z,
                name = label + f" (\u03C3 = {s})",
                line = dict(dash=line_style)
            )
        )
        
# Also add the optimal density
def opt_density(x):
    return sp.stats.multivariate_normal.pdf(x, mean = np.zeros(d), cov = np.ones(d)) * (linear_problem.lsf(x)<=0) / linear_problem.prob_fail_true
fig2.add_trace(
    go.Scatter(
        x = x_range,
        y = opt_density(xx),
        name= "Optinmal Density"
    )
)
fig2.update_layout(xaxis_title = "x",
                  yaxis_title = "I(x,\u03C3)\u03C0(x)",
                  title = f"Approximations of the optimal density for \u03C3 = {s1} and {s2}")
fig2.write_image("optimal_densiy_approximations.png")
fig2
# %%

