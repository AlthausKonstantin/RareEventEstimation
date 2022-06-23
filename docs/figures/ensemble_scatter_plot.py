#%%
import rareeventestimation as ree
import numpy as np
import plotly.graph_objects as go
%load_ext autoreload
%autoreload 2
cbree = ree.CBREE(seed=1, save_history=True, cvar_tgt=1, divergence_check=False)
p = ree.prob_convex.set_sample(1000, seed=1000)
sol = cbree.solve(p)
# %%
# # Make evaluations for contour plot
delta=.1
x_limits = [np.min(sol.ensemble_hist[:, :, 0])-3,
            np.max(sol.ensemble_hist[:, :, 0])+3]
y_limits = [np.min(sol.ensemble_hist[:, :, 1])-3,
            np.max(sol.ensemble_hist[:, :, 1])+3]
xx = np.arange(*x_limits, step=delta)
yy = np.arange(*y_limits, step=delta)
# y is first as rows are stacked vertically
zz_lsf = np.zeros((len(yy), len(xx)))
for (xi, x) in enumerate(xx):
    for(yi, y) in enumerate(yy):
        z = np.array([x, y])
        zz_lsf[yi, xi] = p.lsf(z)



#  Make contour plot of limit state function
col_scale = [[0, "salmon"], [1, "white"]]
contour_style = {"start": 0, "end": 0, "size": 0, "showlabels": True}
c_lsf = go.Contour(z=zz_lsf, x=xx, y=yy, colorscale=col_scale,
                    contours=contour_style, line_width=2, showscale=False)

fig = go.Figure(data=c_lsf)



# Make frames for animation
symbol_dict = {0:4, sol.ensemble_hist.shape[0]-1:0}
for i in [0, sol.ensemble_hist.shape[0]-1]:
    s = go.Scatter(x=sol.ensemble_hist[i, :, 0], y=sol.ensemble_hist[i, :, 1],
                    mode="markers",
                    marker=dict(symbol=symbol_dict[i]),
                    name = f"Iteration {i}")
    fig.add_trace(s)
fig.update_layout(xaxis_title="x", yaxis_title="y", title="CBREE Moves 1000 Samples into Failure Region")
fig.write_image("ensemble_scatter_plot.png")
 # %%
