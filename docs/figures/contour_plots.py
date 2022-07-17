#%% 
import scipy as sp
import rareeventestimation as ree
import numpy as np
import scipy as sp
import sympy as smp
import pandas as pd
import plotly.express as px
from rareeventestimation.evaluation.constants import INDICATOR_APPROX_LATEX_NAME, BM_SOLVER_SCATTER_STYLE, MY_LAYOUT, DF_COLUMNS_TO_LATEX, LATEX_TO_HTML, WRITE_SCALE, CMAP
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rareeventestimation.evaluation.visualization import add_scatter_to_subplots, sr_to_color_dict
import re
%load_ext autoreload
%autoreload 2
# %%
# problem and solver stuff
cvar_tgt=1.5
problem_list = [ree.prob_convex, ree.make_linear_problem(2), ree.make_fujita_rackwitz(2)]
marker_shape_list = ["circle", "square-open", "cross-open"]
#figure stuff
annotation_anchors = [[4,4], [4,4], [-4,-4]]
figs = []
delta = 0.1
x0 = -6
xx = np.arange(x0,-x0, delta)
yy = np.arange(x0,-x0, delta)
col_scale = [[0, "grey"], [1, "white"]]
contour_style = {"start": 0, "end": 0, "size": 0, "showlabels": True}
for i, prob in enumerate(problem_list):
    fig = go.Figure()
    # contour plot
    zz_lsf = np.zeros((len(yy), len(xx)))
    for (xi, x) in enumerate(xx):
        for(yi, y) in enumerate(yy):
            z = np.array([x, y])
            zz_lsf[yi, xi] = prob.lsf(z)

    c_lsf = go.Contour(z=zz_lsf, x=xx, y=yy, colorscale=col_scale,
                        contours=contour_style, line_width=2, showscale=False, showlegend=False)
    fig.add_trace(c_lsf)
    # scatter
    for j, solver in enumerate([ree.CBREE(seed=1, cvar_tgt=cvar_tgt, divergence_check=False), ree.SIS(seed=1, cvar_tgt=cvar_tgt), ree.ENKF(seed=1, cvar_tgt=cvar_tgt)]):
        prob.set_sample(1000, seed=1)
        sol = solver.solve(prob)
        sc = go.Scatter(
            x = sol.ensemble_hist[-1,:,0],
            y = sol.ensemble_hist[-1,:,1],
            name= str(solver),
            mode="markers",
            opacity=1 if j==0 else 0.8,
            marker_symbol = marker_shape_list[j],
        )
        fig.add_trace(sc)
    # style
    fig.update_layout(**MY_LAYOUT)
    fig.update_layout(height=450, width=450)
    fig.add_annotation(x=annotation_anchors[i][0],
                       y=annotation_anchors[i][1],
                       ay=0,
                       ax=0,
                       text="{<i>G</i> \u2264 0}")
    fig.add_annotation(x=-annotation_anchors[i][0],
                       y=-annotation_anchors[i][1],
                       ay=0,
                       ax=0,
                       text="{<i>G</i> > 0}")
    figs.append(fig)
    fig.show()
    fig.write_image(f"{prob.name} scatter plot.png".replace(" ", "_").lower(), scale=WRITE_SCALE)
    fig_description = f"Failure domain of the {prob.name}. \
Also the final ensembles of the CBREE, SiS and EnKF methods respectively are plotted. \
Each method used $J=1000$ samples and the stopping criterion $\\Delta_{{\\text{{Target}}}}$ = {cvar_tgt}. \
The CBREE method performed no divergence check, used the approximation $I_\\text{{alg}}$, the stepsize control $\\epsilon_{{\\text{{Target}}}}=0.5$ and controled the increase of $\\sigma$ with $\\text{{Lip}}(\\sigma) = 1$."
    with open(f"{prob.name} scatter plot desc.tex".replace(" ", "_").lower(), "w") as file:
        file.write(fig_description)
    print(fig_description)
# %%

# %%
