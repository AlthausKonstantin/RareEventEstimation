#%%
from copy import deepcopy
import rareeventestimation as ree
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from layout import *
import numpy as np
%load_ext autoreload
%autoreload 2
#%%
cvar_tgt = .1
stepsize_tolerance = 4
J = 2500
lip_sigma = 2
n_cut = 30
prob = ree.make_linear_problem(10)
prob.set_sample(J, seed=J)
solver = ree.CBREE(seed=1,
             divergence_check=False,
             cvar_tgt=cvar_tgt,
             num_steps=250,
             save_history=True,
             lip_sigma=2,
             return_caches=True,
             return_other=True)
sol = solver.solve(prob)

# %%
ree.plot_cbree_parameters(sol, prob, plot_time=False)
# %%
fig = make_subplots(rows=3,
                    cols=1,
                    shared_xaxes=True,
                    specs=[[{"secondary_y": False}],
                           [{"secondary_y": True}],
                           [{"secondary_y": False}]])
fig_name = "divergence_check"

# add error 
fig.add_trace(
    go.Scatter(
        y = sol.get_rel_err(prob = prob),
        name="Relative Error"
    ),
    row=1,
    col=1
) 
# add parameters
params ={
    "sigma": STR_SIGMA_N,
    "beta": STR_BETA_N,
    "t_step": STR_H_N
}

for p, p_name in params.items():
    secondary = p != "beta"
    if p=="t_step":
        yy = -np.log(sol.other[p])
    else:
        yy = sol.other[p]
    fig.add_trace(
        go.Scatter(
            y=yy,
            name = p_name
        ),
        row = 2,
        col = 1,
        secondary_y=secondary
    )

# add cvar + goal 
fig.add_trace(
    go.Scatter(
        y = sol.other["cvar_is_weights"],
        name = "\u0394(<i><b>r</b><sup>n</sup></i>)"
    ),
    row = 3,
    col = 1,
)
fig.add_hline(y=cvar_tgt,
              line_dash="dot",
              annotation_text=f"\u0394<sub>Target</sub> = {cvar_tgt}", 
              annotation_position="bottom right",
              annotation_y=cvar_tgt-0.65,
              annotation_bgcolor="white",
              row=3,
              
              col=1)


    
# Style fig
fig.update_yaxes(title_text="Rel. Error", type="log", row=1, col=1)
fig.update_yaxes(title_text="\u0394(<i><b>r</b><sup>n</sup></i>)", row=3, col=1, type="log")
fig.update_yaxes(title_text=f"{STR_SIGMA_N} and {STR_H_N}", title_standoff=10, row=2, col=1, secondary_y=True)
fig.update_yaxes(title_text=STR_BETA_N, row=2, col=1, secondary_y=False)
fig.update_xaxes(title_text="Iteration <i>n<i>", row=3, col=1)
fig.update_layout(**my_layout)
fig2 = deepcopy(fig)
fig.add_vrect(0,n_cut, line_width=0.5)
fig.show()
# %%

# zoom in
fig2.update_xaxes(range=[0,n_cut])
# add stops of divergence check
kk = range(2,12,2)
label_xx = np.zeros(0)
label_yy = np.zeros(0)
label_text = np.zeros(0)
for i,k in enumerate(kk):
    solver.divergence_check = True
    solver.observation_window = k
    sol_ref = solver.solve_from_caches(deepcopy(sol.other["cache_list"]))
    n = sol_ref.num_steps
    fig2.add_vline(
        x = n,
        line_width = 0.5,
        #annotation_text=f"<i>K</i><sub>obs</sub> = {k}",
        #annotation_x=sol_ref.num_steps,
    )
    if n not in label_xx:
        label_xx = np.append(label_xx, n)
        label_yy = np.append(label_yy, 10)
        label_text =  np.append(label_text, f"<i>N</i><sub>obs</sub> = {k}")
    else:
        for (idx, x) in enumerate(label_xx):
            if n==x:
                label_text[idx] = f"{label_text[idx]}, {k}"

for i, txt in enumerate(label_text):
    fig2.add_annotation(
        text=txt,
        x=label_xx[i],
        y=np.log(5),
        bgcolor="rgba(2550,255,255,1)",
        textangle=90,
        xref="x",
        yref="y4",
        showarrow=False,
        align="right"
    )
fig2.update_yaxes(range = [0.05, 1.2*np.amax(sol.other["cvar_is_weights"])], row=3, col=1)
fig2.show()
# %%
# save everything
fig_description = f"Solving the {prob.name} with the CBREE method using  \
$J = {J}$ particles, \
stopping criterion $\\Delta_{{\\text{{Target}}}} = {cvar_tgt}$, \
stepsize tolerance $\\epsilon_{{\\text{{Target}}}} = {solver.stepsize_tolerance}$, \
controlling the increase of $\\sigma$ with $\\text{{Lip}}(\\sigma) = {solver.lip_sigma}$ \
and approximating the indicator function with {indicator_approx_latex_names[solver.tgt_fun]}."
print(fig_description)
fig.write_image(fig_name + ".png", scale =WRITE_SCALE)
fig2.write_image(fig_name + "_zoom.png", scale =WRITE_SCALE)
with open(fig_name + "_desc.tex", "w") as file:
    file.write(fig_description)
# %%
