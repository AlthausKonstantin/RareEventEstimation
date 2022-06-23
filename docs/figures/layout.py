import plotly.express as px
cmap = px.colors.qualitative.Safe
my_layout = {
    "template":"plotly_white",
    "font_family":"Serif"}

indicator_approx_latex_names = {
    "sigmoid": f"$I_\\text{{sig}}$",
    "relu": f"$I_\\text{{ReLU}}$",
    "algebraic": f"$I_\\text{{alg}}$",
    "arctan": f"$I_\\text{{arctan}}$",
    "tanh": f"$I_\\text{{tanh}}$",
    "erf": f"$I_\\text{{erf}}$",
}

STR_BETA_N = "\u03B2<sup><i>n</i></sup>"
STR_SIGMA_N = "\u03C3<sup><i>n</i></sup>"
STR_J_ESS = f"<i>J</i><sub>ESS</sub>"