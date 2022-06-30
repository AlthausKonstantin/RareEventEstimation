import plotly.express as px
cmap = px.colors.qualitative.Safe 
import plotly.io as pio
my_template = pio.templates["plotly_white"]
my_template.update({"layout.colorway": cmap})
my_layout = {
    "template":my_template,
    "font_family":"Serif",
    "autosize":False,
    "width":700,
    "height":700*2/3,
    "margin":dict(
        l=50,
        r=50,
        b=50,
        t=10,
        pad=4
    )}

indicator_approx_latex_names = {
    "sigmoid": f"$I_\\text{{sig}}$",
    "relu": f"$I_\\text{{ReLU}}$",
    "algebraic": f"$I_\\text{{alg}}$",
    "arctan": f"$I_\\text{{arctan}}$",
    "tanh": f"$I_\\text{{tanh}}$",
    "erf": f"$I_\\text{{erf}}$",
}

STR_BETA_N = "<i>\u03B2<sup>n</i></sup>"
STR_SIGMA_N = "<i>\u03C3<sup>n</i></sup>"
STR_H_N =  "<i>h<sup>n</i></sup>"
STR_J_ESS = f"<i>J</i><sub>ESS</sub>"

WRITE_SCALE=7