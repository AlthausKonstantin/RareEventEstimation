import plotly.io as pio
import plotly.express as px

CMAP = px.colors.qualitative.D3
# Visualization and Output
WIDTH = 700  # latex command \the\textwidth
MY_TEMPLATE = pio.templates["plotly_white"]
MY_TEMPLATE.update({"layout.colorway": CMAP})
MY_LAYOUT = {
    "template": MY_TEMPLATE,
    "font_family": "Serif",
    "xaxis_exponentformat": "power",
    "yaxis_exponentformat": "power",
    "autosize": False,
    "width": WIDTH,
    "height": WIDTH * 2 / 3,
    "margin": dict(l=50, r=50, b=50, t=25, pad=4),
}

INDICATOR_APPROX_LATEX_NAME = {
    "sigmoid": "$I_\\text{{sig}}$",
    "relu": "$I_\\text{{ReLU}}$",
    "algebraic": "$I_\\text{{alg}}$",
    "arctan": "$I_\\text{{arctan}}$",
    "tanh": "$I_\\text{{tanh}}$",
    "erf": "$I_\\text{{erf}}$",
}

STR_BETA_N = "<i>\u03B2<sup>n</i></sup>"
STR_SIGMA_N = "<i>\u03C3<sup>n</i></sup>"
STR_H_N = "<i>h<sup>n</i></sup>"
STR_J_ESS = "<i>J</i><sub>ESS</sub>"

WRITE_SCALE = 7


BM_SOLVER_SCATTER_STYLE = {
    "EnKF (vMFNM)": {
        "line_dash": "dash",
        "marker_symbol": "square",
        "marker_color": CMAP[4],
    },
    "EnKF (GM)": {
        "line_dash": "dash",
        "marker_symbol": "square",
        "marker_color": CMAP[5],
    },
    "SIS (GM)": {
        "line_dash": "dot",
        "marker_symbol": "diamond",
        "marker_color": CMAP[6],
    },
    "SIS (aCS)": {
        "line_dash": "dot",
        "marker_symbol": "diamond",
        "marker_color": CMAP[7],
    },
    "SIS (vMFNM)": {
        "line_dash": "dot",
        "marker_symbol": "diamond",
        "marker_color": CMAP[7],
    },
}


DF_COLUMNS_TO_LATEX = {
    "stepsize_tolerance": "$\\epsilon_{{\\text{{Target}}}}$",
    "cvar_tgt": "$\\Delta_{{\\text{{Target}}}}$",
    "lip_sigma": "Lip$(\\sigma)$",
    "tgt_fun": "Smoothing Function",
    "observation_window": "$N_{{ \\text{{obs}} }}$",
    "callback": "Method",
}
LATEX_TO_HTML = {
    "$\\epsilon_{{\\text{{Target}}}}$": "<i>\u03B5</i><sub>Target</sub>",
    "$\\Delta_{{\\text{{Target}}}}$": "\u0394<sub>Target</sub>",
    "Lip$(\\sigma)$": "Lip(\u03C3)",
    "$N_{{ \\text{{obs}} }}$": "<i>N</i><sub>obs</sub>",
    "IS Density": "<i> \u03BC<sup>N</sup></i>",
}

# Miscellaneous
DOUBLE_PRECISION = 15
