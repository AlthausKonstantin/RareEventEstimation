from numpy import (
    arange,
    cumsum,
    log,
    sqrt,
    zeros,
    array,
    minimum,
    maximum,
    sum,
    amin,
    amax,
    inf,
    log10,
)
import pandas as pd
from rareeventestimation.evaluation.convergence_analysis import aggregate_df
from rareeventestimation.evaluation.constants import CMAP, MY_LAYOUT
import re
from plotly.graph_objects import Figure, Scatter, Bar, Contour, Layout, Box
from os import path
from rareeventestimation.problem.problem import Problem
from rareeventestimation.solution import Solution
from plotly.subplots import make_subplots
from os.path import commonpath
import plotly.colors
from PIL import ImageColor


def make_accuracy_plots(
    df: pd.DataFrame,
    save_to_resp_path=True,
    plot_all_seeds=False,
    one_plot=False,
    MSE=True,
    CMAP=CMAP,
    layout={},
) -> list:
    """Plot rel. root MSE of estimates vs mean of costs.

    Args:
        df (pd.DataFrame): Dataframe, assumed to come from  add_evaluations and aggregate_df.
        save_to_resp_path (bool, optional): Save plots to resp. path specified in column "Path". Defaults to True.

    Returns:
        list: List with Figures
    """
    out = []
    df = df.set_index(["Problem", "Solver", "Sample Size"])

    # Set up dicts with (solver,color) and (solver,line-style) entries
    solver_colors = {
        s: CMAP[i % len(CMAP)]
        for (i, s) in enumerate(df.index.get_level_values(1).unique())
    }
    if one_plot:
        solver_colors = {
            s: CMAP[i % len(CMAP)]
            for (i, s) in enumerate(df.index.droplevel(2).unique())
        }
    solver_dashes = dict.fromkeys(solver_colors.keys())
    for solver in solver_dashes.keys():
        if re.search("CBREE", str(solver)):
            solver_dashes[solver] = "solid"
        if re.search("EnKF", str(solver)):
            solver_dashes[solver] = "dash"
        if re.search("SiS", str(solver)):
            solver_dashes[solver] = "dot"
    # Make a plot for each problem
    if one_plot:
        one_fig = Figure()
    problems_in_df = df.index.get_level_values(0).unique()
    for problem in problems_in_df:
        fig = Figure()
        applied_solvers = df.loc[problem, :].index.get_level_values(0).unique()
        # Add traces for each solver
        for solver in applied_solvers:
            xvals = (
                df.loc[(problem, solver), "Relative Root MSE"].values
                if MSE
                else df.loc[(problem, solver), ".50 Relative Error"]
            )
            yvals = (
                df.loc[(problem, solver), "Cost Mean"].values
                if MSE
                else df.loc[(problem, solver), ".50 Cost"].values
            )
            rel_root_mse_sc = Scatter(
                y=yvals,
                x=xvals,
                name=solver,
                legendgroup=0,
                legendgrouptitle_text="Method",
                mode="lines + markers",
                line={
                    "color": solver_colors.get(solver),
                    "dash": solver_dashes.get(solver),
                },
                error_x={
                    "array": df.loc[(problem, solver), ".75 Relative Error"].values
                    - xvals,
                    "arrayminus": xvals
                    - df.loc[(problem, solver), ".25 Relative Error"].values,
                    "type": "data",
                    "symmetric": False,
                    "thickness": 0.5,
                },
                error_y={
                    "array": df.loc[(problem, solver), ".75 Cost"].values - yvals,
                    "arrayminus": yvals - df.loc[(problem, solver), ".25 Cost"].values,
                    "type": "data",
                    # "symmetric":True,
                    "thickness": 0.5,
                },
            )
            # rel_err_median = Scatter(
            #     y = df.loc[(problem, solver),".50 Cost"].values,
            #     x = df.loc[(problem, solver),".50 Relative Error"].values,
            #     mode="markers",
            #     marker={"color": solver_colors.get(solver), "symbol":"circle-x"},
            #     showlegend=False
            # )
            if one_plot:
                rel_root_mse_sc.name = solver + " " + problem
                one_fig.add_trace(rel_root_mse_sc)
            if plot_all_seeds:
                # Add scatter for each indivual estimate
                groups = ["Solver", "Problem", "Solver Seed", "Sample Size"]
                df_err_and_cost = (
                    df.groupby(groups)
                    .mean()
                    .loc[(solver, problem), ("Relative Error", "Cost")]
                )
                for s in df_err_and_cost.index.get_level_values(0).unique():
                    sample_sc = Scatter(
                        y=df_err_and_cost.loc[s, "Cost"].values,
                        x=df_err_and_cost.loc[s, "Relative Error"].values,
                        mode="lines + markers",
                        line={
                            "color": solver_colors[solver],
                            "dash": solver_dashes[solver],
                        },
                        opacity=0.2,
                        hovertext=f"Seed {s}",
                        hoverinfo="text",
                        showlegend=False,
                    )
                    fig.add_trace(sample_sc)
            fig.add_trace(rel_root_mse_sc)
            # fig.add_trace(rel_err_median)
            if one_plot:
                one_fig.update_xaxes(
                    {
                        "title": "Relative Root MSE",
                        "type": "log",
                        "showexponent": "all",
                        "exponentformat": "e",
                    }
                )
                one_fig.update_yaxes(
                    {
                        "title": "Cost",
                        "type": "log",
                        "showexponent": "all",
                        "exponentformat": "e",
                    }
                )
                one_fig.update_layout(title="Cost-Error Plot")
            fig.update_xaxes(
                {
                    "title": "Relative Root MSE" if MSE else "Median of rel Error",
                    "type": "log",
                    "showexponent": "all",
                    "exponentformat": "e",
                }
            )
            fig.update_yaxes(
                {
                    "title": "Cost",
                    "type": "log",
                    "showexponent": "all",
                    "exponentformat": "e",
                }
            )
            fig.update_layout(title="Cost-Error Plot for " + problem)
            fig.update_layout(**layout)
        out.append(fig)
    if one_plot:
        return one_fig
    return out


def make_mse_plots(df: pd.DataFrame, save_to_resp_path=True) -> dict:
    df_agg = aggregate_df(df)
    problem_solver_keys = df_agg.index.droplevel(2).unique()
    out = dict.fromkeys(problem_solver_keys)

    for k in problem_solver_keys:
        fig = Figure()
        problem, solver = k
        sc_bias = Scatter(
            y=df_agg.loc[k, "Cost Mean"].values,
            x=df_agg.loc[k, "Estimate Bias"].values ** 2,
            name="Bias Squared",
            mode="lines + markers",
            line={"color": CMAP[0]},
            stackgroup="one",
            orientation="h",
        )
        sc_var = Scatter(
            y=df_agg.loc[k, "Cost Mean"].values,
            x=df_agg.loc[k, "Estimate Variance"].values,
            name="Variance",
            mode="lines + markers",
            line={"color": CMAP[1]},
            stackgroup="one",
            orientation="h",
        )
        fig.add_trace(sc_bias)
        fig.add_trace(sc_var)
        fig.update_xaxes(
            {
                "title": "MSE",
                "type": "log",
                "showexponent": "all",
                "exponentformat": "e",
            }
        )
        fig.update_yaxes(
            {
                "title": "Cost",
                "type": "log",
                "showexponent": "all",
                "exponentformat": "e",
            }
        )
        fig.update_layout(title=f"MSE of {solver} applied to {problem}")
        out[k] = fig

    if save_to_resp_path:
        for k, fig in out.items():
            p = commonpath(df_agg.loc[k, "Path"].values.tolist())
            fig.write_image(path.join(p, fig.layout.title["text"] + ".pdf"))

    return out


def get_color(colorscale_name, loc):
    from _plotly_utils.basevalidators import ColorscaleValidator

    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    # kudos: https://stackoverflow.com/a/67912302
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale_name)

    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str

    kudos: https://stackoverflow.com/a/67912302
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    def hex_to_rgb(c):
        return "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )


def sr_to_color_dict(sr: pd.Series, color_scale_name="Viridis") -> dict:
    keys = sr.unique()
    arr = keys - amin(keys)
    arr = arr / amax(arr)
    vals = get_color(color_scale_name, arr)
    return dict(zip([str(k) for k in keys], vals))


def plot_cbree_parameters(sol: Solution, p2, plot_time=False):
    f = make_subplots(rows=3, shared_xaxes=True)
    if plot_time:
        xx = cumsum(-log(sol.other["t_step"]))
    else:
        xx = arange(len(sol.other["t_step"]))
    # Plot error and error estimate
    rel_err = sol.get_rel_err(p2)
    rel_err_est = sol.other["cvar_is_weights"] / sqrt(p2.sample.shape[0])
    f.add_trace(Scatter(x=xx, y=rel_err, name="Rel. Root Err."), row=1, col=1)
    f.add_trace(
        Scatter(x=xx, y=rel_err_est, name="Rel. Root Err. Estimate"), row=1, col=1
    )
    f.update_yaxes(type="log", row=1, col=1)

    # Plot parameters
    f.add_trace(Scatter(x=xx, y=sol.other["sigma"], name="Sigma"), row=2, col=1)
    f.add_trace(Scatter(x=xx, y=sol.other["beta"], name="Beta"), row=2, col=1)
    f.add_trace(
        Scatter(x=xx, y=-log(sol.other["t_step"]), name="Stepsize"), row=2, col=1
    )
    f.add_trace(
        Scatter(x=xx, y=sol.other["ess"] / p2.sample.shape[0], name="rel.ESS"),
        row=2,
        col=1,
    )
    # Plot monitored quants

    f.add_trace(
        Scatter(
            x=xx, y=sum(sol.lsf_eval_hist <= 0, axis=1) / p2.sample.shape[0], name="SFP"
        ),
        row=3,
        col=1,
    )
    f.add_trace(
        Scatter(
            x=xx,
            y=sol.other["sfp_slope"],
            name="SFP Slope",
            line_color=CMAP[(len(f.data) - 1) % len(CMAP)],
            line_dash="dash",
        ),
        row=3,
        col=1,
    )
    f.add_trace(
        Scatter(
            x=xx,
            y=sol.other["slope_cvar"],
            name="CVAR Slope",
            line_color=CMAP[1],
            line_dash="dash",
        ),
        row=3,
        col=1,
    )
    f.update_layout(hovermode="x unified")
    return f


def add_scatter_to_subplots(fig, rows, cols, **scatter_kwargs):
    if "showlegend" in scatter_kwargs:
        first = scatter_kwargs["showlegend"]
        del scatter_kwargs["showlegend"]
    else:
        first = False
    row_iter = range(rows) if isinstance(rows, int) else rows
    col_iter = range(cols) if isinstance(cols, int) else cols
    for i in row_iter:
        for j in col_iter:
            fig.append_trace(
                Scatter(**scatter_kwargs, showlegend=first), row=i + 1, col=j + 1
            )
            first = False
    return fig


def make_rel_error_plot(self, prob: Problem, **kwargs):
    """
    Make a plot of the relative error of estimated probability of failure.

    Args:
        prob (Problem): Instance of Problem class.

    Returns:
        [plotly.graph_objs._figure.Figure]: Plotly figure with plot.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Cmpute and plot relative error

    self.__compute_rel_error(prob)
    s = Scatter(y=self.prob_fail_r_err, name="Relative Error")
    fig.add_trace(s, secondary_y=False)
    fig.update_layout(showlegend=True, xaxis_title="Iteration")
    fig.update_yaxes(title_text="Relative Error", secondary_y=False, type="log")

    # Maybe compute und plot percentage of particles in fail. domain
    if kwargs.get("show_failure_percentage", False):
        self.__compute_perc_failure()
        b = Bar(
            y=self.perc_failure,
            name="Percentage of Particles in Failure Domain",
            marker={"opacity": 0.5},
        )
        # b = go.Bar(x = arange(1,self.num_steps), y=self.diff_mean, name="Percentage of Particles in Failure Domain", marker={
        #            "opacity": 0.5})
        fig.add_trace(b, secondary_y=True)
        fig.update_yaxes(title_text="Percent", secondary_y=True)

    return fig


def plot_iteration(iter: int, prob: Problem, sol: Solution = None, delta=1) -> Figure:
    """Make a plot of iteration `iter`.


    Args:
        iter (int): Which iteration shall be plotted.
        sol (optional, Solution): solution with info about all iterations.
        prob (Problem): Instance of Problem class.
        delta (optional[float]): stepsize for contour plot.

    Returns:
        [plotly.graph_objs._figure.Figure]: Plotly figure plot.
    """
    iter = maximum(0, minimum(iter, sol.N))
    # # Make evaluations for contour plot
    x_limits = [
        min(sol.ensemble_hist[:, :, 0]) - 3,
        max(sol.ensemble_hist[:, :, 0]) + 3,
    ]
    y_limits = [
        min(sol.ensemble_hist[:, :, 1]) - 3,
        max(sol.ensemble_hist[:, :, 1]) + 3,
    ]
    xx = arange(*x_limits, step=delta)
    yy = arange(*y_limits, step=delta)
    # y is first as rows are stacked vertically
    zz_lsf = zeros((len(yy), len(xx)))
    for xi, x in enumerate(xx):
        for yi, y in enumerate(yy):
            z = array([x, y])
            zz_lsf[yi, xi] = prob.lsf(z)

    #  Make contour plot of limit state function
    col_scale = [[0, "salmon"], [1, "white"]]
    contour_style = {"start": 0, "end": 0, "size": 0, "showlabels": True}
    c_lsf = Contour(
        z=zz_lsf,
        x=xx,
        y=yy,
        colorscale=col_scale,
        contours=contour_style,
        line_width=2,
        showscale=False,
    )
    # Plot ensemble
    s = Scatter(
        x=sol.ensemble_hist[iter, :, 0],
        y=sol.ensemble_hist[iter, :, 1],
        mode="markers",
    )
    layout = Layout(title="Iteration " + str(iter))
    fig = Figure(data=[c_lsf, s], layout=layout)
    # fig = go.Figure(data=[c_lsf, s], layout=l)
    return fig


def update_axes_format(
    fig: Figure,
    fmt_x: str,
    fmt_y: str,
    y_exp_range=range(-10, 10, 1),
    x_exp_range=range(-10, 10, 1),
    minor_ticks=[1],
    secondary_y=False,
) -> Figure:
    """Helper function for buggy `fig.update_layout`.

    Currently `fig.update_layout("yaxis_exponentformat": fmt_y)` is not applied
    to all subplots. This functions does it for y and x axis.
    Also set ticklabels at 1e+n for n in `y_exp_range` but draw gridlines for
    1e+n, 2e+n,...,9e+n,1e+(n+1) on the y axis.
    Same is done for the x axis based on `x_exp_range`
    Args:
        fig (Figure):  Plotly figure.
        fmt (str): Valid exponentformat.
        One of ( "none" | "e" | "E" | "power" | "SI" | "B" ).

    Returns:
        fig: Plotly figure with correct yaxis label format.
    """
    # Set exponentformat
    if fmt_x is not None:
        fig.update_xaxes(exponentformat=fmt_x)
    if fmt_y is not None:
        fig.update_yaxes(exponentformat=fmt_y, secondary_y=secondary_y)
    # Set ticklabels
    axis_ranges = {}
    if x_exp_range is not None and fmt_x is not None:
        axis_ranges = axis_ranges | {"xaxis": x_exp_range}
    if y_exp_range is not None and fmt_y is not None:
        axis_ranges = axis_ranges | {"yaxis": y_exp_range}
    for axis, exp_range in axis_ranges.items():
        tickvals = []
        ticktext = []
        for exponent in exp_range:
            for minor in range(10):
                tickvals.append(minor * 10**exponent)
                if minor in minor_ticks:
                    ticktext.append(
                        ("" if minor == 1 else f"{minor} ")
                        + f"10 <sup>{exponent}</sup>"
                    )
                else:
                    ticktext.append(" ")
        if axis == "xaxis":
            fig.update_xaxes({"tickvals": tickvals, "ticktext": ticktext})
        if axis == "yaxis":
            fig.update_yaxes(
                {"tickvals": tickvals, "ticktext": ticktext}, secondary_y=secondary_y
            )
    return fig


def make_efficiency_plot(
    df_line,
    df_box,
    x,
    y_line,
    y_box,
    facet_row=None,
    facet_col=None,
    shared_secondary_y_axes=False,
    y_log=False,
    secondary_y_log=True,
    facet_col_prefix=None,
    facet_row_prefix=None,
    x_axis_sorting_key=None,
    facet_name_sep=": ",
    labels={},
):
    """_summary_

    Must not containe columns with names "fc" or "fr"

    Args:
        df_line (_type_): _description_
        df_box (_type_): _description_
        x (_type_): _description_
        y_line (_type_): _description_
        y_box (_type_): _description_
        facet_row (_type_, optional): _description_. Defaults to None.
        facet_col (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # set up figure, maybew adapt data with help columns
    if facet_row is None:
        df_line.loc[:, "fr"] = ""
        df_box.loc[:, "fr"] = ""
        facet_row = "fr"
    if facet_col is None:
        df_line.loc[:, "fc"] = ""
        df_box.loc[:, "fc"] = ""
        facet_col = "fc"
    rows = df_line[facet_row].unique()
    cols = df_line[facet_col].unique()
    specs = len(rows) * [len(cols) * [{"secondary_y": True}]]
    # Set titles
    subplot_titles = []
    for row_name in rows:
        for col_name in cols:
            if len(rows) > 1:
                r_title = (
                    (facet_row_prefix + facet_name_sep)
                    if facet_row_prefix is not None
                    else ""
                )
                r_title += str(row_name)
            else:
                r_title = ""
            if len(cols) > 1:
                c_title = (
                    (facet_col_prefix + facet_name_sep)
                    if facet_col_prefix is not None
                    else ""
                )
                c_title += str(col_name)
                title = r_title + ", " + c_title
            else:
                title = r_title
            subplot_titles.append(title)
    fig = make_subplots(
        rows=len(rows),
        cols=1,
        specs=specs,
        subplot_titles=subplot_titles,
        x_title=x,
        shared_xaxes="all",
        shared_yaxes="all",
    )
    # collect limits here, to unify range of secondary y-axis
    secondary_y_limits = [inf, -inf]
    # populate subplots
    counter = 0
    for row_idx, row_name in enumerate(rows):
        for col_idx, col_name in enumerate(cols):
            # plot boxes
            this_df_box = df_box[
                (df_box[facet_row] == row_name) & (df_box[facet_col] == col_name)
            ]
            this_df_box = this_df_box.sort_values(x, key=x_axis_sorting_key)
            xx_box = this_df_box.loc[:, x].values
            yy_box = this_df_box.loc[:, y_box].values
            msk = yy_box > 0
            box = Box(
                x=xx_box[msk],
                y=yy_box[msk],
                marker_color=CMAP[1],
                name="Estimates",
                legendgroup="2",
                showlegend=counter == 0,
            )
            fig.add_trace(box, row=row_idx + 1, col=col_idx + 1, secondary_y=True)

            # plot lines
            this_df_line = df_line[
                (df_line[facet_row] == row_name) & (df_line[facet_col] == col_name)
            ]
            this_df_line = this_df_line.sort_values(x, key=x_axis_sorting_key)
            xx_line = this_df_line[x].values
            yy_line = this_df_line[y_line].values
            line = Scatter(
                x=xx_line,
                y=yy_line,
                marker_color=CMAP[0],
                name="Rel. Efficency",
                legendgroup="1",
                showlegend=counter == 0,
            )
            fig.add_trace(line, row=row_idx + 1, col=col_idx + 1)
            counter += 1
            for idx, fun in zip((0, 1), (amin, amax)):
                try:
                    secondary_y_limits[idx] = fun(
                        [secondary_y_limits[idx], fun(yy_box)]
                    )
                except Exception:
                    Warning(f"bad data")
    # clean up dfs
    for df in [df_line, df_box]:
        df.drop(columns=["fc", "fr"], inplace=True, errors="ignore")
    # style figure
    fig.update_layout(**MY_LAYOUT)
    fig.update_layout(legend=dict(orientation="h"))
    if "yaxis_exponentformat" in MY_LAYOUT.keys() and y_log:
        fig = update_axes_format(fig, None, MY_LAYOUT["yaxis_exponentformat"])
    if "yaxis_exponentformat" in MY_LAYOUT.keys() and secondary_y_log:
        fig = update_axes_format(
            fig,
            None,
            MY_LAYOUT["yaxis_exponentformat"],
            secondary_y=True,
            minor_ticks=[1, 5, 2],
        )
    secondary_y_limits = [
        log10(lim * [0.9, 1.1][i]) for i, lim in enumerate(secondary_y_limits)
    ]
    for row_idx in range(len(rows)):
        for col_idx in range(len(cols)):
            if shared_secondary_y_axes:
                fig.update_yaxes(
                    range=secondary_y_limits,
                    row=row_idx + 1,
                    col=col_idx + 1,
                    secondary_y=True,
                )
            fig.update_yaxes(
                title="Estimates",
                type="log" if secondary_y_log else None,
                row=row_idx + 1,
                col=col_idx + 1,
                showgrid=False,
                ticks="outside",
                secondary_y=True,
            )
            fig.update_yaxes(
                type="log" if y_log else None,
                title="Rel. Efficiency",
                row=row_idx + 1,
                col=col_idx + 1,
                secondary_y=False,
            )
    fig.for_each_annotation(
        lambda a: a.update(
            text=labels.get(a.text) if labels.get(a.text) is not None else a.text
        )
    )
    return fig
