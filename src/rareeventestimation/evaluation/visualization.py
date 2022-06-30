from numpy import arange, cumsum, log, sqrt, zeros, array, minimum, maximum, sum
import pandas as pd
from rareeventestimation.evaluation.convergence_analysis import aggregate_df
import plotly.express as px
import re
from plotly.graph_objects import Figure, Scatter,Bar, Contour, Layout
from os import path
from rareeventestimation.problem.problem import Problem
from rareeventestimation.solution import Solution
from plotly.subplots import make_subplots
from os.path  import commonpath
import plotly.colors
from PIL import ImageColor
cmap = px.colors.qualitative.Plotly


def make_accuracy_plots(df:pd.DataFrame, save_to_resp_path=True,plot_all_seeds=False, one_plot=False, MSE=True, cmap=cmap, layout={}) -> list:
    """Plot rel. root MSE of estimates vs mean of costs.

    Args:
        df (pd.DataFrame): Dataframe, assumed to come from  add_evaluations and aggregate_df.
        save_to_resp_path (bool, optional): Save plots to resp. path specified in column "Path". Defaults to True.

    Returns:
        list: List with Figures
    """
    out = []
    df =  df.set_index(["Problem","Solver","Sample Size"])

    # Set up dicts with (solver,color) and (solver,line-style) entries
    solver_colors = {
        s: cmap[i%len(cmap)] 
        for (i, s) in enumerate(df.index.get_level_values(1).unique())
        }
    if one_plot:
        solver_colors = {
            s: cmap[i%len(cmap)] 
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
        applied_solvers = df.loc[problem,:].index.get_level_values(0).unique()
        # Add traces for each solver
        for solver in applied_solvers:
            xvals = df.loc[(problem, solver),"Relative Root MSE"].values if MSE else df.loc[(problem, solver),".50 Relative Error"]
            yvals = df.loc[(problem, solver),"Cost Mean"].values if MSE else df.loc[(problem, solver),".50 Cost"].values
            rel_root_mse_sc = Scatter(
                y = yvals,
                x = xvals,
                name = solver,
                mode="lines + markers",
               line={"color": solver_colors.get(solver), "dash": solver_dashes.get(solver)},
                error_x={
                    "array": df.loc[(problem, solver),".75 Relative Error"].values - xvals,
                    "arrayminus":xvals-df.loc[(problem, solver),".25 Relative Error"].values,
                    "type": "data",
                    "symmetric":False,
                    "thickness": 0.5
                },
                error_y={
                    "array": df.loc[(problem, solver),".75 Cost"].values - yvals,
                    "arrayminus":yvals-df.loc[(problem, solver),".25 Cost"].values,
                    "type": "data",
                    #"symmetric":True,
                    "thickness": 0.5
                }
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
                df_err_and_cost = df.groupby(groups).mean().loc[(solver, problem),("Relative Error", "Cost")]
                for s in df_err_and_cost.index.get_level_values(0).unique():
                    sample_sc = Scatter(
                        y = df_err_and_cost.loc[s, "Cost"].values,
                        x = df_err_and_cost.loc[s,"Relative Error"].values,
                        mode="lines + markers",
                        line={"color": solver_colors[solver], "dash": solver_dashes[solver]},
                        opacity=0.2,
                        hovertext=f"Seed {s}",
                        hoverinfo="text",
                        showlegend = False
                    )
                    fig.add_trace(sample_sc)
            fig.add_trace(rel_root_mse_sc)
            #fig.add_trace(rel_err_median)
            if one_plot:
                one_fig.update_xaxes({"title":"Relative Root MSE","type":"log","showexponent":'all',"exponentformat" : 'e'})
                one_fig.update_yaxes({"title":"Cost","type":"log","showexponent":'all',"exponentformat" : 'e'})
                one_fig.update_layout(title="Cost-Error Plot")
            fig.update_xaxes({"title":"Relative Root MSE" if MSE else "Median of rel Error",
                              "type":"log",
                              "showexponent":'all',
                              "exponentformat" : 'e'})
            fig.update_yaxes({"title":"Cost","type":"log","showexponent":'all',"exponentformat" : 'e'})
            fig.update_layout(title="Cost-Error Plot for " + problem)
            fig.update_layout(**layout)
        out.append(fig)
    if one_plot:
        return one_fig
    return out


def make_mse_plots(df:pd.DataFrame, save_to_resp_path=True) ->dict:
    df_agg = aggregate_df(df)
    problem_solver_keys = df_agg.index.droplevel(2).unique()
    out = dict.fromkeys(problem_solver_keys)
    
    for k in problem_solver_keys:
        fig = Figure()
        problem, solver = k
        sc_bias = Scatter(
             y = df_agg.loc[k,"Cost Mean"].values,
             x = df_agg.loc[k,"Estimate Bias"].values**2,
            name = "Bias Squared",
            mode="lines + markers",
            line={"color": cmap[0]},
            stackgroup = "one",
            orientation="h"
        )
        sc_var = Scatter(
             y = df_agg.loc[k,"Cost Mean"].values,
             x = df_agg.loc[k,"Estimate Variance"].values,
            name = "Variance",
            mode="lines + markers",
            line={"color": cmap[1]},
            stackgroup = "one",
            orientation="h"
        )
        fig.add_trace(sc_bias)
        fig.add_trace(sc_var)
        fig.update_xaxes({"title":"MSE","type":"log","showexponent":'all',"exponentformat" : 'e'})
        fig.update_yaxes({"title":"Cost","type":"log","showexponent":'all',"exponentformat" : 'e'})
        fig.update_layout(title= f"MSE of {solver} applied to {problem}")
        out[k] = fig
    
    if save_to_resp_path:
        for k, fig in out.items():
            p = commonpath(df_agg.loc[k,"Path"].values.tolist())
            fig.write_image(path.join(p, fig.layout.title["text"] + ".pdf"))
            
    return out


        
        


def get_color(colorscale_name, loc):
    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...] 
    colorscale = cv.validate_coerce(colorscale_name)
    
    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)
        

# Identical to Adam's answer


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
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

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


def plot_cbree_parameters(sol:Solution, p2, plot_time=False):
    f = make_subplots(rows=3, shared_xaxes=True)
    if plot_time:
        xx = cumsum(-log(sol.other["t_step"]))
    else:
        xx = arange(len(sol.other["t_step"]))
    # Plot error and error estimate
    rel_err = sol.get_rel_err(p2)
    rel_err_est = sol.other["cvar_is_weights"]/sqrt(p2.sample.shape[0])
    f.add_trace(Scatter(x=xx, y=rel_err, name="Rel. Root Err."), row=1, col=1)
    f.add_trace(Scatter(x=xx, y=rel_err_est, name="Rel. Root Err. Estimate"), row=1, col=1)
    f.update_yaxes(type="log", row=1,col=1)

    # Plot parameters
    f.add_trace(Scatter(x=xx, y=sol.other["sigma"], name="Sigma"), row=2, col=1)
    f.add_trace(Scatter(x=xx, y=sol.other["beta"], name="Beta"), row=2, col=1)
    f.add_trace(Scatter(x=xx, y=-log(sol.other["t_step"]), name="Stepsize"), row=2, col=1)
    f.add_trace(Scatter(x=xx, y=sol.other["ess"]/p2.sample.shape[0], name="rel.ESS"), row=2, col=1)
    # Plot monitored quants

    f.add_trace(Scatter(x=xx, y=sum(sol.lsf_eval_hist <= 0,axis=1)/p2.sample.shape[0], name= "SFP"),  row=3, col=1)
    f.add_trace(Scatter(x=xx, y=sol.other["sfp_slope"], name= "SFP Slope", line_color = cmap[(len(f.data)-1) % len(cmap)], line_dash="dash"),  row=3, col=1)
    f.add_trace(Scatter(x=xx, y=sol.other["slope_cvar"], name= "CVAR Slope", line_color = cmap[1], line_dash="dash"),  row=3, col=1)
    f.update_layout(hovermode="x unified")
    return f

def add_trace_to_subplots(f,**tr_kwargs):
    
    
    
    
    
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
        fig.update_yaxes(title_text="Relative Error",
                         secondary_y=False, type="log")

        # Maybe compute und plot percentage of particles in fail. domain
        if kwargs.get("show_failure_percentage", False):
            self.__compute_perc_failure()
            b = Bar(y=self.perc_failure, name="Percentage of Particles in Failure Domain", marker={
                       "opacity": 0.5})
            # b = go.Bar(x = arange(1,self.num_steps), y=self.diff_mean, name="Percentage of Particles in Failure Domain", marker={
            #            "opacity": 0.5})
            fig.add_trace(b, secondary_y=True)
            fig.update_yaxes(title_text="Percent", secondary_y=True)

        return fig

    def plot_iteration(self, iter: int, prob: Problem, delta=1):
        """Make a plot of iteration `iter`.

        Args:
            prob (Problem): Instance of Problem class.
            delta (optional[float]): stepsize for contour plot.

        Returns:
            [plotly.graph_objs._figure.Figure]: Plotly figure plot.
        """
        bb = self.temp_hist.squeeze()
        ss = self.other["Sigma"].squeeze()
        iter = maximum(0, minimum(iter, self.num_steps-2))
        # # Make evaluations for contour plot
        x_limits = [min(self.ensemble_hist[:, :, 0])-3,
                    max(self.ensemble_hist[:, :, 0])+3]
        y_limits = [min(self.ensemble_hist[:, :, 1])-3,
                    max(self.ensemble_hist[:, :, 1])+3]
        xx = arange(*x_limits, step=delta)
        yy = arange(*y_limits, step=delta)
        # y is first as rows are stacked vertically
        zz_tgt = zeros((len(yy), len(xx)))
        zz_lsf = zeros((len(yy), len(xx)))
        for (xi, x) in enumerate(xx):
            for(yi, y) in enumerate(yy):
                z = array([x, y])
                zz_lsf[yi, xi] = prob.lsf(z)
                zz_tgt[yi, xi] = self.tgt_fun(prob.lsf(z),prob.e_fun(z),sigma=ss[iter]) ** bb[iter]


        #  Make contour plot of limit state function
        col_scale = [[0, "salmon"], [1, "white"]]
        contour_style = {"start": 0, "end": 0, "size": 0, "showlabels": True}
        c_lsf = Contour(z=zz_lsf, x=xx, y=yy, colorscale=col_scale,
                          contours=contour_style, line_width=2, showscale=False)
        #  Make contour plot for target function
        c_tgt = Contour(z=zz_tgt, x=xx, y=yy, contours_coloring='lines', showscale=False)
        # Make frames for animation
        s = Scatter(x=self.ensemble_hist[iter, :, 0], y=self.ensemble_hist[iter, :, 1],
                       mode="markers")
        l = Layout(title="Iteration " + str(iter))
        fig = Figure(data=[c_lsf, c_tgt, s], layout=l)
        #fig = go.Figure(data=[c_lsf, s], layout=l)
        return fig