"""
MIT License

Copyright (c) 2022 JAKE LEVI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import math
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import matplotlib.lines
import matplotlib.patches
import matplotlib.colors
import matplotlib.cm
import numpy as np
import PIL.Image
from jutility import util

class Plottable:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._expand_abbreviated_keys()
        for k, v in self._get_default_kwargs().items():
            self._kwargs.setdefault(k, v)

    def plot(self, axis: matplotlib.axes.Axes):
        raise NotImplementedError()

    def get_handle(self):
        raise NotImplementedError()

    def set_options(self, **kwargs):
        for k, v in kwargs.items():
            self._kwargs[k] = v

        self._expand_abbreviated_keys()
        return self

    def _get_default_kwargs(self):
        return {"zorder": 10}

    def _get_abbreviated_keys_dict(self):
        return {
            "c": "color",
            "z": "zorder",
            "a": "alpha",
            "m": "marker",
        }

    def _expand_abbreviated_keys(self):
        for k, k_full in self._get_abbreviated_keys_dict().items():
            if k in self._kwargs:
                self._kwargs[k_full] = self._kwargs.pop(k)

    def has_label(self):
        return (self._kwargs.get("label") is not None)

    def __repr__(self):
        args_str   = ", ".join(repr(a) for a in self._args)
        kwargs_str = ", ".join(
            "%s=%r" % (k, v)
            for k, v in self._kwargs.items()
        )
        return ("%s(%s, %s)" % (type(self).__name__, args_str, kwargs_str))

class Line(Plottable):
    """
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.plot(*self._args, **self._kwargs)

    def get_handle(self):
        return matplotlib.lines.Line2D([], [], **self._kwargs)

    def _get_default_kwargs(self):
        return {"zorder": 10, "color": "b"}

class HLine(Line):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axhline.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.axhline(*self._args, **self._kwargs)

class VLine(Line):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axvline.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.axvline(*self._args, **self._kwargs)

class AxLine(Line):
    """
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axline.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.axline(*self._args, **self._kwargs)

class Quiver(Line):
    """
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        if self._kwargs.pop("normalise", False):
            tol = self._kwargs.pop("tol", 1e-5)
            x, y, u, v = self._args[:4]
            dr = np.sqrt(np.square(u) + np.square(v))
            dr_safe = np.maximum(dr, tol)
            u = u / dr_safe
            v = v / dr_safe
            self._args = (x, y, u, v) + self._args[4:]

        axis.quiver(*self._args, **self._kwargs)

class Step(Line):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.step.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.step(*self._args, **self._kwargs)

class Circle(Line):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Circle.html
    and
    https://matplotlib.org/stable/gallery/shapes_and_collections/artist_reference.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        default_lw = matplotlib.rcParams["lines.linewidth"]
        self._kwargs.setdefault("lw", default_lw)
        self._kwargs.setdefault("fill", False)
        circle = matplotlib.patches.Circle(*self._args, **self._kwargs)
        axis.add_artist(circle)

class Scatter(Plottable):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.scatter(*self._args, **self._kwargs)

    def get_handle(self):
        line_kwargs = self._kwargs.copy()
        line_kwargs.setdefault("marker", "o")
        line_kwargs.setdefault("ls", "")
        if "s" in line_kwargs:
            line_kwargs.setdefault("ms", np.sqrt(line_kwargs.pop("s")))

        return matplotlib.lines.Line2D([], [], **line_kwargs)

    def _get_abbreviated_keys_dict(self):
        return {"z": "zorder", "a": "alpha", "m": "marker"}

class Contour(Plottable):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.contour(*self._args, **self._kwargs)

class Text(Plottable):
    """
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html

    Further examples:
    https://matplotlib.org/stable/gallery/text_labels_and_annotations/fancytextbox_demo.html
    https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        if self._kwargs.pop("center_align", False):
            self._kwargs["horizontalalignment"]   = "center"
            self._kwargs["verticalalignment"]     = "center"

        axis.text(*self._args, **self._kwargs)

class _Patch(Plottable):
    def get_handle(self):
        patch_kwargs = {
            k: v for k, v in self._kwargs.items()
            if k not in ["x", "y1", "y2"]
        }
        return matplotlib.patches.Patch(**patch_kwargs)

class FillBetween(_Patch):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.fill_between(*self._args, **self._kwargs)

    def _get_default_kwargs(self):
        return {"zorder": 10, "color": "b", "ec": None}

class HSpan(FillBetween):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axhspan.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.axhspan(*self._args, **self._kwargs)

class VSpan(FillBetween):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axvspan.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.axvspan(*self._args, **self._kwargs)

class Bar(_Patch):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.bar(*self._args, **self._kwargs)

    def _get_default_kwargs(self):
        return {"zorder": 10, "color": "b", "ec": "k"}

class Hist(Bar):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.hist(*self._args, **self._kwargs)

class Polygon(Plottable):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.fill.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.fill(*self._args, **self._kwargs)

    def _get_default_kwargs(self):
        return {"zorder": 10, "fc": "b", "ec": "k", "lw": 5}

class ColourMesh(Plottable):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.pcolormesh(*self._args, **self._kwargs)

class ContourFilled(Plottable):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.contourf(*self._args, **self._kwargs)

class ImShow(Plottable):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis_off = self._kwargs.pop("axis_off", True)
        axis.imshow(*self._args, **self._kwargs)
        if axis_off:
            axis.set_axis_off()

class Legend(Plottable):
    """
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        zorder = self._kwargs.pop("zorder", None)
        legend = axis.legend(*self._args, **self._kwargs)

        if zorder is not None:
            legend.set_zorder(zorder)

def confidence_bounds(
    data_list,
    n_sigma=1,
    split_dim=None,
    num_split=100,
):
    if split_dim is not None:
        data_array = np.array(data_list)
        num_split = min(num_split, data_array.shape[split_dim])
        data_list = np.array_split(data_array, num_split, split_dim)

    mean = np.array([np.mean(x) for x in data_list])
    std  = np.array([np.std( x) for x in data_list])
    ucb = mean + (n_sigma * std)
    lcb = mean - (n_sigma * std)
    return mean, ucb, lcb

class NoisyData:
    def __init__(self, log_x=False, log_y=False, x_index=False):
        self._results_list_dict: dict[float, list] = dict()
        self._log_x = log_x
        self._log_y = log_y
        self._x_index = x_index
        self._x_index_list = []

    def update(self, x, y):
        if self._x_index:
            if x not in self._x_index_list:
                self._x_index_list.append(x)

            x = self._x_index_list.index(x)

        if self._log_y:
            y = np.log(y)
        if x in self._results_list_dict:
            self._results_list_dict[x].append(y)
        else:
            self._results_list_dict[x] = [y]

    def get_all_data(self):
        all_results_pairs = [
            [x, y]
            for x, result_list in self._results_list_dict.items()
            for y in result_list
        ]
        all_x, all_y = np.split(np.array(all_results_pairs), 2, axis=1)
        if self._log_y:
            all_y = np.exp(all_y)

        return all_x.flatten(), all_y.flatten()

    def get_statistics(self, n_sigma=1):
        x = sorted(
            x_i for x_i, results_list in self._results_list_dict.items()
            if len(results_list) > 0
        )
        results_list_list = [self._results_list_dict[x_i] for x_i in x]
        mean, ucb, lcb = confidence_bounds(results_list_list, n_sigma)
        if self._log_y:
            mean, ucb, lcb = np.exp([mean, ucb, lcb])

        return np.array(x), mean, ucb, lcb

    def plot(self, c="b", label=None, n_sigma=1):
        x, mean, ucb, lcb = self.get_statistics(n_sigma)
        return (
            Scatter([], [], label=label, color=c),
            Scatter(*self.get_all_data(),   a=0.5, z=20, color=c),
            Line(x, mean,                   a=1.0, z=30, c=c),
            FillBetween(x, lcb, ucb,        a=0.2, z=10, c=c),
        )

    def get_xtick_kwargs(self):
        xtick_kwargs = dict()
        if self._x_index:
            xtick_kwargs["xticks"] = list(range(len(self._x_index_list)))
            xtick_kwargs["xticklabels"] = self._x_index_list

        return xtick_kwargs

    def predict(self, x_pred: np.ndarray, eps=1e-5):
        x, y = self.get_all_data()
        if self._log_x:
            x_pred = np.log(x_pred)
            x = np.log(x)
        if self._log_y:
            y = np.log(y)

        xm = x.mean()
        ym = y.mean()
        xc = x - xm
        yc = y - ym

        w = np.sum(yc * xc) / (np.sum(xc * xc) + eps)
        b = ym - w * xm
        y_pred = w * x_pred + b

        if self._log_y:
            y_pred = np.exp(y_pred)

        return y_pred

    def predict_line(self, x0: float, x1: float, eps=1e-5, **line_kwargs):
        y0, y1 = self.predict(np.array([x0, x1]), eps)
        line_kwargs.setdefault("ls", "--")
        return AxLine([x0, y0], [x1, y1], **line_kwargs)

class ColourPicker:
    """
    See https://matplotlib.org/stable/users/explain/colors/colormaps.html
    """
    def __init__(
        self,
        num_colours,
        cyclic=True,
        cmap_name=None,
        dynamic_range=None,
    ):
        if cmap_name is None:
            if cyclic:
                cmap_name = "hsv"
            else:
                cmap_name = "cool"
        if cyclic:
            endpoint = False
        else:
            endpoint = True

        self._cmap = plt.get_cmap(cmap_name)
        if dynamic_range is not None:
            lo, hi = dynamic_range
            colour_list = self._cmap(np.linspace(lo, hi, num_colours))
            self._cmap = matplotlib.colors.ListedColormap(colour_list)

        cmap_sample_points = np.linspace(0, 1, num_colours, endpoint)
        self._colours = [self._cmap(i) for i in cmap_sample_points]
        self.reset()

    def __call__(self, colour_ind):
        return self._colours[colour_ind]

    def next(self):
        c = self._colours[self._index % len(self._colours)]
        self._index += 1
        return c

    def reset(self):
        self._index = 0

    def get_cmap(self):
        return self._cmap

    def get_colourbar(self, vmin=0, vmax=None, **kwargs):
        """
        See
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
        """
        if vmax is None:
            vmax = len(self._colours)

        return ColourBar(vmin, vmax, cmap=self._cmap, **kwargs)

    def colour_name(self, colour_ind):
        return "c%i" % colour_ind

    def define_colours(self):
        definition_list = [
            "\\definecolor{%s}{rgb}{%f,%f,%f}"
            % (self.colour_name(i), c[0], c[1], c[2])
            for i, c in enumerate(self._colours)
        ]
        definition_str = "\n".join(definition_list)
        return definition_str

class AxisProperties:
    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        log_x=False,
        log_y=False,
        symlog_x=False,
        symlog_y=False,
        xticks=None,
        xticklabels=None,
        rotate_xticklabels=False,
        axis_equal=False,
        axis_square=False,
        axis_off=False,
        grid=True,
        title=None,
        title_colour=None,
        wrap_title=True,
        colour=None,
    ):
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._xlim = xlim
        self._ylim = ylim
        self._log_x = log_x
        self._log_y = log_y
        self._symlog_x = symlog_x
        self._symlog_y = symlog_y
        self._xticks = xticks
        self._xticklabels = xticklabels
        self._rotate_xticklabels = rotate_xticklabels
        self._axis_equal = axis_equal
        self._axis_square = axis_square
        self._axis_off = axis_off
        self._grid = grid
        self._title = title
        self._title_colour = title_colour
        self._wrap_title = wrap_title
        self._colour = colour

    def set_default_title(self, title):
        if self._title is None:
            self._title = title

    def apply(self, axis: matplotlib.axes.Axes):
        if self._xlabel is not None:
            axis.set_xlabel(self._xlabel)
        if self._ylabel is not None:
            axis.set_ylabel(self._ylabel)
        if self._xticks is not None:
            axis.set_xticks(self._xticks, self._xticklabels)
        if self._xlim is not None:
            axis.set_xlim(self._xlim)
        if self._ylim is not None:
            axis.set_ylim(self._ylim)
        if self._axis_equal:
            axis.axis("equal")
        if self._axis_square:
            axis.axis("square")
        if self._axis_off:
            axis.set_axis_off()
        if self._log_x:
            axis.set_xscale("log")
        if self._log_y:
            axis.set_yscale("log")
        if self._symlog_x:
            axis.set_xscale("symlog")
        if self._symlog_y:
            axis.set_yscale("symlog")
        if self._grid:
            axis.grid(True, which="both")
        if self._rotate_xticklabels:
            for xtl in axis.get_xticklabels():
                xtl.set(rotation=-45, ha="left")
        if self._title is not None:
            colour_dict = dict()
            if self._title_colour is not None:
                colour_dict["color"] = self._title_colour
            if self._wrap_title:
                self._title = util.wrap_string(self._title)
            axis.set_title(self._title, **colour_dict)
        if self._colour is not None:
            axis.set_facecolor(self._colour)

class FigureProperties:
    def __init__(
        self,
        num_rows=None,
        num_cols=None,
        figsize=None,
        sharex=False,
        sharey=False,
        width_ratios=None,
        height_ratios=None,
        tight_layout=True,
        constrained_layout=False,
        colour=None,
        title=None,
        title_font_size=25,
        title_colour=None,
        wrap_title=True,
        top_space=None,
        bottom_space=None,
        legend: "FigureLegend"=None
    ):
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._figsize = figsize
        self._sharex = sharex
        self._sharey = sharey
        self._width_ratios = width_ratios
        self._height_ratios = height_ratios

        if constrained_layout:
            tight_layout = False
            self._layout = "constrained"
        else:
            self._layout = None

        self._tight_layout = tight_layout
        self._colour = colour
        self._title = title
        self._title_font_size = title_font_size
        self._title_colour = title_colour
        self._wrap_title = wrap_title
        self._top_space = top_space
        self._bottom_space = bottom_space
        self._legend = legend

    def get_figure_and_axes(self, num_subplots):
        if self._num_rows is None:
            if self._num_cols is None:
                self._num_cols = math.ceil(math.sqrt(num_subplots))
            self._num_rows = math.ceil(num_subplots / self._num_cols)
        if self._num_cols is None:
            self._num_cols = math.ceil(num_subplots / self._num_rows)

        if self._figsize is None:
            self._figsize = [6 * self._num_cols, 4 * self._num_rows]

        gridspec_kw = {
            "width_ratios": self._width_ratios,
            "height_ratios": self._height_ratios,
        }
        figure, axes = plt.subplots(
            self._num_rows,
            self._num_cols,
            figsize=self._figsize,
            sharex=self._sharex,
            sharey=self._sharey,
            gridspec_kw=gridspec_kw,
            squeeze=False,
            layout=self._layout,
        )
        return figure, axes.flat

    def apply(self, figure: matplotlib.figure.Figure):
        if self._tight_layout:
            figure.tight_layout()
        if self._colour is not None:
            figure.patch.set_facecolor(self._colour)
        if self._title is not None:
            if self._wrap_title:
                self._title = util.wrap_string(self._title)
            figure.suptitle(
                self._title,
                fontsize=self._title_font_size,
                color=self._title_colour,
            )
        if self._top_space is not None:
            figure.subplots_adjust(top=(1 - self._top_space))
        if self._bottom_space is not None:
            figure.subplots_adjust(bottom=self._bottom_space)
        if self._legend is not None:
            self._legend.plot(figure)

class Subplot:
    def __init__(
        self,
        *lines: Plottable,
        axis_properties=None,
        **axis_kwargs,
    ):
        self._lines = lines
        if axis_properties is None:
            axis_properties = AxisProperties(**axis_kwargs)
        self._axis_properties = axis_properties

    def plot(self, axis: matplotlib.axes.Axes):
        for line in self._lines:
            line.plot(axis)

        self._axis_properties.apply(axis)

class LegendSubplot(Subplot):
    """
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    """
    def __init__(self, *lines: Plottable, **legend_kwargs):
        handles = [
            line.get_handle() for line in lines if line.has_label()
        ]
        legend_kwargs.setdefault("handles", handles)
        legend_kwargs.setdefault("loc", "center")
        self._legend_plottable = Legend(**legend_kwargs)

    def plot(self, axis: matplotlib.axes.Axes):
        self._legend_plottable.plot(axis)
        axis.set_axis_off()

class FigureLegend(Subplot):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.legend.html
    """
    def __init__(
        self,
        *lines: Plottable,
        ncols=1,
        loc="outside lower center",
        **legend_kwargs,
    ):
        handles = [
            line.get_handle() for line in lines if line.has_label()
        ]
        self._kwargs = {
            "ncols":    ncols,
            "loc":      loc,
        }
        if len(handles) > 0:
            self._kwargs["handles"] = handles

        self._kwargs.update(legend_kwargs)

    def plot(self, figure: matplotlib.figure.Figure):
        figure.legend(**self._kwargs)

class ColourBar(Subplot):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
    """
    def __init__(self, vmin, vmax, cmap=None, **kwargs):
        norm = matplotlib.colors.Normalize(vmin, vmax)
        self._sm = matplotlib.cm.ScalarMappable(norm, cmap=cmap)
        self._axes = []
        self._colourbar = None
        self._kwargs = kwargs

    def plot(self, axis: matplotlib.axes.Axes):
        if self._colourbar is not None:
            self._colourbar.remove()

        axis.set_axis_off()
        self._axes.append(axis)
        self._colourbar = plt.colorbar(
            mappable=self._sm,
            ax=self._axes,
            fraction=1,
            location="left",
            **self._kwargs,
        )

class Empty(Subplot):
    def plot(self, axis: matplotlib.axes.Axes):
        axis.set_axis_off()

def plot(
    *lines: Plottable,
    axis_properties=None,
    legend=False,
    figsize=None,
    plot_name=None,
    dir_name=None,
    save_close=True,
    pdf=False,
    **axis_kwargs,
):
    if axis_properties is None:
        axis_properties = AxisProperties(**axis_kwargs)
    if figsize is None:
        figsize = [10, 6] if legend else [8, 6]
    if plot_name is not None:
        axis_properties.set_default_title(plot_name)

    if legend:
        wr =  [1, 0.2]
        fig_properties = FigureProperties(1, 2, figsize, width_ratios=wr)
        multi_plot = MultiPlot(
            Subplot(*lines, axis_properties=axis_properties),
            LegendSubplot(*lines),
            figure_properties=fig_properties,
        )
    else:
        fig_properties = FigureProperties(1, 1, figsize)
        multi_plot = MultiPlot(
            Subplot(*lines, axis_properties=axis_properties),
            figure_properties=fig_properties,
        )

    if save_close:
        multi_plot.save(plot_name, dir_name, pdf=pdf)

    return multi_plot

class MultiPlot:
    def __init__(
        self,
        *subplots: Subplot,
        figure_properties=None,
        **figure_kwargs,
    ):
        if figure_properties is None:
            figure_properties = FigureProperties(**figure_kwargs)

        self._fig, axes = figure_properties.get_figure_and_axes(len(subplots))

        if len(subplots) < len(axes):
            subplots += tuple([Empty()]) * (len(axes) - len(subplots))

        for subplot, axis in zip(subplots, axes):
            subplot.plot(axis)

        figure_properties.apply(self._fig)

        self.full_path = None

    def save(
        self,
        plot_name=None,
        dir_name=None,
        verbose=True,
        file_ext=None,
        pdf=False,
        close=True,
    ):
        if plot_name is None:
            plot_name = "Output"
        if file_ext is None:
            file_ext = "pdf" if pdf else "png"

        self.full_path = util.get_full_path(
            plot_name,
            dir_name,
            file_ext=file_ext,
            verbose=verbose,
        )
        self._fig.savefig(self.full_path)

        if close:
            self.close()

        return self.full_path

    def get_rgb_bytes(self):
        self._fig.canvas.draw()
        rgb_bytes       = self._fig.canvas.tostring_rgb()
        width, height   = self._fig.canvas.get_width_height()
        return rgb_bytes, width, height

    def get_rgba_bytes(self):
        self._fig.canvas.draw()
        rgba_bytes      = self._fig.canvas.buffer_rgba().tobytes()
        width, height   = self._fig.canvas.get_width_height()
        return rgba_bytes, width, height

    def get_pil_image(self):
        rgba_bytes, width, height = self.get_rgba_bytes()
        pil_image = PIL.Image.frombytes(
            mode="RGBA",
            size=[width, height],
            data=rgba_bytes,
        )
        return pil_image

    def get_image_array(self):
        return np.array(self.get_pil_image())

    def close(self):
        plt.close(self._fig)

class Gif:
    def __init__(self):
        self._frame_list: list[PIL.Image.Image] = []

    def add_pil_image_frame(self, pil_image):
        self._frame_list.append(pil_image)

    def add_rgb_bytes_frame(self, rgb_bytes, width, height):
        pil_image = PIL.Image.frombytes(
            mode="RGB",
            size=[width, height],
            data=rgb_bytes,
        )
        self.add_pil_image_frame(pil_image)

    def add_rgba_bytes_frame(self, rgba_bytes, width, height):
        pil_image = PIL.Image.frombytes(
            mode="RGBA",
            size=[width, height],
            data=rgba_bytes,
        )
        self.add_pil_image_frame(pil_image)

    def add_multiplot_frame(self, multi_plot: MultiPlot):
        self.add_pil_image_frame(multi_plot.get_pil_image())

    def add_plot_frame(self, *lines, save=False, **plot_kwargs):
        plot_kwargs.setdefault("save_close", False)
        mp = plot(*lines, **plot_kwargs)
        self.add_multiplot_frame(mp)
        if save:
            mp.save(plot_kwargs.get("plot_name"), plot_kwargs.get("dir_name"))
        mp.close()

    def add_rgb_array_frame(self, ndarray_hwc, vmin=0, vmax=1):
        util.check_type(ndarray_hwc, np.ndarray, "ndarray_hwc")
        if (ndarray_hwc.ndim != 3) or (ndarray_hwc.shape[2] != 3):
            raise ValueError(
                "Expected shape (H, W, C=3), but received shape %s"
                % ndarray_hwc.shape
            )

        ndarray_scaled = 255 * (ndarray_hwc - vmin) / (vmax - vmin)
        ndarray_clipped = np.clip(ndarray_scaled, 0, 255)
        ndarray_int8 = ndarray_clipped.astype(np.uint8)
        pil_image = PIL.Image.fromarray(ndarray_int8, mode="RGB")
        self.add_pil_image_frame(pil_image)

    def add_bw_array_frame(self, ndarray_hw, vmin=0, vmax=1):
        util.check_type(ndarray_hw, np.ndarray, "ndarray_hw")
        if ndarray_hw.ndim != 2:
            raise ValueError(
                "Expected shape (H, W), but received shape %s"
                % ndarray_hw.shape
            )

        ndarray_scaled = 255 * (ndarray_hw - vmin) / (vmax - vmin)
        ndarray_clipped = np.clip(ndarray_scaled, 0, 255)
        ndarray_int8 = ndarray_clipped.astype(np.uint8)
        pil_image = PIL.Image.fromarray(ndarray_int8, mode="L")
        self.add_pil_image_frame(pil_image)

    def add_rgb_array_sequence(self, ndarray_lhwc, vmin=0, vmax=1):
        util.check_type(ndarray_lhwc, np.ndarray, "ndarray_lhwc")
        if (ndarray_lhwc.ndim != 4) or (ndarray_lhwc.shape[3] != 3):
            raise ValueError(
                "Expected shape (L, H, W, C=3), but received shape %s"
                % ndarray_lhwc.shape
            )

        for i in range(ndarray_lhwc.shape[0]):
            self.add_rgb_array_frame(ndarray_lhwc[i], vmin, vmax)

    def add_bw_array_sequence(self, ndarray_lhw, vmin=0, vmax=1):
        util.check_type(ndarray_lhw, np.ndarray, "ndarray_lhw")
        if ndarray_lhw.ndim != 3:
            raise ValueError(
                "Expected shape (L, H, W), but received shape %s"
                % ndarray_lhw.shape
            )

        for i in range(ndarray_lhw.shape[0]):
            self.add_bw_array_frame(ndarray_lhw[i], vmin, vmax)

    def add_image_file_frame(self, filename, dir_name=None):
        if dir_name is None:
            dir_name = util.RESULTS_DIR
        full_path = os.path.join(dir_name, filename)
        self.add_pil_image_frame(PIL.Image.open(full_path))

    def save(
        self,
        output_name=None,
        dir_name=None,
        frame_duration_ms=100,
        optimise=False,
        loop_forever=True,
        n_loops=1,
        verbose=True,
    ):
        if output_name is None:
            output_name = "Output"

        self.full_path = util.get_full_path(
            output_name,
            dir_name,
            file_ext="gif",
            verbose=verbose,
        )

        if loop_forever:
            n_loops = 0

        self._frame_list[0].save(
            self.full_path,
            format="gif",
            save_all=True,
            append_images=self._frame_list[1:],
            duration=frame_duration_ms,
            optimise=optimise,
            loop=n_loops,
        )

        return self.full_path

def set_latex_params(use_tex=True):
    latex_params_dict = {
        "font.family":          "serif",
        "font.serif":           ["Computer Modern"],
        "text.usetex":          True,
        "text.latex.preamble":  "\\usepackage{amsmath}",
        "legend.edgecolor":     "k",
        "legend.fancybox":      False,
        "legend.framealpha":    1,
    }

    for key, value in latex_params_dict.items():
        if use_tex:
            matplotlib.rcParams[key] = value
        else:
            matplotlib.rcParams[key] = matplotlib.rcParamsDefault[key]
