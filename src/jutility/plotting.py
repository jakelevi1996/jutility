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
from jutility import util, properties

class Plottable:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = dict()
        self.set_options(**self._get_default_kwargs())
        self.set_options(**kwargs)

    def plot(self, axis: matplotlib.axes.Axes):
        raise NotImplementedError()

    def set_options(self, **kwargs):
        key_map = self._get_abbreviated_keys_dict()
        kwargs = {
            key_map.get(k, k): v
            for k, v in kwargs.items()
        }
        self._kwargs.update(kwargs)
        return self

    def _get_default_kwargs(self):
        return {"z": 10}

    def _get_abbreviated_keys_dict(self):
        return {
            "c": "color",
            "z": "zorder",
            "a": "alpha",
            "m": "marker",
        }

    def _get_handle_args(self):
        return [
            None if (a is None) else [np.nan]
            for a in (self._args if (len(self._args) > 0) else [[]])
        ]

    def get_handle(self):
        plot_args = self._args
        self._args = self._get_handle_args()
        self.plot(_temp_axis.get_axis())
        self._args = plot_args
        return tuple(_temp_axis.pop_artists())

    def get_label(self):
        return self._kwargs.get("label")

    def has_label(self):
        return (self.get_label() is not None)

    def __repr__(self):
        return util.format_type(type(self), **self._kwargs)

class Line(Plottable):
    """
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.plot(*self._args, **self._kwargs)

    def _get_default_kwargs(self):
        return {"z": 10, "c": "b"}

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

    def _get_handle_args(self):
        return [[np.nan, np.nan], [np.nan, np.nan]]

class Arrow(Line):
    """
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.arrow.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.arrow(*self._args, **self._kwargs)

    def _get_abbreviated_keys_dict(self):
        return {
            "c": "color",
            "z": "zorder",
            "a": "alpha",
            "hw": "head_width",
            "lih": "length_includes_head",
        }

    def _get_default_kwargs(self):
        return {"z": 10, "c": "b", "lih": True, "hw": 0.05}

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

class Circle(Plottable):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Circle.html
    and
    https://matplotlib.org/stable/gallery/shapes_and_collections/artist_reference.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        circle = matplotlib.patches.Circle(*self._args, **self._kwargs)
        axis.add_artist(circle)

    def _get_default_kwargs(self):
        default_lw = matplotlib.rcParams["lines.linewidth"]
        return {"z": 10, "ec": "b", "fc": "w", "lw": default_lw}

    def _get_handle_args(self):
        return [[np.nan, np.nan], np.nan]

class ErrorBar(Plottable):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.errorbar(*self._args, **self._kwargs)

    def get_handle(self):
        args = self._get_handle_args()
        handle = _temp_axis.get_axis().errorbar(*args, **self._kwargs)
        _temp_axis.pop_artists()
        return handle

    def _get_default_kwargs(self):
        return {"z": 10, "c": "k", "capsize": 5}

class Scatter(Plottable):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.scatter(*self._args, **self._kwargs)

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
        axis.text(*self._args, **self._kwargs)

    def _get_abbreviated_keys_dict(self):
        return {
            "c": "color",
            "z": "zorder",
            "a": "alpha",
            "fs": "fontsize",
            "ha": "horizontalalignment",
            "va": "verticalalignment",
        }

    def _get_default_kwargs(self):
        return {"z": 10, "ha": "center", "va": "center"}

class FillBetween(Plottable):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_between.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.fill_between(*self._args, **self._kwargs)

    def _get_default_kwargs(self):
        return {"z": 10, "c": "b", "ec": None}

class FillBetweenx(FillBetween):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.fill_betweenx.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.fill_betweenx(*self._args, **self._kwargs)

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

class Bar(Plottable):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
    """
    def plot(self, axis: matplotlib.axes.Axes):
        axis.bar(*self._args, **self._kwargs)

    def _get_default_kwargs(self):
        return {"z": 10, "c": "b", "ec": "k"}

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
        return {"z": 10, "fc": "b", "ec": "k", "lw": 5}

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
        if "plottables" in self._kwargs:
            plottables = self._kwargs.pop("plottables")
            plottables = self.filter_plottables(plottables)
            self._kwargs.update(self.get_kwargs(plottables))

        zorder = self._kwargs.pop("zorder", None)
        legend = axis.legend(*self._args, **self._kwargs)
        if zorder is not None:
            legend.set_zorder(zorder)

    @staticmethod
    def filter_plottables(plottables: list[Plottable]):
        return [p for p in plottables if p.has_label()]

    @staticmethod
    def get_kwargs(plottables: list[Plottable]):
        return {
            "handles": [p.get_handle() for p in plottables],
            "labels":  [p.get_label()  for p in plottables],
        }

    @classmethod
    def from_plottables(cls, *plottables: Plottable, **kwargs):
        return cls(plottables=plottables, **kwargs)

class PlottableGroup(Plottable):
    def __init__(
        self,
        *plottables: Plottable,
        label=None,
    ):
        self._plottables = plottables
        self._label = label

    def plot(self, axis):
        for plottable in self._plottables:
            plottable.plot(axis)

    def set_options(self, **kwargs):
        if "label" in kwargs:
            self._label = kwargs.pop("label")

        for plottable in self._plottables:
            plottable.set_options(**kwargs)

    def get_handle(self):
        return tuple(p.get_handle() for p in self._plottables)

    def get_label(self):
        return self._label

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
        self._results_list_dict: dict[float, list[float]] = dict()
        self._log_x = log_x
        self._log_y = log_y
        self._x_index = x_index
        self._x_index_list = []

    def update(self, x, y):
        if self._x_index:
            if x not in self._x_index_list:
                self._x_index_list.append(x)

            x = self._x_index_list.index(x)

        if x in self._results_list_dict:
            self._results_list_dict[x].append(y)
        else:
            self._results_list_dict[x] = [y]

    def get_all_data(self):
        all_results_pairs = [
            [x, y]
            for x, y_list in self._results_list_dict.items()
            for y in y_list
        ]
        n = len(all_results_pairs)
        xy_n2 = np.array(all_results_pairs).reshape(n, 2)
        x_n1, y_n1 = np.split(xy_n2, 2, axis=1)
        return x_n1.flatten(), y_n1.flatten()

    def get_statistics(self, n_sigma=1):
        x_list = sorted(
            x for x, y_list in self._results_list_dict.items()
            if len(y_list) > 0
        )
        y_list_list = [self._results_list_dict[x] for x in x_list]

        if self._log_y:
            y_list_list = [np.log(y_list) for y_list in y_list_list]

        mean, ucb, lcb = confidence_bounds(y_list_list, n_sigma)

        if self._log_y:
            mean, ucb, lcb = np.exp([mean, ucb, lcb])

        return np.array(x_list), mean, ucb, lcb

    def argmax(self):
        best_y_dict = {
            max(y_list): x
            for x, y_list in self._results_list_dict.items()
            if len(y_list) > 0
        }
        best_y = max(best_y_dict.keys())
        best_x = best_y_dict[best_y]
        best_repeat = self._results_list_dict[best_x].index(best_y)
        if self._x_index:
            best_x = self._x_index_list[best_x]

        return best_x, best_repeat, best_y

    def argmin(self):
        best_y_dict = {
            min(y_list): x
            for x, y_list in self._results_list_dict.items()
            if len(y_list) > 0
        }
        best_y = min(best_y_dict.keys())
        best_x = best_y_dict[best_y]
        best_repeat = self._results_list_dict[best_x].index(best_y)
        if self._x_index:
            best_x = self._x_index_list[best_x]

        return best_x, best_repeat, best_y

    def plot(self, c="b", label=None, n_sigma=1):
        x, mean, ucb, lcb = self.get_statistics(n_sigma)
        return PlottableGroup(
            Scatter(*self.get_all_data(),   a=0.5, z=20, color=c),
            Line(x, mean,                   a=1.0, z=30, color=c),
            FillBetween(x, lcb, ucb,        a=0.2, z=10, color=c),
            label=label,
        )

    def get_xtick_kwargs(self):
        if self._x_index:
            ticks = list(range(len(self._x_index_list)))
            labels = self._x_index_list
        else:
            ticks = sorted(self._results_list_dict.keys())
            labels = ticks

        return {"xticks": ticks, "xticklabels": labels}

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

    def __repr__(self):
        return util.format_type(type(self), self._results_list_dict)

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

    @classmethod
    def from_colourise(cls, plottables: list[Plottable], *args, **kwargs):
        self = cls(len(plottables), *args, **kwargs)
        self.colourise(plottables)
        return self

    def colourise(
        self,
        plottables: list[Plottable],
        colour_arg_name="color",
    ):
        for p in plottables:
            kwargs = {colour_arg_name: self.next()}
            p.set_options(**kwargs)

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

class AxisProperties(properties.PropertyDict):
    def apply(self, axis: matplotlib.axes.Axes):
        if self._has("xlabel"):
            axis.set_xlabel(self._get("xlabel"))
        if self._has("ylabel"):
            axis.set_ylabel(self._get("ylabel"))
        if self._get_default("axis_equal", False):
            axis.axis("equal")
        if self._get_default("axis_square", False):
            axis.axis("square")
        if self._get_default("axis_off", False):
            axis.set_axis_off()
        if self._get_default("log_x", False):
            axis.set_xscale("log")
        if self._get_default("log_y", False):
            axis.set_yscale("log")
        if self._get_default("symlog_x", False):
            axis.set_xscale("symlog")
        if self._get_default("symlog_y", False):
            axis.set_yscale("symlog")
        if self._get_default("grid", True):
            grid_x = self._get_default("grid_x", "both")
            grid_y = self._get_default("grid_y", "both")
            if grid_x:
                axis.grid(visible=True, which=grid_x, axis="x")
            if grid_y:
                axis.grid(visible=True, which=grid_y, axis="y")
        if self._has("xticks"):
            ticks  = self._get("xticks")
            labels = self._get_default("xticklabels", None)
            axis.set_xticks(ticks, labels)
        if self._has("yticks"):
            ticks  = self._get("yticks")
            labels = self._get_default("yticklabels", None)
            axis.set_yticks(ticks, labels)
        if self._has("xlim"):
            axis.set_xlim(self._get("xlim"))
        if self._has("ylim"):
            axis.set_ylim(self._get("ylim"))
        if self._get_default("rotate_xticklabels", False):
            for xtl in axis.get_xticklabels():
                xtl.set(rotation=-45, ha="left")
        title = self._get_default("title", None)
        if title is not None:
            colour_dict = dict()
            if self._has("title_colour"):
                colour_dict["color"] = self._get("title_colour")
            if self._get_default("wrap_title", True):
                title = util.wrap_string(title)
            axis.set_title(
                title,
                fontsize=self._get_default("title_font_size", 12),
                **colour_dict,
            )
        if self._has("colour"):
            axis.set_facecolor(self._get("colour"))

class FigureProperties:
    def __init__(
        self,
        num_subplots,
        num_rows=None,
        num_cols=None,
        figsize=None,
        sharex=False,
        sharey=False,
        width_ratios=None,
        height_ratios=None,
        constrained_layout=True,
        tight_layout=False,
        layout=None,
        colour=None,
        title=None,
        title_font_size=25,
        title_colour=None,
        title_wrap_len=None,
        top_space=None,
        bottom_space=None,
        legend: "FigureLegend"=None,
        pad=0.1,
        space=0,
        dpi=None,
    ):
        if num_rows is None:
            if num_cols is None:
                num_cols = math.ceil(math.sqrt(num_subplots))
            num_rows = math.ceil(num_subplots / num_cols)
        if num_cols is None:
            num_cols = math.ceil(num_subplots / num_rows)
        if figsize is None:
            figsize = [6 * num_cols, 4 * num_rows]
        if layout is not None:
            constrained_layout = False
            tight_layout = False
        if tight_layout:
            constrained_layout = False
            layout = None
        if constrained_layout:
            tight_layout = False
            layout = "constrained"
        if title_wrap_len is not None:
            title = util.wrap_string(
                title,
                max_len=title_wrap_len,
                wrap_len=title_wrap_len,
            )

        self._num_rows = num_rows
        self._num_cols = num_cols
        self._figsize = figsize
        self._sharex = sharex
        self._sharey = sharey
        self._width_ratios = width_ratios
        self._height_ratios = height_ratios
        self._constrained_layout = constrained_layout
        self._tight_layout = tight_layout
        self._layout = layout
        self._colour = colour
        self._title = title
        self._title_font_size = title_font_size
        self._title_colour = title_colour
        self._top_space = top_space
        self._bottom_space = bottom_space
        self._legend = legend
        self._pad = pad
        self._space = space
        self._dpi = dpi

    def get_num_axes(self):
        return self._num_rows * self._num_cols

    def get_figure(self):
        figure = plt.figure(
            figsize=self._figsize,
            dpi=self._dpi,
            layout=self._layout,
        )
        if self._constrained_layout:
            layout_engine = figure.get_layout_engine()
            layout_engine.set(
                w_pad=self._pad,
                h_pad=self._pad,
            )

        return figure

    def get_axes(
        self,
        figure: matplotlib.figure.Figure,
    ) -> list[matplotlib.axes.Axes]:
        axis_array = figure.subplots(
            nrows=self._num_rows,
            ncols=self._num_cols,
            sharex=self._sharex,
            sharey=self._sharey,
            squeeze=False,
            width_ratios=self._width_ratios,
            height_ratios=self._height_ratios,
            gridspec_kw=dict(
                wspace=self._space,
                hspace=self._space,
            ),
        )
        axis_list = axis_array.flatten().tolist()
        return axis_list

    def apply(self, figure: matplotlib.figure.Figure):
        if self._tight_layout:
            figure.tight_layout()
        if self._colour is not None:
            figure.patch.set_facecolor(self._colour)
        if self._title is not None:
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

    def get_subplot_axes(
        self,
        axis: matplotlib.axes.Axes,
    ) -> list[matplotlib.axes.Axes]:
        if self._title is not None:
            axis.set_title(self._title, fontsize=self._title_font_size)

        subplot_spec = axis.get_subplotspec()
        subgrid_spec = subplot_spec.subgridspec(
            nrows=self._num_rows,
            ncols=self._num_cols,
            width_ratios=self._width_ratios,
            height_ratios=self._height_ratios,
            wspace=self._space,
            hspace=self._space,
        )
        axes_array = subgrid_spec.subplots(
            sharex=self._sharex,
            sharey=self._sharey,
            squeeze=False,
        )
        axis_list = axes_array.flatten().tolist()
        return axis_list

class Subplot:
    def __init__(
        self,
        *lines: Plottable,
        **axis_kwargs,
    ):
        self._lines  = lines
        self._kwargs = axis_kwargs

    def plot(self, axis: matplotlib.axes.Axes):
        for line in self._lines:
            line.plot(axis)

        properties = AxisProperties(**self._kwargs)
        properties.apply(axis)
        properties.check_unused_kwargs()

    def set_options(self, **kwargs):
        self._kwargs.update(kwargs)

    def __repr__(self):
        return util.format_type(type(self), **self._kwargs)

class LegendSubplot(Subplot):
    """
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    """
    def __init__(self, *lines: Plottable, loc="center", **legend_kwargs):
        self._lines = Legend.filter_plottables(lines)
        self._kwargs = legend_kwargs
        self._kwargs["loc"] = loc

    def plot(self, axis: matplotlib.axes.Axes):
        if len(self._lines) > 0:
            self._kwargs.update(Legend.get_kwargs(self._lines))

        axis.legend(**self._kwargs)
        axis.set_axis_off()

class FigureLegend:
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.legend.html
    """
    def __init__(
        self,
        *lines: Plottable,
        num_rows=1,
        loc="outside lower center",
        **legend_kwargs,
    ):
        self._lines = Legend.filter_plottables(lines)
        self._kwargs = legend_kwargs
        self._kwargs["loc"] = loc
        if (num_rows is not None) and (len(self._lines) > 0):
            self._kwargs["ncols"] = math.ceil(len(self._lines) / num_rows)

    def plot(self, figure: matplotlib.figure.Figure):
        if len(self._lines) > 0:
            self._kwargs.update(Legend.get_kwargs(self._lines))

        figure.legend(**self._kwargs)

class ColourBar(Subplot):
    """
    See
    https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.colorbar.html
    """
    def __init__(
        self,
        vmin,
        vmax,
        cmap=None,
        horizontal=False,
        log=False,
        **kwargs,
    ):
        if log:
            norm = matplotlib.colors.LogNorm(vmin, vmax)
        else:
            norm = matplotlib.colors.Normalize(vmin, vmax)

        self._sm = matplotlib.cm.ScalarMappable(norm, cmap=cmap)
        self._kwargs = kwargs
        if horizontal:
            self._kwargs["orientation"] = "horizontal"

    def plot(self, axis: matplotlib.axes.Axes):
        axis.figure.colorbar(
            mappable=self._sm,
            cax=axis,
            **self._kwargs,
        )

class Empty(Subplot):
    def plot(self, axis: matplotlib.axes.Axes):
        axis.set_axis_off()

def plot(
    *lines: Plottable,
    legend=False,
    figsize=None,
    plot_name=None,
    dir_name=None,
    save_close=True,
    pdf=False,
    **axis_kwargs,
):
    if figsize is None:
        figsize = [10, 6] if legend else [8, 6]

    axis_kwargs.setdefault("title", plot_name)

    if legend:
        multi_plot = MultiPlot(
            Subplot(*lines, **axis_kwargs),
            LegendSubplot(*lines),
            figsize=figsize,
            width_ratios=[1, 0.2],
        )
    else:
        multi_plot = MultiPlot(
            Subplot(*lines, **axis_kwargs),
            figsize=figsize,
        )

    if save_close:
        multi_plot.save(plot_name, dir_name, pdf=pdf)

    return multi_plot

class MultiPlot(Subplot):
    def __init__(
        self,
        *subplots: Subplot,
        **figure_kwargs,
    ):
        figure_properties = FigureProperties(len(subplots), **figure_kwargs)
        num_axes = figure_properties.get_num_axes()
        if len(subplots) < num_axes:
            subplots += tuple([Empty()]) * (num_axes - len(subplots))

        self.full_path      = None
        self._fig           = None
        self._properties    = figure_properties
        self._subplots      = subplots

    def _make_figure(self):
        if self._fig is not None:
            return

        fig  = self._properties.get_figure()
        axes = self._properties.get_axes(fig)

        for subplot, axis in zip(self._subplots, axes):
            subplot.plot(axis)

        self._properties.apply(fig)
        self._fig = fig

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
        self._make_figure()
        self._fig.savefig(self.full_path)

        if close:
            self.close()

        return self.full_path

    def plot(self, axis: matplotlib.axes.Axes):
        axis.set_axis_off()
        axis_list = self._properties.get_subplot_axes(axis)
        for subplot, axis in zip(self._subplots, axis_list):
            subplot.plot(axis)

    def get_rgb_bytes(self):
        self._make_figure()
        self._fig.canvas.draw()
        rgb_bytes       = self._fig.canvas.tostring_rgb()
        width, height   = self._fig.canvas.get_width_height()
        return rgb_bytes, width, height

    def get_rgba_bytes(self):
        self._make_figure()
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
        if self._fig is not None:
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

    def shuffle(self, rng: np.random.Generator=None):
        if rng is None:
            rng = np.random.default_rng()

        perm = rng.permutation(len(self._frame_list)).tolist()
        self._frame_list = [self._frame_list[i] for i in perm]

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

class _TempAxis:
    def __init__(self):
        self._axis = None
        self._old_children = None

    def get_axis(self):
        if self._axis is None:
            self._axis = plt.figure().gca()
            self._old_children = set(self._axis.get_children())

        return self._axis

    def pop_artists(self):
        new_children = set(self._axis.get_children()) - self._old_children
        for a in new_children:
            a.remove()

        return new_children

_temp_axis = _TempAxis()
