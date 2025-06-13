import matplotlib
import matplotlib.axes
import matplotlib.patches
import numpy as np
from jutility import util
from jutility.plotting.temp_axis import _temp_axis

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
