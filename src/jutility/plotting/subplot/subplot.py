import matplotlib.axes
import matplotlib.figure
from jutility import util
from jutility.plotting.plottable import Plottable
from jutility.plotting.subplot.axis_props import AxisProperties

class Subplot:
    def __init__(
        self,
        *lines: Plottable,
        **axis_kwargs,
    ):
        self._lines  = lines
        self._kwargs = axis_kwargs

    def plot_axis(self, axis: matplotlib.axes.Axes):
        for line in self._lines:
            line.plot(axis)

        properties = AxisProperties(**self._kwargs)
        properties.apply(axis)
        properties.check_unused_kwargs()

    def plot_fig(self, fig: matplotlib.figure.Figure):
        axis = fig.subplots(nrows=1, ncols=1, squeeze=True)
        self.plot_axis(axis)

    def set_options(self, **kwargs):
        self._kwargs.update(kwargs)

    def is_leaf(self) -> bool:
        return True

    def __repr__(self):
        return util.format_type(type(self), **self._kwargs)
