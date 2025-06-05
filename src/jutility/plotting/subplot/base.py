import matplotlib.axes
from jutility import util
from jutility.plotting.plottable import Plottable
from jutility.plotting.axis_props import AxisProperties

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
