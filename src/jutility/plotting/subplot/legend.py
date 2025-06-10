import matplotlib.axes
from jutility.plotting.plottable import Plottable, Legend
from jutility.plotting.subplot.subplot import Subplot

class LegendSubplot(Subplot):
    """
    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html
    """
    def __init__(self, *lines: Plottable, loc="center", **legend_kwargs):
        self._lines = Legend.filter_plottables(lines)
        self._kwargs = legend_kwargs
        self._kwargs["loc"] = loc

    def plot_axis(self, axis: matplotlib.axes.Axes):
        if len(self._lines) > 0:
            self._kwargs.update(Legend.get_kwargs(self._lines))

        axis.legend(**self._kwargs)
        axis.set_axis_off()
