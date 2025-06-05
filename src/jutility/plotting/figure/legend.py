import math
import matplotlib.figure
from jutility.plotting.plottable import Plottable, Legend

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
