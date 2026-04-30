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
        *lines:     Plottable,
        num_rows:   (int | None)=1,
        loc:        (str | None)="outside lower center",
        **legend_kwargs,
    ):
        self._lines = Legend.filter_plottables(lines)
        self._kwargs = legend_kwargs
        self._kwargs["loc"] = loc
        if (num_rows is not None) and (len(self._lines) > 0):
            self._kwargs["ncols"] = math.ceil(len(self._lines) / num_rows)
        if len(self._lines) > 0:
            self._kwargs.update(Legend.get_kwargs(self._lines))

    @classmethod
    def centre_right(
        cls,
        *lines:     Plottable,
        num_rows:   (int | None)=None,
        loc:        (str | None)="outside center right",
        **legend_kwargs,
    ) -> "FigureLegend":
        return cls(
            *lines,
            num_rows=num_rows,
            loc=loc,
            **legend_kwargs,
        )

    @classmethod
    def bottom_centre(
        cls,
        *lines:     Plottable,
        num_rows:   (int | None)=1,
        loc:        (str | None)="outside lower center",
        **legend_kwargs,
    ) -> "FigureLegend":
        return cls(
            *lines,
            num_rows=num_rows,
            loc=loc,
            **legend_kwargs,
        )

    def plot(self, figure: matplotlib.figure.Figure):
        figure.legend(**self._kwargs)
