import matplotlib.colors
import matplotlib.cm
import matplotlib.axes
from jutility.plotting.subplot.subplot import Subplot

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

    def plot_axis(self, axis: matplotlib.axes.Axes):
        axis.figure.colorbar(
            mappable=self._sm,
            cax=axis,
            **self._kwargs,
        )
