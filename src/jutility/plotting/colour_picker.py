import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from jutility.plotting.plottable import Plottable, Line
from jutility.plotting.subplot.colour_bar import ColourBar

class ColourPicker:
    """
    See https://matplotlib.org/stable/users/explain/colors/colormaps.html
    """
    def __init__(
        self,
        num_colours,
        cyclic=True,
        cmap_name=None,
        offset:         (float | None)=None,
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
        cmap_sample_points = np.linspace(0, 1, num_colours, endpoint)

        if offset is not None:
            cmap_sample_points += offset
            cmap_sample_points %= 1.0

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

    def get_legend_lines(self, *labels: str, **kwargs) -> list[Line]:
        return [
            Line(c=c, label=s, **kwargs)
            for c, s in zip(self, labels)
        ]

    def __iter__(self):
        return iter(self._colours)
