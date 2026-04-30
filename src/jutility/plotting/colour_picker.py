import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from jutility.plotting.plottable import Plottable, Line, PlottableGroup
from jutility.plotting.subplot.colour_bar import ColourBar
import jutility.plotting.noisy.sweep as jsweep

class ColourPicker:
    """
    See https://matplotlib.org/stable/users/explain/colors/colormaps.html
    """
    def __init__(
        self,
        num_colours:    int,
        cyclic:         bool=True,
        cmap_name:      (str | None)=None,
        offset:         (float | None)=None,
        colour_list:    (list | None)=None,
    ):
        if colour_list is None:
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

            colour_list = [self._cmap(i) for i in cmap_sample_points]
        else:
            self._cmap = matplotlib.colors.ListedColormap(colour_list)

        self._colours = colour_list
        self.reset()

    @classmethod
    def ibm(cls):
        """
        See https://davidmathlogic.com/colorblind/
        """
        return ColourPicker(
            num_colours=5,
            colour_list=[
                "#DC267F",
                "#648FFF",
                "#785EF0",
                "#FFB000",
                "#FE6100",
            ],
        )

    @classmethod
    def ibm_2_colour(cls):
        """
        See https://davidmathlogic.com/colorblind/
        """
        return ColourPicker(
            num_colours=2,
            colour_list=[
                "#648FFF",
                "#DC267F",
            ],
        )

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

    def get_legend_sweeps(
        self,
        *labels: str,
        **kwargs,
    ) -> list[PlottableGroup]:
        ns = jsweep.NoisySweep()
        for s in labels:
            ns.update(s, 0, 0)

        return ns.plot(self, **kwargs)

    def __iter__(self):
        return iter(self._colours)
