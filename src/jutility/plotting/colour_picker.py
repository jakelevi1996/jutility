import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from jutility.plotting.plottable import Plottable, Line, PlottableGroup
from jutility.plotting.subplot.colour_bar import ColourBar
import jutility.plotting.noisy.sweep

class ColourPicker:
    """
    See https://matplotlib.org/stable/users/explain/colors/colormaps.html
    """
    def __init__(
        self,
        colour_list:    list[tuple | float | str],
        cmap:           matplotlib.colors.Colormap,
    ):
        self._colours = colour_list
        self._cmap = cmap
        self._num_colours = len(colour_list)
        self.reset()

    @classmethod
    def cool(cls, num_colours: int) -> "ColourPicker":
        return cls.from_linear_cmap("cool", num_colours)

    @classmethod
    def hsv(
        cls,
        num_colours:    int,
        offset:         (float | None)=None,
    ) -> "ColourPicker":
        return cls.from_cyclic_cmap("hsv", num_colours, offset)

    @classmethod
    def contrast(cls) -> "ColourPicker":
        return cls.from_colour_list(
            "#ff0000",
            "#0000ff",
            "#00ff00",
            "#9900ff",
            "#0099ff",
            "#ff9900",
        )

    @classmethod
    def ibm(cls) -> "ColourPicker":
        """
        See https://davidmathlogic.com/colorblind/
        """
        return cls.from_colour_list(
            "#DC267F",
            "#648FFF",
            "#785EF0",
            "#FFB000",
            "#FE6100",
        )

    @classmethod
    def from_colour_list(
        cls,
        *colours: (tuple | float | str),
    ) -> "ColourPicker":
        cmap = matplotlib.colors.ListedColormap(colours)
        return cls(list(colours), cmap)

    @classmethod
    def from_linear_cmap(
        cls,
        cmap_name:      str,
        num_colours:    int,
    ) -> "ColourPicker":
        sample_points = np.linspace(0, 1, num_colours)
        return cls.from_cmap(cmap_name, sample_points)

    @classmethod
    def from_cyclic_cmap(
        cls,
        cmap_name:      str,
        num_colours:    int,
        offset:         (float | None)=None,
    ) -> "ColourPicker":
        sample_points = np.linspace(0, 1, num_colours, endpoint=False)
        if offset is not None:
            sample_points += offset
            sample_points %= 1.0

        return cls.from_cmap(cmap_name, sample_points)

    @classmethod
    def from_cmap(
        cls,
        cmap_name:      str,
        sample_points:  list[float],
    ) -> "ColourPicker":
        cmap = plt.get_cmap(cmap_name)
        colour_list = [cmap(x) for x in sample_points]
        return cls(colour_list, cmap)

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
        ns = jutility.plotting.noisy.sweep.NoisySweep()
        for s in labels:
            ns.update(s, 0, 0)

        return ns.plot(self, **kwargs)

    def __iter__(self):
        return iter(self._colours)

    def __call__(self, colour_ind: int):
        return self._colours[colour_ind]

    def __len__(self) -> int:
        return self._num_colours
