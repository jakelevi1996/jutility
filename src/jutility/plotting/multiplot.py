import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import matplotlib.backends.backend_agg
import numpy as np
import PIL.Image
from jutility import util
from jutility.plotting.temp_axis import _temp_axis
from jutility.plotting.subplot.subplot import Subplot
from jutility.plotting.subplot.empty import Empty
from jutility.plotting.figure.fig_props import FigureProperties
from jutility.plotting.figure.grid_props import GridProperties

class MultiPlot(Subplot):
    def __init__(
        self,
        *subplots: Subplot,
        **kwargs,
    ):
        """
        See [`jutility.plotting.GridProperties`](
        https://github.com/jakelevi1996/jutility/blob/main/src/jutility/plotting/figure/grid_props.py
        ) and [`jutility.plotting.FigureProperties`](
        https://github.com/jakelevi1996/jutility/blob/main/src/jutility/plotting/figure/fig_props.py
        )
        """
        fig_kwargs, grid_kwargs = FigureProperties.get_figure_kwargs(kwargs)

        self._subplots      = subplots
        self._kwargs        = kwargs
        self._fig_kwargs    = fig_kwargs
        self._grid_kwargs   = grid_kwargs
        self._fig           = None
        self._full_path     = None

    def save(
        self,
        plot_name:  (str | None)=None,
        dir_name:   (str | None)=None,
        file_ext:   (str | None)=None,
        pdf:        bool=False,
        verbose:    bool=True,
        close:      bool=True,
    ) -> str:
        if plot_name is None:
            plot_name = "output"
        if file_ext is None:
            file_ext = "pdf" if pdf else "png"

        self._full_path = util.get_full_path(
            plot_name,
            dir_name,
            file_ext=file_ext,
            verbose=verbose,
        )
        self._make_figure()
        self._fig.savefig(self._full_path)

        if close:
            self.close()

        return self._full_path

    def get_full_path(self) -> str | None:
        return self._full_path

    def _make_figure(self):
        if self._fig is not None:
            return

        fig_props = FigureProperties(**self._fig_kwargs)
        self._fig = fig_props.get_figure()
        self.plot_fig(self._fig)

        fig_props.apply(self._fig)
        fig_props.check_unused_kwargs()

    def plot_axis(self, axis: matplotlib.axes.Axes):
        raise NotImplementedError()

    def plot_fig(self, fig: matplotlib.figure.Figure):
        all_leaves = all(sp.is_leaf() for sp in self._subplots)

        grid_props = GridProperties(**self._grid_kwargs)
        num_empty = grid_props.init_size(len(self._subplots))
        subplots_pad  = tuple(Empty() for _ in range(num_empty))
        subplots_grid = self._subplots + subplots_pad

        if all_leaves:
            axes = grid_props.get_axes(fig)
            for subplot, axis in zip(subplots_grid, axes):
                subplot.plot_axis(axis)
        else:
            subfigs = grid_props.get_subfigs(fig)
            for subplot, subfig in zip(subplots_grid, subfigs):
                subplot.plot_fig(subfig)

        grid_props.apply(fig)
        grid_props.check_unused_kwargs()

    def is_leaf(self) -> bool:
        return False

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)

    def show(self):
        self._make_figure()
        _temp_axis.close()
        plt.show()

    def _get_canvas(self) -> matplotlib.backends.backend_agg.FigureCanvasAgg:
        self._make_figure()
        self._fig.canvas.draw()
        return self._fig.canvas

    def get_rgb_bytes(self) -> tuple[bytes, int, int]:
        canvas = self._get_canvas()
        rgb_bytes = canvas.tostring_rgb()
        width, height = canvas.get_width_height()
        return rgb_bytes, width, height

    def get_rgba_bytes(self) -> tuple[bytes, int, int]:
        canvas = self._get_canvas()
        rgba_bytes = canvas.buffer_rgba().tobytes()
        width, height = canvas.get_width_height()
        return rgba_bytes, width, height

    def get_pil_image(self) -> PIL.Image.Image:
        rgba_bytes, width, height = self.get_rgba_bytes()
        pil_image = PIL.Image.frombytes(
            mode="RGBA",
            size=[width, height],
            data=rgba_bytes,
        )
        return pil_image

    def get_image_array(self) -> np.ndarray:
        return np.array(self.get_pil_image())
