import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import numpy as np
import PIL.Image
from jutility import util
from jutility.plotting.temp_axis import _temp_axis
from jutility.plotting.subplot.base import Subplot
from jutility.plotting.subplot.empty import Empty
from jutility.plotting.figure.props import FigureProperties

class MultiPlot(Subplot):
    def __init__(
        self,
        *subplots: Subplot,
        **figure_kwargs,
    ):
        figure_properties = FigureProperties(len(subplots), **figure_kwargs)
        num_axes = figure_properties.get_num_axes()
        if len(subplots) < num_axes:
            subplots += tuple([Empty()]) * (num_axes - len(subplots))

        self.full_path      = None
        self._fig           = None
        self._properties    = figure_properties
        self._subplots      = subplots
        self._kwargs        = figure_kwargs

    def _make_figure(self):
        if self._fig is not None:
            return

        fig = self._properties.get_figure()
        self.plot_fig(fig)

        self._properties.apply(fig)
        self._fig = fig

    def save(
        self,
        plot_name=None,
        dir_name=None,
        verbose=True,
        file_ext=None,
        pdf=False,
        close=True,
    ):
        if plot_name is None:
            plot_name = "output"
        if file_ext is None:
            file_ext = "pdf" if pdf else "png"

        self.full_path = util.get_full_path(
            plot_name,
            dir_name,
            file_ext=file_ext,
            verbose=verbose,
        )
        self._make_figure()
        self._fig.savefig(self.full_path)

        if close:
            self.close()

        return self.full_path

    def show(self):
        self._make_figure()
        _temp_axis.close()
        plt.show()

    def plot_axis(self, axis: matplotlib.axes.Axes):
        raise NotImplementedError()

    def plot_fig(self, fig: matplotlib.figure.Figure):
        all_leaves = all(sp.is_leaf() for sp in self._subplots)

        if all_leaves:
            axes = self._properties.get_axes(fig)
            for subplot, axis in zip(self._subplots, axes):
                subplot.plot_axis(axis)
        else:
            subfigs = self._properties.get_subfigs(fig)
            for subplot, subfig in zip(self._subplots, subfigs):
                subplot.plot_fig(subfig)

    def is_leaf(self) -> bool:
        return False

    def get_rgb_bytes(self):
        self._make_figure()
        self._fig.canvas.draw()
        rgb_bytes       = self._fig.canvas.tostring_rgb()
        width, height   = self._fig.canvas.get_width_height()
        return rgb_bytes, width, height

    def get_rgba_bytes(self):
        self._make_figure()
        self._fig.canvas.draw()
        rgba_bytes      = self._fig.canvas.buffer_rgba().tobytes()
        width, height   = self._fig.canvas.get_width_height()
        return rgba_bytes, width, height

    def get_pil_image(self):
        rgba_bytes, width, height = self.get_rgba_bytes()
        pil_image = PIL.Image.frombytes(
            mode="RGBA",
            size=[width, height],
            data=rgba_bytes,
        )
        return pil_image

    def get_image_array(self):
        return np.array(self.get_pil_image())

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
