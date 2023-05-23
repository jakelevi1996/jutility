"""
MIT License

Copyright (c) 2022 JAKE LEVI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import math
import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.patches
import matplotlib.colors
import matplotlib.cm
import numpy as np
import PIL
from jutility import util

class Line:
    def __init__(self, x=None, y=None, **kwargs):
        if (y is None) and (x is not None):
            y = x
            x = None
        if (x is None) and (y is not None):
            x = np.arange(np.array(y).shape[0])

        self._x = x
        self._y = y
        self._kwargs = kwargs

    def plot(self, axis):
        if (self._x is not None) and (self._y is not None):
            axis.plot(self._x, self._y, **self._kwargs)

    def has_label(self):
        return ("label" in self._kwargs)

    def get_handle(self):
        if self.has_label():
            return self._get_handle_from_kwargs(self._kwargs)

    def _get_handle_from_kwargs(self, kwargs):
        return matplotlib.lines.Line2D([], [], **kwargs)

class Scatter(Line):
    def plot(self, axis):
        axis.scatter(self._x, self._y, **self._kwargs)

    def _get_handle_from_kwargs(self, kwargs):
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("ls", "")
        return matplotlib.lines.Line2D([], [], **kwargs)

class HVLine(Line):
    def __init__(self, h=None, v=None, **kwargs):
        self._h = h
        self._v = v
        self._kwargs = kwargs

    def plot(self, axis):
        if self._h is not None:
            axis.axhline(self._h, **self._kwargs)
        if self._v is not None:
            axis.axvline(self._v, **self._kwargs)

class Quiver(Line):
    def __init__(self, x, y, u, v, **kwargs):
        self._x = x
        self._y = y
        self._u = u
        self._v = v
        self._kwargs = kwargs

    def plot(self, axis):
        axis.quiver(self._x, self._y, self._u, self._v, **self._kwargs)

class Step(Line):
    def plot(self, axis):
        axis.step(self._x, self._y, **self._kwargs)

class Contour(Line):
    def __init__(self, x, y, z, levels, **kwargs):
        self._x = x
        self._y = y
        self._z = z
        self._levels = levels
        self._kwargs = kwargs

    def plot(self, axis):
        axis.contour(self._x, self._y, self._z, self._levels, **self._kwargs)

class FillBetween(Line):
    def __init__(self, x, y1, y2, **kwargs):
        self._x = x
        self._y1 = y1
        self._y2 = y2
        self._kwargs = kwargs

    def plot(self, axis):
        axis.fill_between(self._x, self._y1, self._y2, **self._kwargs)

    def get_handle(self):
        if self.has_label():
            return matplotlib.patches.Patch(**self._kwargs)

class Bar(FillBetween):
    def __init__(self, x, height, **kwargs):
        self._x = x
        self._height = height
        self._kwargs = kwargs

    def plot(self, axis):
        axis.bar(self._x, self._height, **self._kwargs)

class HVSpan(FillBetween):
    def __init__(self, xlims=None, ylims=None, **kwargs):
        self._xlims = xlims
        self._ylims = ylims
        self._kwargs = kwargs

    def plot(self, axis):
        if self._xlims is not None:
            axis.axvspan(*self._xlims, **self._kwargs)
        if self._ylims is not None:
            axis.axhspan(*self._ylims, **self._kwargs)

class ColourMesh(FillBetween):
    def __init__(self, x, y, c, **kwargs):
        self._x = x
        self._y = y
        self._c = c
        self._kwargs = kwargs

    def plot(self, axis):
        axis.pcolormesh(self._x, self._y, self._c, **self._kwargs)

class ContourFilled(FillBetween):
    def __init__(self, x, y, c, **kwargs):
        self._x = x
        self._y = y
        self._c = c
        self._kwargs = kwargs

    def plot(self, axis):
        axis.contourf(self._x, self._y, self._c, **self._kwargs)

class ImShow(FillBetween):
    def __init__(self, c, **kwargs):
        self._c = c
        self._kwargs = kwargs

    def plot(self, axis):
        axis.imshow(self._c, **self._kwargs)

class NoisyData:
    def __init__(self):
        self._results_list_dict = dict()

    def update(self, x, y):
        if x in self._results_list_dict:
            self._results_list_dict[x].append(y)
        else:
            self._results_list_dict[x] = [y]

    def get_lines(
        self,
        n_sigma=2,
        colour="b",
        name="Result",
        result_alpha=0.3,
        results_line_kwargs=None,
        mean_line_kwargs=None,
        std_line_kwargs=None,
    ):
        all_results_pairs = [
            [x, y]
            for x, result_list in self._results_list_dict.items()
            for y in result_list
        ]
        all_x, all_y = zip(*all_results_pairs)

        x_list = [
            x for x in self._results_list_dict.keys()
            if len(self._results_list_dict[x]) > 0
        ]
        results_list_list = [self._results_list_dict[x] for x in x_list]
        mean_array = np.array([np.mean(y) for y in results_list_list])
        std_array  = np.array([np.std( y) for y in results_list_list])

        if results_line_kwargs is None:
            results_line_kwargs = {
                "c":        colour,
                "label":    name,
                "alpha":    result_alpha,
                "zorder":   20,
            }
        if mean_line_kwargs is None:
            mean_line_kwargs = {
                "c":        colour,
                "label":    "Mean",
                "zorder":   30,
            }
        if std_line_kwargs is None:
            std_line_kwargs = {
                "color":    colour,
                "label":    "$\\pm %s \\sigma$" % n_sigma,
                "alpha":    0.3,
                "zorder":   10,
            }
        results_line = Scatter(all_x, all_y, **results_line_kwargs)
        mean_line = Line(x_list, mean_array, **mean_line_kwargs)
        std_line = FillBetween(
            x_list,
            mean_array + (n_sigma * std_array),
            mean_array - (n_sigma * std_array),
            **std_line_kwargs,
        )
        return results_line, mean_line, std_line

class ColourPicker:
    def __init__(self, num_colours, cyclic=True, cmap_name=None):
        if cmap_name is None:
            if cyclic:
                cmap_name = "hsv"
            else:
                cmap_name = "cool"
        if cyclic:
            endpoint = False
        else:
            endpoint = True

        cmap = plt.get_cmap(cmap_name)
        cmap_sample_points = np.linspace(0, 1, num_colours, endpoint)
        self._colours = [cmap(i) for i in cmap_sample_points]

    def __call__(self, colour_ind):
        return self._colours[colour_ind]

class AxisProperties:
    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        log_xscale=False,
        log_yscale=False,
        rotate_xticklabels=False,
        axis_equal=False,
        axis_off=False,
        grid=True,
        title=None,
        wrap_title=True,
        colour=None,
    ):
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._xlim = xlim
        self._ylim = ylim
        self._log_xscale = log_xscale
        self._log_yscale = log_yscale
        self._rotate_xticklabels = rotate_xticklabels
        self._axis_equal = axis_equal
        self._axis_off = axis_off
        self._grid = grid
        self._title = title
        self._wrap_title = wrap_title
        self._colour = colour

    def set_title(self, title):
        self._title = title

    def apply(self, axis):
        if self._xlabel is not None:
            axis.set_xlabel(self._xlabel)
        if self._ylabel is not None:
            axis.set_ylabel(self._ylabel)
        if self._xlim is not None:
            axis.set_xlim(self._xlim)
        if self._axis_equal:
            axis.axis("equal")
        if self._axis_off:
            axis.axis("off")
        if self._ylim is not None:
            axis.set_ylim(self._ylim)
        if self._log_xscale:
            axis.set_xscale("log")
        if self._log_yscale:
            axis.set_yscale("log")
        if self._grid:
            axis.grid(True, which="both")
        if self._rotate_xticklabels:
            for xtl in axis.get_xticklabels():
                xtl.set(rotation=-45, ha="left")
        if self._title is not None:
            if self._wrap_title:
                self._title = util.wrap_string(self._title)
            axis.set_title(self._title)
        if self._colour is not None:
            axis.set_facecolor(self._colour)

class FigureProperties:
    def __init__(
        self,
        num_rows=None,
        num_cols=None,
        figsize=None,
        sharex=False,
        sharey=False,
        width_ratios=None,
        height_ratios=None,
        tight_layout=True,
        colour=None,
        title=None,
        title_font_size=25,
        title_colour=None,
        wrap_title=True,
        top_space=None,
        layout=None,
    ):
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._figsize = figsize
        self._sharex = sharex
        self._sharey = sharey
        self._width_ratios = width_ratios
        self._height_ratios = height_ratios
        self._tight_layout = tight_layout
        self._colour = colour
        self._title = title
        self._title_font_size = title_font_size
        self._title_colour = title_colour
        self._wrap_title = wrap_title
        self._top_space = top_space
        self._layout = layout

    def get_figure_and_axes(self, num_subplots):
        if self._num_rows is None:
            if self._num_cols is None:
                self._num_cols = math.ceil(math.sqrt(num_subplots))
            self._num_rows = math.ceil(num_subplots / self._num_cols)
        if self._num_cols is None:
            self._num_cols = math.ceil(num_subplots / self._num_rows)

        if self._figsize is None:
            self._figsize = [6 * self._num_cols, 6 * self._num_rows]

        gridspec_kw = {
            "width_ratios": self._width_ratios,
            "height_ratios": self._height_ratios,
        }
        figure, axes = plt.subplots(
            self._num_rows,
            self._num_cols,
            figsize=self._figsize,
            sharex=self._sharex,
            sharey=self._sharey,
            gridspec_kw=gridspec_kw,
            squeeze=False,
            layout=self._layout,
        )
        return figure, axes.flat

    def apply(self, figure):
        if self._tight_layout:
            figure.tight_layout()
        if self._colour is not None:
            figure.patch.set_facecolor(self._colour)
        if self._title is not None:
            if self._wrap_title:
                self._title = util.wrap_string(self._title)
            figure.suptitle(
                self._title,
                fontsize=self._title_font_size,
                color=self._title_colour,
            )
        if self._top_space is not None:
            figure.subplots_adjust(top=(1 - self._top_space))

class Subplot:
    def __init__(self, *lines, axis_properties=None):
        self._lines = lines
        if axis_properties is None:
            axis_properties = AxisProperties()
        self._axis_properties = axis_properties

    def plot(self, axis):
        for line in self._lines:
            line.plot(axis)

        self._axis_properties.apply(axis)

class Legend(Subplot):
    def __init__(self, *lines):
        self._lines = lines

    def plot(self, axis):
        handles = [
            line.get_handle() for line in self._lines if line.has_label()
        ]
        axis.legend(handles=handles, loc="center")
        axis.axis("off")

class ColourBar(Subplot):
    def __init__(self, vmin, vmax):
        norm = matplotlib.colors.Normalize(vmin, vmax)
        self._sm = matplotlib.cm.ScalarMappable(norm)
        self._axes = []
        self._colourbar = None

    def plot(self, axis):
        if self._colourbar is not None:
            self._colourbar.remove()

        axis.axis("off")
        self._axes.append(axis)
        self._colourbar = plt.colorbar(
            mappable=self._sm,
            ax=self._axes,
            fraction=1,
            location="left",
        )

class Empty(Subplot):
    def plot(self, axis):
        axis.axis("off")

def plot(
    *lines,
    axis_properties=None,
    legend=False,
    plot_name=None,
    dir_name=None,
    save_close=True,
):
    if axis_properties is None:
        axis_properties = AxisProperties()
    if plot_name is not None:
        axis_properties.set_title(plot_name)

    if legend:
        figsize = [10, 6]
        wr =  [1, 0.2]
        fig_properties = FigureProperties(1, 2, figsize, width_ratios=wr)
        multi_plot = MultiPlot(
            Subplot(*lines, axis_properties=axis_properties),
            Legend(*lines),
            figure_properties=fig_properties,
        )
    else:
        figsize = [8, 6]
        fig_properties = FigureProperties(1, 1, figsize)
        multi_plot = MultiPlot(
            Subplot(*lines, axis_properties=axis_properties),
            figure_properties=fig_properties,
        )

    if save_close:
        multi_plot.save(plot_name, dir_name)
        multi_plot.close()

    return multi_plot

class MultiPlot:
    def __init__(self, *subplots, figure_properties=None):
        if figure_properties is None:
            figure_properties = FigureProperties()

        self._fig, axes = figure_properties.get_figure_and_axes(len(subplots))

        if len(subplots) < len(axes):
            subplots += tuple([Empty()]) * (len(axes) - len(subplots))

        for subplot, axis in zip(subplots, axes):
            subplot.plot(axis)

        figure_properties.apply(self._fig)

        self.filename = None

    def save(
        self,
        plot_name=None,
        dir_name=None,
        verbose=True,
        file_ext="png",
    ):
        if plot_name is None:
            plot_name = "Output"

        self.filename = util.get_full_path(
            plot_name,
            dir_name,
            for_saving=True,
            file_ext=file_ext,
        )

        if verbose:
            print("Saving image in \"%s\"" % self.filename)

        self._fig.savefig(self.filename)

    def get_rgb_bytes(self):
        self._fig.canvas.draw()
        rgb_bytes       = self._fig.canvas.tostring_rgb()
        width, height   = self._fig.canvas.get_width_height()
        return rgb_bytes, width, height

    def close(self):
        plt.close(self._fig)

class Gif:
    def __init__(self):
        self._frame_list = []

    def add_pil_image_frame(self, pil_image):
        self._frame_list.append(pil_image)

    def add_rgb_bytes_frame(self, rgb_bytes, width, height):
        pil_image = PIL.Image.frombytes(
            mode="RGB",
            size=[width, height],
            data=rgb_bytes,
        )
        self.add_pil_image_frame(pil_image)

    def add_multiplot_frame(self, multi_plot):
        self.add_rgb_bytes_frame(*multi_plot.get_rgb_bytes())

    def add_plot_frame(self, *lines, save=False, **plot_kwargs):
        plot_kwargs.setdefault("save_close", False)
        mp = plot(*lines, **plot_kwargs)
        self.add_multiplot_frame(mp)
        if save:
            mp.save(plot_kwargs.get("plot_name"), plot_kwargs.get("dir_name"))
        mp.close()

    def add_rgb_array_frame(self, rgb_array):
        self.add_pil_image_frame(PIL.Image.fromarray(rgb_array))

    def add_image_file_frame(self, filename, dir_name=None):
        if dir_name is None:
            dir_name = util.RESULTS_DIR
        full_path = os.path.join(dir_name, filename)
        self.add_pil_image_frame(PIL.Image.open(full_path))

    def save(
        self,
        output_name=None,
        dir_name=None,
        frame_duration_ms=100,
        optimise=False,
        loop_forever=True,
        n_loops=1,
        verbose=True,
    ):
        if output_name is None:
            output_name = "Output"

        self.filename = util.get_full_path(
            output_name,
            dir_name,
            for_saving=True,
            file_ext="gif",
        )

        if loop_forever:
            n_loops = 0

        if verbose:
            print("Saving GIF in \"%s\"" % self.filename)

        self._frame_list[0].save(
            self.filename,
            format="gif",
            save_all=True,
            append_images=self._frame_list[1:],
            duration=frame_duration_ms,
            optimise=optimise,
            loop=n_loops,
        )
