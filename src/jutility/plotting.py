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
    def __init__(self, x=None, y=None, batch_first=False, **kwargs):
        if (y is None) and (x is not None):
            y = x
            x = None
        if y is not None:
            y = np.array(y)
            if batch_first:
                y = y.T
            if x is None:
                x = np.arange(y.shape[0])

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
    def __init__(self, x, y, u, v, normalise=False, tol=1e-5, **kwargs):
        self._x = x
        self._y = y

        if normalise:
            dr = np.sqrt(np.square(u) + np.square(v))
            dr_safe = np.maximum(dr, tol)
            self._u = u / dr_safe
            self._v = v / dr_safe
        else:
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

        c = kwargs.pop("c", None)
        if c is not None:
            kwargs["color"] = c

        self._kwargs = kwargs

    def plot(self, axis):
        axis.fill_between(self._x, self._y1, self._y2, **self._kwargs)

    def get_handle(self):
        if self.has_label():
            return matplotlib.patches.Patch(**self._kwargs)

class Text(Line):
    def __init__(self, *args, center_align=False, **kwargs):
        """
        See
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html

        Further examples:
        https://matplotlib.org/stable/gallery/text_labels_and_annotations/fancytextbox_demo.html
        https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html
        """
        if center_align:
            kwargs["horizontalalignment"]   = "center"
            kwargs["verticalalignment"]     = "center"

        self._args = args
        self._kwargs = kwargs

    def plot(self, axis):
        axis.text(*self._args, **self._kwargs)

class Bar(FillBetween):
    def __init__(self, x, height, **kwargs):
        self._x = x
        self._height = height
        self._kwargs = kwargs

    def plot(self, axis):
        axis.bar(self._x, self._height, **self._kwargs)

class Hist(FillBetween):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def plot(self, axis):
        axis.hist(*self._args, **self._kwargs)

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
    def __init__(self, c, axis_off=True, **kwargs):
        self._c = c
        self._axis_off = axis_off
        self._kwargs = kwargs

    def plot(self, axis):
        axis.imshow(self._c, **self._kwargs)
        if self._axis_off:
            axis.set_axis_off()

def get_noisy_data_lines(
    noisy_data,
    n_sigma=1,
    colour="b",
    name="Result",
    result_alpha=0.3,
    results_line_kwargs=None,
    mean_line_kwargs=None,
    std_line_kwargs=None,
    plot_all_data=True,
    mean_std_labels=True,
):
    line_list = []
    if plot_all_data:
        all_x, all_y = noisy_data.get_all_data()
        if results_line_kwargs is None:
            results_line_kwargs = {
                "color":    colour,
                "label":    name,
                "alpha":    result_alpha,
                "zorder":   20,
            }
        results_line = Scatter(all_x, all_y, **results_line_kwargs)
        line_list.append(results_line)

    x, mean, ucb, lcb = noisy_data.get_statistics(n_sigma)
    if mean_line_kwargs is None:
        mean_line_kwargs = {
            "color":    colour,
            "zorder":   30,
        }
        if mean_std_labels:
            mean_line_kwargs["label"] = "Mean"
    if std_line_kwargs is None:
        std_line_kwargs = {
            "color":    colour,
            "alpha":    0.3,
            "zorder":   10,
        }
        if mean_std_labels:
            std_line_kwargs["label"] = "$\\pm %s \\sigma$" % n_sigma
    mean_line = Line(x, mean, **mean_line_kwargs)
    std_line = FillBetween(x, ucb, lcb, **std_line_kwargs)
    line_list.append(mean_line)
    line_list.append(std_line)
    return line_list

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
        self._index = 0

    def __call__(self, colour_ind):
        return self._colours[colour_ind]

    def next(self):
        c = self._colours[self._index]
        self._index += 1
        if self._index >= len(self._colours):
            self._index = 0

        return c

    def colour_name(self, colour_ind):
        return "c%i" % colour_ind

    def define_colours(self):
        definition_list = [
            "\\definecolor{%s}{rgb}{%f,%f,%f}"
            % (self.colour_name(i), c[0], c[1], c[2])
            for i, c in enumerate(self._colours)
        ]
        definition_str = "\n".join(definition_list)
        return definition_str

class AxisProperties:
    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        log_xscale=False,
        log_yscale=False,
        symlogx=False,
        symlogy=False,
        rotate_xticklabels=False,
        axis_equal=False,
        axis_off=False,
        grid=True,
        title=None,
        wrap_title=True,
        colour=None,
        legend=False,
        legend_properties=None,
    ):
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._xlim = xlim
        self._ylim = ylim
        self._log_xscale = log_xscale
        self._log_yscale = log_yscale
        self._symlogx = symlogx
        self._symlogy = symlogy
        self._rotate_xticklabels = rotate_xticklabels
        self._axis_equal = axis_equal
        self._axis_off = axis_off
        self._grid = grid
        self._title = title
        self._wrap_title = wrap_title
        self._colour = colour
        self._legend = legend
        self._legend_properties = legend_properties

    def set_default_title(self, title):
        if self._title is None:
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
            axis.set_axis_off()
        if self._ylim is not None:
            axis.set_ylim(self._ylim)
        if self._log_xscale:
            axis.set_xscale("log")
        if self._log_yscale:
            axis.set_yscale("log")
        if self._symlogx:
            axis.set_xscale("symlog")
        if self._symlogy:
            axis.set_yscale("symlog")
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
        if self._legend or (self._legend_properties is not None):
            if self._legend_properties is None:
                self._legend_properties = LegendProperties()
            self._legend_properties.apply(axis)

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
        constrained_layout=False,
        colour=None,
        title=None,
        title_font_size=25,
        title_colour=None,
        wrap_title=True,
        top_space=None,
    ):
        self._num_rows = num_rows
        self._num_cols = num_cols
        self._figsize = figsize
        self._sharex = sharex
        self._sharey = sharey
        self._width_ratios = width_ratios
        self._height_ratios = height_ratios

        if constrained_layout:
            tight_layout = False
            self._layout = "constrained"
        else:
            self._layout = None

        self._tight_layout = tight_layout
        self._colour = colour
        self._title = title
        self._title_font_size = title_font_size
        self._title_colour = title_colour
        self._wrap_title = wrap_title
        self._top_space = top_space

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

class LegendProperties:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def apply(self, axis):
        axis.legend(*self._args, **self._kwargs)

class Subplot:
    def __init__(self, *lines, axis_properties=None, **axis_kwargs):
        self._lines = lines
        if axis_properties is None:
            axis_properties = AxisProperties(**axis_kwargs)
        self._axis_properties = axis_properties

    def plot(self, axis):
        for line in self._lines:
            line.plot(axis)

        self._axis_properties.apply(axis)

class Legend(Subplot):
    def __init__(self, *lines, **legend_kwargs):
        handles = [
            line.get_handle() for line in lines if line.has_label()
        ]
        self._legend_properties = LegendProperties(
            handles=handles,
            loc="center",
            **legend_kwargs,
        )

    def plot(self, axis):
        self._legend_properties.apply(axis)
        axis.set_axis_off()

class ColourBar(Subplot):
    def __init__(self, vmin, vmax, cmap=None, **kwargs):
        norm = matplotlib.colors.Normalize(vmin, vmax)
        self._sm = matplotlib.cm.ScalarMappable(norm, cmap=cmap)
        self._axes = []
        self._colourbar = None
        self._kwargs = kwargs

    def plot(self, axis):
        if self._colourbar is not None:
            self._colourbar.remove()

        axis.set_axis_off()
        self._axes.append(axis)
        self._colourbar = plt.colorbar(
            mappable=self._sm,
            ax=self._axes,
            fraction=1,
            location="left",
            **self._kwargs,
        )

class Empty(Subplot):
    def plot(self, axis):
        axis.set_axis_off()

def plot(
    *lines,
    axis_properties=None,
    legend=False,
    figsize=None,
    plot_name=None,
    dir_name=None,
    save_close=True,
    pdf=False,
    **axis_kwargs,
):
    if axis_properties is None:
        axis_properties = AxisProperties(**axis_kwargs)
    if figsize is None:
        figsize = [10, 6] if legend else [8, 6]
    if plot_name is not None:
        axis_properties.set_default_title(plot_name)

    if legend:
        wr =  [1, 0.2]
        fig_properties = FigureProperties(1, 2, figsize, width_ratios=wr)
        multi_plot = MultiPlot(
            Subplot(*lines, axis_properties=axis_properties),
            Legend(*lines),
            figure_properties=fig_properties,
        )
    else:
        fig_properties = FigureProperties(1, 1, figsize)
        multi_plot = MultiPlot(
            Subplot(*lines, axis_properties=axis_properties),
            figure_properties=fig_properties,
        )

    if save_close:
        multi_plot.save(plot_name, dir_name, pdf=pdf)
        multi_plot.close()

    return multi_plot

class MultiPlot:
    def __init__(self, *subplots, figure_properties=None, **figure_kwargs):
        if figure_properties is None:
            figure_properties = FigureProperties(**figure_kwargs)

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
        file_ext=None,
        pdf=False,
    ):
        if plot_name is None:
            plot_name = "Output"
        if file_ext is None:
            file_ext = "pdf" if pdf else "png"

        self.filename = util.get_full_path(
            plot_name,
            dir_name,
            file_ext=file_ext,
            verbose=verbose,
        )

        self._fig.savefig(self.filename)

    def get_rgb_bytes(self):
        self._fig.canvas.draw()
        rgb_bytes       = self._fig.canvas.tostring_rgb()
        width, height   = self._fig.canvas.get_width_height()
        return rgb_bytes, width, height

    def get_rgba_bytes(self):
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

    def add_rgba_bytes_frame(self, rgba_bytes, width, height):
        pil_image = PIL.Image.frombytes(
            mode="RGBA",
            size=[width, height],
            data=rgba_bytes,
        )
        self.add_pil_image_frame(pil_image)

    def add_multiplot_frame(self, multi_plot):
        self.add_pil_image_frame(multi_plot.get_pil_image())

    def add_plot_frame(self, *lines, save=False, **plot_kwargs):
        plot_kwargs.setdefault("save_close", False)
        mp = plot(*lines, **plot_kwargs)
        self.add_multiplot_frame(mp)
        if save:
            mp.save(plot_kwargs.get("plot_name"), plot_kwargs.get("dir_name"))
        mp.close()

    def add_rgb_array_frame(self, ndarray_hwc, vmin=0, vmax=1):
        util.check_type(ndarray_hwc, np.ndarray, "ndarray_hwc")
        if (ndarray_hwc.ndim != 3) or (ndarray_hwc.shape[2] != 3):
            raise ValueError(
                "Expected shape (H, W, C=3), but received shape %s"
                % ndarray_hwc.shape
            )

        ndarray_scaled = 255 * (ndarray_hwc - vmin) / (vmax - vmin)
        ndarray_clipped = np.clip(ndarray_scaled, 0, 255)
        ndarray_int8 = ndarray_clipped.astype(np.uint8)
        pil_image = PIL.Image.fromarray(ndarray_int8, mode="RGB")
        self.add_pil_image_frame(pil_image)

    def add_bw_array_frame(self, ndarray_hw, vmin=0, vmax=1):
        util.check_type(ndarray_hw, np.ndarray, "ndarray_hw")
        if ndarray_hw.ndim != 2:
            raise ValueError(
                "Expected shape (H, W), but received shape %s"
                % ndarray_hw.shape
            )

        ndarray_scaled = 255 * (ndarray_hw - vmin) / (vmax - vmin)
        ndarray_clipped = np.clip(ndarray_scaled, 0, 255)
        ndarray_int8 = ndarray_clipped.astype(np.uint8)
        pil_image = PIL.Image.fromarray(ndarray_int8, mode="L")
        self.add_pil_image_frame(pil_image)

    def add_rgb_array_sequence(self, ndarray_lhwc, vmin=0, vmax=1):
        util.check_type(ndarray_lhwc, np.ndarray, "ndarray_lhwc")
        if (ndarray_lhwc.ndim != 4) or (ndarray_lhwc.shape[3] != 3):
            raise ValueError(
                "Expected shape (L, H, W, C=3), but received shape %s"
                % ndarray_lhwc.shape
            )

        for i in range(ndarray_lhwc.shape[0]):
            self.add_rgb_array_frame(ndarray_lhwc[i], vmin, vmax)

    def add_bw_array_sequence(self, ndarray_lhw, vmin=0, vmax=1):
        util.check_type(ndarray_lhw, np.ndarray, "ndarray_lhw")
        if ndarray_lhw.ndim != 3:
            raise ValueError(
                "Expected shape (L, H, W), but received shape %s"
                % ndarray_lhw.shape
            )

        for i in range(ndarray_lhw.shape[0]):
            self.add_bw_array_frame(ndarray_lhw[i], vmin, vmax)

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
            file_ext="gif",
            verbose=verbose,
        )

        if loop_forever:
            n_loops = 0

        self._frame_list[0].save(
            self.filename,
            format="gif",
            save_all=True,
            append_images=self._frame_list[1:],
            duration=frame_duration_ms,
            optimise=optimise,
            loop=n_loops,
        )

def set_latex_params(use_tex=True):
    latex_params_dict = {
        "font.family":          "serif",
        "font.serif":           ["Computer Modern"],
        "text.usetex":          True,
        "text.latex.preamble":  "\\usepackage{amsmath}",
        "legend.edgecolor":     "k",
        "legend.fancybox":      False,
        "legend.framealpha":    1,
    }

    for key, value in latex_params_dict.items():
        if use_tex:
            matplotlib.rcParams[key] = value
        else:
            matplotlib.rcParams[key] = matplotlib.rcParamsDefault[key]
