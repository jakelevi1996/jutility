import textwrap
import numpy as np
from jutility import util

SHADER_FLAT         = "flat"
SHADER_FLAT_CORNER  = "flat corner"
SHADER_INTERP       = "interp"

def plot(
    *lines,
    axis_properties=None,
    plot_name="output",
    dir_name=None,
    autocompile=False,
):
    if axis_properties is None:
        axis_properties = AxisProperties()

    printer = util.Printer(
        plot_name,
        dir_name,
        file_ext="tex",
        print_to_console=False,
    )
    indent = Indenter(printer)
    counter = util.Counter()

    printer("\\documentclass[tikz]{standalone}")
    printer()
    printer("\\usepackage{pgfplots}")
    printer("\\pgfplotsset{compat=1.18}")
    printer("\\usepgfplotslibrary{fillbetween}")
    printer()

    axis_properties.preamble(printer)

    printer("\\begin{document}")
    printer("\\begin{tikzpicture}")
    printer("\\begin{axis}[")
    with indent.new_block():
        axis_properties.apply(indent)
    printer("]")
    printer()

    with indent.new_block():
        for line in lines:
            line.plot(indent, counter)
            printer()

    printer("\\end{axis}")
    printer("\\end{tikzpicture}")
    printer("\\end{document}")

class _Plottable:
    def __init__(self):
        """
        Initialise object to be plotted. Optional keyword arguments:

        - `c`: colour of line/marker/patch
        - `alpha`: opacity, which satisfies `0 <= alpha <= 1`, where 1 = opaque
          and 0 = transparent
        - `marker`: marker to use for each point, EG `"*"` for filled marker,
          or `"o"` for unfilled marker. Default is `None`, in which case no
          marker is plotted
        - `label`: entry to be added to legend
        - `name`: name which is applied to this object, allowing it to be
          referred to by other objects, EG filled areas
        """
        raise NotImplementedError()

    def _set_kwargs(
        self,
        c=None,
        alpha=None,
        marker=None,
        name=None,
        label=None,
    ):
        self._colour    = c
        self._alpha     = alpha
        self._marker    = marker
        self._name      = name
        self._label     = label

    def plot(self, indent, counter):
        raise NotImplementedError()

    def _print_latex_plot_options(self, indent, forgettable=True):
        if self._colour is not None:
            indent.print("color=%s," % self._colour)
        if self._alpha is not None:
            indent.print("opacity=%s," % self._alpha)
        if self._marker is not None:
            indent.print("mark=%s," % self._marker)
        if self._name is not None:
            indent.print("name path=%s," % self._name)
        if forgettable and (self._label is None):
            indent.print("forget plot,")

class Line(_Plottable):
    def __init__(self, x, y, **kwargs):
        self._set_kwargs(**kwargs)
        self._x_list = x
        self._y_list = y

    def plot(self, indent, counter):
        indent.print("\\addplot[")
        with indent.new_block():
            self._print_latex_plot_options(indent)

        indent.print("]")
        indent.print("table {")
        with indent.new_block():
            indent.print(format_table_row("x", "y"))
            for x, y in zip(self._x_list, self._y_list):
                indent.print(format_table_row(x, y))

        indent.print("};")

        if self._label is not None:
            indent.print("\\addlegendentry{%s}" % self._label)

class Quiver(_Plottable):
    def __init__(self, x, y, dx, dy, norm_length=None, tol=1e-5, **kwargs):
        self._set_kwargs(**kwargs)

        if norm_length is not None:
            dr = np.sqrt(np.square(dx) + np.square(dy))
            scale = norm_length / np.maximum(dr, tol)
            dx = dx * scale
            dy = dy * scale

        self._data_table = [x, y, dx, dy]

    def plot(self, indent, counter):
        indent.print("\\addplot[")
        with indent.new_block():
            self._print_latex_plot_options(indent)

            indent.print("quiver={")
            with indent.new_block():
                indent.print("u=\\thisrow{u},")
                indent.print("v=\\thisrow{v},")
            indent.print("},")
            indent.print("-stealth,")

        indent.print("]")
        indent.print("table {")
        with indent.new_block():
            indent.print(format_table_row("x", "y", "u", "v"))
            for x, y, dx, dy in zip(*self._data_table):
                indent.print(format_table_row(x, y, dx, dy))

        indent.print("};")

        if self._label is not None:
            indent.print("\\addlegendentry{%s}" % self._label)

class FillBetween(_Plottable):
    def __init__(self, x, y1, y2, **kwargs):
        self._set_kwargs(**kwargs)
        self._x_list = x
        self._y_list = [y1, y2]

    def plot(self, indent, counter):
        names = ["y%i" % counter() for _ in range(2)]
        lines = [
            Line(x=self._x_list, y=y, alpha=0, name=name)
            for y, name in zip(self._y_list, names)
        ]
        lines[0].plot(indent, counter)
        lines[1].plot(indent, counter)

        indent.print("\\addplot[")
        with indent.new_block():
            self._print_latex_plot_options(indent)

        indent.print("]")
        indent.print("fill between[of=%s and %s];" % tuple(names))
        if self._label is not None:
            indent.print("\\addlegendentry{%s}" % self._label)

class _ConstantLine(_Plottable):
    def plot(self, indent, counter):
        indent.print("\\draw[")
        with indent.new_block():
            self._print_latex_plot_options(indent, forgettable=False)

        indent.print("]")
        self._print_line_data(indent)
        if self._label is not None:
            indent.print("\\addlegendimage{")
            with indent.new_block():
                self._print_latex_plot_options(indent)

            indent.print("}")
            indent.print("\\addlegendentry{%s}" % self._label)

    def _print_line_data(self, indent):
        raise NotImplementedError()

class HLine(_ConstantLine):
    def __init__(self, y, **kwargs):
        self._set_kwargs(**kwargs)
        self._y = y

    def _print_line_data(self, indent):
        indent.print(
            "(current axis.east |- 0,%s) -- (current axis.west |- 0,%s);"
            % (self._y, self._y)
        )

class VLine(_ConstantLine):
    def __init__(self, x, **kwargs):
        self._set_kwargs(**kwargs)
        self._x = x

    def _print_line_data(self, indent):
        indent.print(
            "(%s,0 |- current axis.south) -- (%s,0 |- current axis.north);"
            % (self._x, self._x)
        )

class ColourMesh(_Plottable):
    def __init__(self, x, y, z):
        self._x_list = x
        self._y_list = y
        self._z_list = z

    def plot(self, indent, counter):
        indent.print("\\addplot[surf,point meta=\\thisrow{z}]")
        indent.print("table {")
        with indent.new_block():
            indent.print(format_table_row("x", "y", "z"))
            for i, x in enumerate(self._x_list):
                for j, y in enumerate(self._y_list):
                    indent.print(format_table_row(x, y, self._z_list[j, i]))

                indent.blank_line()

        indent.print("};")

class AxisProperties:
    def __init__(
        self,
        title=None,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        figsize_cm=None,
        legend_pos="outer north east",
        legend_text_align="left",
        grid=False,
        grid_style="solid",
        shader=SHADER_FLAT,
        colour_map_name="viridis",
        colour_bar=False,
        use_times=False,
        colour_picker=None,
    ):
        self._title             = title
        self._xlabel            = xlabel
        self._ylabel            = ylabel
        self._xlim              = xlim
        self._ylim              = ylim
        self._figsize_cm        = figsize_cm
        self._legend_pos        = legend_pos
        self._legend_text_align = legend_text_align
        self._grid              = grid
        self._grid_style        = grid_style
        self._shader            = shader
        self._colour_map_name   = colour_map_name
        self._colour_bar        = colour_bar
        self._use_times         = use_times
        self._colour_picker     = colour_picker

    def preamble(self, printer):
        if self._use_times:
            printer("\\usepackage{times}")
            printer()
        if self._colour_picker is not None:
            printer(self._colour_picker.define_colours())
            printer()

    def apply(self, indent):
        if self._title is not None:
            indent.print("title={%s}," % self._title)
        if self._xlabel is not None:
            indent.print("xlabel={%s}," % self._xlabel)
        if self._ylabel is not None:
            indent.print("ylabel={%s}," % self._ylabel)
        if self._xlim is not None:
            x_min, x_max = self._xlim
            indent.print("xmin=%s," % x_min)
            indent.print("xmax=%s," % x_max)
        if self._ylim is not None:
            y_min, y_max = self._ylim
            indent.print("ymin=%s," % y_min)
            indent.print("ymax=%s," % y_max)
        if self._figsize_cm is not None:
            width, height = self._figsize_cm
            indent.print("width=%scm,"     % width)
            indent.print("height=%scm,"    % height)
        if self._legend_pos is not None:
            indent.print("legend pos=%s," % self._legend_pos)
        if self._legend_text_align is not None:
            indent.print("legend cell align={%s}," % self._legend_text_align)
        if self._grid:
            indent.print("xmajorgrids=true,")
            indent.print("ymajorgrids=true,")
            indent.print("grid style=%s," % self._grid_style)
        if self._shader is not None:
            indent.print("shader=%s," % self._shader)
        if self._colour_map_name is not None:
            indent.print("colormap name=%s," % self._colour_map_name)
        if self._colour_bar:
            indent.print("colorbar,")

class Indenter:
    def __init__(self, printer=None, indent_str=None, initial_indent=0):
        if printer is None:
            printer = util.Printer()
        if indent_str is None:
            indent_str = "".ljust(4)

        self._print         = printer
        self._indent_str    = indent_str
        self._num_indent    = initial_indent

    def new_block(self):
        new_block_context = util.CallbackContext(
            lambda: self._add_indent( 1),
            lambda: self._add_indent(-1),
        )
        return new_block_context

    def _add_indent(self, n):
        self._num_indent += n

    def indent(self, s):
        prefix = self._indent_str * self._num_indent
        s_indent = textwrap.indent(str(s), prefix)
        return s_indent

    def print(self, s):
        self._print(self.indent(s))

    def blank_line(self):
        self._print()

def format_table_row(*entries, width=25, sep=""):
    row_str = sep.join(str(i).ljust(width) for i in entries)
    return row_str
