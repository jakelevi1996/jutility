import textwrap
from jutility import util

def plot(
    *lines,
    axis_properties=None,
    plot_name="output",
    dir_name=None,
    autocompile=False,
):
    if axis_properties is None:
        axis_properties = AxisProperties()

    printer = util.Printer(plot_name, dir_name, file_ext="tex")
    indent = Indenter()

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
        axis_properties.apply(printer, indent)
    printer("]")
    printer()

    with indent.new_block():
        for line in lines:
            line.plot(printer, indent)

    printer("\\end{axis}")
    printer("\\end{tikzpicture}")
    printer("\\end{document}")

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
        grid=False,
        grid_style="solid",
        shader="flat corner",
        use_times=False,
        colour_defines_str=None,
    ):
        self._title                 = title
        self._xlabel                = xlabel
        self._ylabel                = ylabel
        self._xlim                  = xlim
        self._ylim                  = ylim
        self._figsize_cm            = figsize_cm
        self._legend_pos            = legend_pos
        self._grid                  = grid
        self._grid_style            = grid_style
        self._shader                = shader
        self._use_times             = use_times
        self._colour_defines_str    = colour_defines_str

    def preamble(self, printer):
        if self._use_times:
            printer("\\usepackage{times}")
            printer()
        if self._colour_defines_str is not None:
            printer(self._colour_defines_str)
            printer()

    def apply(self, printer, indent):
        if self._title is not None:
            printer(indent("title={%s}," % self._title))
        if self._xlabel is not None:
            printer(indent("xlabel={%s}," % self._xlabel))
        if self._ylabel is not None:
            printer(indent("ylabel={%s}," % self._ylabel))
        if self._xlim is not None:
            x_min, x_max = self._xlim
            printer(indent("xmin=%s," % x_min))
            printer(indent("xmax=%s," % x_max))
        if self._ylim is not None:
            y_min, y_max = self._ylim
            printer(indent("ymin=%s," % y_min))
            printer(indent("ymax=%s," % y_max))
        if self._figsize_cm is not None:
            width, height = self._figsize_cm
            printer(indent("width=%scm,"     % width))
            printer(indent("height=%scm,"    % height))
        if self._legend_pos is not None:
            printer(indent("legend pos=%s," % self._legend_pos))
        if self._grid:
            printer(indent("xmajorgrids=true,"))
            printer(indent("ymajorgrids=true,"))
            printer(indent("grid style=%s," % self._grid_style))
        if self._shader is not None:
            printer(indent("shader=%s," % self._shader))

class Indenter:
    def __init__(self, indent_str=None, initial_indent=0):
        if indent_str is None:
            indent_str = "".ljust(4)
        self._indent_str = indent_str
        self._num_indent = initial_indent

    def new_block(self):
        new_block_context = util.CallbackContext(
            lambda: self._add_indent( 1),
            lambda: self._add_indent(-1),
        )
        return new_block_context

    def _add_indent(self, n):
        self._num_indent += n

    def __call__(self, s):
        prefix = self._indent_str * self._num_indent
        s_indent = textwrap.indent(str(s), prefix)
        return s_indent
