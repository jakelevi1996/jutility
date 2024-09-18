import os
import subprocess
import textwrap
import math
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
    lualatex=False,
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

    if autocompile:
        printer(flush=True, end="")
        output_path = compile_latex(printer.get_filename(), lualatex)
    else:
        output_path = printer.get_filename()

    return output_path

class _Plottable:
    def __init__(self):
        """
        Initialise object to be plotted. Optional keyword arguments:

        - `c`: colour of line/marker/patch, either as a string, or as a
          sequence of at least 3 floats in the range [0, 1], in which the first
          3 elements refer to the red, green, and blue colour components (any
          subsequent elements are ignored)
        - `alpha`: opacity, which satisfies `0 <= alpha <= 1`, where 1 = opaque
          and 0 = transparent
        - `marker`: marker to use for each point, EG `"*"` for filled marker,
          or `"o"` for unfilled marker. Default is `None`, in which case no
          marker is plotted
        - `only_marks`: if True, only markers are added to the plot for each
          point, and no line is drawn between them. If `only_marks` is True,
          then by default a filled circle (`*`) is used as the marker, even if
          no marker is specified. Default is `only_marks=False`
        - `line_width`: width of line in pt as a float. Default is None, which
          uses pgfplots' default line width
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
        only_marks=False,
        line_width=None,
        name=None,
        label=None,
    ):
        self._colour        = c
        self._alpha         = alpha
        self._marker        = marker
        self._only_marks    = only_marks
        self._line_width    = line_width
        self._name          = name
        self._label         = label

    def plot(self, indent, counter):
        raise NotImplementedError()

    def _print_latex_plot_options(self, indent, forgettable=True):
        if self._colour is not None:
            if isinstance(self._colour, str):
                indent.print("color=%s," % self._colour)
            else:
                indent.print(
                    "color={rgb,1:red,%f;green,%f;blue,%f},"
                    % (self._colour[0], self._colour[1], self._colour[2])
                )
        if self._alpha is not None:
            indent.print("opacity=%s," % self._alpha)
        if self._marker is not None:
            indent.print("mark=%s," % self._marker)
        if self._only_marks:
            indent.print("only marks,")
        if self._line_width is not None:
            indent.print("line width=%fpt," % self._line_width)
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

def get_noisy_data_lines(
    noisy_data,
    n_sigma=1,
    colour="blue",
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
                "c":            colour,
                "label":        name,
                "alpha":        result_alpha,
                "only_marks":   True,
            }
        results_line = Line(all_x, all_y, **results_line_kwargs)
        line_list.append(results_line)

    x, mean, ucb, lcb = noisy_data.get_statistics(n_sigma)
    if mean_line_kwargs is None:
        mean_line_kwargs = {"c": colour}
        if mean_std_labels:
            mean_line_kwargs["label"] = "Mean"
    if std_line_kwargs is None:
        std_line_kwargs = {"c": colour, "alpha": 0.3}
        if mean_std_labels:
            std_line_kwargs["label"] = "$\\pm %s \\sigma$" % n_sigma
    mean_line = Line(x, mean, **mean_line_kwargs)
    std_line = FillBetween(x, ucb, lcb, **std_line_kwargs)
    line_list.append(mean_line)
    line_list.append(std_line)
    return line_list

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
        log_x=False,
        log_y=False,
        shader=SHADER_INTERP,
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
        self._log_x             = log_x
        self._log_y             = log_y
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
        if self._log_x:
            indent.print("xmode=log,")
        if self._log_y:
            indent.print("ymode=log,")
        if self._shader is not None:
            indent.print("shader=%s," % self._shader)
        if self._colour_map_name is not None:
            indent.print("colormap name=%s," % self._colour_map_name)
        if self._colour_bar:
            indent.print("colorbar,")

class Subfigure:
    def __init__(
        self,
        full_path,
        width="",
        caption=None,
        label=None,
    ):
        self._full_path = full_path
        self._width     = width
        self._caption   = caption
        self._label     = label

    def plot(self, indent, graphics_path, only_subfigure):
        indent.print("\\centering")
        width_str = self.get_width_str() if only_subfigure else ""
        rel_path = os.path.relpath(self._full_path, graphics_path)
        rel_path = rel_path.replace("\\", "/")
        indent.print(
            "\\includegraphics[width=%s\\textwidth]{%s}"
            % (width_str, rel_path)
        )
        if self._caption is not None:
            indent.print("\\caption{%s}" % self._caption)
        if self._label is not None:
            indent.print("\\label{fig:%s}" % self._label)

    def get_width_str(self):
        return str(self._width)

def plot_figure(
    *subfigures,
    graphics_path=None,
    num_cols=None,
    caption=None,
    label=None,
    starred_env=False,
    fig_name="figure",
    dir_name=None,
):
    printer = util.Printer(
        fig_name,
        dir_name,
        file_ext="tex",
        print_to_console=False,
    )
    indent = Indenter(printer)
    star_str = "*" if starred_env else ""
    indent.print("\\begin{figure%s}" % star_str)
    if num_cols is None:
        num_cols = math.ceil(math.sqrt(len(subfigures)))

    num_rows = math.ceil(len(subfigures) / num_cols)
    use_hspace = ((num_rows * num_cols) != len(subfigures))
    if use_hspace:
        pre_hspace_ind = num_cols * int(len(subfigures) / num_cols)
        post_hspace_ind = len(subfigures) - 1

    with indent.new_block():
        if len(subfigures) == 1:
            [subfig] = subfigures
            subfig.plot(indent, graphics_path, True)
        else:
            indent.print("\\centering")
            indent.print("\\captionsetup{justification=centering}")

            for i, subfig in enumerate(subfigures):
                if use_hspace and (i == pre_hspace_ind):
                    indent.print("\\hspace*{\\fill}")

                w = subfig.get_width_str()
                indent.print("\\begin{subfigure}[t]{%s\\textwidth}" % w)

                with indent.new_block():
                    subfig.plot(indent, graphics_path, False)

                indent.print("\\end{subfigure}")

                if use_hspace and (i == post_hspace_ind):
                    indent.print("\\hspace*{\\fill}")

                if (i + 1) == len(subfigures):
                    break

                if ((i + 1) % num_cols) == 0:
                    indent.print("\\newline")
                else:
                    indent.print("\\hfill")

            if caption is not None:
                indent.print("\\caption{%s}" % caption)
            if label is not None:
                indent.print("\\label{%s}" % label)

    indent.print("\\end{figure%s}" % star_str)
    return printer.get_filename()

def standalone_document(
    *input_paths,
    graphics_path=None,
    document_name="document",
    dir_name=None,
    autocompile=False,
    lualatex=False,
):
    printer = util.Printer(
        document_name,
        dir_name,
        file_ext="tex",
        print_to_console=False,
    )
    document_dir = os.path.dirname(printer.get_filename())

    printer("\\documentclass{article}")
    printer("\\usepackage{subcaption}")
    printer("\\usepackage{graphicx}")
    printer("\\pagestyle{empty}")
    if graphics_path is not None:
        graphics_rel_path = os.path.relpath(graphics_path, document_dir)
        graphics_rel_path = graphics_rel_path.replace("\\", "/")
        printer("\\graphicspath{{%s}}" % graphics_rel_path)

    printer()
    printer("\\begin{document}")

    for input_path in input_paths:
        input_rel_path = os.path.relpath(input_path, document_dir)
        input_rel_path = input_rel_path.replace("\\", "/")
        printer("\\input{%s}" % input_rel_path)

    printer("\\end{document}")

    if autocompile:
        printer(flush=True, end="")
        output_path = compile_latex(printer.get_filename(), lualatex)
    else:
        output_path = printer.get_filename()

    return output_path

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

def compile_latex(full_path, lualatex=False):
    tex_dir_name, tex_filename = os.path.split(full_path)
    tex_root, _ = os.path.splitext(tex_filename)
    pdf_path = util.get_full_path(tex_root, tex_dir_name, "pdf")

    program_str = "lualatex" if lualatex else "pdflatex"
    completed_process = subprocess.run(
        [program_str, tex_filename],
        cwd=tex_dir_name,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed_process.returncode != 0:
        print(completed_process.stdout.decode())
        completed_process.check_returncode()

    return pdf_path

def format_table_row(*entries, width=25, sep=""):
    row_str = sep.join(str(i).ljust(width) for i in entries)
    return row_str
