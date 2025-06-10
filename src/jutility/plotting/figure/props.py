import math
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
from jutility import util
from jutility.plotting.figure.legend import FigureLegend

class FigureProperties:
    def __init__(
        self,
        num_subplots,
        num_rows=None,
        num_cols=None,
        figsize=None,
        sharex=False,
        sharey=False,
        width_ratios=None,
        height_ratios=None,
        constrained_layout=True,
        tight_layout=False,
        layout=None,
        colour=None,
        title=None,
        title_font_size=25,
        title_colour=None,
        title_wrap_len=None,
        top_space=None,
        bottom_space=None,
        legend: FigureLegend=None,
        pad=0.1,
        space=0,
        dpi=None,
    ):
        if num_rows is None:
            if num_cols is None:
                num_cols = math.ceil(math.sqrt(num_subplots))
            num_rows = math.ceil(num_subplots / num_cols)
        if num_cols is None:
            num_cols = math.ceil(num_subplots / num_rows)
        if figsize is None:
            figsize = [6, 4]
        if layout is not None:
            constrained_layout = False
            tight_layout = False
        if tight_layout:
            constrained_layout = False
            layout = None
        if constrained_layout:
            tight_layout = False
            layout = "constrained"
        if title_wrap_len is not None:
            title = util.wrap_string(
                title,
                max_len=title_wrap_len,
                wrap_len=title_wrap_len,
            )

        self._num_rows = num_rows
        self._num_cols = num_cols
        self._figsize = figsize
        self._sharex = sharex
        self._sharey = sharey
        self._width_ratios = width_ratios
        self._height_ratios = height_ratios
        self._constrained_layout = constrained_layout
        self._tight_layout = tight_layout
        self._layout = layout
        self._colour = colour
        self._title = title
        self._title_font_size = title_font_size
        self._title_colour = title_colour
        self._top_space = top_space
        self._bottom_space = bottom_space
        self._legend = legend
        self._pad = pad
        self._space = space
        self._dpi = dpi

    def get_num_axes(self):
        return self._num_rows * self._num_cols

    def get_figure(self):
        figure = plt.figure(
            figsize=self._figsize,
            dpi=self._dpi,
            layout=self._layout,
        )
        if self._constrained_layout:
            layout_engine = figure.get_layout_engine()
            layout_engine.set(
                w_pad=self._pad,
                h_pad=self._pad,
            )

        return figure

    def get_axes(
        self,
        figure: matplotlib.figure.Figure,
    ) -> list[matplotlib.axes.Axes]:
        axis_array = figure.subplots(
            nrows=self._num_rows,
            ncols=self._num_cols,
            sharex=self._sharex,
            sharey=self._sharey,
            squeeze=False,
            width_ratios=self._width_ratios,
            height_ratios=self._height_ratios,
            gridspec_kw=dict(
                wspace=self._space,
                hspace=self._space,
            ),
        )
        axis_list = axis_array.flatten().tolist()
        return axis_list

    def get_subfigs(
        self,
        figure: matplotlib.figure.Figure,
    ) -> list[matplotlib.figure.SubFigure]:
        subfig_array = figure.subfigures(
            nrows=self._num_rows,
            ncols=self._num_cols,
            squeeze=False,
            wspace=self._space,
            hspace=self._space,
            width_ratios=self._width_ratios,
            height_ratios=self._height_ratios,
        )
        subfig_list = subfig_array.flatten().tolist()
        return subfig_list

    def apply(self, figure: matplotlib.figure.Figure):
        if self._tight_layout:
            figure.tight_layout()
        if self._colour is not None:
            figure.patch.set_facecolor(self._colour)
        if self._title is not None:
            figure.suptitle(
                self._title,
                fontsize=self._title_font_size,
                color=self._title_colour,
            )
        if self._top_space is not None:
            figure.subplots_adjust(top=(1 - self._top_space))
        if self._bottom_space is not None:
            figure.subplots_adjust(bottom=self._bottom_space)
        if self._legend is not None:
            self._legend.plot(figure)
