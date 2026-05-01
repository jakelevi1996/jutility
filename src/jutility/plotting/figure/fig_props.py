import math
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt
from jutility import util
from jutility.plotting.properties import PropertyDict
from jutility.plotting.figure.legend import FigureLegend

class FigureProperties(PropertyDict):
    def _get_abbreviated_keys_dict(self) -> dict:
        return {
            "fs": "figsize",
            "nr": "num_rows",
            "nc": "num_cols",
            "wr": "width_ratios",
            "hr": "height_ratios",
            "wp": "w_pad",
            "hp": "h_pad",
            "ws": "wspace",
            "hs": "hspace",
            "tfs": "title_font_size",
        }

    def get_figure(self):
        constrained_layout  = self._get_default("constrained_layout", True)
        tight_layout        = self._get_default("tight_layout", False)
        layout              = self._get_default("layout", None)

        if layout is not None:
            constrained_layout = False
            tight_layout = False
        if tight_layout:
            constrained_layout = False
            layout = None
        if constrained_layout:
            tight_layout = False
            layout = "constrained"

        figure = plt.figure(
            figsize=self._get_default("figsize", [6, 4]),
            dpi=self._get_default("dpi", None),
            layout=layout,
        )
        if constrained_layout:
            pad = self._get_default("pad", 0.1)
            layout_engine = figure.get_layout_engine()
            layout_engine.set(
                w_pad=self._get_default("w_pad", pad),
                h_pad=self._get_default("h_pad", pad),
            )

        self._set("tight_layout", tight_layout)
        return figure

    def init_size(self, num_subplots: int) -> int:
        num_rows = self._get_default("num_rows", None)
        num_cols = self._get_default("num_cols", None)
        if (num_rows is None) and (num_cols is None):
            num_cols = math.ceil(math.sqrt(num_subplots))
        if num_rows is None:
            num_rows = math.ceil(num_subplots / num_cols)
        if num_cols is None:
            num_cols = math.ceil(num_subplots / num_rows)

        self._set("num_rows", num_rows)
        self._set("num_cols", num_cols)
        num_empty = (num_rows * num_cols) - num_subplots
        return num_empty

    def get_axes(
        self,
        figure: matplotlib.figure.Figure,
    ) -> list[matplotlib.axes.Axes]:
        space = self._get_default("space", 0.0)
        axis_array = figure.subplots(
            nrows=self._get("num_rows"),
            ncols=self._get("num_cols"),
            width_ratios=self._get_default("width_ratios", None),
            height_ratios=self._get_default("height_ratios", None),
            sharex=self._get_default("sharex", False),
            sharey=self._get_default("sharey", False),
            gridspec_kw={
                "wspace": self._get_default("wspace", space),
                "hspace": self._get_default("hspace", space),
            },
            squeeze=False,
        )
        axis_list = axis_array.flatten().tolist()
        return axis_list

    def get_subfigs(
        self,
        figure: matplotlib.figure.Figure,
    ) -> list[matplotlib.figure.SubFigure]:
        space = self._get_default("space", 0.0)
        subfig_array = figure.subfigures(
            nrows=self._get("num_rows"),
            ncols=self._get("num_cols"),
            width_ratios=self._get_default("width_ratios", None),
            height_ratios=self._get_default("height_ratios", None),
            wspace=space,
            hspace=space,
            squeeze=False,
        )
        subfig_list = subfig_array.flatten().tolist()
        return subfig_list

    def apply(self, figure: matplotlib.figure.Figure):
        if self._has("colour"):
            figure.patch.set_facecolor(self._get("colour"))
        if self._has("legend"):
            self._plot_legend(self._get("legend"), figure)
        if self._has("title"):
            title = self._get("title")
            if self._has("title_wrap_len"):
                title_wrap_len = self._get("title_wrap_len")
                title = util.wrap_string(
                    title,
                    max_len=title_wrap_len,
                    wrap_len=title_wrap_len,
                )

            figure.suptitle(
                title,
                fontsize=self._get_default("title_font_size", 25),
                color=self._get_default("title_colour", None),
            )

        if self._get_default("tight_layout", False):
            figure.tight_layout()

    def _plot_legend(
        self,
        legend: FigureLegend,
        figure: matplotlib.figure.Figure,
    ):
        legend.plot(figure)
