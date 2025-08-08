import math
import matplotlib.axes
import matplotlib.figure
from jutility import util
from jutility.plotting.properties import PropertyDict
from jutility.plotting.figure.legend import FigureLegend

class GridProperties(PropertyDict):
    def _get_abbreviated_keys_dict(self) -> dict:
        return {
            "nr": "num_rows",
            "nc": "num_cols",
            "wr": "width_ratios",
            "hr": "height_ratios",
        }

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
            gridspec_kw=dict(
                wspace=space,
                hspace=space,
            ),
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

    def _plot_legend(
        self,
        legend: FigureLegend,
        figure: matplotlib.figure.Figure,
    ):
        legend.plot(figure)
