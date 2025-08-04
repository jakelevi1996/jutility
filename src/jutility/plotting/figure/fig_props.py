import matplotlib.pyplot as plt
import matplotlib.figure
from jutility.plotting.properties import PropertyDict

class FigureProperties(PropertyDict):
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

    def apply(self, figure: matplotlib.figure.Figure):
        if self._get("tight_layout"):
            figure.tight_layout()

    @classmethod
    def get_figure_kwargs(cls, all_kwargs: dict) -> tuple[dict, dict]:
        grid_kwargs = all_kwargs.copy()
        fig_kwargs = dict()
        keys_str = (
            "figsize dpi layout constrained_layout tight_layout "
            "pad w_pad h_pad"
        )
        keys = keys_str.split()

        for k in keys:
            if k in grid_kwargs:
                fig_kwargs[k] = grid_kwargs.pop(k)

        return fig_kwargs, grid_kwargs
