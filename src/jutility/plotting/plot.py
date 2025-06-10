from jutility.plotting.plottable import Plottable
from jutility.plotting.multiplot import MultiPlot
from jutility.plotting.subplot.subplot import Subplot
from jutility.plotting.subplot.legend import LegendSubplot

def plot(
    *lines:     Plottable,
    legend:     bool=False,
    figsize:    (tuple[float, float] | None)=None,
    plot_name:  (str | None)=None,
    dir_name:   (str | None)=None,
    show:       bool=False,
    save_close: bool=True,
    pdf:        bool=False,
    **axis_kwargs,
):
    if "title" not in axis_kwargs:
        axis_kwargs["title"] = plot_name

    if legend:
        mp = MultiPlot(
            Subplot(*lines, **axis_kwargs),
            LegendSubplot(*lines),
            figsize=figsize,
            width_ratios=[1, 0.2],
        )
    else:
        mp = MultiPlot(
            Subplot(*lines, **axis_kwargs),
            figsize=figsize,
        )

    if show:
        mp.show()
    if save_close:
        mp.save(plot_name, dir_name, pdf=pdf)

    return mp
