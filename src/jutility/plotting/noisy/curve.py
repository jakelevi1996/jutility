import numpy as np
from jutility.plotting.plottable import (
    PlottableGroup,
    Line,
    FillBetween,
)
from jutility.plotting.noisy.bounds import confidence_bounds

class NoisyCurve:
    def __init__(
        self,
        log_y:  bool=False,
        curves: (list[list[float]] | None)=None,
    ):
        if curves is None:
            curves = []

        self._log_y = log_y
        self._curves = curves

    def update(self, y: list[float]):
        self._curves.append(y)

    def plot(
        self,
        x:          list[float],
        c:          str="b",
        label:      (str | None)=None,
        n_sigma:    float=1.0,
        alpha_line: float=0.2,
        alpha_mean: float=1.0,
        alpha_fill: float=0.2,
    ) -> PlottableGroup:
        y = np.array(self._curves)
        y_list = np.array_split(y, y.shape[1], 1)
        mean, ucb, lcb = confidence_bounds(
            y_list,
            n_sigma=n_sigma,
            log=self._log_y,
        )
        return PlottableGroup(
            Line(x, y.T,                c=c, z=20, a=alpha_line),
            Line(x, mean,               c=c, z=30, a=alpha_mean),
            FillBetween(x, lcb, ucb,    c=c, z=10, a=alpha_fill),
            label=label,
        )

    def __iter__(self):
        return (
            yi
            for y in self._curves
            for yi in y
        )
