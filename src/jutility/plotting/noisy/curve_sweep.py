from jutility import util
from jutility.plotting.plottable import PlottableGroup
from jutility.plotting.colour_picker import ColourPicker
from jutility.plotting.noisy.curve import NoisyCurve

class NoisyCurveSweep:
    def __init__(
        self,
        noisy_curves: (dict[float, NoisyCurve] | None)=None,
        **kwargs,
    ):
        if noisy_curves is None:
            noisy_curves = dict()

        self._noisy_curves = noisy_curves
        self._kwargs = kwargs

    def update(self, key: float, y: list[float]):
        if key not in self._noisy_curves:
            self._noisy_curves[key] = NoisyCurve(*self._kwargs)

        self._noisy_curves[key].update(y)

    def plot(
        self,
        x:          list[float],
        cp:         (ColourPicker | None)=None,
        label_fmt:  (util.StringFormatter | None)=None,
        **kwargs,
    ) -> list[PlottableGroup]:
        if cp is None:
            cp = ColourPicker(len(self._noisy_curves))
        if label_fmt is None:
            label_fmt = util.NoFormat()

        return [
            noisy_curve.plot(
                x=x,
                c=cp.next(),
                label=label_fmt.format(key),
                **kwargs,
            )
            for key, noisy_curve in self._noisy_curves.items()
        ]

    def __len__(self) -> int:
        return len(self._noisy_curves)

    def __iter__(self):
        return (
            y
            for nc in self._noisy_curves.values()
            for y in nc
        )
