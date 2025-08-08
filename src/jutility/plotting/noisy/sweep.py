from jutility.plotting.noisy.data import NoisyData
from jutility.plotting.colour_picker import ColourPicker
from jutility.plotting.plottable import (
    PlottableGroup,
    ColourMesh,
)

class NoisySweep:
    def __init__(
        self,
        sweeps:     (dict[str, NoisyData] | None)=None,
        key_order:  (list[str] | None)=None,
        **kwargs,
    ):
        if sweeps is None:
            sweeps = dict()
        if key_order is None:
            key_order = []

        self._sweeps = sweeps
        self._key_order = key_order
        self._kwargs = kwargs

    def update(self, key: str, x: float, y: float):
        if key not in self._sweeps:
            self._sweeps[key] = NoisyData(**self._kwargs)
            self._key_order.append(key)

        self._sweeps[key].update(x, y)

    def plot(
        self,
        cp:         (ColourPicker | None)=None,
        key_order:  (list[str] | None)=None,
        **kwargs,
    ) -> list[PlottableGroup]:
        if key_order is None:
            key_order = self._key_order
        if cp is None:
            cp = ColourPicker(len(key_order))

        return [
            self._sweeps[key].plot(
                c=cp.next(),
                label=key,
                **kwargs,
            )
            for key in key_order
        ]

    def colour_mesh(
        self,
        x:          (list[float] | None)=None,
        key_order:  (list[str] | None)=None,
        **kwargs,
    ):
        """
        See [`plt.pcolormesh`](
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html
        ) and
        [`tests/test_plotting/test_noisy/test_sweep.py::test_colour_mesh`](
        https://github.com/jakelevi1996/jutility/blob/main/tests/test_plotting/test_noisy/test_sweep.py
        )
        """
        if x is None:
            x = self.get_x()
        if key_order is None:
            key_order = self._key_order

        z = [
            [self._sweeps[k].get_mean(xi) for xi in x]
            for k in key_order
        ]
        return ColourMesh(x, key_order, z, **kwargs)

    def get_x(self) -> list[float]:
        all_x = [
            x
            for nd in self._sweeps.values()
            for x in nd.get_x()
        ]
        return sorted(set(all_x))

    def __iter__(self):
        return (
            y
            for nd in self._sweeps.values()
            for y in nd
        )

    def __len__(self) -> int:
        return len(self._sweeps)
