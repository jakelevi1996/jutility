import numpy as np
from jutility import util
from jutility.plotting.noisy.data import NoisyData
from jutility.plotting.colour_picker import ColourPicker
from jutility.plotting.plottable import (
    PlottableGroup,
    ColourMesh,
)
from jutility.plotting.subplot.colour_bar import ColourBar

class NoisySweep:
    def __init__(
        self,
        log_z:      bool=False,
        sweeps:     (dict[str, NoisyData] | None)=None,
        key_order:  (list[str] | None)=None,
        **kwargs,
    ):
        if sweeps is None:
            sweeps = dict()
        if key_order is None:
            key_order = []

        self._log_z = log_z
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
        label_fmt:  (util.StringFormatter | None)=None,
        **kwargs,
    ) -> list[PlottableGroup]:
        if key_order is None:
            key_order = self._key_order
        if cp is None:
            cp = ColourPicker(len(key_order))
        if label_fmt is None:
            label_fmt = util.NoFormat()

        return [
            self._sweeps[key].plot(
                c=cp.next(),
                label=label_fmt.format(key),
                **kwargs,
            )
            for key in key_order
        ]

    def colour_mesh(
        self,
        x:          (list[float] | None)=None,
        key_order:  (list[str] | None)=None,
        set_vlims:  bool=True,
        **kwargs,
    ) -> ColourMesh:
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
        z = np.array(z)

        if self._log_z:
            z = np.log(np.abs(z))
        if set_vlims:
            kwargs.setdefault("vmin", z.min())
            kwargs.setdefault("vmax", z.max())

        return ColourMesh(x, key_order, z, **kwargs)

    def colour_bar(
        self,
        x:          (list[float] | None)=None,
        key_order:  (list[str] | None)=None,
        **kwargs,
    ) -> ColourBar:
        """
        See [`jutility.plotting.ColourBar`](
        https://github.com/jakelevi1996/jutility/blob/main/src/jutility/plotting/subplot/colour_bar.py
        )
        """
        if x is None:
            x = self.get_x()
        if key_order is None:
            key_order = self._key_order

        z = [
            self._sweeps[k].get_mean(xi)
            for k in key_order
            for xi in x
        ]
        return ColourBar(
            vmin=min(z),
            vmax=max(z),
            log=self._log_z,
            **kwargs,
        )

    def get_x(self) -> list[float]:
        all_x = [
            x
            for nd in self._sweeps.values()
            for x in nd.get_x()
        ]
        return sorted(set(all_x))

    def get_keys(self) -> list[float]:
        return sorted(self._sweeps.keys())

    def transpose(self) -> "NoisySweep":
        ns = NoisySweep(log_z=self._log_z, **self._kwargs)
        for k, nd in self._sweeps.items():
            x, y = nd.get_all_data()
            for xi, yi in zip(x, y):
                ns.update(xi, k, yi)

        return ns

    def inverse(self, y: float) -> set[tuple[float, float, int]]:
        return set(
            (k, x, repeat)
            for k, nd in self._sweeps.items()
            for x, repeat in nd.inverse(y)
        )

    def __iter__(self):
        return (
            y
            for nd in self._sweeps.values()
            for y in nd
        )

    def __len__(self) -> int:
        return len(self._sweeps)
