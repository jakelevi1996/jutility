# jutility

Collection of Python utilities intended to be useful for machine learning research and experiments.

![](https://raw.githubusercontent.com/jakelevi1996/jutility/main/images/logo_black.png)

## Contents

- [jutility](#jutility)
  - [Contents](#contents)
  - [Install with `pip`](#install-with-pip)
  - [Usage examples](#usage-examples)
    - [`plotting`](#plotting)
      - [Making a simple plot](#making-a-simple-plot)
      - [Multiple lines in a single Line](#multiple-lines-in-a-single-line)
      - [Vector fields (with optional normalisation)](#vector-fields-with-optional-normalisation)
      - [Shared colour bar](#shared-colour-bar)
    - [`util`](#util)
    - [`sweep`](#sweep)
  - [Unit tests](#unit-tests)
  - [Build package locally](#build-package-locally)
  - [Updating package on PyPI](#updating-package-on-pypi)

## Install with `pip`

The `jutility` package is available as [a Python package on PyPI](https://pypi.org/project/jutility/), and can be installed with `pip` using the following commands:

```
python -m pip install -U pip
python -m pip install -U jutility
```

Alternatively, `jutility` can be installed in "editable mode" from the GitHub repository:

```
git clone https://github.com/jakelevi1996/jutility.git
cd jutility
python -m pip install -U pip
python -m pip install -e .
```

## Usage examples

### `plotting`

#### Making a simple plot

```python
import numpy as np
from jutility import plotting

x = np.linspace(-10, 10, 100, dtype=np.float32)
plotting.plot(
    plotting.Line(x, np.log(1 + np.exp(x)), label="Softplus", c="b"),
    plotting.Line(x, np.maximum(x, 0),      label="ReLU",     c="r", ls="--"),
    legend=True,
    plot_name="Softplus vs ReLU",
    dir_name="images",
)
```

![](images/Softplus_vs_ReLU.png)

```python
import numpy as np
from jutility import plotting

rng = np.random.default_rng(0)
x = np.linspace(0, 2)
f = lambda x: x + 0.1 * rng.normal(size=x.shape)

plotting.plot(
    plotting.Line(x, f(x), c="b", marker="o", label="Blue data"),
    plotting.Line(x, np.exp(-f(x)), c="r", marker="o", label="Red data"),
    axis_properties=plotting.AxisProperties("x label", "y label"),
    legend=True,
    plot_name="Simple plot",
    dir_name="images",
)
```

![](images/Simple_plot.png)

#### Multiple lines in a single Line

```python
import numpy as np
from jutility import plotting, util

n = 100
batch_size = 20
rng = np.random.default_rng(0)
noise = rng.normal(size=[n, batch_size])

x = np.linspace(-1, 4, n).reshape(n, 1)
y = np.exp(-x) + 0.1 * noise

plotting.plot(
    plotting.Line(x, y, alpha=0.2,       c="b", zorder=10),
    plotting.Line(x, np.mean(y, axis=1), c="r", zorder=20),
    plot_name="Multiple lines in a single Line",
    dir_name="images",
)
```

![](images/Multiple_lines_in_a_single_Line.png)

#### Vector fields (with optional normalisation)

```python
import numpy as np
from jutility import plotting, util

n = 25
x = np.linspace(-2, 2, n).reshape(1, n)
y = np.linspace(-2, 2, n).reshape(n, 1)
dx = y - x
dy = x + y
mp = plotting.MultiPlot(
    plotting.Subplot(
        plotting.Quiver(x, y, dx, dy, zorder=10, normalise=False),
        axis_properties=plotting.AxisProperties(title="normalise=False")
    ),
    plotting.Subplot(
        plotting.Quiver(x, y, dx, dy, zorder=10, normalise=True),
        axis_properties=plotting.AxisProperties(title="normalise=True")
    ),
    figure_properties=plotting.FigureProperties(figsize=[10, 4])
)
mp.save(plot_name="Vector field", dir_name="images")
```

![](images/Vector_field.png)

#### Shared colour bar

```python
import numpy as np
from jutility import plotting

rng = np.random.default_rng(0)
z1 = rng.random((100, 200)) + 5
z2 = rng.random((100, 200)) + 2
v_min = min(z1.min(), z2.min())
v_max = max(z1.max(), z2.max())

colour_bar = plotting.ColourBar(v_min, v_max)

mp = plotting.MultiPlot(
    plotting.ImShow(c=z1, vmin=v_min, vmax=v_max),
    colour_bar,
    plotting.ImShow(c=z2, vmin=v_min, vmax=v_max),
    colour_bar,
    figure_properties=plotting.FigureProperties(
        num_rows=2,
        num_cols=2,
        width_ratios=[1, 0.2],
        tight_layout=False,
        title="Shared colour bar",
    ),
)
mp.save("Shared colour bar", dir_name="images")
```

![](images/Shared_colour_bar.png)

*More complex examples coming soon*

### `util`

*Coming soon*

### `sweep`

*Coming soon*

(in the meantime, see [`scripts/make_logo.py`](scripts/make_logo.py) which made the logo above, and [unit tests](tests/) for [`util`](tests/test_util.py), [`plotting`](tests/test_plotting.py), and [`sweep`](tests/test_sweep.py))

## Unit tests

To run all unit tests, install [`pytest`](https://pypi.org/project/pytest/) (these tests have previously been run with `pytest` version 5.4.1), and run the following command (at the time of writing, this takes about 17 seconds to run 42 unit tests, because several unit tests involve saving images or GIFs to disk):

```
pytest
```

## Build package locally

`jutility` can be installed in "editable mode" using `pip` using the following command:

```
python -m pip install -e .
```

Alternatively, `jutility` can be built and installed locally using the following commands, replacing `$WHEEL_NAME` with the name of the wheel built by the `python -m build` command (for example, `jutility-0.0.5-py3-none-any.whl`):

```
python -m build
python -m pip install --force-reinstall --no-deps dist/$WHEEL_NAME
```

Or alternatively, run the script `build_local.py`, which automatically extracts the version number to infer the name of the wheel and then builds it:

```
python build_local.py
```

## Updating package on PyPI

This package was uploaded to PyPI following [the Packaging Python Projects tutorial in the official Python documentation](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

To update PyPI with a newer version, update the `version` tag in [`setup.cfg`](setup.cfg), and then use the following commands:

```
rm -rf dist/*
python -m build
python -m twine upload dist/*
```

When prompted by `twine`, enter `__token__` as the username, and paste an API token from the [PyPI account management webpage](https://pypi.org/manage/account/) as the password (including the `pypi-` prefix).
