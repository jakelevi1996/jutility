# jutility

A Judicious collection of Python Utilities (including CLI configuration, plotting, and tables) and component of the [JUML](https://github.com/jakelevi1996/juml) framework.

![](https://raw.githubusercontent.com/jakelevi1996/jutility/main/images/logo_black.png)

## Contents

- [jutility](#jutility)
  - [Contents](#contents)
  - [Install with `pip`](#install-with-pip)
  - [Usage examples](#usage-examples)
  - [Unit tests](#unit-tests)

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

*Outdated; TODO*

(in the meantime, see [`scripts/make_logo.py`](scripts/make_logo.py) which made the logo above, and [unit tests](tests/) for [`util`](tests/test_util.py), [`plotting`](tests/test_plotting.py), and [`cli`](tests/test_cli.py))

## Unit tests

To run all unit tests, install [`pytest`](https://pypi.org/project/pytest/) (these tests have previously been run with `pytest` version 5.4.1), and run the following command (at the time of writing, this takes about 17 seconds to run 42 unit tests, because several unit tests involve saving images or GIFs to disk):

```
pytest
```
