import os
import numpy as np
import pytest
from jutility import transform, plotting, util
import test_utils

OUTPUT_DIR = test_utils.get_output_dir("test_transform")

TOL = 1e-8

def test_least_squares_linear():
    rng = util.Seeder().get_rng("test_least_squares_linear")
    printer = util.Printer("test_least_squares_linear", OUTPUT_DIR)

    nx = 4
    ny = 6
    nd = 10
    x = rng.normal(size=[nx, nd])
    w = rng.normal(size=[ny, nx])
    y = w @ x

    f = transform.least_squares_linear(x, y, reg=1e-10)

    printer(y, f(x), y - f(x), sep="\n")
    printer(w, f.w , w - f.w , sep="\n")
    assert np.max(np.abs(y - f(x))) < TOL
    assert np.max(np.abs(w - f.w )) < TOL

def test_least_squares_affine():
    rng = util.Seeder().get_rng("test_least_squares_affine")
    printer = util.Printer("test_least_squares_affine", OUTPUT_DIR)

    nx = 4
    ny = 6
    nd = 10
    x = rng.normal(size=[nx, nd])
    w = rng.normal(size=[ny, nx])
    b = rng.normal(size=[ny, 1])
    y = w @ x + b

    f = transform.least_squares_affine(x, y, reg=1e-10)
    w_ls, b_ls = f.w, f.b

    printer(y, f(x), y - f(x), sep="\n")
    printer(w, w_ls, w - w_ls, sep="\n")
    printer(b, b_ls, b - b_ls, sep="\n")
    assert np.max(np.abs(y - f(x))) < TOL
    assert np.max(np.abs(w - w_ls)) < TOL
    assert np.max(np.abs(b - b_ls)) < TOL
