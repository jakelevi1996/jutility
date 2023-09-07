import os
import numpy as np
import pytest
from jutility import transform, plotting, util
import tests.util

OUTPUT_DIR = tests.util.get_output_dir("test_latex")

TOL = 1e-8

def test_least_squares_affine_transform():
    rng = util.Seeder().get_rng("test_least_squares_affine_transform")
    printer = util.Printer("test_least_squares_affine_transform", OUTPUT_DIR)

    nx = 4
    ny = 6
    nd = 10
    x = rng.normal(size=[nx, nd])
    w = rng.normal(size=[ny, nx])
    b = rng.normal(size=[ny, 1])
    y = w @ x + b

    f = transform.least_squares_affine_transform(x, y, reg=1e-10)
    w_ls, b_ls = f.w, f.b

    printer(y - f(x))
    printer(w - w_ls)
    printer(b - b_ls)
    assert np.max(np.abs(y - f(x))) < TOL
    assert np.max(np.abs(w - w_ls)) < TOL
    assert np.max(np.abs(b - b_ls)) < TOL
