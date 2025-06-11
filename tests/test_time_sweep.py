import os
import time
import numpy as np
import pytest
from jutility import util, plotting
import test_utils

OUTPUT_DIR = test_utils.get_output_dir("test_time_sweep")

def test_time_sweep():
    rng = util.Seeder().get_rng("test_time_sweep")
    printer = util.Printer("table", OUTPUT_DIR)

    class Matmul(plotting.time_sweep.Experiment):
        def __init__(self, rng):
            self.rng = rng

        def setup(self, n):
            self.x1 = rng.normal(size=[n, n])
            self.x2 = rng.normal(size=[n, n])

        def run(self):
            return self.x1 @ self.x2

    class Solve(plotting.time_sweep.Experiment):
        def __init__(self, rng):
            self.rng = rng

        def setup(self, n):
            self.a = rng.normal(size=[n, n])
            self.b = rng.normal(size=n)

        def run(self):
            return np.linalg.solve(self.a, self.b)

    class LstSq(plotting.time_sweep.Experiment):
        def __init__(self, rng):
            self.rng = rng

        def setup(self, n):
            self.a = rng.normal(size=[n, n])
            self.b = rng.normal(size=n)

        def run(self):
            return np.linalg.lstsq(self.a, self.b, rcond=None)

    class Eig(plotting.time_sweep.Experiment):
        def __init__(self, rng):
            self.rng = rng

        def setup(self, n):
            self.x = rng.normal(size=[n, n])

        def run(self):
            return np.linalg.eig(self.x)

    plotting.time_sweep.run(
        *[t(rng) for t in [Matmul, Solve, LstSq, Eig]],
        n_list=util.log_range(10, 100, 10, unique_integers=True),
        printer=printer,
        dir_name=OUTPUT_DIR,
    )

