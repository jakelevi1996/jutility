import numpy as np
from jutility import util

rng = np.random.default_rng(0)

table = util.Table(
    util.TimeColumn(),
    util.CountColumn(width=10),
    util.Column("k", ".5f", 20),
    util.Column("v", ".5f", 20),
    print_interval=util.TimeInterval(1)
)

with util.Timer("table") as t:
    while t.get_time_taken() < 5:
        table.update(k=rng.normal(), v=rng.normal())

with util.Timer("get_data"):
    t = table.get_data("t")
    c = table.get_data("c")

print("Time per update = %.6f ms" % ((t[-1] / c[-1]) * 1e3))
