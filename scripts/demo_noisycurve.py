import numpy as np
from jutility import plotting, util

rng = np.random.default_rng(0)

nc = plotting.NoisyCurve()

n = 100
x = np.linspace(-1, 3, n)

for i in range(10):
    y = np.exp(-x) + rng.normal(0, 0.1, n) + 0.1*i
    nc.update(y)

nc_plot = nc.plot(x, label="NoisyCurve")

plotting.plot(
    nc_plot,
    plotting.Legend.from_plottables(nc_plot),
    plot_name="demo_noisycurve",
    dir_name="images",
)
