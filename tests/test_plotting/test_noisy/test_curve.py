import numpy as np
from jutility import plotting, util

OUTPUT_DIR = util.get_test_output_dir("test_plotting/test_noisy/test_curve")

def test_noisycurve():
    rng = util.get_numpy_rng("test_noisycurve")

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
        plot_name="test_noisycurve",
        dir_name=OUTPUT_DIR,
    )
