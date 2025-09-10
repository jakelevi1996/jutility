import numpy as np
from jutility import plotting, util

OUTPUT_DIR = util.get_test_output_dir(
    "test_plotting/test_noisy/test_curve_sweep",
)

def test_noisycurvesweep():
    rng = util.get_numpy_rng("test_noisycurvesweep")

    ncs = plotting.NoisyCurveSweep()

    n = 100
    x = np.linspace(0, 3, n)

    for slope in np.linspace(1, 2, 4):
        for _ in range(rng.integers(3, 10)):
            y = slope * np.exp(-x) - slope + rng.normal(0, 0.1, n)
            ncs.update(slope, y)

    ncs_plot = ncs.plot(x, label_fmt=util.FloatFormatter(3))

    plotting.plot(
        *ncs_plot,
        plotting.Legend.from_plottables(*ncs_plot),
        plot_name="test_noisycurvesweep",
    )
