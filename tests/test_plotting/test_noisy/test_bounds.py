import numpy as np
from jutility import plotting, util

OUTPUT_DIR = util.get_test_output_dir("test_plotting/test_noisy/test_bounds")

def test_summarise():
    rng = util.get_numpy_rng("test_summarise")

    x = np.sort(rng.uniform(0, 5, (200, 10)), axis=0)
    y = np.exp(-x) + rng.normal(0, 0.1, x.shape)

    xm,  _,  _ = plotting.summarise(x, num_split=50)
    ym, yu, yl = plotting.summarise(y, num_split=50, n_sigma=2)

    plotting.plot(
        plotting.Line(x, y, a=0.2, z=20),
        plotting.Line(xm, ym, a=1.0, z=30),
        plotting.FillBetween(xm, yl, yu, a=0.3, z=10),
        plot_name="test_summarise",
        dir_name=OUTPUT_DIR,
    )
