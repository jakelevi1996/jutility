import numpy as np
from jutility import plotting, util

OUTPUT_DIR = util.get_test_output_dir("test_plotting/test_noisy/test_sweep")

def test_colour_mesh():
    rng = util.get_numpy_rng("test_colour_mesh")

    nx = 20
    ny = 30
    x = np.linspace(-3, 3, nx).reshape(1, nx)
    y = np.linspace(-3, 3, ny).reshape(ny, 1)

    ns = plotting.NoisySweep()

    for xi in x.flatten().tolist():
        for yi in y.flatten().tolist():
            for _ in range(rng.integers(3, 5)):
                z = np.exp(-(xi*xi + yi*yi)) + rng.normal(0, 0.02)

            ns.update(yi, xi, z)

    mp = plotting.MultiPlot(
        plotting.Subplot(
            ns.colour_mesh(vmin=min(ns), vmax=max(ns)),
        ),
        plotting.ColourBar(
            vmin=min(ns),
            vmax=max(ns),
            ticks=[min(ns), max(ns), 0.5],
        ),
        width_ratios=[1, 0.1],
        title="NoisySweep.colour_mesh",

    )
    mp.save("test_colour_mesh", OUTPUT_DIR)
