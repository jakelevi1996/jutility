import numpy as np
from jutility import plotting, util

OUTPUT_DIR = util.get_test_output_dir("test_plotting/test_noisy/test_sweep")

def test_label_fmt():
    rng = util.get_numpy_rng("test_label_fmt")

    ns = plotting.NoisySweep()

    for k in np.linspace(11, 31, 7).tolist():
        for x in np.linspace(-5, 12, 11).tolist():
            for _ in range(3):
                y = k*x*x + rng.normal(0, 200)
                ns.update(k, x, y)

    default_lines   = ns.plot()
    formatted_lines = ns.plot(label_fmt=util.FloatFormatter(precision=2))

    mp = plotting.MultiPlot(
        plotting.Subplot(
            *default_lines,
            plotting.Legend.from_plottables(*default_lines),
            title="Default",
        ),
        plotting.Subplot(
            *formatted_lines,
            plotting.Legend.from_plottables(*formatted_lines),
            title="Formatted",
        ),
    )
    mp.save("test_label_fmt", OUTPUT_DIR)

def test_colour_mesh():
    rng = util.get_numpy_rng("test_colour_mesh")

    nx = 20
    ny = 30
    x = np.linspace(-3, 3, nx).reshape(1, nx)
    y = np.linspace(-3, 3, ny).reshape(ny, 1)

    ns = plotting.NoisySweep()
    ns_log = plotting.NoisySweep(log_z=True)

    for xi in x.flatten().tolist():
        for yi in y.flatten().tolist():
            for _ in range(rng.integers(3, 5)):
                z = np.exp(-(xi*xi + yi*yi))

                ns.update(yi, xi, z + rng.normal(0, 0.05))
                ns_log.update(yi, xi, z)

    mp = plotting.MultiPlot(
        plotting.Subplot(ns.colour_mesh()),
        plotting.Subplot(ns_log.colour_mesh()),
        ns.colour_bar(horizontal=True),
        ns_log.colour_bar(horizontal=True),
        hr=[1, 0.1],
        title="test_colour_mesh",
    )
    mp.save("test_colour_mesh", OUTPUT_DIR)
