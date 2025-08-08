import numpy as np
from jutility import plotting, util

OUTPUT_DIR = util.get_test_output_dir("test_plotting/test_noisy_data")

def test_plot_alphas():
    rng = util.get_numpy_rng("test_plot_alphas")

    nd = plotting.NoisyData()

    for x in np.linspace(-1, 1, 25):
        for _ in range(5):
            nd.update(x, rng.normal(x*x, 0.1))

    mp = plotting.MultiPlot(
        plotting.Subplot(nd.plot()),
        plotting.Subplot(nd.plot(alpha_scat=0)),
        plotting.Subplot(nd.plot(alpha_line=0)),
        plotting.Subplot(nd.plot(alpha_fill=0)),
        figsize=[10, 8],
    )
    mp.save("test_plot_alphas", OUTPUT_DIR)

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
