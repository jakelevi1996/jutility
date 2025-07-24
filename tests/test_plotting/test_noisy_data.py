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
