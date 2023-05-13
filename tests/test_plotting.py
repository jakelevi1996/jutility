import os
import numpy as np
import pytest
from jutility import plotting, util
import tests.util

OUTPUT_DIR = tests.util.get_output_dir("test_plotting")

@pytest.mark.parametrize("num_subplots", range(1, 9))
def test_multiplot(num_subplots):
    rng = util.Seeder().get_rng("test_multiplot", num_subplots)
    x = np.linspace(0, 1)
    f = lambda x: x + 0.1 * rng.normal(size=x.shape)

    subplots = [
        plotting.Subplot(
            plotting.Scatter(x, f(x), c="b"),
            plotting.Scatter(x, f(x), c="r"),
            axis_properties=plotting.AxisProperties(
                title="Subplot %i" % subplot_ind,
            ),
        )
        for subplot_ind in range(num_subplots)
    ]
    plot_name = "%i subplots" % num_subplots
    mp = plotting.MultiPlot(
        *subplots,
        figure_properties=plotting.FigureProperties(
            title=plot_name,
            top_space=0.15,
        ),
    )
    mp.save(plot_name, OUTPUT_DIR)
    assert os.path.isfile(mp.filename)

@pytest.mark.parametrize("save_frames", [True, False])
def test_gif_add_plot_frame(save_frames):
    rng = util.Seeder().get_rng("test_gif_add_plot_frame", save_frames)
    x = np.linspace(0, 1)
    f = lambda x: x + 0.1 * rng.normal(size=x.shape)
    output_name = "test_gif_add_plot_frame, save_frames = %s" % save_frames

    gif = plotting.Gif()
    for i in range(5):
        gif.add_plot_frame(
            plotting.Scatter(x, f(x), c="b", label="Blue data"),
            plotting.Scatter(x, f(x), c="r", label="Red data"),
            plot_name="%s, frame %i" % (output_name, i),
            dir_name=OUTPUT_DIR,
            axis_properties=plotting.AxisProperties(ylim=[-0.2, 1.2]),
            legend=True,
            save=save_frames,
        )

    gif.save(output_name, OUTPUT_DIR, frame_duration_ms=500)
    assert os.path.isfile(gif.filename)

def test_gif_add_image_file_frame():
    rng = util.Seeder().get_rng("test_gif_add_image_file_frame")
    x = np.linspace(0, 1)
    f = lambda x: x + 0.1 * rng.normal(size=x.shape)
    output_name = "test_gif_add_image_file_frame"
    filename_list = []

    for i in range(5):
        mp = plotting.plot(
            plotting.Scatter(x, f(x), c="b", label="Blue data"),
            plotting.Scatter(x, f(x), c="r", label="Red data"),
            plot_name="%s, frame %i" % (output_name, i),
            dir_name=OUTPUT_DIR,
            axis_properties=plotting.AxisProperties(ylim=[-0.2, 1.2]),
            legend=True,
        )
        filename_list.append(mp.filename)

    gif = plotting.Gif()
    for f in filename_list:
        gif.add_image_file_frame(os.path.basename(f), os.path.dirname(f))
    gif.save(output_name, OUTPUT_DIR, frame_duration_ms=500)
    assert os.path.isfile(gif.filename)

@pytest.mark.parametrize("result_alpha", [0, 0.3])
def test_noisy_data(result_alpha):
    rng = util.Seeder().get_rng("test_noisy_data", result_alpha)
    noisy_data_blue = plotting.NoisyData()
    noisy_data_red  = plotting.NoisyData()
    x_list = np.linspace(0, 1)

    for x in x_list:
        for num_repeats in range(rng.integers(10)):
            noisy_data_blue.update(x, x + 0.04 * rng.normal())
        for num_repeats in range(rng.integers(10)):
            noisy_data_red.update(x, 0.3 + (0.3 * x) + (0.04 * rng.normal()))

    mp = plotting.plot(
        *noisy_data_blue.get_lines(
            colour="b",
            name="Blue data",
            result_alpha=result_alpha,
        ),
        *noisy_data_red.get_lines(
            colour="r",
            name="Red data",
            result_alpha=result_alpha,
        ),
        plot_name="test_noisy_data, result_alpha = %s" % result_alpha,
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties("x", "y", ylim=[-0.2, 1.2]),
        legend=True,
    )
    assert os.path.isfile(mp.filename)
