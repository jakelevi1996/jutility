import os
import numpy as np
import pytest
from jutility import plotting, util
import tests.util

OUTPUT_DIR = tests.util.get_output_dir("test_plotting")

def test_plot_lines():
    line_list = [
        plotting.Line([1, 2, 3], [4, 5, 7], c="b"),
        plotting.Line([1.6, 1.3, 1.8], [3.1, 5.6, 4], marker="o", c="r"),
        plotting.Line([1.4, 2.5], [3.5, 3.9], ls="--", c="g"),
        plotting.HVLine(h=5.3, v=2.2, c="m", zorder=-10, lw=10, alpha=0.4),
    ]
    mp = plotting.plot(
        *line_list,
        plot_name="test_plot_lines",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(xlabel="x", ylabel="y"),
    )
    assert os.path.isfile(mp.filename)

def test_line_shapes_defaults():
    nx = 150
    nf = 3

    x = np.linspace(0, 5, nx).reshape(1, nx)
    f = np.linspace(1, 2, nf).reshape(nf, 1)
    y = np.sin(x * f)

    x = x.reshape(nx)
    y = y.T

    assert list(x.shape) == [nx]
    assert list(y.shape) == [nx, nf]

    plotting.plot(
        plotting.Line(x, y, c="b"),
        plot_name="test_line_shapes_defaults with x and y",
        dir_name=OUTPUT_DIR,
    )
    plotting.plot(
        plotting.Line(y, c="b"),
        plot_name="test_line_shapes_defaults, x = None",
        dir_name=OUTPUT_DIR,
    )
    plotting.plot(
        plotting.Line(c="b"),
        plot_name="test_line_shapes_defaults, x = y = None",
        dir_name=OUTPUT_DIR,
    )

def test_plot_fill():
    mp = plotting.plot(
        plotting.FillBetween(
            x=[1, 2, 2.5],
            y1=[1.5, 2, 3],
            y2=[4, 3, 4.5],
            color="b",
            alpha=0.3,
        ),
        plotting.FillBetween(
            x=[1.3, 2.1, 3],
            y1=[4, 2, 3],
            y2=[5.5, 4, 4.5],
            color="r",
            alpha=0.3,
        ),
        plot_name="test_plot_fill",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(xlabel="x", ylabel="y"),
    )
    assert os.path.isfile(mp.filename)

def test_legend():
    line_list = [
        plotting.Line([1, 2], [1, 2], marker="o", c="r", label="Red line"),
        plotting.Line([1.2, 1.8], [1.8, 1.2], c="g", label="Green line"),
        plotting.Line([1.3, 1.7], [1.5, 1.6], marker="o", c="y"),
        plotting.HVLine(h=1.7, c="m", ls="--", label="hline"),
        plotting.FillBetween(
            x=[1.3, 1.6],
            y1=[1.2, 1.3],
            y2=[1.1, 1.0],
            fc="b",
            alpha=0.5,
            label="Patch",
        ),
    ]
    axis_properties = plotting.AxisProperties(xlabel="x", ylabel="y")
    mp = plotting.plot(
        *line_list,
        plot_name="test_legend",
        dir_name=OUTPUT_DIR,
        axis_properties=axis_properties,
        legend=True,
    )
    assert os.path.isfile(mp.filename)

def test_plot_bar():
    x1 = "Red" * 10
    x2 = "Green" * 5
    mp = plotting.plot(
        plotting.Bar(x1, 3.1, color="r", zorder=10, label="Bar 1"),
        plotting.Bar(x2, 4.3, color="g", zorder=10, label="Bar 2"),
        plot_name="test_plot_bar",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(
            xlabel="Category",
            ylabel="Height",
            rotate_xticklabels=True,
        ),
        legend=True,
    )
    assert os.path.isfile(mp.filename)

def test_log_axes():
    x1 = [1, 2, 3, 4, 5, 6]
    y1 = 1e-3 * np.array([1.2, 6, 120, 600, 1e4, 9e4])
    mp = plotting.plot(
        plotting.Line(x1, y1, c="b", marker="o"),
        plot_name="test_log_axes - log y axis",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties("x", "y", log_yscale=True),
    )
    assert os.path.isfile(mp.filename)

    x2 = [0.1, 1, 10, 100, 1000]
    y2 = [3.8, 3.2, 1.8, 1.2, -1.2]
    mp = plotting.plot(
        plotting.Line(x2, y2, c="b", marker="o"),
        plot_name="test_log_axes - log x axis",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties("x", "y", log_xscale=True),
    )
    assert os.path.isfile(mp.filename)

    x3 = [1, 10, 100, 1000]
    noise = np.array([0.4, 1.8, 0.3, 2.2])
    y3 = 1e-4 * np.power(x3, 2.3) * noise
    mp = plotting.plot(
        plotting.Line(x3, y3, c="b", marker="o"),
        plot_name="test_log_axes - log both axes",
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(
            xlabel="x",
            ylabel="y",
            log_xscale=True,
            log_yscale=True,
        ),
    )
    assert os.path.isfile(mp.filename)

@pytest.mark.parametrize("num_colours, cyclic", [[5, True], [7, False]])
def test_colour_picker(num_colours, cyclic):
    cp = plotting.ColourPicker(num_colours, cyclic)
    x = np.linspace(-1, 7, 100)
    line_list = [
        plotting.Line(
            x=x,
            y=((1 + (i/10)) * np.sin(x + (i / num_colours))),
            c=cp(i),
            label="Line %i" % i,
        )
        for i in range(num_colours)
    ]
    mp = plotting.plot(
        *line_list,
        plot_name="test_colour_picker, cyclic=%s" % cyclic,
        dir_name=OUTPUT_DIR,
        legend=True,
    )
    assert os.path.isfile(mp.filename)

def test_title():
    title = (
        "This is a very long title containing /|\\*:<\"$pecial?\">:*/|\\ "
        "characters which wraps multiple lines because it is too long for "
        "one line. It also contains $\\sum_{{i}}{{\\left[\\frac{{latex}}{{"
        "\\alpha_i^\\beta}}\\right]}}$"
    )
    mp = plotting.plot(
        plotting.Line(
            x=[1, 2, 3],
            y=[4, 4.5, 6],
            c="b",
            marker="o",
            label="$\\beta ^ \\varepsilon$",
        ),
        plot_name=title,
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties(
            xlabel="$x_1$",
            ylabel="$x_2$",
        ),
        legend=True,
    )
    assert os.path.isfile(mp.filename)

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

def test_colourbar():
    rng = util.Seeder().get_rng("test_colourbar")
    z1 = rng.random((100, 200)) + 5
    z2 = rng.random((100, 200)) + 2
    v_min = min(z1.min(), z2.min())
    v_max = max(z1.max(), z2.max())

    colour_bar = plotting.ColourBar(v_min, v_max)

    mp = plotting.MultiPlot(
        plotting.ImShow(c=z1, vmin=v_min, vmax=v_max),
        colour_bar,
        plotting.ImShow(c=z2, vmin=v_min, vmax=v_max),
        colour_bar,
        figure_properties=plotting.FigureProperties(
            num_rows=2,
            num_cols=2,
            width_ratios=[1, 0.2],
            tight_layout=False,
            title="Shared colour bar",
        ),
    )
    mp.save("test_colourbar", OUTPUT_DIR)
    assert os.path.isfile(mp.filename)
