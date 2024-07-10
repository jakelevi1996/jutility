import os
import numpy as np
import pytest
from jutility import plotting, util
import tests

OUTPUT_DIR = tests.get_output_dir("test_plotting")

def test_plot_lines():
    line_list = [
        plotting.Line([1, 2, 3], [4, 5, 7], c="b"),
        plotting.Line([1.6, 1.3, 1.8], [3.1, 5.6, 4], marker="o", c="r"),
        plotting.Line([1.4, 2.5], [3.5, 3.9], ls="--", c="g"),
        plotting.HLine(5.3, c="m", zorder=-10, lw=10, alpha=0.4),
        plotting.VLine(2.2, c="m", zorder=-10, lw=10, alpha=0.4),
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
        plotting.HLine(1.7, c="m", ls="--", label="hline"),
        plotting.FillBetween(
            x=[1.3, 1.6],
            y1=[1.2, 1.3],
            y2=[1.1, 1.0],
            fc="b",
            alpha=0.5,
            label="Patch",
        ),
    ]
    mp = plotting.plot(
        *line_list,
        plot_name="test_legend",
        dir_name=OUTPUT_DIR,
        xlabel="x",
        ylabel="y",
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
        xlabel="Category",
        ylabel="Height",
        rotate_xticklabels=True,
        legend=True,
    )
    assert os.path.isfile(mp.filename)

def test_log_axes():
    x1 = [1, 2, 3, 4, 5, 6]
    y1 = 1e-3 * np.array([1.2, 6, 120, 600, 1e4, 9e4])
    sp_list = []
    sp = plotting.Subplot(
        plotting.Line(x1, y1, c="b", marker="o"),
        axis_properties=plotting.AxisProperties("x", "y", log_yscale=True),
    )
    sp_list.append(sp)

    x2 = [0.1, 1, 10, 100, 1000]
    y2 = [3.8, 3.2, 1.8, 1.2, -1.2]
    sp = plotting.Subplot(
        plotting.Line(x2, y2, c="b", marker="o"),
        axis_properties=plotting.AxisProperties("x", "y", log_xscale=True),
    )
    sp_list.append(sp)

    x3 = [1, 10, 100, 1000]
    noise = np.array([0.4, 1.8, 0.3, 2.2])
    y3 = 1e-4 * np.power(x3, 2.3) * noise
    sp = plotting.Subplot(
        plotting.Line(x3, y3, c="b", marker="o"),
        xlabel="x",
        ylabel="y",
        log_xscale=True,
        log_yscale=True,
    )
    sp_list.append(sp)
    mp = plotting.MultiPlot(*sp_list, num_rows=1)
    mp.save("test_log_axes", OUTPUT_DIR)
    assert os.path.isfile(mp.filename)

@pytest.mark.parametrize("num_colours, cyclic", [[5, True], [7, False]])
def test_colour_picker(num_colours, cyclic):
    cp = plotting.ColourPicker(num_colours, cyclic)
    x = np.linspace(-1, 7, 100)
    line_list = [
        plotting.Line(
            x,
            ((1 + (i/10)) * np.sin(x + (i / num_colours))),
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

def test_colour_picker_next():
    num_colours = 5
    cp = plotting.ColourPicker(num_colours)
    x = np.linspace(-1, 7, 200)
    line_list = [
        plotting.Line(
            x,
            np.sin((1 + i / num_colours) * x),
            c=cp.next(),
            label="Line %i" % i,
        )
        for i in range(2 * num_colours)
    ]
    mp = plotting.plot(
        *line_list,
        plot_name="test_colour_picker_next",
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
            [1, 2, 3],
            [4, 4.5, 6],
            c="b",
            marker="o",
            label="$\\beta ^ \\varepsilon$",
        ),
        plot_name=title,
        dir_name=OUTPUT_DIR,
        xlabel="$x_1$",
        ylabel="$x_2$",
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
            ylim=[-0.2, 1.2],
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
            ylim=[-0.2, 1.2],
            legend=True,
        )
        filename_list.append(mp.filename)

    gif = plotting.Gif()
    for f in filename_list:
        gif.add_image_file_frame(os.path.basename(f), os.path.dirname(f))
    gif.save(output_name, OUTPUT_DIR, frame_duration_ms=500)
    assert os.path.isfile(gif.filename)

@pytest.mark.parametrize("constrained_layout", [True, False])
def test_colourbar(constrained_layout):
    rng = util.Seeder().get_rng("test_colourbar")
    z1 = rng.random((100, 200)) + 5
    z2 = rng.random((100, 200)) + 2
    v_min = min(z1.min(), z2.min())
    v_max = max(z1.max(), z2.max())

    colour_bar = plotting.ColourBar(v_min, v_max)

    plot_name = "test_colourbar, constrained_layout=%s" % constrained_layout
    mp = plotting.MultiPlot(
        plotting.ImShow(z1, vmin=v_min, vmax=v_max),
        colour_bar,
        plotting.ImShow(z2, vmin=v_min, vmax=v_max),
        colour_bar,
        figure_properties=plotting.FigureProperties(
            num_rows=2,
            num_cols=2,
            width_ratios=[1, 0.2],
            tight_layout=False,
            constrained_layout=constrained_layout,
            title=plot_name,
        ),
    )
    mp.save(plot_name, OUTPUT_DIR)
    assert os.path.isfile(mp.filename)

def test_quiver():
    n = 25
    x = np.linspace(-2, 2, n).reshape(1, n)
    y = np.linspace(-2, 2, n).reshape(n, 1)
    dx = y - x
    dy = x + y
    mp = plotting.MultiPlot(
        plotting.Subplot(
            plotting.Quiver(x, y, dx, dy, zorder=10, normalise=False),
            axis_properties=plotting.AxisProperties(title="normalise=False")
        ),
        plotting.Subplot(
            plotting.Quiver(x, y, dx, dy, zorder=10, normalise=True),
            axis_properties=plotting.AxisProperties(title="normalise=True")
        ),
        figure_properties=plotting.FigureProperties(figsize=[10, 4])
    )
    mp.save(plot_name="test_quiver", dir_name=OUTPUT_DIR)
    assert os.path.isfile(mp.filename)

def test_text():
    plotting.plot(
        plotting.Text(0.5, 0.5, "Text example", size=60, center_align=True),
        axis_properties=plotting.AxisProperties(axis_off=True, title=""),
        plot_name="test_text",
        dir_name=OUTPUT_DIR,
    )

def test_save_pdf():
    n = 10
    x = np.linspace(0, 4, n).reshape(1, n)
    y = np.linspace(0, 4, n).reshape(n, 1)
    dx = y - x
    dy = x + y
    for pdf in [True, False]:
        plotting.plot(
            plotting.Line([1, 2, 3], [1, 3, 2], c="b", marker="o"),
            plotting.Text(2, 3, "Example text", size=30),
            plotting.Quiver(x, y, dx, dy, zorder=10, normalise=False),
            plot_name="test_save_pdf, pdf = %s" % pdf,
            dir_name=OUTPUT_DIR,
            pdf=pdf,
        )

@pytest.mark.parametrize("pdf", [True, False])
def test_set_latex_params(pdf):
    for use_tex in [True, False]:
        plotting.set_latex_params(use_tex=use_tex)

        plot_name = (
            "test_set_latex_params, use_tex=%s, pdf=%s" % (use_tex, pdf)
        )

        plotting.plot(
            plotting.Line(
                [1, 3, 2, 3, 4],
                [1, 2, 3, 3, 4],
                c="b",
                marker="o",
                label="label",
                zorder=20,
            ),
            plotting.FillBetween(
                [0, 5],
                [-1, 1],
                [1, 8],
                c="b",
                alpha=0.3,
                label="$\\sigma$",
                lw=0,
                zorder=10,
            ),
            plotting.Text(1, 6, "Example text", size=10),
            grid=False,
            xlabel="$x$ label",
            ylabel="$\\mathcal{L}$",
            figsize=[5, 3],
            legend=True,
            plot_name=plot_name,
            dir_name=OUTPUT_DIR,
            pdf=pdf,
        )

    plotting.set_latex_params(use_tex=False)

def test_get_image_array():
    plot_name = "test_get_image_array"
    mp = plotting.plot(
        plotting.Text(0.5, 0.5, plot_name, center_align=True, size=10),
        axis_properties=plotting.AxisProperties(axis_off=True),
        figsize=[2, 2],
        save_close=False,
    )
    image_array = mp.get_image_array()
    mp.close()
    sp = plotting.Subplot(
        plotting.ImShow(image_array),
        axis_properties=plotting.AxisProperties(axis_off=True),
    )
    sp_list = [sp] * 9
    mp = plotting.MultiPlot(*sp_list)
    mp.save(
        plot_name=plot_name,
        dir_name=OUTPUT_DIR,
    )

def test_legend_in_existing_subplot():
    x = [1, 2, 3]
    mp = plotting.MultiPlot(
        plotting.Subplot(
            plotting.Line(x, [1, 3, 2], c="b", marker="o", label="Blue"),
            plotting.Line(x, [2, 1, 3], c="r", marker="o", label="Red"),
            plotting.Legend(),
            grid=False,
            title="First plot",
        ),
        plotting.Subplot(
            plotting.Line(x, [3, 1, 2], c="g", marker="o", label="Green"),
            plotting.Line(x, [2, 3, 1], c="c", marker="o", label="Cyan"),
            plotting.Legend(loc="lower left"),
            grid=False,
            title="Second plot",
        ),
        figsize=[6, 3],
    )
    mp.save(
        plot_name="test_legend_in_existing_subplot",
        dir_name=OUTPUT_DIR,
    )

def test_gif_add_bw_array_frame():
    test_name = "test_gif_add_bw_array_frame"
    rng = util.Seeder().get_rng(test_name)

    gif = plotting.Gif()
    for i in range(20):
        x = rng.uniform(0, 1, [50, 100])
        x[:i] = 0
        gif.add_bw_array_frame(x)

    gif.save(test_name, OUTPUT_DIR, frame_duration_ms=100)

def test_gif_add_rgb_array_frame():
    test_name = "test_gif_add_rgb_array_frame"
    rng = util.Seeder().get_rng(test_name)

    gif = plotting.Gif()
    for i in range(20):
        x = rng.uniform(0, 1, [50, 100, 3])
        x[:i] = 0
        gif.add_rgb_array_frame(x)

    gif.save(test_name, OUTPUT_DIR, frame_duration_ms=100)

def test_gif_add_bw_array_sequence():
    test_name = "test_gif_add_bw_array_sequence"
    rng = util.Seeder().get_rng(test_name)

    num_frames = 20
    width = 50
    height = 100
    x = rng.uniform(0, 1, [num_frames, width, height])
    for i in range(num_frames):
        x[i, :, :i] = 0

    gif = plotting.Gif()
    gif.add_bw_array_sequence(x)
    gif.save(test_name, OUTPUT_DIR, frame_duration_ms=100)

def test_gif_add_rgb_array_sequence():
    test_name = "test_gif_add_rgb_array_sequence"
    rng = util.Seeder().get_rng(test_name)

    num_frames = 20
    width = 50
    height = 100
    x = rng.uniform(0, 1, [num_frames, width, height, 3])
    for i in range(num_frames):
        x[i, :, :i] = 0

    gif = plotting.Gif()
    gif.add_rgb_array_sequence(x)
    gif.save(test_name, OUTPUT_DIR, frame_duration_ms=100)

def test_axline():
    rng = util.Seeder().get_rng("test_axline")
    num_lines = 10
    cp = plotting.ColourPicker(num_lines)
    plotting.plot(
        *[
            plotting.AxLine(
                rng.uniform(-1, 1, 2),
                rng.uniform(-1, 1, 2),
                c=cp(i),
            )
            for i in range(num_lines)
        ],
        plot_name="test_axline",
        dir_name=OUTPUT_DIR,
    )

def test_circle():
    rng = util.Seeder().get_rng("test_circle")
    num_circles = 10
    cp = plotting.ColourPicker(num_circles)
    plotting.plot(
        *[
            plotting.Circle(
                rng.uniform(-1, 1, 2),
                np.exp(rng.uniform(-1, 1)),
                c=cp(i),
            )
            for i in range(num_circles)
        ],
        xlim=[-5, 5],
        ylim=[-5, 5],
        axis_equal=True,
        grid=False,
        plot_name="test_circle",
        dir_name=OUTPUT_DIR,
    )

def test_scatter():
    rng = util.Seeder().get_rng("test_scatter")
    n = lambda s: rng.normal(size=s)
    plotting.plot(
        plotting.Scatter(n(50), n(50), c=n(50), cmap="binary"),
        plotting.Scatter(n(20), n(20), c=n(20)),
        plotting.Scatter(n(30), n(30), c="r", label="red"),
        legend=True,
        plot_name="test_scatter",
        dir_name=OUTPUT_DIR,
    )

def test_imshow_axis_off():
    rng = util.Seeder().get_rng("test_imshow_axis_off")
    x = rng.normal(size=[20, 20])
    mp = plotting.MultiPlot(
        plotting.Subplot(plotting.ImShow(x)),
        plotting.Subplot(plotting.ImShow(x, axis_off=True)),
        plotting.Subplot(plotting.ImShow(x, axis_off=False)),
        plotting.Subplot(plotting.ImShow(x, axis_off=False, z=0)),
        title="test_imshow_axis_off",
        constrained_layout=True,
    )
    mp.save("test_imshow_axis_off", OUTPUT_DIR)

def test_default_colours():
    rng = util.Seeder().get_rng("test_default_colours")
    n = lambda s: rng.normal(size=s)
    mp = plotting.MultiPlot(
        plotting.Subplot(plotting.Line(n(20), n(20))),
        plotting.Subplot(plotting.Step(n(20), n(20))),
        plotting.Subplot(plotting.Scatter(n(20), n(20), c="b")),
        plotting.Subplot(plotting.Scatter(n(20), n(20), c=n(20))),
        plotting.Subplot(
            plotting.Circle([1, 2], 3),
            plotting.Circle([2, 1], 3),
            plotting.Circle([1, 1], 1, c="r"),
            plotting.Text(-4, -4, "Circles", size=50),
            xlim=[-6, 6],
            ylim=[-6, 6],
        ),
        plotting.Subplot(plotting.Bar(np.arange(50), n(50))),
        plotting.Subplot(plotting.Bar(np.arange(50), n(50), c="r")),
        plotting.Subplot(plotting.Hist(n(1000), 100)),
        plotting.Subplot(
            plotting.FillBetween(np.arange(20), n(20), n(20), a=0.2),
        ),
        plotting.Subplot(
            plotting.FillBetween(np.arange(20), n(20), n(20), a=0.2, c="r"),
            plotting.FillBetween(np.arange(20), n(20), n(20), a=0.2, ec="b"),
        ),
        plotting.Subplot(plotting.Contour(n([50, 50])), axis_off=True),
        plotting.Subplot(plotting.ContourFilled(n([50, 50])), axis_off=True),
        title="test_default_colours",
        constrained_layout=True,
        figsize=[15, 10],
    )
    mp.save("test_default_colours", OUTPUT_DIR)

def test_legend_labels():
    rng = util.Seeder().get_rng("test_legend_labels")
    n = lambda s: rng.normal(size=s)
    plotting.plot(
        plotting.Scatter(n(20), n(20), c="r"),
        plotting.Scatter(n(20), n(20), color="c"),
        plotting.Scatter(n(20), n(20), c=n(20), cmap="binary"),
        plotting.Scatter([], [],            c="r",  label="red"),
        plotting.Line(sorted(n(20)), n(20),         label="blue"),
        plotting.Line(                              label="blue2"),
        plotting.Line(                      c="g",  label="green"),
        plotting.Line([], [],               c="c",  label="cyan"),
        plotting.FillBetween([], [], a=0.5, c="m",  label="patch"),
        plotting.Legend(),
        legend=True,
        grid=False,
        figsize=[6, 4],
        plot_name="test_legend_labels",
        dir_name=OUTPUT_DIR,
    )
