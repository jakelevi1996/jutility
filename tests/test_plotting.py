import os
import numpy as np
import pytest
from jutility import plotting, util
import test_utils

OUTPUT_DIR = test_utils.get_output_dir("test_plotting")

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
        xlabel="x",
        ylabel="y",
    )
    assert os.path.isfile(mp.full_path)

def test_plottable_repr():
    line = plotting.Line([1, 2], [3, 4], c="g", z=23, a=0.8, ls="--")
    scatter = plotting.Scatter([5, 6, 7], [8, 9, 10], color="r")
    assert repr(line) == "Line(alpha=0.8, color='g', ls='--', zorder=23)"
    assert repr(scatter) == "Scatter(color='r', zorder=10)"

def test_subplot_repr():
    line = plotting.Line([1, 2], [3, 4], c="g")
    assert repr(plotting.Subplot(line)) == "Subplot()"
    assert repr(plotting.Subplot(line, title="Subplot title")) == (
        "Subplot(title='Subplot title')"
    )

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
        xlabel="x",
        ylabel="y",
    )
    assert os.path.isfile(mp.full_path)

def test_legend():
    line_list = [
        plotting.Line([1, 2], [1, 2], marker="o", c="r", label="Red line"),
        plotting.Line([1.2, 1.8], [1.8, 1.2], c="g", label="Green line"),
        plotting.Line([1.3, 1.7], [1.5, 1.6], marker="o", c="y"),
        plotting.HLine(1.7, c="m", ls="--", label="hline"),
        plotting.FillBetween(
            [1.3, 1.6],
            [1.2, 1.3],
            [1.1, 1.0],
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
    assert os.path.isfile(mp.full_path)

def test_plot_bar():
    x1 = "Red" * 10
    x2 = "Green" * 5
    results = dict()
    for rotate in [True, False]:
        mp = plotting.plot(
            plotting.Bar(x1, 3.1, color="r", zorder=10, label="Bar 1"),
            plotting.Bar(x2, 4.3, color="g", zorder=10, label="Bar 2"),
            plot_name="test_plot_bar %s" % rotate,
            dir_name=OUTPUT_DIR,
            xlabel="Category",
            ylabel="Height",
            rotate_xticklabels=rotate,
            legend=True,
        )
        results[rotate] = mp.get_image_array()

    assert      np.all(results[True ] == results[True ])
    assert      np.all(results[False] == results[False])
    assert not  np.all(results[True ] == results[False])

def test_log_axes():
    x1 = [1, 2, 3, 4, 5, 6]
    y1 = 1e-3 * np.array([1.2, 6, 120, 600, 1e4, 9e4])
    sp_list = []
    sp = plotting.Subplot(
        plotting.Line(x1, y1, c="b", marker="o"),
        xlabel="x",
        ylabel="y",
        log_y=True,
    )
    sp_list.append(sp)

    x2 = [0.1, 1, 10, 100, 1000]
    y2 = [3.8, 3.2, 1.8, 1.2, -1.2]
    sp = plotting.Subplot(
        plotting.Line(x2, y2, c="b", marker="o"),
        xlabel="x",
        ylabel="y",
        log_x=True,
    )
    sp_list.append(sp)

    x3 = [1, 10, 100, 1000]
    noise = np.array([0.4, 1.8, 0.3, 2.2])
    y3 = 1e-4 * np.power(x3, 2.3) * noise
    sp = plotting.Subplot(
        plotting.Line(x3, y3, c="b", marker="o"),
        xlabel="x",
        ylabel="y",
        log_x=True,
        log_y=True,
    )
    sp_list.append(sp)
    mp = plotting.MultiPlot(*sp_list, num_rows=1)
    mp.save("test_log_axes", OUTPUT_DIR)
    assert os.path.isfile(mp.full_path)

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
    assert os.path.isfile(mp.full_path)

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
    assert os.path.isfile(mp.full_path)

def test_colour_picker_colourise():
    rng = util.Seeder().get_rng("test_colour_picker_colourise")
    lines = [
        plotting.Line(
            rng.uniform(0, 1, 20),
            label="Line %i" % i,
        )
        for i in range(5)
    ]
    nd = plotting.NoisyData()
    for x in range(20):
        for _ in range(5):
            nd.update(x, rng.uniform(1, 2))

    lines.append(nd.plot(label="NoisyData"))
    cp = plotting.ColourPicker(len(lines))
    cp.colourise(lines)
    plotting.plot(
        *lines,
        plotting.Legend.from_plottables(*lines, z=50),
        plot_name="test_colour_picker_colourise",
        dir_name=OUTPUT_DIR,
    )

    plotting.ColourPicker.from_colourise(lines, False)
    plotting.plot(
        *lines,
        plotting.Legend.from_plottables(*lines, z=50),
        plot_name="test_colour_picker_from_colourise",
        dir_name=OUTPUT_DIR,
    )

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
    assert os.path.isfile(mp.full_path)

@pytest.mark.parametrize("num_subplots", range(1, 9))
def test_multiplot(num_subplots):
    rng = util.Seeder().get_rng("test_multiplot", num_subplots)
    x = np.linspace(0, 1)
    f = lambda x: x + 0.1 * rng.normal(size=x.shape)

    subplots = [
        plotting.Subplot(
            plotting.Scatter(x, f(x), c="b"),
            plotting.Scatter(x, f(x), c="r"),
            title="Subplot %i" % subplot_ind,
        )
        for subplot_ind in range(num_subplots)
    ]
    plot_name = "%i subplots" % num_subplots
    mp = plotting.MultiPlot(
        *subplots,
        title=plot_name,
    )
    mp.save(plot_name, OUTPUT_DIR)
    assert os.path.isfile(mp.full_path)

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
    assert os.path.isfile(gif.full_path)

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
        filename_list.append(mp.full_path)

    gif = plotting.Gif()
    for f in filename_list:
        gif.add_image_file_frame(os.path.basename(f), os.path.dirname(f))
    gif.save(output_name, OUTPUT_DIR, frame_duration_ms=500)
    assert os.path.isfile(gif.full_path)

def test_colourbar():
    rng = util.Seeder().get_rng("test_colourbar")
    z1 = rng.random((100, 200)) + 5
    z2 = rng.random((100, 200)) + 2
    v_min = min(z1.min(), z2.min())
    v_max = max(z1.max(), z2.max())

    mp = plotting.MultiPlot(
        plotting.MultiPlot(
            plotting.Subplot(plotting.ImShow(z1, vmin=v_min, vmax=v_max)),
            plotting.Subplot(plotting.ImShow(z2, vmin=v_min, vmax=v_max)),
            num_cols=1,
        ),
        plotting.MultiPlot(
            plotting.ColourBar(v_min, v_max, label="ColourBar"),
        ),
        width_ratios=[1, 0.2],
        title="Shared colour bar",
        figsize=[8, 6],
    )
    mp.save("test_colourbar", OUTPUT_DIR)
    assert os.path.isfile(mp.full_path)

def test_colourbar_horizontal():
    rng = util.Seeder().get_rng("test_colourbar_horizontal")
    z1 = rng.random((100, 200)) + 5
    z2 = rng.random((100, 200)) + 2
    v_min = min(z1.min(), z2.min())
    v_max = max(z1.max(), z2.max())

    mp = plotting.MultiPlot(
        plotting.MultiPlot(
            plotting.Subplot(plotting.ImShow(z1, vmin=v_min, vmax=v_max)),
            plotting.Subplot(plotting.ImShow(z2, vmin=v_min, vmax=v_max)),
        ),
        plotting.MultiPlot(
            plotting.ColourBar(v_min, v_max, horizontal=True),
        ),
        num_cols=1,
        height_ratios=[1, 0.4],
        title="Shared colour bar",
        figsize=[8, 4],
    )
    mp.save("test_colourbar_horizontal", OUTPUT_DIR)
    assert os.path.isfile(mp.full_path)

def test_log_colourbar():
    z = 100
    x = np.linspace(0, z, 500)
    y = lambda n, z: (z**n - x**n)**(1/n)
    ticks = list(range(-z, z+1, 20))
    n_list = util.log_range(0.1, 10, 21)
    cp_n = plotting.ColourPicker(len(n_list))

    mp = plotting.MultiPlot(
        plotting.Subplot(
            *[
                line
                for i, n in enumerate(n_list)
                for line in
                [
                    plotting.Line( x,  y(n, z), c=cp_n(i)),
                    plotting.Line( x, -y(n, z), c=cp_n(i)),
                    plotting.Line(-x,  y(n, z), c=cp_n(i)),
                    plotting.Line(-x, -y(n, z), c=cp_n(i)),
                ]
            ],
            axis_equal=True,
            grid=False,
            xticks=ticks,
            yticks=ticks,
        ),
        plotting.ColourBar(
            n_list.min(),
            n_list.max(),
            "hsv",
            log=True,
            label="n",
        ),
        figsize=[8, 6],
        width_ratios=[1, 0.1],
    )
    mp.save("test_log_colourbar", OUTPUT_DIR)

def test_quiver():
    n = 25
    x = np.linspace(-2, 2, n).reshape(1, n)
    y = np.linspace(-2, 2, n).reshape(n, 1)
    dx = y - x
    dy = x + y
    mp = plotting.MultiPlot(
        plotting.Subplot(
            plotting.Quiver(x, y, dx, dy, zorder=10, normalise=False),
            title="normalise=False",
        ),
        plotting.Subplot(
            plotting.Quiver(x, y, dx, dy, zorder=10, normalise=True),
            title="normalise=True",
        ),
        figsize=[10, 4],
    )
    mp.save(plot_name="test_quiver", dir_name=OUTPUT_DIR)
    assert os.path.isfile(mp.full_path)

def test_text():
    plotting.plot(
        plotting.Text(0.5, 0.5, "Text example", fs=60),
        axis_off=True,
        title=None,
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
    with util.CallbackContext(
        exit_callback=(lambda: plotting.set_latex_params(use_tex=False)),
    ):
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
        plotting.Text(0.5, 0.5, plot_name, fs=10),
        axis_off=True,
        figsize=[2, 2],
        save_close=False,
    )
    image_array = mp.get_image_array()
    mp.close()
    sp = plotting.Subplot(
        plotting.ImShow(image_array),
        axis_off=True,
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
    lines = [
        plotting.AxLine(
            rng.uniform(-1, 1, 2),
            rng.uniform(-1, 1, 2),
            c=cp(i),
            label="lines[%i]" % i,
        )
        for i in range(num_lines)
    ]
    plotting.plot(
        *lines,
        plotting.Legend.from_plottables(*lines),
        plot_name="test_axline",
        dir_name=OUTPUT_DIR,
    )

def test_circle():
    plotting.plot(
        plotting.AxLine([0, 0], [1, 1]),
        plotting.Circle([0.0, 0.0], 0.1),
        plotting.Circle([0.1, 0.1], 0.1, fill=False),
        plotting.Circle([0.2, 0.2], 0.1, fc=None),
        plotting.Circle([0.3, 0.3], 0.1, ec=None),
        plotting.Circle([0.4, 0.4], 0.1, lw=5),
        plotting.Circle([0.5, 0.5], 0.1, ec="g"),
        plotting.Circle([0.6, 0.6], 0.1, fc="r"),
        plotting.Circle([0.7, 0.7], 0.1, fc=[0, 1, 0], ec=None),
        axis_equal=True,
        xlim=[-0.2, 1],
        ylim=[-0.2, 1],
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
        plotting.Line([], [],               c="c",  label="cyan"),
        plotting.FillBetween([], [], a=0.5, c="m",  label="patch"),
        plotting.Legend(),
        legend=True,
        grid=False,
        figsize=[6, 4],
        plot_name="test_legend_labels",
        dir_name=OUTPUT_DIR,
    )

def test_axis_title_colour():
    rng = util.Seeder().get_rng("test_axis_title_colour")

    n = 25
    x = np.linspace(0, 1, n)
    f = lambda x: x + rng.normal(0, 0.1, n)

    mp = plotting.MultiPlot(
        *[
            plotting.Subplot(
                plotting.Line(x, f(x), c=c, marker="o"),
                title="Plot title %i" % i,
                title_colour=c,
            )
            for i, c in enumerate(["k", "r", "g", "b"])
        ],
        figsize=[8, 6],
    )
    mp.save("test_axis_title_colour", OUTPUT_DIR)

def test_confidence_bounds():
    rng = util.Seeder().get_rng("test_confidence_bounds")
    printer = util.Printer("test_confidence_bounds", dir_name=OUTPUT_DIR)
    m = 19
    n = 7
    y1 = np.arange(m).reshape(m, 1) + 100
    y2 = np.arange(n).reshape(1, n) + 20
    y = y1 + y2 + rng.normal(size=[m, n])
    printer(y)
    subplots = []
    for split_dim in [0, 1]:
        mean, ucb, lcb = plotting.confidence_bounds(y, split_dim=split_dim)
        x = np.arange(mean.size)
        sp = plotting.Subplot(
            plotting.Line(x, mean, c="b", marker="o"),
            plotting.FillBetween(x, ucb, lcb, c="b", alpha=0.2),
        )
        subplots.append(sp)

    mp = plotting.MultiPlot(*subplots)
    mp.save("test_confidence_bounds", OUTPUT_DIR)

def test_confidence_bounds_split_dim():
    num_split = 5
    sp_list = []
    for n in [3, 456]:
        x = np.linspace(0, 5, n)
        y = np.tile(np.exp(x), [3, 1])
        assert list(x.shape) == [n]
        assert list(y.shape) == [3, n]

        mean, ucb, lcb = plotting.confidence_bounds(
            y,
            split_dim=1,
            num_split=num_split,
        )
        x_ds, _, _ = plotting.confidence_bounds(
            x,
            split_dim=0,
            num_split=num_split,
        )
        for i in [x_ds, mean, ucb, lcb]:
            assert isinstance(i, np.ndarray)
            assert list(i.shape) == [min(n, num_split)]

        sp = plotting.Subplot(
            plotting.Line(x, y[0], c="r", ls="--", z=40),
            plotting.Line(x_ds, mean, a=1.0, z=30, c="b", marker="o"),
            plotting.FillBetween(x_ds, lcb, ucb, a=0.2, z=10, c="b"),
            title="n = %i" % n,
        )
        sp_list.append(sp)

    mp = plotting.MultiPlot(*sp_list)
    mp.save("test_confidence_bounds_split_dim", OUTPUT_DIR)

def test_noisy_data():
    printer = util.Printer("test_noisy_data", OUTPUT_DIR)
    rng = util.Seeder().get_rng("test_noisy_data")
    noisy_data = plotting.NoisyData()
    x_list = sorted(rng.uniform(-3, 3, size=20))

    for x in x_list:
        for num_repeats in range(rng.integers(10)):
            noisy_data.update(x, x + 0.3 * rng.normal())

    printer(*noisy_data.get_all_data(), sep="\n")
    printer.hline()
    printer(*noisy_data.get_statistics(), sep="\n")
    printer.hline()
    printer(*noisy_data.get_statistics(1.5), sep="\n")

    all_x, all_y = noisy_data.get_all_data()
    x, mean, ucb, lcb = noisy_data.get_statistics()

    for i in [all_x, all_y, x, mean, ucb, lcb]:
        assert isinstance(i, np.ndarray)
        assert len(i.shape) == 1

    p_kw = {"ls": "--", "z": 50}
    x_kw = {"ls": "-",  "z": 40, "c": "r", "a": 0.2, "lw": 10}
    lines = [
        noisy_data.plot("b", label="Data"),
        noisy_data.predict_line(0, 5, label="Regression", **p_kw),
        plotting.AxLine([0, 0], [1, 1], label="y = x", **x_kw),
    ]
    plotting.plot(
        *lines,
        plotting.Legend.from_plottables(*lines),
        plot_name="test_noisy_data",
        dir_name=OUTPUT_DIR,
    )

def test_noisy_log_data():
    rng = util.Seeder().get_rng("test_noisy_log_data")
    noisy_log_data = plotting.NoisyData(log_y=True)
    noisy_data = plotting.NoisyData()
    x_list = util.log_range(0.01, 10, 50)
    for x in x_list:
        for repeat in range(rng.integers(10)):
            y = x * np.exp(rng.normal())
            noisy_log_data.update(x, y)
            noisy_data.update(x, y)


    g1 = noisy_log_data.plot(c="b", label="`log_y=True`")
    g2 = noisy_data.plot(c="r", label="`log_y=False`")
    mp = plotting.MultiPlot(
        plotting.Subplot(
            g1,
            title="noisy_log_data, log_y=True",
            log_x=True,
            log_y=True,
        ),
        plotting.Subplot(
            g2,
            title="noisy_data, log_y=False",
            log_x=True,
            log_y=True,
        ),
        legend=plotting.FigureLegend(g1, g2),
    )
    mp.save("test_noisy_log_data", OUTPUT_DIR)

def test_noisy_data_predict_log():
    rng = util.Seeder().get_rng("test_noisy_data_predict_log")
    printer = util.Printer("test_noisy_data_predict_log", OUTPUT_DIR)
    subplots = []
    for log_x in [False, True]:
        for log_y in [False, True]:
            data = plotting.NoisyData(log_x=log_x, log_y=log_y)

            for x in np.linspace(0, 3, 20):
                for _ in range(rng.integers(2, 10)):
                    x_data = x
                    y_data = 0.3 + 2.1*x + 0.1*rng.normal()
                    if log_x:
                        x_data = np.exp(x_data)
                    if log_y:
                        y_data = np.exp(y_data)

                    data.update(x_data, y_data)

            x = np.array([1, 6])
            y = data.predict(x)
            printer(log_x, log_y, *data.get_all_data(), sep="\n")
            printer.hline()

            sp = plotting.Subplot(
                data.plot("b", "Data"),
                plotting.AxLine(*zip(x, y), ls="--"),
                log_x=log_x,
                log_y=log_y,
                title="log_x=%s, log_y=%s" % (log_x, log_y),
            )
            subplots.append(sp)

    mp = plotting.MultiPlot(*subplots)
    mp.save("test_noisy_data_predict_log", OUTPUT_DIR)

def test_noisy_data_argmax_argmin():
    nd = plotting.NoisyData()

    for x, y in [
        (3, 10),
        (3, 14),
        (3, 13),
        (5, 9),
        (5, 20),
        (5, 12),
        (5, 11),
        (8, 19),
        (8, 22),
    ]:
        nd.update(x, y)

    max_x, max_repeat, max_y = nd.argmax()
    min_x, min_repeat, min_y = nd.argmin()

    assert max_x == 8
    assert max_repeat == 1
    assert max_y == 22
    assert min_x == 5
    assert min_repeat == 0
    assert min_y == 9

    plotting.plot(
        nd.plot(),
        nd.predict_line(2, 10, a=0.5),
        plotting.VLine(min_x, c="r", ls="--"),
        plotting.HLine(min_y, c="r", ls="--"),
        plotting.VLine(max_x, c="g", ls="--"),
        plotting.HLine(max_y, c="g", ls="--"),
        plot_name="test_noisy_data_argmax_argmin",
        dir_name=OUTPUT_DIR,
    )

def test_noisy_data_argmax_argmin_x_index():
    nd = plotting.NoisyData(x_index=True)

    for x, y in [
        ("abc", 10),
        ("abc", 14),
        ("abc", 13),
        ("defg", 9),
        ("defg", 20),
        ("defg", 12),
        ("defg", 11),
        ("hi", 19),
        ("hi", 22),
    ]:
        nd.update(x, y)

    max_x, max_repeat, max_y = nd.argmax()
    min_x, min_repeat, min_y = nd.argmin()

    print(nd.argmax(), nd.argmin())

    assert max_x == "hi"
    assert max_repeat == 1
    assert max_y == 22
    assert min_x == "defg"
    assert min_repeat == 0
    assert min_y == 9

    plotting.plot(
        nd.plot(),
        plotting.HLine(min_y, c="r", ls="--"),
        plotting.HLine(max_y, c="g", ls="--"),
        **nd.get_xtick_kwargs(),
        plot_name="test_noisy_data_argmax_argmin_x_index",
        dir_name=OUTPUT_DIR,
    )

@pytest.mark.parametrize("handles", [True, False])
def test_figure_legend(handles):
    test_name = "test_figure_legend, handles=%s" % handles
    rng = util.Seeder().get_rng(test_name)

    x = np.linspace(0, 1)
    y1 = rng.normal(x,      0.1)
    y2 = rng.normal(1 - x,  0.1)

    if handles:
        legend = plotting.FigureLegend(
            plotting.Line([], c="b", label="y1"),
            plotting.Line([], c="r", label="y2"),
        )
    else:
        legend = plotting.FigureLegend()

    mp = plotting.MultiPlot(
        plotting.Subplot(plotting.Line(x, y1, c="b", label="y1")),
        plotting.Subplot(plotting.Line(x, y2, c="r", label="y2")),
        legend=legend,
        title=test_name,
    )
    mp.save(test_name, OUTPUT_DIR)

def test_polygon():
    plotting.plot(
        plotting.Polygon(
            [-1, 1, 0],
            [0, 0, 2],
        ),
        plotting.Polygon(
            [3, 6, 2, 7],
            [7, 2, 6, 6],
            fc="r",
            ec="g",
            lw=10,
        ),
        plot_name="test_polygon",
        dir_name=OUTPUT_DIR,
    )

def test_random_polygon():
    rng = util.Seeder().get_rng("test_random_polygon")
    num_polygons = 7
    num_vertices = 5
    cp = plotting.ColourPicker(num_polygons)
    plotting.plot(
        *[
            plotting.Polygon(
                rng.uniform(0, 1, num_vertices),
                rng.uniform(0, 1, num_vertices),
                fc=cp.next(),
            )
            for _ in range(num_polygons)
        ],
        axis_off=True,
        title="",
        plot_name="test_random_polygon",
        dir_name=OUTPUT_DIR,
    )

def test_scatter_legend():
    rng = util.Seeder().get_rng("test_scatter_legend")
    x = lambda: rng.uniform(0, 1, 10)
    cp = plotting.ColourPicker(6)
    plotting.plot(
        plotting.Line(x(), x(), marker="o", c=cp.next(), label="line"),
        plotting.Line(x(), x(), marker="o", c=cp.next(), label="line", ms=6),
        plotting.Line(x(), x(), marker="o", c=cp.next(), label="line", ms=15),
        plotting.Scatter(x(), x(), color=cp.next(), label="scatter"),
        plotting.Scatter(x(), x(), color=cp.next(), label="scatter", s=36),
        plotting.Scatter(x(), x(), color=cp.next(), label="scatter", s=225),
        legend=True,
        plot_name="test_scatter_legend",
        dir_name=OUTPUT_DIR,
    )

def test_xticks():
    rng = util.Seeder().get_rng("test_xticks")

    line = plotting.Line(np.linspace(0, 1, 20), rng.normal(0, 1, 20))
    mp = plotting.MultiPlot(
        plotting.Subplot(
            line,
        ),
        plotting.Subplot(
            line,
            xticks=np.linspace(-1, 2, 11),
        ),
        plotting.Subplot(
            line,
            xlim=[0, 1],
        ),
        plotting.Subplot(
            line,
            xticks=np.linspace(-1, 2, 11),
            xlim=[0, 1],
        ),
    )
    mp.save("test_xticks", OUTPUT_DIR)

def test_xticklabels():
    rng = util.Seeder().get_rng("test_xticklabels")

    means = {
        "relu": 4,
        "sigmoid": 10,
        "ite": 6,
    }

    results = plotting.NoisyData()
    for _ in range(5):
        for i, name in enumerate(sorted(means.keys())):
            results.update(i, means[name] + rng.normal())

    plotting.plot(
        results.plot(),
        xticks=[0, 1, 2],
        xticklabels=sorted(means.keys()),
        plot_name="test_xticklabels",
        dir_name=OUTPUT_DIR,
    )

def test_yticks():
    plotting.plot(
        plotting.Line([1, 3, 2]),
        yticks=[0.3, 1.1, 1.2, 2.5, 3.1],
        plot_name="test_yticks",
        dir_name=OUTPUT_DIR,
    )

def test_yticklabels():
    plotting.plot(
        plotting.Line([1, 3, 2]),
        yticks=[0.3, 1.1, 1.2, 2.5, 3.1],
        yticklabels=["frog", "bog", "grog", "log", "dog"],
        plot_name="test_yticklabels",
        dir_name=OUTPUT_DIR,
    )

def test_noisydata_x_index():
    rng = util.Seeder().get_rng("test_noisydata_x_index")

    means = {
        "relu": 4,
        "sigmoid": 10,
        "ite": 6,
    }

    results = plotting.NoisyData()
    results_index = plotting.NoisyData(x_index=True)
    for _ in range(5):
        for i, name in enumerate(sorted(means.keys())):
            y = means[name] + rng.normal()
            results.update(i, y)
            results_index.update(name, y)

    mp = plotting.MultiPlot(
        plotting.Subplot(
            results.plot(),
            **results.get_xtick_kwargs(),
        ),
        plotting.Subplot(
            results_index.plot(),
            **results_index.get_xtick_kwargs(),
        ),
    )
    mp.save("test_noisydata_x_index", OUTPUT_DIR)

def test_noisy_data_get_xtick_kwargs():
    test_name = "test_noisy_data_get_xtick_kwargs"
    sp_list = []
    for x_index in [False, True]:
        rng = util.Seeder().get_rng(test_name)
        nd = plotting.NoisyData(x_index=x_index)

        for x in rng.integers(0, 100, [10]):
            for repeat in range(rng.integers(2, 10)):
                nd.update(x, 0.8*x + 0.2 + rng.normal(0, 10))

        sp = plotting.Subplot(
            nd.plot(),
            nd.predict_line(0, 1),
            **nd.get_xtick_kwargs(),
            title="x_index = %s" % x_index,
        )
        sp_list.append(sp)

    mp = plotting.MultiPlot(*sp_list, title=test_name)
    mp.save(test_name, OUTPUT_DIR)

def test_noisy_data_get_xtick_kwargs_log_x():
    test_name = "test_noisy_data_get_xtick_kwargs_log_x"
    sp_list = []
    for x in [
        2*np.arange(5),
        2**np.arange(5),
        [1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3],
    ]:
        for log_x in [True, False]:
            nd = plotting.NoisyData(True)

            for i, xi in enumerate(x, 1):
                for j in range(i):
                    nd.update(xi, j)

            sp = plotting.Subplot(
                nd.plot(),
                **nd.get_xtick_kwargs(),
                log_x=log_x,
                title="x=%s, log_x=%s" % (x, log_x)
            )
            sp_list.append(sp)

    mp = plotting.MultiPlot(*sp_list, num_cols=2, title=test_name)
    mp.save(test_name, OUTPUT_DIR)

def test_noisy_data_repr():
    n = plotting.NoisyData()
    n.update(1, 1)
    n.update(1, 2)
    n.update(3, 4)
    assert repr(n) == "NoisyData({1: [1, 2], 3: [4]})"

def test_nested_multiplot():
    rng = util.Seeder().get_rng("test_nested_multiplot")

    def get_random_subplot(title=None, n=20):
        x = np.linspace(0, 1, n)
        return plotting.Subplot(
            plotting.Line(x, x + rng.normal(0, 0.1, n)),
            plotting.Scatter(x, rng.uniform(0, 1, n), c="r"),
            title=title,
        )

    mp = plotting.MultiPlot(
        plotting.MultiPlot(
            *[get_random_subplot("Subplot %i/2" % (i+1)) for i in range(2)],
        ),
        plotting.MultiPlot(
            *[get_random_subplot() for _ in range(4)],
        ),
        plotting.MultiPlot(
            plotting.MultiPlot(
                *[get_random_subplot() for _ in range(2)],
            ),
            plotting.MultiPlot(
                *[get_random_subplot() for _ in range(3)],
                num_rows=1,
            ),
            num_cols=1,
        ),
        plotting.MultiPlot(
            *[get_random_subplot() for _ in range(9)],
        ),
        width_ratios=[1, 0.5,]
    )
    for pdf in [True, False]:
        mp.save("test_nested_multiplot", OUTPUT_DIR, pdf=pdf)

def test_nested_multiplot_titles():
    cp = plotting.ColourPicker(7)
    line_list = [
        plotting.Line([1, 3, 2], c=cp.next(), label="[1, 3, 2]"),
        plotting.Line([2, 1, 3], c=cp.next(), label="[2, 1, 3]"),
        plotting.Line([2, 3, 1], c=cp.next(), label="[2, 3, 1]"),
        plotting.Line([3, 1, 2], c=cp.next(), label="[3, 1, 2]"),
        plotting.Line([1, 3, 1], c=cp.next(), label="[1, 3, 1]"),
        plotting.Line([3, 1, 3], c=cp.next(), label="[3, 1, 3]"),
    ]
    lines = util.circular_iterator(line_list)

    mp = plotting.MultiPlot(
        plotting.MultiPlot(
            plotting.Subplot(next(lines), next(lines)),
            title="Subplot 1",
            title_font_size=12,
        ),
        plotting.MultiPlot(
            plotting.MultiPlot(plotting.Subplot(next(lines), next(lines))),
            plotting.MultiPlot(plotting.Subplot(next(lines), next(lines))),
            plotting.MultiPlot(plotting.Subplot(next(lines), next(lines))),
            plotting.MultiPlot(
                plotting.Subplot(next(lines), next(lines)),
                plotting.Subplot(next(lines), next(lines)),
                plotting.Subplot(next(lines), next(lines)),
                plotting.Subplot(
                    next(lines),
                    next(lines),
                    title="Subsubsubplot 4",
                    title_font_size=8,
                ),
            ),
            title="Subplot 2",
            height_ratios=[1, 2],
        ),
        plotting.MultiPlot(
            plotting.Subplot(next(lines), next(lines)),
            plotting.Subplot(next(lines), next(lines)),
            title="Subplot 3",
            title_font_size=12,
        ),
        plotting.MultiPlot(
            plotting.Subplot(next(lines), next(lines)),
            title="Subplot 4",
            title_font_size=12,
        ),
        legend=plotting.FigureLegend(*line_list, ncols=len(line_list)),
        pad=0.1,
        title="Figure",
        height_ratios=[2, 1],
    )
    for pdf in [True, False]:
        mp.save("test_nested_multiplot_titles", OUTPUT_DIR, pdf=pdf)

def test_nested_multiplot_space():
    cp = plotting.ColourPicker(7)
    line_list = [
        plotting.Line([1, 3, 2], c=cp.next(), label="[1, 3, 2]"),
        plotting.Line([2, 1, 3], c=cp.next(), label="[2, 1, 3]"),
        plotting.Line([2, 3, 1], c=cp.next(), label="[2, 3, 1]"),
        plotting.Line([3, 1, 2], c=cp.next(), label="[3, 1, 2]"),
        plotting.Line([1, 3, 1], c=cp.next(), label="[1, 3, 1]"),
        plotting.Line([3, 1, 3], c=cp.next(), label="[3, 1, 3]"),
    ]
    lines = util.circular_iterator(line_list)

    mp = plotting.MultiPlot(
        plotting.MultiPlot(
            *[plotting.Subplot(next(lines)) for _ in range(9)],
            title="Group 1",
        ),
        plotting.MultiPlot(
            *[plotting.Subplot(next(lines)) for _ in range(9)],
            space=0.2,
        ),
        legend=plotting.FigureLegend(*line_list, ncols=len(line_list)),
        title="Figure",
        figsize=[10, 6],
        space=0.4,
    )
    for pdf in [True, False]:
        mp.save("test_nested_multiplot_space", OUTPUT_DIR, pdf=pdf)

def test_dpi():
    printer = util.Printer("test_dpi", OUTPUT_DIR)

    size_dict = dict()
    for dpi in [None, 100, 50, 25]:
        name = "test_dpi_%s" % dpi
        full_path = util.get_full_path(name, OUTPUT_DIR, "png")
        if os.path.exists(full_path):
            os.remove(full_path)
            printer("Removed", full_path)

        sp = plotting.Subplot(plotting.Line([1, 2], [3, 4]))
        mp = plotting.MultiPlot(sp, dpi=dpi)
        mp.save(name, OUTPUT_DIR)
        size_dict[dpi] = os.path.getsize(full_path)

        printer("name =", name)
        printer("dpi =", dpi)
        printer("size (bytes) =", size_dict[dpi])
        printer.hline()

    assert size_dict[25] < size_dict[50]
    assert size_dict[25] < size_dict[100]
    assert size_dict[25] < size_dict[None]

    assert size_dict[50] < size_dict[100]
    assert size_dict[50] < size_dict[None]

    assert size_dict[100] == size_dict[None]

def test_grid_xy():
    line = plotting.Line(
        [1e-2, 3e-1, 3e0, 1e2],
        [1e-2, 3e-2, 3e1, 1e2],
        marker="o",
    )
    kwargs = {"log_x": True, "log_y": True}
    mp = plotting.MultiPlot(
        plotting.Subplot(line, **kwargs),
        plotting.Subplot(line, **kwargs, grid=False),
        plotting.Subplot(line, **kwargs, grid_x=None, grid_y="major"),
        plotting.Subplot(line, **kwargs, grid_x=None),
        plotting.Subplot(line, **kwargs, grid_x="major"),
        plotting.Subplot(line, **kwargs, grid_x="major", grid_y="major"),
        figsize=[10, 6],
    )
    mp.save("test_grid_xy", OUTPUT_DIR)

def test_fillbetweenx():
    plotting.plot(
        plotting.FillBetweenx(
            [3, 5, 6, 7, 9],
            [2, 1, 3, 2, 3],
            [4, 2, 6, 7, 4],
            c="r",
            a=0.3,
            z=20,
        ),
        plotting.FillBetween(
            [0, 2, 4, 5, 7],
            [7, 9, 8, 7, 5],
            [6, 6, 7, 5, 4],
            c="b",
            a=0.3,
        ),
        plot_name="test_fillbetweenx",
        dir_name=OUTPUT_DIR,
    )

def test_errorbar():
    rng = util.Seeder().get_rng("test_errorbar")

    n = 10
    x = np.linspace(-2, 5, n)
    y1 = rng.uniform(1, 2, n) + x
    e1 = rng.uniform(0.1, 0.5, n)
    y2 = rng.uniform(7, 8, n) - x
    e2 = rng.uniform(0.2, 0.6, n)
    xs = rng.uniform(-2, 5, 20)
    ys = rng.uniform( 0, 9, 20)

    lines = [
        plotting.ErrorBar(x, y1, e1, label="Vertical", ls="--"),
        plotting.ErrorBar(x, y2, None, e2, label="Horizontal", c="r", m="o"),
        plotting.Scatter(xs, ys, label="Scatter", c="b", z=20),
    ]
    mp = plotting.MultiPlot(
        plotting.Subplot(*lines, plotting.Legend()),
        legend=plotting.FigureLegend(*lines),
    )
    mp.save(
        plot_name="test_errorbar",
        dir_name=OUTPUT_DIR,
    )

def test_plottable_group():
    rng = util.Seeder().get_rng("test_plottable_group")

    n = 10
    r = 5
    x = np.linspace(-2, 5, n)
    xn = rng.uniform(-0.1, 0.1, [n, r]) + x.reshape(n, 1)
    y1 = rng.uniform(1, 3, [n, r]) + xn
    m1 = y1.mean(-1)
    s1 = y1.std(-1)
    y2 = rng.uniform(6, 8, [n, r]) - xn
    m2 = y2.mean(-1)
    s2 = y2.std(-1)

    pg1 = plotting.PlottableGroup(
        plotting.Scatter(xn, y1),
        plotting.Line(x, m1),
        plotting.FillBetween(x, m1-s1, m1+s1, a=0.2),
    )
    pg2 = plotting.PlottableGroup(
        plotting.Scatter(xn, y2),
        plotting.Line(x, m2),
        plotting.FillBetween(x, m2-s2, m2+s2, a=0.2),
    )
    pg1.set_options(c="b", label="Blue")
    pg2.set_options(c="r", label="Red")
    plotting.plot(
        pg1,
        pg2,
        plotting.Legend.from_plottables(pg1, pg2),
        plot_name="test_plottable_group",
        dir_name=OUTPUT_DIR,
    )

def test_empty_noisy_data_plot():
    n = 5
    cp = plotting.ColourPicker(n, False)
    nd = plotting.NoisyData()
    mp = plotting.MultiPlot(
        plotting.LegendSubplot(
            *[
                nd.plot(c=cp.next(), label="Series %i" % i)
                for i in range(n)
            ],
            title="test_empty_noisy_data_plot",
        ),
        title="test_empty_noisy_data_plot",
        title_font_size=15,
        figsize=[4, 2],
    )
    mp.save("test_empty_noisy_data_plot", OUTPUT_DIR)

def test_arrow():
    rng = util.Seeder().get_rng("test_arrow")
    n = 20
    x = rng.normal(0, 1, [n, 4]).tolist()
    w = np.exp(rng.normal(-2, 0.5, [n])).tolist()
    plotting.plot(
        *[plotting.Arrow(*x[i], c="k", hw=w[i]) for i in range(n)],
        plot_name="test_arrow",
        dir_name=OUTPUT_DIR,
    )

def test_bad_axis_properties():
    mp = plotting.MultiPlot(
        plotting.Subplot(
            plotting.Line([1, 3, 2]),
            bad_kwarg="something",
        ),
    )
    with pytest.raises(ValueError):
        mp.save("test_bad_axis_properties", OUTPUT_DIR)

def test_subplot_set_options():
    test_name = "test_subplot_set_options"
    sp = plotting.Subplot(plotting.Line([1, 3, 2]))
    assert repr(sp) == "Subplot()"

    mp1 = plotting.MultiPlot(sp)
    mp1.save(test_name, OUTPUT_DIR)
    im1 = mp1.get_image_array()

    sp.set_options(colour="r")
    assert repr(sp) == "Subplot(colour='r')"

    mp2 = plotting.MultiPlot(sp)
    mp2.save(test_name + "_red", OUTPUT_DIR)
    im2 = mp2.get_image_array()

    assert      np.all(im1 == im1)
    assert not  np.all(im1 == im2)

    sp.set_options(bad_kwarg="something")
    assert repr(sp) == "Subplot(bad_kwarg='something', colour='r')"

    mp3 = plotting.MultiPlot(sp)
    with pytest.raises(ValueError):
        mp3.save(test_name + "bad", OUTPUT_DIR)

def test_figure_legend_lines_no_data():
    mp = plotting.MultiPlot(
        plotting.Subplot(),
        plotting.Subplot(),
        legend=plotting.FigureLegend(
            plotting.Line(c="b", marker="o", label="Target"),
            plotting.Line(c="r", marker="o", label="Prediction"),
        ),
        title="Reconstructions",
        figsize=[6, 4],
    )
    mp.save("test_figure_legend_lines_no_data", OUTPUT_DIR)
