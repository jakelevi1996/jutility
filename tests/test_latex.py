import os
import numpy as np
import pytest
from jutility import latex, plotting, util
import test_utils

OUTPUT_DIR = test_utils.get_output_dir("test_latex")

@pytest.mark.parametrize("indent_str", [None, "+ "])
@pytest.mark.parametrize("initial_indent", range(3))
def test_indenter(indent_str, initial_indent):
    test_name = "test_indenter, %s, %s" % (indent_str, initial_indent)
    test_dir = os.path.join(OUTPUT_DIR, "test_indenter")
    printer = util.Printer(test_name, test_dir)

    indent = latex.Indenter(printer, indent_str, initial_indent)

    x = iter([x ** 2 for x in range(20)])

    indent.print("Hello")
    with indent.new_block():
        indent.print(next(x))
        indent.print("world")
    indent.print(next(x))
    with indent.new_block():
        indent.print(next(x))
        indent.print(next(x))
        with indent.new_block():
            indent.print(next(x))
            indent.print(next(x))
        indent.print(next(x))
    indent.print(next(x))

def test_plot_colour_picker():
    rng = util.Seeder().get_rng("test_plot_colour_picker")

    x = np.linspace(0, 1, 100)
    f = lambda x, slope: slope * x + 0.01 * rng.normal(size=x.shape)

    n = 5
    cp = plotting.ColourPicker(n)
    lines = [
        latex.Line(x, f(x, slope), c=cp.colour_name(i), label="Line %i" % i)
        for i, slope in enumerate(np.linspace(0, 1, n))
    ]

    latex.plot(
        *lines,
        axis_properties=latex.AxisProperties(
            title="Plot title",
            xlabel="$x$ label",
            ylabel="$y$ label",
            colour_picker=cp,
        ),
        plot_name="test_plot_colour_picker_latex",
        dir_name=os.path.join(OUTPUT_DIR, "test_plot_colour_picker"),
    )

def test_plottables():
    cp = plotting.ColourPicker(4)

    latex.plot(
        latex.Line(
            x=[0.083, 0.205, 1.349, 3.185, 4.066],
            y=[0.263, 0.857, 1.822, 2.833, 3.434],
            c=cp(0),
            marker="*",
            alpha=0.5,
            label="Legend entry 1",
        ),
        latex.Line(
            x=[0.721, 1.559, 2.559, 4.743, 4.752],
            y=[0.944, 1.291, 2.85,  4.926, 4.899],
            c=cp(1),
            marker="*",
            label="Legend entry 2",
        ),
        latex.Line(
            x=[3, 3.5, 4],
            y=[5, 6, 5],
            c="black",
            only_marks=True,
            label="Scatter",
        ),
        latex.FillBetween(
            x=[0, 2, 5],
            y1=[-1, 0, 4],
            y2=[1, 4, 6],
            c=cp(2),
            alpha=0.2,
            label="Patch",
        ),
        latex.Quiver(
            x=[1, 4, 0],
            y=[-1, 0, 3],
            dx=[2, 1, 2],
            dy=[1, 2, 2],
            c=cp(3),
            label="Arrow",
        ),
        latex.VLine(
            x=1.2,
            c="red",
            label="Vertical line",
        ),
        latex.HLine(
            y=5.5,
            c="blue",
            label="Horizontal line",
        ),
        axis_properties=latex.AxisProperties(
            title="Title of plot",
            xlabel="$x$ axis label",
            ylabel="$y$ axis label",
            figsize_cm=[8, 6],
        ),
        plot_name="test_plottables_latex",
        dir_name=os.path.join(OUTPUT_DIR, "test_plottables"),
    )

def test_colour_mesh():
    nx = 11
    ny = 21
    x = np.linspace(-2, 2, nx).reshape(1, nx)
    y = np.linspace(-2, 2, ny).reshape(ny, 1)
    z = 100 * np.exp(-x*x-y*y)

    latex.plot(
        latex.ColourMesh(x.reshape(nx), y.reshape(ny), z),
        axis_properties=latex.AxisProperties(
            xlim=[-2, 2],
            ylim=[-2, 2],
            colour_bar=True,
            figsize_cm=[6, 5],
        ),
        plot_name="test_colour_mesh_latex",
        dir_name=os.path.join(OUTPUT_DIR, "test_colour_mesh"),
    )

@pytest.mark.parametrize("normalise", [True, False])
def test_quiver(normalise):
    nx = 15
    ny = 25
    n = nx * ny
    x = np.tile(np.linspace(-2, 2, nx).reshape(1, nx), [ny, 1])
    y = np.tile(np.linspace(-2, 2, ny).reshape(ny, 1), [1, nx])
    dx = (y - x) / 10
    dy = (x + y) / 10

    norm_length = 0.2 if normalise else None
    plot_name = "test quiver latex, normalise = %s" % normalise
    latex.plot(
        latex.Quiver(
            x.reshape(n),
            y.reshape(n),
            dx.reshape(n),
            dy.reshape(n),
            norm_length=norm_length,
            label="Arrow",
        ),
        axis_properties=latex.AxisProperties(
            title=plot_name,
            xlabel="$x$",
            ylabel="$y$",
        ),
        plot_name=plot_name,
        dir_name=os.path.join(OUTPUT_DIR, "test_quiver"),
    )

@pytest.mark.parametrize("log_x", [True, False])
@pytest.mark.parametrize("log_y", [True, False])
def test_log_axes(log_x, log_y):
    x = np.linspace(1e-1, 10, 20)
    y = np.exp(x)

    plot_name = "test log axes, log x = %s, log y = %s" % (log_x, log_y)
    latex.plot(
        latex.Line(x, y, c="blue", marker="*"),
        axis_properties=latex.AxisProperties(
            title=plot_name,
            xlabel="$x$",
            ylabel="$y$",
            log_x=log_x,
            log_y=log_y,
        ),
        plot_name=plot_name,
        dir_name=os.path.join(OUTPUT_DIR, "test_log_axes"),
    )

@pytest.mark.parametrize("plot_all_data", [True, False])
def test_noisy_data(plot_all_data):
    rng = util.Seeder().get_rng("test_noisy_data", plot_all_data)
    noisy_data_blue = plotting.NoisyData()
    noisy_data_red  = plotting.NoisyData()
    x_list = np.linspace(0, 1)

    for x in x_list:
        for num_repeats in range(rng.integers(10)):
            noisy_data_blue.update(x, x + 0.04 * rng.normal())
        for num_repeats in range(rng.integers(10)):
            noisy_data_red.update(x, 0.3 + (0.3 * x) + (0.04 * rng.normal()))

    mp = latex.plot(
        *latex.get_noisy_data_lines(
            noisy_data_blue,
            colour="blue",
            name="Blue data",
            plot_all_data=plot_all_data,
        ),
        *latex.get_noisy_data_lines(
            noisy_data_red,
            colour="red",
            name="Red data",
            plot_all_data=plot_all_data,
        ),
        dir_name=os.path.join(
            OUTPUT_DIR,
            "test_noisy_data, plot_all_data = %s" % plot_all_data,
        ),
        axis_properties=latex.AxisProperties(
            xlabel="$x$",
            ylabel="$y$",
            ylim=[-0.2, 1.2],
        ),
    )

@pytest.mark.parametrize("n_plots", [1, 5, 6, 8, 9])
def test_plot_figure(n_plots, compile_pdf=False):
    test_name       = "test_plot_figure_%i" % n_plots
    test_dir        = os.path.join(OUTPUT_DIR,      test_name)
    graphics_dir    = os.path.join(test_dir,        "figures")
    fig_dir         = os.path.join(graphics_dir,    "test_figure")
    subfig_dir      = os.path.join(fig_dir,         "test_subfigure")

    if compile_pdf:
        x = np.linspace(0, 1, 250)
        f = lambda x, freq: np.sin(2 * np.pi * freq * x)

        for i in range(n_plots):
            latex.plot(
                latex.Line(x, f(x, i + 1), c="blue"),
                plot_name="subfig_%i" % i,
                dir_name=subfig_dir,
                autocompile=True,
            )

    subfigures = [
        latex.Subfigure(
            os.path.join(subfig_dir, "subfig_%i.pdf" % i),
            width=0.3,
            caption="Test subfigure caption %i" % i,
            label="test subfig label %i" % i,
        )
        for i in range(n_plots)
    ]
    fig_path = latex.plot_figure(
        *subfigures,
        graphics_path=graphics_dir,
        num_cols=3,
        caption="Test figure caption",
        label="test fig label",
        fig_name="fig name",
        dir_name=fig_dir,
    )
    latex.standalone_document(
        fig_path,
        graphics_path=graphics_dir,
        document_name="test_plot_figure_latex",
        dir_name=test_dir,
    )
