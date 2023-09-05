import os
import numpy as np
import pytest
from jutility import latex, plotting, util
import tests.util

OUTPUT_DIR = tests.util.get_output_dir("test_latex")

@pytest.mark.parametrize("indent_str", [None, "+ "])
@pytest.mark.parametrize("initial_indent", range(3))
def test_indenter(indent_str, initial_indent):
    test_name = "test_indenter, %s, %s" % (indent_str, initial_indent)
    printer = util.Printer(test_name, OUTPUT_DIR)

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

def test_plot():
    rng = util.Seeder().get_rng("test_plot")

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
        plot_name="test_plot_latex",
        dir_name=os.path.join(OUTPUT_DIR, "test_plot"),
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
            log_xscale=log_x,
            log_yscale=log_y,
        ),
        plot_name=plot_name,
        dir_name=os.path.join(OUTPUT_DIR, "test_log_axes"),
    )
