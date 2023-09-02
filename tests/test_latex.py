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
        dir_name=os.path.join(OUTPUT_DIR, "test_plot")
    )
