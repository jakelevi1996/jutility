import numpy as np
import pytest
from jutility import util, latex
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
