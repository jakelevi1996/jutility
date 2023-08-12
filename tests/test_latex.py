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

    indent = latex.Indenter(indent_str, initial_indent)

    x = iter([x ** 2 for x in range(20)])

    printer(indent("Hello"))
    with indent.new_block:
        printer(indent(next(x)))
        printer(indent("world"))
    printer(indent(next(x)))
    with indent.new_block:
        printer(indent(next(x)))
        printer(indent(next(x)))
        with indent.new_block:
            printer(indent(next(x)))
            printer(indent(next(x)))
        printer(indent(next(x)))
    printer(indent(next(x)))
