import os
import time
import numpy as np
import pytest
from jutility import util, plotting, cli
import tests

OUTPUT_DIR = tests.get_output_dir("test_cli")

def get_parser():
    parser = cli.ObjectParser(
        cli.Arg("int",   "in", default=10,            type=int),
        cli.Arg("float", "fl", default=-4.7,          type=float),
        cli.Arg("list",  "li", default=[30, 20, 10],  type=int, nargs="*"),
        cli.Arg("none",  "no", default=None),
        cli.Arg("true",  "tr", default=True),
        cli.Arg("false", "fa", default=False),
        cli.Arg("no_abbrev",   default="random"),
        cli.ObjectArg(
            Adam,
            cli.Arg("lr",   "lr", type=float, default=1e-3),
            cli.Arg("beta", "be", type=float, default=[0.9, 0.999], nargs=2),
            abbreviation="op",
            init_requires=["params"],
        ),
    )
    return parser

def test_parse():
    printer = util.Printer("test_parse", OUTPUT_DIR)

    parser = get_parser()
    args = parser.parse_args([])

    printer(parser)
    printer(args)

    assert parser.parse_args([]).int == 10
    assert parser.parse_args([]).list == [30, 20, 10]
    assert parser.parse_args(["--int=-3"]).int == -3
    assert parser.parse_args(["--list", "3", "4", "5"]).list == [3, 4, 5]

def test_print_help():
    printer = util.Printer("test_print_help", OUTPUT_DIR)

    parser = get_parser()
    parser.print_help(printer._file)

def test_get_args_summary():
    printer = util.Printer("test_get_args_summary", OUTPUT_DIR)

    parser = get_parser()
    args = parser.parse_args([])
    s = cli.get_args_summary(args)

    assert s == "faFfl-4.7in10li30,20,10noNop.be0.9,0.999op.lr0.001trT"

    s2 = cli.get_args_summary(parser.parse_args(["--no_abbrev=123"]))
    assert s2 == s

    s3 = cli.get_args_summary(parser.parse_args(["--no_abbrev=4567"]))
    assert (s3 == s2) and (s3 == s)

    s4 = cli.get_args_summary(parser.parse_args(["--int=20"]))
    assert s4 != s

    s5 = cli.get_args_summary(parser.parse_args(["--Adam.lr=3e-3"]))
    assert s5 != s

    printer(s, s2, s3, s4, s5, sep="\n")

def test_init_object():
    printer = util.Printer("test_init_object", OUTPUT_DIR)

    parser = get_parser()
    parser.parse_args([])
    optimiser = parser.init_object("Adam", params=[1, 2, 3])

    printer(optimiser, vars(optimiser))
    assert isinstance(optimiser, Adam)
    assert optimiser.lr == 1e-3
    assert optimiser.inner_params == [1, 2, 3]

    parser.parse_args(["--Adam.lr=3e-3"])
    optimiser: Adam = parser.init_object("Adam", params=[1, 2, 3])
    assert optimiser.lr == 3e-3

    parser.parse_args([])
    optimiser: Adam = parser.init_object("Adam", params=[20, 30])
    assert optimiser.inner_params == [20, 30]

class Adam:
    def __init__(self, params, lr=1e-3, beta=None):
        self.inner_params   = params
        self.lr             = lr
        self.beta           = beta
