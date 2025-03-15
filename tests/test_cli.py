import pytest
from jutility import util, cli
import test_utils

OUTPUT_DIR = test_utils.get_output_dir("test_cli")

def test_parsed_args():
    class A:
        def __init__(self, x: int):
            self.x = x

    class B:
        def __init__(self, a: A, y: float):
            self.a = a
            self.y = y

    parser = cli.Parser(
        cli.ObjectChoice(
            "m",
            cli.ObjectArg(A, cli.Arg("x", type=int, default=1)),
            cli.ObjectArg(
                B,
                cli.ObjectArg(A, cli.Arg("x", type=int, default=2), name="a"),
                cli.Arg("y", type=float, default=3.4),
            ),
            default="A",
        ),
        cli.Arg("c", type=str, default="abc"),
        cli.Arg("d", action="store_true"),
    )

    args = parser.parse_args([])
    assert isinstance(args, cli.ParsedArgs)
    assert repr(args) == "ParsedArgs()"
    assert args.get_summary() == "cABCdFmAmx1"
    assert args.get_value_dict() == {
        "m":        "A",
        "m.A.x":    1,
        "c":        "abc",
        "d":        False,
    }

    c = args.get_arg("c")
    assert isinstance(c, cli.Arg)
    assert c.value == "abc"

    assert args.get_value("c") == "abc"
    assert args.get_value("m.A.x") == 1

    assert args.get_kwargs() == {"c": "abc", "d": False}

    args.update({"c": "defg"})
    assert repr(args) == "ParsedArgs()"
    assert args.get_value("c") == "defg"
    assert args.get_summary() == "cDEFGdFmAmx1"
    assert args.get_value_dict() == {
        "m":        "A",
        "m.A.x":    1,
        "c":        "defg",
        "d":        False,
    }

    assert args.get_value("m.A") is None
    assert not isinstance(args.get_value("m.A"), A)

    a = args.init_object("m")
    assert isinstance(a, A)
    assert a.x == 1
    assert not args.get_value("m.A") is None
    assert isinstance(args.get_value("m.A"), A)
    assert repr(args) == "ParsedArgs()"
    assert args.get_summary() == "cDEFGdFmAmx1"
    assert args.get_value_dict() == {
        "m":        "A",
        "m.A.x":    1,
        "c":        "defg",
        "d":        False,
    }

    args.reset_object_cache()
    assert args.get_value("m.A") is None
    assert not isinstance(args.get_value("m.A"), A)

    args.update({"m.A.x": 3})
    a = args.init_object("m")
    assert isinstance(a, A)
    assert a.x == 3
    assert repr(args) == "ParsedArgs()"
    assert args.get_summary() == "cDEFGdFmAmx3"
    assert args.get_value_dict() == {
        "m":        "A",
        "m.A.x":    3,
        "c":        "defg",
        "d":        False,
    }

    new_args = parser.parse_args(
        "--m B --m.B.a.x 5 --m.B.y 6.7 --c hijk --d".split(),
    )
    assert isinstance(new_args, cli.ParsedArgs)
    assert repr(new_args) == "ParsedArgs()"
    assert new_args.get_value_dict() == {
        "m":        "B",
        "m.B.a.x":  5,
        "m.B.y":    6.7,
        "c":        "hijk",
        "d":        True,
    }
    assert new_args.get_summary() == "cHIJKdTmBmax5my6.7"
    assert new_args.get_value("c") == "hijk"
    assert new_args.get_kwargs() == {"c": "hijk", "d": True}
    b = new_args.init_object("m")
    assert isinstance(b, B)
    assert isinstance(b.a, A)
    assert b.a.x == 5
    assert b.y == 6.7

def test_parser_help():
    printer = util.Printer("test_parser_help", dir_name=OUTPUT_DIR)

    class A:
        def __init__(self, x: int, z: str):
            self.x = x
            self.z = z

    class B:
        def __init__(self, a: A, y: float, z: str):
            self.a = a
            self.y = y
            self.z = z

    parser = cli.Parser(
        cli.ObjectChoice(
            "model",
            cli.ObjectArg(A, cli.Arg("x", type=int, default=1)),
            cli.ObjectArg(
                B,
                cli.ObjectArg(A, cli.Arg("x", type=int, default=2), name="a"),
                cli.Arg("y", type=float, default=3.4),
                name="Mlp",
            ),
            shared_args=[cli.Arg("z", type=str, default="abc")],
            default="A",
        ),
        cli.Arg("c", type=str, default="defg"),
        cli.Arg("d", action="store_true"),
        cli.Arg("e", type=int, nargs="*", default=[]),
        cli.PositionalArg("f", type=float),
        cli.PositionalArg("g", type=int),
        cli.BooleanArg("h"),
        cli.BooleanArg("i", default=True),
        cli.BooleanArg("j", default=False),
        cli.JsonArg("k", default=[1, [2, 3]]),
        cli.JsonArg("l", default=[4, [5, 6]], nargs="*"),
        prog="test_parser_help",
    )

    printer(parser.help())

    assert parser.help() == (
        "usage: test_parser_help [-h] [--model {A,Mlp}] [--model.z Z] "
        "[--model.A.x X]\n"
        "                        [--model.Mlp.a.x X] [--model.Mlp.y Y] "
        "[--c C] [--d]\n"
        "                        [--e [E ...]] [--h | --no-h] "
        "[--i | --no-i]\n"
        "                        [--j | --no-j] [--k K] [--l [L ...]]\n"
        "                        F G\n"
        "\n"
        "positional arguments:\n"
        "  F                  type=<class 'float'>\n"
        "  G                  type=<class 'int'>\n"
        "\n"
        "options:\n"
        "  -h, --help         show this help message and exit\n"
        "  --model {A,Mlp}    default=A, required=False\n"
        "  --model.z Z        default='abc', type=<class 'str'>\n"
        "  --model.A.x X      default=1, type=<class 'int'>\n"
        "  --model.Mlp.a.x X  default=2, type=<class 'int'>\n"
        "  --model.Mlp.y Y    default=3.4, type=<class 'float'>\n"
        "  --c C              default='defg', type=<class 'str'>\n"
        "  --d                action='store_true'\n"
        "  --e [E ...]        default=[], nargs='*', type=<class 'int'>\n"
        "  --h, --no-h        (default: None)\n"
        "  --i, --no-i        (default: True)\n"
        "  --j, --no-j        (default: False)\n"
        "  --k K              default=[1, [2, 3]] (format: JSON string)\n"
        "  --l [L ...]        default=[4, [5, 6]], nargs='*' "
        "(format: JSON string)\n"
    )

def test_object_arg():
    printer = util.Printer("test_object_arg", dir_name=OUTPUT_DIR)
    cli.verbose.set_printer(printer)

    class A:
        def __init__(self, b: int, c: str):
            self.b = b
            self.c = c

        def __repr__(self):
            return util.format_type(type(self))

    class D(A):
        def __init__(self, e: list[int], f: float, g: A, **kw):
            self.e = e
            self.f = f
            self.g = g
            self.kw = kw

    class H(A):
        def __init__(self, a: A, d: D, i: int):
            self.a = a
            self.d = d
            self.i = i

    parser = cli.Parser(
        cli.ObjectArg(
            H,
            cli.ObjectArg(
                A,
                cli.Arg("b", type=int, default=1),
                cli.Arg("c", type=str, default="abc"),
                name="a",
            ),
            cli.ObjectArg(
                D,
                cli.Arg("e", type=int, default=[2, 3], nargs="*"),
                cli.Arg("f", type=float, default=4.5),
                cli.ObjectArg(
                    A,
                    cli.Arg("b", type=int),
                    name="g",
                    init_const_kwargs={"c": "defg"},
                ),
                name="d",
                init_requires=["k"],
            ),
            cli.Arg("i", type=int, default=6),
        ),
        cli.Arg("j", type=float, default=-7.8),
    )

    printer.heading("Default args")

    args = parser.parse_args([])
    assert args.get_summary() == "hab1hacABChde2,3hdf4.5hdgbNhi6j-7.8"
    assert args.get_value_dict() == {
        "H.a.b": 1,
        "H.a.c": "abc",
        "H.d.e": [
            2,
            3
        ],
        "H.d.f": 4.5,
        "H.d.g.b": None,
        "H.i": 6,
        "j": -7.8
    }
    with cli.verbose:
        with pytest.raises(ValueError):
            h = args.init_object("H")
        with pytest.raises(ValueError):
            d = args.init_object("H.d")
        d = args.init_object("H.d", k=9)
        h = args.init_object("H")
    assert isinstance(d, D)
    assert d.e == [2, 3]
    assert d.f == 4.5
    assert isinstance(d.g, A)
    assert d.g.b == None
    assert d.g.c == "defg"
    assert d.kw == {"k": 9}
    assert isinstance(h, H)
    assert d.kw == {"k": 9}

    printer.heading("reset_object_cache")

    with cli.verbose:
        d = args.init_object("H.d", k=1000, m="hij")
        assert isinstance(d, D)
        assert d.kw == {"k": 9}
        args.reset_object_cache()
        d = args.init_object("H.d", k=1000, m="hij")
        assert isinstance(d, D)
        assert d.kw == {"k": 1000, "m": "hij"}
        h = args.init_object("H")
    assert isinstance(h, H)
    assert h.d is d
    assert h.d.kw == {"k": 1000, "m": "hij"}
    assert h.a.b == 1
    assert h.a.c == "abc"

    printer.heading("Different CLI args")

    new_args = parser.parse_args("--H.a.c lmnop --H.d.g.b 13".split())
    with cli.verbose:
        with pytest.raises(ValueError):
            new_args.init_object("H")
        new_args.init_object("H.d", k=[55, 66], n=77)
        h = new_args.init_object("H")
    assert isinstance(h, H)
    assert h.a.c == "lmnop"
    assert h.d.g.b == 13
    assert h.d.kw == {"k": [55, 66], "n": 77}
    assert h.i == 6

    printer.heading("parser.help")

    printer(parser.help())

def test_object_choice():
    printer = util.Printer("test_object_choice", dir_name=OUTPUT_DIR)
    cli.verbose.set_printer(printer)

    class A:
        def __init__(self, b: int, c: str):
            self.b = b
            self.c = c

        def __repr__(self):
            return util.format_type(type(self))

    class D(A):
        def __init__(self, e: list[int], f: float, g: A, **kw):
            self.e = e
            self.f = f
            self.g = g
            self.kw = kw

    parser = cli.Parser(
        cli.ObjectChoice(
            "model",
            cli.ObjectArg(
                A,
                cli.Arg("c", type=str, default="abc"),
                init_ignores=["e"],
            ),
            cli.ObjectArg(
                D,
                cli.Arg("e", type=int, default=[2, 3], nargs="*"),
                cli.Arg("f", type=float, default=4.5),
                cli.ObjectChoice(
                    "g",
                    cli.ObjectArg(
                        A,
                        cli.Arg("c", type=str, default="abc"),
                        init_ignores=["e"],
                    ),
                    cli.ObjectArg(
                        D,
                        cli.Arg("f", type=float, default=3.141),
                        cli.ObjectArg(
                            A,
                            cli.Arg("b", type=int),
                            name="g",
                            init_const_kwargs={"c": "deep arg"},
                        ),
                        init_ignores=["b"],
                        init_requires=["e"],
                    ),
                    shared_args=[cli.Arg("b", type=int, default=1)],
                    default="A",
                    init_const_kwargs={"c": "deep kwarg"},
                ),
                init_ignores=["b"],
            ),
            shared_args=[cli.Arg("b", type=int, default=1)],
            init_requires=["e"],
            init_const_kwargs={"c": "defg"},
        ),
    )

    args = parser.parse_args([])
    with pytest.raises(ValueError):
        args.init_object("model")

    printer.heading("model = A")

    args = parser.parse_args("--model A".split())
    assert args.get_summary() == "mAmb1mcABC"
    assert args.get_value_dict() == {
        'model': 'A',
        'model.A.c': 'abc',
        'model.b': 1,
    }
    with cli.verbose:
        a = args.init_object("model")
        assert isinstance(a, A)
        assert a.c == "abc"
        assert a.b == 1

    printer.heading("args.update")

    with cli.verbose:
        args.update({'model.A.c': 'defg', 'model.b': 2})
        a = args.init_object("model")
        assert isinstance(a, A)
        assert a.c == "defg"
        assert a.b == 2

    printer.heading("model = A, non-default options")

    args = parser.parse_args("--model A --model.b 3 --model.A.c xyz".split())
    assert args.get_summary() == "mAmb3mcXYZ"
    with cli.verbose:
        a = args.init_object("model")
        assert isinstance(a, A)
        assert a.c == "xyz"
        assert a.b == 3

    printer.heading("model = D")

    args = parser.parse_args("--model D".split())
    assert args.get_summary() == "mDme2,3mf4.5mgAmgb1mgcABC"
    assert args.get_arg("model").get_summary() == "e2,3f4.5gAgb1gcABC"
    assert args.get_arg("model.D.g").get_summary() == "b1cABC"
    assert args.get_value_dict() == {
        "model": "D",
        "model.D.e": [
            2,
            3
        ],
        "model.D.f": 4.5,
        "model.D.g": "A",
        "model.D.g.A.c": "abc",
        "model.D.g.b": 1
    }
    with cli.verbose:
        d = args.init_object("model")
        assert isinstance(d, D)
        assert d.e == [2, 3]
        assert d.f == 4.5
        assert     isinstance(d.g, A)
        assert not isinstance(d.g, D)
        assert d.g.b == 1
        assert d.g.c == "abc"
        assert d.kw == {"c": "defg"}

    printer.heading("model = D, model.D.g = D")

    args = parser.parse_args(
        "--model D --model.D.g D --model.D.e 7 8 9".split()
    )
    assert args.get_summary() == "mDme7,8,9mf4.5mgDmgf3.141mggbN"
    assert args.get_value_dict() == {
        "model": "D",
        "model.D.e": [
            7,
            8,
            9
        ],
        "model.D.f": 4.5,
        "model.D.g": "D",
        "model.D.g.D.f": 3.141,
        "model.D.g.D.g.b": None
    }
    with cli.verbose:
        with pytest.raises(ValueError):
            d = args.init_object("model")

    printer.heading("init_object model.D.g")

    with cli.verbose:
        args.init_object("model.D.g", e=[1, 10, 100])
        d = args.init_object("model")
        assert     isinstance(d, D)
        assert     isinstance(d.g, D)
        assert     isinstance(d.g.g, A)
        assert not isinstance(d.g.g, D)
        assert d.e == [7, 8, 9]
        assert d.f == 4.5
        assert d.g.e == [1, 10, 100]
        assert d.g.f == 3.141
        assert d.g.g.b == None
        assert d.g.g.c == "deep arg"
        assert d.g.kw == {"c": "deep kwarg"}
        assert d.kw == {"c": "defg"}

    printer.heading("parser.help")

    printer(parser.help())

def test_unknown_arg():
    printer = util.Printer("test_unknown_arg", dir_name=OUTPUT_DIR)

    parser = cli.Parser(
        cli.Arg("a", type=int,      default=1),
        cli.Arg("b", type=float,    default=2.3),
        cli.Arg("c", type=str,      default="abc"),
    )
    args = parser.parse_args([])

    assert repr(args) == "ParsedArgs()"
    assert args.get_summary() == "a1b2.3cABC"
    assert args.get_value_dict() == {"a": 1, "b": 2.3, "c": "abc"}

    new_arg_dict = {"a": 4, "d": "xyz", "e.f": 5.67}
    with pytest.raises(ValueError):
        args.update(new_arg_dict)

    args.update(new_arg_dict, allow_new_keys=True)
    assert repr(args) == "ParsedArgs()"
    assert args.get_summary() == "a4b2.3cABC"
    assert args.get_value_dict() == {
        "a":    4,
        "b":    2.3,
        "c":    "abc",
        "d":    "xyz",
        "e.f":  5.67,
    }
    assert args.get_kwargs() == {"a": 4, "b": 2.3, "c": "abc"}

    assert isinstance(args.get_arg("a"),    cli.Arg)
    assert isinstance(args.get_arg("d"),    cli._UnknownArg)
    assert isinstance(args.get_arg("e.f"),  cli._UnknownArg)

    assert repr(args.get_arg("a")) == (
        "Arg(full_name='a', name='a', value=4)"
    )
    assert repr(args.get_arg("d")) == (
        "_UnknownArg(full_name='d', name='d', value='xyz')"
    )
    assert repr(args.get_arg("e.f")) == (
        "_UnknownArg(full_name='e.f', name='e.f', value=5.67)"
    )

    printer(args, parser, parser.help(), sep="\n\n")

def test_cli_verbose():
    printer = util.Printer("test_cli_verbose", dir_name=OUTPUT_DIR)
    cli.verbose.set_printer(printer)

    class A:
        def __init__(self, x: int):
            self.x = x

        def __repr__(self):
            return util.format_type(type(self))

    class B(A):
        def __init__(self, a: A, y: float):
            self.a = a
            self.y = y

    parser = cli.Parser(
        cli.ObjectArg(A, cli.Arg("x", type=int, default=1)),
        cli.ObjectArg(
            B,
            cli.ObjectArg(A, cli.Arg("x", type=int, default=2), name="a"),
            cli.Arg("y", type=float, default=3.4),
        ),
    )
    args = parser.parse_args([])

    args.init_object("A", x=5)
    with cli.verbose:
        args.init_object("B", y=6.6)
        with cli.verbose:
            args.init_object("B.a", x=7)

        printer.heading("args.reset_object_cache")
        args.reset_object_cache()
        args.init_object("B.a", x=8)
        args.init_object("B.a", x=9)
        args.init_object("B", y=-10.11)
        args.init_object("B", y=12.13)
        args.init_object("A", x=14)
        args.init_object("A", x=15)

    args.init_object("A", x=15)

    assert printer.read() == (
        "cli: A(x=2)\n"
        "cli: B(a=A(), y=6.6)\n"
        "cli: `B.a` retrieved from cache\n"
        "\n"
        "------------------------- (1) args.reset_object_cache "
        "-------------------------\n"
        "\n"
        "cli: A(x=8)\n"
        "cli: `B.a` retrieved from cache\n"
        "cli: `B.a` retrieved from cache\n"
        "cli: B(a=A(), y=-10.11)\n"
        "cli: `B` retrieved from cache\n"
        "cli: A(x=14)\n"
        "cli: `A` retrieved from cache\n"
    )

def test_get_kwargs():
    class C:
        def __init__(self, x):
            self.x = x

    parser = cli.Parser(
        cli.Arg("a", type=int, default=1),
        cli.Arg("b", type=int, default=2),
        cli.Arg("c", type=int, default=3, is_kwarg=False),
        cli.ObjectArg(
            C,
            cli.Arg("x", default=4),
        ),
    )

    args = parser.parse_args([])
    assert args.get_value_dict() == {"a": 1, "b": 2, "c": 3, "C.x": 4}
    assert args.get_kwargs() == {"a": 1, "b": 2}
    c = args.init_object("C")
    assert isinstance(c, C)
    assert c.x == 4

    args = parser.parse_args("--a 4 --b 5 --c 6".split())
    assert args.get_value_dict() == {"a": 4, "b": 5, "c": 6, "C.x": 4}
    assert args.get_kwargs() == {"a": 4, "b": 5}

def test_object_choice_init():
    class A:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class B:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    parser = cli.Parser(
        cli.ObjectChoice(
            "top",
            cli.ObjectArg(
                A,
                cli.Arg("z", type=int, default=10),
                init_const_kwargs={"x": 20},
            ),
            cli.ObjectArg(
                B,
                init_const_kwargs={"y": 30},
            ),
            init_const_kwargs={"x": 40},
            init_requires=["y"],
        ),
    )
    args = parser.parse_args("--top A".split())
    a = args.init_object("top", y=50)
    assert isinstance(a, A)
    assert a.kwargs == {"x": 20, "y": 50, "z": 10}

    args = parser.parse_args("--top A".split())
    with pytest.raises(ValueError):
        a = args.init_object("top")

    args = parser.parse_args("--top A --top.A.z 70".split())
    a = args.init_object("top", y=50, x=60)
    assert isinstance(a, A)
    assert a.kwargs == {"x": 60, "y": 50, "z": 70}

    args = parser.parse_args("--top B".split())
    b = args.init_object("top")
    assert isinstance(b, B)
    assert b.kwargs == {"x": 40, "y": 30}

def test_get_type():
    printer = util.Printer("test_get_type", dir_name=OUTPUT_DIR)

    class A:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class B:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    parser = cli.Parser(
        cli.ObjectChoice(
            "ab",
            cli.ObjectArg(
                A,
                cli.Arg("a", type=int, default=123),
                cli.Arg("b", type=str, default="abc"),
            ),
            cli.ObjectArg(
                B,
                cli.Arg("c", type=float, default=4.5),
            ),
            default="A",
        ),
        cli.ObjectArg(
            A,
            cli.Arg("d", type=int, default=[6, 7], nargs="*"),
        ),
        cli.JsonArg("e"),
        cli.BooleanArg("f", default=True),
    )

    args = parser.parse_args("--ab A".split())
    assert issubclass(args.get_type("ab"), A)
    assert not issubclass(args.get_type("ab"), B)
    assert issubclass(args.get_type("ab.A"), A)
    assert issubclass(args.get_type("ab.B"), B)
    assert issubclass(args.get_type("ab.A.a"), int)
    assert issubclass(args.get_type("ab.A.b"), str)
    assert issubclass(args.get_type("ab.B.c"), float)
    assert issubclass(args.get_type("A.d"), list)
    assert issubclass(args.get_type("e"), type(None))
    assert issubclass(args.get_type("f"), bool)

    args = parser.parse_args("--ab B --e [1,2,{\"a\":3.4}]".split())
    assert not issubclass(args.get_type("ab"), A)
    assert issubclass(args.get_type("ab"), B)
    assert issubclass(args.get_type("e"), list)

    args = parser.parse_args("--e {\"a\":1} --no-f".split())
    assert issubclass(args.get_type("ab"), A)
    assert not issubclass(args.get_type("ab"), B)
    assert issubclass(args.get_type("e"), dict)
    assert issubclass(args.get_type("f"), bool)

    printer(parser.help())
    printer(args)

def test_duplicate_names():
    class A:
        def __init__(self, x: float):
            self.x = x

    class B:
        def __init__(self, y: float, a: A):
            self.y = y
            self.a = a

    with pytest.raises(ValueError):
        cli.Parser(
            cli.Arg("arg_name", type=int),
            cli.Arg("arg_name", type=float),
        )
    with pytest.raises(ValueError):
        cli.Parser(
            cli.ObjectArg(
                B,
                cli.Arg("y"),
                cli.ObjectArg(
                    A,
                    cli.Arg("x"),
                    name="y",
                ),
                name="model",
            ),
        )
    with pytest.raises(ValueError):
        cli.Parser(
            cli.ObjectChoice(
                "model",
                cli.ObjectArg(A, cli.Arg("x")),
                cli.ObjectArg(A, cli.Arg("x")),
            ),
        )
    with pytest.raises(ValueError):
        cli.Parser(
            cli.ObjectChoice(
                "model",
                cli.ObjectArg(A, cli.Arg("x"), name="arg_name"),
                shared_args=[
                    cli.Arg(name="arg_name"),
                ],
            ),
        )

def test_arg_registered_twice():
    class C:
        def __init__(self, x: float):
            self.x = x

    arg = cli.Arg("x", type=float, default=1.2)
    with pytest.raises(RuntimeError):
        parser = cli.Parser(
            cli.ObjectArg(
                C,
                arg,
                arg,
            ),
        )

    arg = cli.Arg("x", type=float, default=3.45)
    with pytest.raises(RuntimeError):
        parser = cli.Parser(
            arg,
            cli.ObjectArg(
                C,
                arg,
            ),
        )

def test_positional_args():
    class A:
        def __init__(self, x: float):
            self.x = x

    class B:
        def __init__(self, y: float):
            self.y = y

    parser = cli.Parser(
        cli.PositionalArg("a", type=int),
        cli.PositionalArg("b", type=int, default=3, nargs="?"),
        cli.Arg("c", type=int, default=4),
        cli.ObjectArg(A, cli.Arg("x", type=float, default=3.2)),
        cli.ObjectChoice(
            "a_or_b",
            cli.ObjectArg(A, cli.Arg("x", type=float, default=4.5)),
            cli.ObjectArg(B, cli.Arg("y", type=float, default=6.7)),
            default="B",
        ),
    )

    with pytest.raises(SystemExit):
        args = parser.parse_args([])

    args = parser.parse_args(["5"])
    assert args.get_kwargs() == {"a": 5, "b": 3, "c": 4}

    args = parser.parse_args(["5", "6"])
    assert args.get_kwargs() == {"a": 5, "b": 6, "c": 4}

    with pytest.raises(SystemExit):
        args = parser.parse_args(["5", "6", "7"])

    args = parser.parse_args(["5", "6", "--c", "7"])
    assert args.get_kwargs() == {"a": 5, "b": 6, "c": 7}

def test_autotag_edge_cases():
    parser = cli.Parser(
        cli.Arg("ab_c", default=123),
        cli.Arg("a_bc", default=456),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "a456a123"

    parser = cli.Parser(
        cli.Arg("abc", default=123),
        cli.Arg("ABC", default=456),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "a456a123"

    parser = cli.Parser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "abc123abcd456"

    parser = cli.Parser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
        cli.Arg("ABC",  default=789),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "abc789abc123abcd456"

    parser = cli.Parser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
        cli.Arg("ABC",  default=789, tag="abc"),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "abc789abc123abcd456"

    parser = cli.Parser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
        cli.Arg("ABC",  default=789, tag="abc_"),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "abc789abc123abcd456"

    parser = cli.Parser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
        cli.Arg("ABC",  default=789, tag="a_b_c"),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "abc789abc123abcd456"

    parser = cli.Parser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
        cli.Arg("ABC",  default=789, tag="random"),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "abc123abcd456random789"

    parser = cli.Parser(
        cli.Arg("abc", default=123),
        cli.Arg("abz", default=456),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "abc123abz456"

    parser = cli.Parser(
        cli.Arg("abc", default=123),
        cli.Arg("abzz", default=456),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "abc123abz456"

    parser = cli.Parser(
        cli.Arg("abc", default=123),
        cli.Arg("azc", default=456),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "ab123az456"

    parser = cli.Parser(
        cli.Arg("abc", default=123),
        cli.Arg("azz", default=456),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "ab123az456"

    arg_names = [
        "abc",
        "abcdef",
        "a",
        "num_epochs",
        "num_steps",
        "ne",
        "ns",
        "p",
        "plid",
        "plimits",
        "b",
    ]
    parser = cli.Parser(
        cli.Arg("plim", default="plim", tag="plim"),
        *[
            cli.Arg(n, default=n)
            for n in arg_names
        ],
    )
    args = parser.parse_args([])
    assert args.get_summary() == (
        "aAabcABCabcdABCDEFbBneNEnsNSnumeNUMEPOCHSnumsNUMSTEPSpPplidPLID"
        "plimPLIMplimiPLIMITS"
    )

def test_autotag_edge_cases_objectchoice():
    class A:
        def __init__(self, x):
            self.x = x

    class B:
        def __init__(self, y):
            self.y = y

    parser = cli.Parser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
        cli.ObjectChoice(
            "ABC",
            cli.ObjectArg(A, cli.Arg("x", default="78")),
            cli.ObjectArg(B, cli.Arg("y", default="90")),
            default="A",
        ),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "abcAabc123abcd456abcx78"

    parser = cli.Parser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
        cli.ObjectChoice(
            "ABC",
            cli.ObjectArg(A, cli.Arg("x", default="78")),
            cli.ObjectArg(B, cli.Arg("y", default="90")),
            default="A",
            tag="a_b_c",
        ),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "abcAabc123abcd456abcx78"

    parser = cli.Parser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
        cli.ObjectChoice(
            "a_or_b",
            cli.ObjectArg(A, cli.Arg("x", default="78")),
            cli.ObjectArg(B, cli.Arg("y", default="90")),
            default="A",
        ),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "abc123abcd456aoAaox78"

    args = parser.parse_args(["--a_or_b", "B"])
    assert args.get_summary() == "abc123abcd456aoBaoy90"

    parser = cli.Parser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
        cli.ObjectChoice(
            "model",
            cli.ObjectArg(A, cli.Arg("x", default="78")),
            cli.ObjectArg(B, cli.Arg("y", default="90")),
            default="A",
        ),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "abc123abcd456mAmx78"

    args = parser.parse_args(["--model", "B"])
    assert args.get_summary() == "abc123abcd456mBmy90"

def test_boolean_arg():
    parser = cli.Parser(
        cli.BooleanArg("a"),
        cli.BooleanArg("b", default=True),
        cli.BooleanArg("c", default=False),
        cli.Arg("x", type=float, default=1.23),
    )
    assert parser.parse_args([]).get_kwargs() == (
        {"a": None, "b": True, "c": False, "x": 1.23}
    )
    assert parser.parse_args(["--a"]).get_kwargs() == (
        {"a": True, "b": True, "c": False, "x": 1.23}
    )
    assert parser.parse_args(["--no-a"]).get_kwargs() == (
        {"a": False, "b": True, "c": False, "x": 1.23}
    )
    assert parser.parse_args(["--b"]).get_kwargs() == (
        {"a": None, "b": True, "c": False, "x": 1.23}
    )
    assert parser.parse_args(["--no-b"]).get_kwargs() == (
        {"a": None, "b": False, "c": False, "x": 1.23}
    )
    assert parser.parse_args(["--c"]).get_kwargs() == (
        {"a": None, "b": True, "c": True, "x": 1.23}
    )
    assert parser.parse_args(["--no-c"]).get_kwargs() == (
        {"a": None, "b": True, "c": False, "x": 1.23}
    )

def test_json_arg():
    printer = util.Printer("test_json_arg", OUTPUT_DIR)

    parser = cli.Parser(
        cli.JsonArg("a"),
        cli.JsonArg("b", required=True, nargs="+"),
        prog="test_json_arg",
    )

    assert parser.help() == (
        "usage: test_json_arg [-h] [--a A] --b B [B ...]\n"
        "\n"
        "options:\n"
        "  -h, --help     show this help message and exit\n"
        "  --a A          Format: JSON string\n"
        "  --b B [B ...]  nargs='+', required=True (format: JSON string)\n"
    )

    args = parser.parse_args("--b 3.4 \"abc\" 7 [1,2,3]".split())
    assert args.get_value("a") is None
    assert args.get_value("b") == [3.4, "abc", 7, [1, 2, 3]]

    args = parser.parse_args("--b 1.2 --a 3".split())
    assert isinstance(args.get_value("a"), int)
    assert args.get_value("a") == 3
    assert args.get_value("b") == [1.2]

    args = parser.parse_args("--b 1.2 --a 4.5".split())
    assert isinstance(args.get_value("a"), float)
    assert args.get_value("a") == 4.5
    assert args.get_value("b") == [1.2]

    args = parser.parse_args("--b 1.2 --a [6,7,8]".split())
    assert isinstance(args.get_value("a"), list)
    assert args.get_value("a") == [6, 7, 8]
    assert args.get_value("b") == [1.2]

    args = parser.parse_args("--b 1.2 --a [6,7,8,[9,10]]".split())
    assert isinstance(args.get_value("a"), list)
    assert args.get_value("a") == [6, 7, 8, [9, 10]]
    assert args.get_value("b") == [1.2]

    args = parser.parse_args("--b 1.2 --a \"abc\"".split())
    assert isinstance(args.get_value("a"), str)
    assert args.get_value("a") == "abc"
    assert args.get_value("b") == [1.2]

def test_subcommand():
    parser = cli.Parser(
        sub_commands=cli.SubCommandGroup(
            cli.SubCommand(
                "train",
                cli.Arg("model",   default="mlp"),
                cli.Arg("dataset", default="mnist"),
            ),
            cli.SubCommand(
                "sweep",
                cli.Arg("sweep_arg_name", default="hidden_dim"),
            ),
        ),
        prog="test_subcommand",
    )
    assert parser.help() == (
        "usage: test_subcommand [-h] {train,sweep} ...\n"
        "\n"
        "options:\n"
        "  -h, --help     show this help message and exit\n"
        "\n"
        "command:\n"
        "  {train,sweep}\n"
    )

    with pytest.raises(SystemExit):
        parser.parse_args([])

    args = parser.parse_args("train".split())
    assert isinstance(args.get_command(), cli.SubCommand)
    assert repr(args.get_command()) == (
        "SubCommand('train', "
        "Arg(full_name='model', name='model', value='mlp'), "
        "Arg(full_name='dataset', name='dataset', value='mnist'))"
    )
    assert args.get_kwargs() == {
        "model": "mlp",
        "dataset": "mnist",
    }
    assert args.get_command().get_kwargs() == {
        "model": "mlp",
        "dataset": "mnist",
    }

    args = parser.parse_args("train --model CNN".split())
    assert isinstance(args.get_command(), cli.SubCommand)
    assert repr(args.get_command()) == (
        "SubCommand('train', "
        "Arg(full_name='model', name='model', value='CNN'), "
        "Arg(full_name='dataset', name='dataset', value='mnist'))"
    )
    assert args.get_kwargs() == {
        "model": "CNN",
        "dataset": "mnist",
    }
    assert args.get_command().get_kwargs() == {
        "model": "CNN",
        "dataset": "mnist",
    }

    args = parser.parse_args("train --model CNN --dataset CIFAR".split())
    assert isinstance(args.get_command(), cli.SubCommand)
    assert repr(args.get_command()) == (
        "SubCommand('train', "
        "Arg(full_name='model', name='model', value='CNN'), "
        "Arg(full_name='dataset', name='dataset', value='CIFAR'))"
    )
    assert args.get_kwargs() == {
        "model": "CNN",
        "dataset": "CIFAR",
    }
    assert args.get_command().get_kwargs() == {
        "model": "CNN",
        "dataset": "CIFAR",
    }

    args = parser.parse_args("sweep".split())
    assert isinstance(args.get_command(), cli.SubCommand)
    assert repr(args.get_command()) == (
        "SubCommand('sweep', Arg(full_name='sweep_arg_name', "
        "name='sweep_arg_name', value='hidden_dim'))"
    )
    assert args.get_kwargs() == {
        "sweep_arg_name": "hidden_dim",
    }
    assert args.get_command().get_kwargs() == {
        "sweep_arg_name": "hidden_dim",
    }

    arg_str = "sweep --sweep_arg_name num_hidden_layers"
    args = parser.parse_args(arg_str.split())
    assert isinstance(args.get_command(), cli.SubCommand)
    assert repr(args.get_command()) == (
        "SubCommand('sweep', Arg(full_name='sweep_arg_name', "
        "name='sweep_arg_name', value='num_hidden_layers'))"
    )
    assert args.get_kwargs() == {
        "sweep_arg_name": "num_hidden_layers",
    }
    assert args.get_command().get_kwargs() == {
        "sweep_arg_name": "num_hidden_layers",
    }

    with pytest.raises(AttributeError):
        args.get_command().run(args)

def test_subcommand_get_value_dict():
    class Mlp:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Cnn:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    parser = cli.Parser(
        cli.Arg("seed", default=0, type=int),
        cli.Arg("gpu",  action="store_true"),
        sub_commands=cli.SubCommandGroup(
            cli.SubCommand(
                "train",
                cli.Arg("dataset", default="mnist"),
                cli.ObjectChoice(
                    "model",
                    cli.ObjectArg(
                        Mlp,
                        cli.Arg("hidden_dim", type=int, default=64),
                    ),
                    cli.ObjectArg(
                        Cnn,
                        cli.Arg("kernel_size", type=int, default=5),
                    ),
                    default="Mlp",
                ),
            ),
            cli.SubCommand(
                "sweep",
                cli.Arg("sweep_arg_name", default="hidden_dim"),
            ),
        ),
    )

    with pytest.raises(SystemExit):
        parser.parse_args([])

    args = parser.parse_args(["train"])
    assert args.get_value_dict() == {
        "seed": 0,
        "gpu": False,
        "dataset": "mnist",
        "model": "Mlp",
        "model.Mlp.hidden_dim": 64,
    }
    assert args.get_kwargs() == {
        "seed": 0,
        "gpu": False,
        "dataset": "mnist",
    }
    assert args.get_command().get_kwargs() == {
        "dataset": "mnist",
    }

    args = parser.parse_args(["sweep"])
    assert args.get_value_dict() == {
        "seed": 0,
        "gpu": False,
        "sweep_arg_name": "hidden_dim",
    }
    assert args.get_kwargs() == {
        "seed": 0,
        "gpu": False,
        "sweep_arg_name": "hidden_dim",
    }
    assert args.get_command().get_kwargs() == {
        "sweep_arg_name": "hidden_dim",
    }

    arg_str = "--seed 123 --gpu train --model Cnn --dataset cifar10"
    args = parser.parse_args(arg_str.split())
    assert args.get_value_dict() == {
        "seed": 123,
        "gpu": True,
        "dataset": "cifar10",
        "model": "Cnn",
        "model.Cnn.kernel_size": 5,
    }
    assert args.get_kwargs() == {
        "seed": 123,
        "gpu": True,
        "dataset": "cifar10",
    }
    assert args.get_command().get_kwargs() == {
        "dataset": "cifar10",
    }

def test_no_tag_arg():
    printer = util.Printer("test_no_tag_arg", dir_name=OUTPUT_DIR)

    class C:
        def __init__(self, x: int, y: float):
            self.x = x
            self.y = y

    parser = cli.Parser(
        cli.Arg("a", type=int,      default=1),
        cli.Arg("b", type=float,    default=2.3),
        cli.Arg("c", type=str,      default="abc"),
        cli.NoTagArg("d", type=int,      default=4),
        cli.NoTagArg("e", type=float,    default=5.67),
        cli.ObjectArg(
            C,
            cli.Arg("x", type=int,      default=-8),
            cli.NoTagArg("y", type=float,    default=-9.99),
        ),
        prog="test_no_tag_arg",
    )

    assert parser.help() == (
        "usage: test_no_tag_arg [-h] [--a A] [--b B] [--c C] [--d D] "
        "[--e E] [--C.x X]\n"
        "                       [--C.y Y]\n"
        "\n"
        "options:\n"
        "  -h, --help  show this help message and exit\n"
        "  --a A       default=1, type=<class 'int'>\n"
        "  --b B       default=2.3, type=<class 'float'>\n"
        "  --c C       default='abc', type=<class 'str'>\n"
        "  --d D       default=4, type=<class 'int'>\n"
        "  --e E       default=5.67, type=<class 'float'>\n"
        "  --C.x X     default=-8, type=<class 'int'>\n"
        "  --C.y Y     default=-9.99, type=<class 'float'>\n"
    )

    args = parser.parse_args([])
    assert util.format_dict(args.get_value_dict()) == (
        "C.x=-8, C.y=-9.99, a=1, b=2.3, c='abc', d=4, e=5.67"
    )
    assert args.get_summary() == "a1b2.3cABCcx-8"
    c = args.init_object("C")
    assert isinstance(c, C)
    assert c.x == -8
    assert c.y == -9.99

    args = parser.parse_args("--a 2 --e 3.14 --C.x -9 --C.y 8.88".split())
    assert util.format_dict(args.get_value_dict()) == (
        "C.x=-9, C.y=8.88, a=2, b=2.3, c='abc', d=4, e=3.14"
    )
    assert args.get_summary() == "a2b2.3cABCcx-9"
    c = args.init_object("C")
    assert isinstance(c, C)
    assert c.x == -9
    assert c.y == 8.88

    printer(args, parser, parser.help(), sep="\n\n")

def test_arg_group():
    printer = util.Printer("test_arg_group", dir_name=OUTPUT_DIR)

    class Trainer:
        def __init__(self, num_epochs: int):
            self.num_epochs = num_epochs

    class Mlp:
        def __init__(self, num_layers: int):
            self.num_layers = num_layers

    class Cnn:
        def __init__(self, num_layers: int, kernel_size: int):
            self.num_layers = num_layers
            self.kernel_size = kernel_size

    def get_parser(is_group: bool):
        return cli.Parser(
            cli.ObjectArg(
                Trainer,
                cli.Arg("num_epochs", type=int, default=10),
                is_group=is_group,
            ),
            cli.ObjectChoice(
                "model",
                cli.ObjectArg(
                    Mlp,
                    cli.Arg("num_layers", type=int, default=5),
                ),
                cli.ObjectArg(
                    Cnn,
                    cli.Arg("num_layers", type=int, default=5),
                    cli.Arg("kernel_size", type=int, default=3),
                ),
                is_group=is_group,
            ),
            cli.Arg("output_name"),
            prog="test_arg_group",
        )

    parser = get_parser(is_group=False)
    printer(parser.help())
    assert parser.help() == (
        "usage: test_arg_group [-h] [--Trainer.num_epochs N] "
        "[--model {Mlp,Cnn}]\n"
        "                      [--model.Mlp.num_layers N] "
        "[--model.Cnn.num_layers N]\n"
        "                      [--model.Cnn.kernel_size K] "
        "[--output_name O]\n"
        "\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --Trainer.num_epochs N\n"
        "                        default=10, type=<class 'int'>\n"
        "  --model {Mlp,Cnn}     default=None, required=True\n"
        "  --model.Mlp.num_layers N\n"
        "                        default=5, type=<class 'int'>\n"
        "  --model.Cnn.num_layers N\n"
        "                        default=5, type=<class 'int'>\n"
        "  --model.Cnn.kernel_size K\n"
        "                        default=3, type=<class 'int'>\n"
        "  --output_name O\n"
    )

    args = parser.parse_args([])
    with pytest.raises(ValueError):
        args.get_value_dict()

    assert parser.parse_args("--model Mlp".split()).get_value_dict() == {
        'Trainer.num_epochs': 10,
        'model': 'Mlp',
        'model.Mlp.num_layers': 5,
        'output_name': None,
    }

    parser = get_parser(is_group=True)
    printer(parser.help())
    assert parser.help() == (
        "usage: test_arg_group [-h] [--Trainer.num_epochs N] "
        "[--model {Mlp,Cnn}]\n"
        "                      [--model.Mlp.num_layers N] "
        "[--model.Cnn.num_layers N]\n"
        "                      [--model.Cnn.kernel_size K] "
        "[--output_name O]\n"
        "\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --output_name O\n"
        "\n"
        "Trainer:\n"
        "  --Trainer.num_epochs N\n"
        "                        default=10, type=<class 'int'>\n"
        "\n"
        "model:\n"
        "  --model {Mlp,Cnn}     default=None, required=True\n"
        "\n"
        "model.Mlp:\n"
        "  --model.Mlp.num_layers N\n"
        "                        default=5, type=<class 'int'>\n"
        "\n"
        "model.Cnn:\n"
        "  --model.Cnn.num_layers N\n"
        "                        default=5, type=<class 'int'>\n"
        "  --model.Cnn.kernel_size K\n"
        "                        default=3, type=<class 'int'>\n"
    )

    assert parser.parse_args("--model Mlp".split()).get_value_dict() == {
        'Trainer.num_epochs': 10,
        'model': 'Mlp',
        'model.Mlp.num_layers': 5,
        'output_name': None,
    }

def test_dubplicate_tags():
    printer = util.Printer("test_dubplicate_tags", dir_name=OUTPUT_DIR)

    class Cnn:
        def __init__(self, kernel_size, stride, padding):
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding

    class ConvNext:
        def __init__(self, kernel_size, stride, padding):
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding

    parser = cli.Parser(
        cli.ObjectChoice(
            "model",
            cli.ObjectArg(
                Cnn,
                cli.Arg("kernel_size", default=5),
                cli.Arg("stride",  default=6),
                cli.Arg("padding", default=7),
            ),
            cli.ObjectArg(
                ConvNext,
                cli.Arg("kernel_size", default=1),
                cli.Arg("stride",  default=2),
                cli.Arg("padding", default=3),
            ),
            default="Cnn",
            is_group=True,
        ),
    )

    args = parser.parse_args([])
    assert args.get_summary() == "mCNmk5mp7ms6"
    assert args.get_value_dict() == {
        'model': 'Cnn',
        'model.Cnn.kernel_size': 5,
        'model.Cnn.stride': 6,
        'model.Cnn.padding': 7,
    }

    args = parser.parse_args("--model ConvNext".split())
    assert args.get_summary() == "mCOmk1mp3ms2"
    assert args.get_value_dict() == {
        'model': 'ConvNext',
        'model.ConvNext.kernel_size': 1,
        'model.ConvNext.stride': 2,
        'model.ConvNext.padding': 3,
    }

    printer(parser.help())

def test_get_value_summary():
    printer = util.Printer("test_get_value_summary", dir_name=OUTPUT_DIR)
    cli.verbose.set_printer(printer)

    class Abc:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __repr__(self):
            return util.format_type(type(self), **self.kwargs)

    class Abcd(Abc):    pass
    class Abcdef(Abc):  pass
    class Abcdghi(Abc): pass
    class Lmnop(Abc):   pass

    parser = cli.Parser(
        cli.Arg("x", type=int,   default=123),
        cli.Arg("y", type=float, default=4.5),
        cli.ObjectChoice(
            "top",
            cli.ObjectArg(Abc),
            cli.ObjectArg(Abcd),
            cli.ObjectArg(Abcdef),
            cli.ObjectArg(Abcdghi),
            cli.ObjectArg(
                Lmnop,
                cli.Arg("xx", type=int,   default=6),
                cli.Arg("xy", type=float, default=7.8),
                cli.ObjectChoice(
                    "bottom",
                    cli.ObjectArg(Abc),
                    cli.ObjectArg(Abcd),
                    cli.ObjectArg(Abcdef),
                    cli.ObjectArg(Abcdghi),
                    cli.ObjectArg(
                        Lmnop,
                        cli.Arg("xy", type=int,   default=9),
                        cli.Arg("yy", type=float, default=10.11),
                    ),
                    is_group=True,
                ),
            ),
        ),
        prog="test_get_value_summary",
    )
    printer(parser.help())

    args = parser.parse_args([])
    assert args.get_arg("x").get_value_summary() == "123"
    assert args.get_arg("y").get_value_summary() == "4.5"
    assert args.get_arg("top").value is None
    with pytest.raises(ValueError):
        args.get_arg("top").get_value_summary()
    with pytest.raises(ValueError):
        args.get_summary()
    with pytest.raises(ValueError):
        args.get_value_dict()
    with pytest.raises(ValueError):
        args.init_object("top")

    args = parser.parse_args("--top Abc".split())
    assert args.get_arg("top").get_value_summary() == "ABC"
    assert args.get_summary() == "tABCx123y4.5"
    assert args.get_value_dict() == {"top": "Abc", "x": 123, "y": 4.5}
    with cli.verbose:
        assert repr(args.init_object("top")) == "Abc()"

    args = parser.parse_args("--top Abcd".split())
    assert args.get_arg("top").get_value_summary() == "ABCD"
    assert args.get_summary() == "tABCDx123y4.5"

    args = parser.parse_args("--top Abcdef".split())
    assert args.get_arg("top").get_value_summary() == "ABCDE"
    assert args.get_summary() == "tABCDEx123y4.5"

    args = parser.parse_args("--top Abcdghi".split())
    assert args.get_arg("top").get_value_summary() == "ABCDG"
    assert args.get_summary() == "tABCDGx123y4.5"

    args = parser.parse_args("--top Lmnop".split())
    assert args.get_arg("top").get_value_summary() == "L"
    with pytest.raises(ValueError):
        args.get_summary()
    with pytest.raises(ValueError):
        args.get_value_dict()
    with pytest.raises(ValueError):
        args.init_object("top")

    args = parser.parse_args("--top Lmnop --top.Lmnop.bottom Abc".split())
    assert args.get_arg("top").get_value_summary() == "L"
    assert args.get_arg("top.Lmnop.bottom").get_value_summary() == "ABC"
    assert args.get_summary() == "tLtbABCtxx6txy7.8x123y4.5"

    args = parser.parse_args("--top Lmnop --top.Lmnop.bottom Abcdghi".split())
    assert args.get_arg("top").get_value_summary() == "L"
    assert args.get_arg("top.Lmnop.bottom").get_value_summary() == "ABCDG"
    assert args.get_summary() == "tLtbABCDGtxx6txy7.8x123y4.5"

    args = parser.parse_args("--top Lmnop --top.Lmnop.bottom Lmnop".split())
    assert args.get_arg("top").get_value_summary() == "L"
    assert args.get_arg("top.Lmnop.bottom").get_value_summary() == "L"
    assert args.get_summary() == "tLtbLtbx9tby10.11txx6txy7.8x123y4.5"
    assert args.get_value_dict() == {
        "x": 123,
        "y": 4.5,
        "top": "Lmnop",
        "top.Lmnop.xx": 6,
        "top.Lmnop.xy": 7.8,
        "top.Lmnop.bottom": "Lmnop",
        "top.Lmnop.bottom.Lmnop.xy": 9,
        "top.Lmnop.bottom.Lmnop.yy": 10.11,
    }
    with cli.verbose:
        assert repr(args.init_object("top")) == (
            "Lmnop(bottom=Lmnop(xy=9, yy=10.11), xx=6, xy=7.8)"
        )
        assert repr(args.init_object("top.Lmnop.bottom")) == (
            "Lmnop(xy=9, yy=10.11)"
        )

def test_get_value_summary_tag():
    class Base:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __repr__(self) -> str:
            return util.format_type(type(self), **self.kwargs)

    class Adam(Base):           pass
    class AdamW(Base):          pass
    class Chamfer(Base):        pass
    class CrossEntropy(Base):   pass

    parser = cli.Parser(
        cli.ObjectChoice(
            "optimiser",
            cli.ObjectArg(
                Adam,
                cli.Arg("lr", type=float, default=1e-3),
            ),
            cli.ObjectArg(
                AdamW,
                cli.Arg("lr", type=float, default=1e-3),
                cli.Arg("wd", type=float, default=1e-2),
                tag="AW",
            ),
            default="Adam",
        ),
        cli.ObjectChoice(
            "loss",
            cli.ObjectArg(CrossEntropy),
            cli.ObjectArg(Chamfer, tag="CH"),
            default="CrossEntropy",
        ),
    )
    args = parser.parse_args([])
    assert args.get_summary() == "lCoAol0.001"
    assert args.get_arg("loss").get_value_summary() == "C"
    assert args.get_arg("optimiser").get_value_summary() == "A"

    args = parser.parse_args("--optimiser AdamW".split())
    assert args.get_summary() == "lCoAWol0.001ow0.01"
    assert args.get_arg("loss").get_value_summary() == "C"
    assert args.get_arg("optimiser").get_value_summary() == "AW"

    args = parser.parse_args("--optimiser AdamW --loss Chamfer".split())
    assert args.get_summary() == "lCHoAWol0.001ow0.01"
    assert args.get_arg("loss").get_value_summary() == "CH"
    assert args.get_arg("optimiser").get_value_summary() == "AW"

def test_get_value_summary_name():
    class A:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __repr__(self) -> str:
            return util.format_type(type(self), **self.kwargs)

    class B(A):
        pass

    parser = cli.Parser(
        cli.ObjectChoice(
            "c1",
            cli.ObjectArg(
                A,
                cli.Arg("a", type=float, default=12),
                name="A_name",
            ),
            cli.ObjectArg(
                B,
                cli.Arg("b", type=float, default=3.4),
                name="B_name",
            ),
            default="A_name",
        ),
    )
    args = parser.parse_args([])
    assert args.get_value_dict()    == {"c1": "A_name", "c1.A_name.a": 12}
    assert args.get_summary()       == "cAca12"

    c1_arg = args.get_arg("c1")
    assert c1_arg.value == "A_name"
    assert c1_arg.get_value_dict()      == {"c1.A_name.a": 12}
    assert c1_arg.get_summary()         == "a12"
    assert c1_arg.get_value_summary()   == "A"

    c1_arg = args.get_arg("c1.A_name.a")
    assert c1_arg.value == 12
    assert c1_arg.get_value_dict()      == dict()
    assert c1_arg.get_summary()         == ""
    assert c1_arg.get_value_summary()   == "12"

def test_set_default_choice():
    printer = util.Printer("test_set_default_choice", dir_name=OUTPUT_DIR)

    class A:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __repr__(self) -> str:
            return util.format_type(type(self), **self.kwargs)

    class B(A):
        pass

    parser = cli.Parser(
        cli.ObjectChoice(
            "c1",
            cli.ObjectArg(A, cli.Arg("a", type=float, default=12)),
            cli.ObjectArg(B, cli.Arg("b", type=float, default=3.4)),
        ),
        cli.ObjectChoice(
            "c2",
            cli.ObjectArg(A, cli.Arg("c", type=float, default=5)),
            cli.ObjectArg(B, cli.Arg("d", type=float, default=67)),
            required=False,
        ),
        cli.ObjectChoice(
            "c3",
            cli.ObjectArg(A, cli.Arg("e", type=float, default=-8)),
            cli.ObjectArg(B, cli.Arg("f", type=float, default=9.9)),
            default="A",
        ),
        prog="test_set_default_choice",
    )

    printer(parser.help())
    assert parser.help() == (
        "usage: test_set_default_choice [-h] [--c1 {A,B}] [--c1.A.a A] "
        "[--c1.B.b B]\n"
        "                               [--c2 {A,B}] [--c2.A.c C] "
        "[--c2.B.d D]\n"
        "                               [--c3 {A,B}] [--c3.A.e E] "
        "[--c3.B.f F]\n"
        "\n"
        "options:\n"
        "  -h, --help  show this help message and exit\n"
        "  --c1 {A,B}  default=None, required=True\n"
        "  --c1.A.a A  default=12, type=<class 'float'>\n"
        "  --c1.B.b B  default=3.4, type=<class 'float'>\n"
        "  --c2 {A,B}  default=None, required=False\n"
        "  --c2.A.c C  default=5, type=<class 'float'>\n"
        "  --c2.B.d D  default=67, type=<class 'float'>\n"
        "  --c3 {A,B}  default=A, required=False\n"
        "  --c3.A.e E  default=-8, type=<class 'float'>\n"
        "  --c3.B.f F  default=9.9, type=<class 'float'>\n"
    )

    args = parser.parse_args([])
    with pytest.raises(ValueError):
        args.init_object("c1")

    with pytest.raises(ValueError):
        args.init_object("c2")

    assert repr(args.init_object("c3")) == "A(e=-8)"

    args = parser.parse_args([])
    c1_B = args.get_arg("c1.B")
    with pytest.raises(ValueError):
        c1_B.set_default_choice("B")

    c1_B_b = args.get_arg("c1.B.b")
    with pytest.raises(ValueError):
        c1_B_b.set_default_choice("B")

    args.get_arg("c1").set_default_choice("B")
    assert repr(args.init_object("c1")) == "B(b=3.4)"

    args.get_arg("c1").set_default_choice("A")
    assert repr(args.init_object("c1")) == "B(b=3.4)"

    args.get_arg("c2").set_default_choice(None)
    with pytest.raises(ValueError):
        args.init_object("c2")

    args.get_arg("c2").set_default_choice("A")
    assert repr(args.init_object("c2")) == "A(c=5)"

    args.get_arg("c2").set_default_choice(None)
    assert repr(args.init_object("c3")) == "A(e=-8)"

    args.get_arg("c2").set_default_choice("B")
    assert repr(args.init_object("c3")) == "A(e=-8)"
