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

    assert repr(args) == "ParsedArgs(c='abc', d=False, m='A', m.A.x=1)"

    assert args.get_value_dict() == {
        "m": "A",
        "m.A.x": 1,
        "c": "abc",
        "d": False,
    }

    assert args.get_summary() == "cABCdFmAm.x1"

    c = args.get_arg("c")
    assert isinstance(c, cli.Arg)
    assert c.value == "abc"

    assert args.get_value("c") == "abc"
    assert args.get_value("m.A.x") == 1

    assert args.get_kwargs() == {"c": "abc", "d": False}

    args.update({"c": "defg"})
    assert args.get_value("c") == "defg"
    assert repr(args) == "ParsedArgs(c='defg', d=False, m='A', m.A.x=1)"
    assert args.get_summary() == "cDEFGdFmAm.x1"

    assert args.get_value("m.A") is None
    assert not isinstance(args.get_value("m.A"), A)
    a = args.init_object("m")
    assert isinstance(a, A)
    assert a.x == 1
    assert not args.get_value("m.A") is None
    assert isinstance(args.get_value("m.A"), A)
    assert repr(args) == "ParsedArgs(c='defg', d=False, m='A', m.A.x=1)"
    assert args.get_summary() == "cDEFGdFmAm.x1"

    args.reset_object_cache()
    assert args.get_value("m.A") is None
    assert not isinstance(args.get_value("m.A"), A)

    args.update({"m.A.x": 3})
    a = args.init_object("m")
    assert isinstance(a, A)
    assert a.x == 3

    assert repr(args) == "ParsedArgs(c='defg', d=False, m='A', m.A.x=3)"
    assert args.get_value_dict() == {
        "m": "A",
        "m.A.x": 3,
        "c": "defg",
        "d": False,
    }
    assert args.get_summary() == "cDEFGdFmAm.x3"

    new_args = parser.parse_args(
        "--m B --m.B.a.x 5 --m.B.y 6.7 --c hijk --d".split(),
    )
    assert isinstance(new_args, cli.ParsedArgs)
    assert repr(new_args) == (
        "ParsedArgs(c='hijk', d=True, m='B', m.B.a.x=5, m.B.y=6.7)"
    )
    assert new_args.get_value_dict() == {
        "m": "B",
        "m.B.a.x": 5,
        "m.B.y": 6.7,
        "c": "hijk",
        "d": True,
    }
    assert new_args.get_summary() == "cHIJKdTmBm.a.x5m.y6.7"
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
    )

    printer(parser.help())

    assert parser.help() == (
        "usage: run_pytest_script.py [-h] [--model {A,Mlp}] "
        "[--model.A.x MODEL.A.X]\n"
        "                            [--model.Mlp.a.x MODEL.MLP.A.X]\n"
        "                            [--model.Mlp.y MODEL.MLP.Y] "
        "[--model.z MODEL.Z]\n"
        "                            [--c C] [--d] [--e [E ...]] "
        "[--h | --no-h]\n"
        "                            [--i | --no-i] [--j | --no-j] [--k K]\n"
        "                            [--l [L ...]]\n"
        "                            f g\n"
        "\n"
        "positional arguments:\n"
        "  f                     type=<class 'float'>\n"
        "  g                     type=<class 'int'>\n"
        "\n"
        "options:\n"
        "  -h, --help            show this help message and exit\n"
        "  --model {A,Mlp}\n"
        "  --model.A.x MODEL.A.X\n"
        "                        default=1, type=<class 'int'>\n"
        "  --model.Mlp.a.x MODEL.MLP.A.X\n"
        "                        default=2, type=<class 'int'>\n"
        "  --model.Mlp.y MODEL.MLP.Y\n"
        "                        default=3.4, type=<class 'float'>\n"
        "  --model.z MODEL.Z     default='abc', type=<class 'str'>\n"
        "  --c C                 default='defg', type=<class 'str'>\n"
        "  --d                   action='store_true'\n"
        "  --e [E ...]           default=[], nargs='*', type=<class 'int'>\n"
        "  --h, --no-h           (default: None)\n"
        "  --i, --no-i           (default: True)\n"
        "  --j, --no-j           (default: False)\n"
        "  --k K                 default=[1, [2, 3]] (format: JSON string)\n"
        "  --l [L ...]           default=[4, [5, 6]], nargs='*' "
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
    assert args.get_summary() == (
        "h.a.b1h.a.cABCh.d.e2,3h.d.f4.5h.d.g.bNh.i6j-7.8"
    )
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

    with pytest.raises(SystemExit):
        args = parser.parse_args([])

    printer.heading("model = A")

    args = parser.parse_args("--model A".split())
    assert args.get_summary() == "mAm.b1m.cABC"
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
    assert args.get_summary() == "mAm.b3m.cXYZ"
    with cli.verbose:
        a = args.init_object("model")
        assert isinstance(a, A)
        assert a.c == "xyz"
        assert a.b == 3

    printer.heading("model = D")

    args = parser.parse_args("--model D".split())
    assert args.get_summary() == "mDm.e2,3m.f4.5m.gAm.g.b1m.g.cABC"
    assert args.get_arg("model").get_summary() == "e2,3f4.5gAg.b1g.cABC"
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
    assert args.get_summary() == "mDm.e7,8,9m.f4.5m.gDm.g.f3.141m.g.g.bN"
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

    assert repr(args) == "ParsedArgs(a=1, b=2.3, c='abc')"
    assert args.get_summary() == "a1b2.3cABC"
    assert args.get_value_dict() == {"a": 1, "b": 2.3, "c": "abc"}

    new_arg_dict = {"a": 4, "d": "xyz", "e.f": 5.67}
    with pytest.raises(ValueError):
        args.update(new_arg_dict)

    args.update(new_arg_dict, allow_new_keys=True)
    assert repr(args) == "ParsedArgs(a=4, b=2.3, c='abc', d='xyz', e.f=5.67)"
    assert args.get_summary() == "a4b2.3cABC"
    assert args.get_value_dict() == {
        "a": 4,
        "b": 2.3,
        "c": "abc",
        "d": "xyz",
        "e.f": 5.67,
    }
    assert args.get_kwargs() == {"a": 4, "b": 2.3, "c": "abc"}

    assert isinstance(args.get_arg("a"),    cli.Arg)
    assert isinstance(args.get_arg("d"),    cli._UnknownArg)
    assert isinstance(args.get_arg("e.f"),  cli._UnknownArg)

    assert repr(args.get_arg("a")) == (
        "Arg(full_name='a', name='a', value=4)"
    )
    assert repr(args.get_arg("d")) == (
        "_UnknownArg(full_name='d', name=None, value='xyz')"
    )
    assert repr(args.get_arg("e.f")) == (
        "_UnknownArg(full_name='e.f', name=None, value=5.67)"
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
        "cli: retrieving \"B.a\" from cache\n"
        "\n"
        "------------------------ (1) args.reset_object_cache "
        "-------------------------\n"
        "\n"
        "cli: A(x=8)\n"
        "cli: retrieving \"B.a\" from cache\n"
        "cli: retrieving \"B.a\" from cache\n"
        "cli: B(a=A(), y=-10.11)\n"
        "cli: retrieving \"B\" from cache\n"
        "cli: A(x=14)\n"
        "cli: retrieving \"A\" from cache\n"
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
    assert args.get_summary() == "abcAabc123abc.x78abcd456"

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
    assert args.get_summary() == "abcAabc123abc.x78abcd456"

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
    assert args.get_summary() == "abc123abcd456aoAao.x78"

    args = parser.parse_args(["--a_or_b", "B"])
    assert args.get_summary() == "abc123abcd456aoBao.y90"

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
    assert args.get_summary() == "abc123abcd456mAm.x78"

    args = parser.parse_args(["--model", "B"])
    assert args.get_summary() == "abc123abcd456mBm.y90"

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
    )

    assert parser.help() == (
        "usage: run_pytest_script.py [-h] [--a A] --b B [B ...]\n"
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
