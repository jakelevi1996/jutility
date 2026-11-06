from jutility import util, cli

OUTPUT_DIR = util.get_test_output_dir("test_cli/test_args")

def test_boolean_arg():
    parser = cli.Parser(
        cli.BooleanArg("a"),
        cli.BooleanArg("b", default=True),
        cli.BooleanArg("c", help="Help!"),
        cli.BooleanArg("d", help="Help!", default=True),
        prog="test_boolean_arg",
    )

    args = parser.parse_args([])
    assert args.get_kwargs() == {"a": None, "b": True, "c": None, "d": True}
    args = parser.parse_args(["--a"])
    assert args.get_kwargs() == {"a": True, "b": True, "c": None, "d": True}
    args = parser.parse_args(["--no-a"])
    assert args.get_kwargs() == {"a": False, "b": True, "c": None, "d": True}
    args = parser.parse_args(["--no-a", "--no-b"])
    assert args.get_kwargs() == {"a": False, "b": False, "c": None, "d": True}

    assert util.strings_equal_except_whitespace(
        parser.help(),
        (
            "usage: test_boolean_arg [-h] [--a | --no-a] [--b | --no-b] "
            "[--c | --no-c] [--d | --no-d]"
            ""
            "options:"
            "  -h, --help   show this help message and exit"
            "  --a, --no-a  (default: None)"
            "  --b, --no-b"
            "  --c, --no-c  Help! (default: None)"
            "  --d, --no-d  Help!"
        ),
    )

def test_json_arg():
    parser = cli.Parser(
        cli.JsonArg("a"),
        cli.JsonArg("b"),
        cli.JsonArg("c"),
        cli.JsonArg("d"),
    )
    args = [
        "--a",
        "{1: 2, 3: 4}",
        "--b",
        "{'1': 2, '3': 4}",
        "--c",
        "['a', ['b', [1, 2, 3.4]]]",
    ]
    kwargs = parser.parse_args(args).get_kwargs()

    assert kwargs["a"] == {1: 2, 3: 4}
    assert kwargs["b"] == {'1': 2, '3': 4}
    assert kwargs["c"] == ['a', ['b', [1, 2, 3.4]]]
    assert kwargs["d"] == None
