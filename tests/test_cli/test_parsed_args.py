import numpy as np
from jutility import util, cli

OUTPUT_DIR = util.get_test_output_dir("test_cli/test_parsed_args")

def test_save_json():
    class A:
        def __init__(self, x: float, y: int):
            pass

    parser = cli.Parser(
        cli.Arg("abc",  type=str, default="asldk"),
        cli.Arg("de",   type=int, default="345"),
        cli.BooleanArg("fghi"),
        cli.JsonArg("jk"),
        cli.ObjectArg(
            A,
            cli.Arg("x", type=float,    default=6.7),
            cli.Arg("y", type=int,      default=-89),
        )
    )
    args = parser.parse_args([])
    output_path = args.save_json(OUTPUT_DIR)
    assert util.load_json(output_path) == {
        "abc":  "asldk",
        "de":   345,
        "fghi": None,
        "jk":   None,
        "A.x":  6.7,
        "A.y":  -89,
    }

    args = parser.parse_args(
        "--abc def --no-fghi --jk {\"a\":1,\"b\":[2,3]} --A.y 3".split(),
    )
    output_path = args.save_json(OUTPUT_DIR, "test_save_json")
    assert util.load_json(output_path) == {
        "abc":  "def",
        "de":   345,
        "fghi": False,
        "jk":   {"a": 1, "b": [2, 3]},
        "A.x":  6.7,
        "A.y":  3,
    }

