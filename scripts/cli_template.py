from jutility import plotting, util, cli

class ExampleClass:
    def __init__(self, p: float):
        self.p = p

def main(
    args: cli.ParsedArgs,
    x: float,
    y: str,
    z: bool,
):
    assert isinstance(x, float)
    assert isinstance(y, str)
    assert isinstance(z, bool)

    with cli.verbose:
        cn = cli.init_object(args, "ExampleClass")
        assert isinstance(cn, ExampleClass)

    print(x, y, z, cn.p, sep=", ")

    ...

if __name__ == "__main__":
    parser = cli.ObjectParser(
        cli.Arg("x", type=float, default=3.45),
        cli.Arg("y", type=str,   default="\"default\""),
        cli.Arg("z", action="store_true"),
        cli.ObjectArg(
            ExampleClass,
            cli.Arg("p", type=float, default=6.78),
        ),
    )
    args = parser.parse_args()

    with util.Timer("main"):
        main(args, **args.get_kwargs())
