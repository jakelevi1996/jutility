import pytest
from jutility import util, cli
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
            tag="op",
            init_requires=["params"],
        ),
    )
    return parser

def test_print_help():
    printer = util.Printer("test_print_help", OUTPUT_DIR)

    parser = get_parser()
    parser.print_help(printer.get_file())

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

    args = parser.parse_args([])
    cli.init_object(args, "Adam", params=[1, 2, 3])
    s6 = cli.get_args_summary(args)
    assert s6 == s

    printer(s, s2, s3, s4, s5, sep="\n")

def test_init_object():
    printer = util.Printer("test_init_object", OUTPUT_DIR)

    parser = get_parser()
    args = parser.parse_args([])

    with pytest.raises(ValueError):
        optimiser = parser.init_object("not.an.arg")
    with pytest.raises(ValueError):
        optimiser = parser.init_object("Adam")

    optimiser = parser.init_object("Adam", params=[1, 2, 3])

    printer(optimiser, vars(optimiser))
    assert isinstance(optimiser, Adam)
    assert optimiser.lr == 1e-3
    assert optimiser.inner_params == [1, 2, 3]
    assert cli.init_object(args, "list") == args.list
    assert cli.init_object(args, "Adam.lr") == optimiser.lr

    parser.parse_args(["--Adam.lr=3e-3"])
    optimiser: Adam = parser.init_object("Adam", params=[1, 2, 3])
    assert optimiser.lr == 3e-3

    parser.parse_args([])
    optimiser: Adam = parser.init_object("Adam", params=[20, 30])
    assert optimiser.inner_params == [20, 30]

def get_nested_object_parser():
    parser = cli.ObjectParser(
        cli.Arg("num_epochs", "ne", type=int, default=10),
        cli.ObjectArg(
            Mlp,
            cli.Arg("hidden_dim",           "hd", type=int, default=100),
            cli.Arg("num_hidden_layers",    "nh", type=int, default=5),
            cli.ObjectArg(
                DeepSet,
                cli.Arg("hidden_dim",           "hd", type=int, default=20),
                cli.Arg("num_hidden_layers",    "nh", type=int, default=2),
                name="encoder",
                tag="en",
                init_requires=["input_dim", "output_dim"],
                init_parsed_kwargs={"output_dim": "model.hidden_dim"},
                init_const_kwargs={"input_dim": 4},
            ),
            name="model",
            tag="ml",
            init_requires=["output_dim"],
        ),
    )
    return parser

def test_init_nested_objects():
    printer = util.Printer("test_init_nested_objects", OUTPUT_DIR)

    parser = get_nested_object_parser()
    args = parser.parse_args([])
    arg_dict: dict = vars(args)
    printer(*arg_dict.items(), cli.get_args_summary(args), "-"*100, sep="\n")

    assert args.num_epochs == 10
    assert arg_dict["model.hidden_dim"] == 100
    assert arg_dict["model.encoder.num_hidden_layers"] == 2
    assert "model"          not in arg_dict
    assert "model.encoder"  not in arg_dict

    encoder = cli.init_object(args, "model.encoder")
    printer(*arg_dict.items(), cli.get_args_summary(args), "-"*100, sep="\n")
    assert "model.encoder" in arg_dict
    assert "model" not in arg_dict
    assert isinstance(encoder, DeepSet)
    assert encoder.num_hidden_layers == 2
    assert encoder.input_dim == 4
    assert encoder.output_dim == arg_dict["model.hidden_dim"]

    with pytest.raises(ValueError):
        model = cli.init_object(args, "model")

    model = cli.init_object(args, "model", output_dim=23)
    printer(*arg_dict.items(), cli.get_args_summary(args), "-"*100, sep="\n")
    assert "model"          in arg_dict
    assert "model.encoder"  in arg_dict
    assert isinstance(model, Mlp)
    assert isinstance(model.encoder, DeepSet)
    assert model.encoder is encoder
    assert model.output_dim == 23
    assert model.num_hidden_layers == 5
    assert model.encoder.hidden_dim == 20

    argv = ["--model.num_hidden_layers=10", "--model.encoder.hidden_dim=25"]
    args = parser.parse_args(argv)
    arg_dict: dict = vars(args)
    printer(*arg_dict.items(), cli.get_args_summary(args), "-"*100, sep="\n")
    assert "model"          not in arg_dict
    assert "model.encoder"  not in arg_dict
    model = cli.init_object(args, "model", output_dim=29)
    assert "model"          in arg_dict
    assert "model.encoder"  in arg_dict
    assert isinstance(model, Mlp)
    assert isinstance(model.encoder, DeepSet)
    assert model.output_dim == 29
    assert model.num_hidden_layers == 10
    assert model.encoder.hidden_dim == 25

    util.save_json(arg_dict, "test_init_nested_objects", OUTPUT_DIR)

def test_init_object_override_kwargs():
    printer = util.Printer("test_init_object_override_kwargs", OUTPUT_DIR)

    parser = get_nested_object_parser()
    args = parser.parse_args([])

    encoder = cli.init_object(args, "model.encoder")
    assert isinstance(encoder, DeepSet)
    assert encoder.hidden_dim == 20
    assert encoder.num_hidden_layers == 2
    assert encoder.input_dim == 4
    assert encoder.output_dim == 100
    printer(*vars(args).items(), "-"*100, sep="\n")

    overrides = {
        "hidden_dim": 25,
        "num_hidden_layers": 3,
        "input_dim": 5,
        "output_dim": 90,
    }
    new_args = parser.parse_args([])
    encoder = cli.init_object(new_args, "model.encoder", **overrides)
    assert isinstance(encoder, DeepSet)
    assert encoder.hidden_dim == 25
    assert encoder.num_hidden_layers == 3
    assert encoder.input_dim == 5
    assert encoder.output_dim == 90
    printer(*vars(new_args).items(), "-"*100, sep="\n")

def test_init_parsed_kwargs():
    printer = util.Printer("test_init_parsed_kwargs", OUTPUT_DIR)

    parser = get_nested_object_parser()

    args = parser.parse_args([])
    encoder = cli.init_object(args, "model.encoder")
    assert isinstance(encoder, DeepSet)
    assert encoder.output_dim == 100
    printer(*vars(args).items(), "-"*100, sep="\n")

    args = parser.parse_args(["--model.hidden_dim=50"])
    encoder = cli.init_object(args, "model.encoder")
    assert isinstance(encoder, DeepSet)
    assert encoder.output_dim == 50
    printer(*vars(args).items(), "-"*100, sep="\n")

def test_get_set_arg_dict():
    printer = util.Printer("test_get_set_arg_dict", OUTPUT_DIR)

    parser = get_nested_object_parser()
    args = parser.parse_args(["--model.hidden_dim=99", "--num_epochs=9"])
    arg_dict_pre_init  = cli.get_arg_dict(args)
    model: Mlp         = cli.init_object( args, "model", output_dim=19)
    arg_dict_post_init = cli.get_arg_dict(args)
    printer(model)
    assert model.hidden_dim == 99
    assert model.output_dim == 19
    assert arg_dict_pre_init == arg_dict_post_init
    assert cli.get_arg_dict(args) == parser.get_arg_dict()
    arg_dict = cli.get_arg_dict(args)
    full_path = util.save_json(arg_dict, "test_get_set_arg_dict", OUTPUT_DIR)

    new_parser = get_nested_object_parser()
    new_args = new_parser.parse_args([])
    cli.set_arg_dict(new_args, util.load_json(full_path))
    new_model: Mlp = new_parser.init_object("model", output_dim=19)
    printer(new_model)
    assert new_model.hidden_dim == 99
    assert new_model.output_dim == 19
    assert parser.get_arg_dict() == new_parser.get_arg_dict()

    default_args = get_nested_object_parser().parse_args([])
    default_model: Mlp = cli.init_object(default_args, "model", output_dim=19)
    printer(default_model)
    printer(cli.get_arg_dict(args))
    printer(cli.get_arg_dict(default_args))
    assert default_model.hidden_dim == 100
    assert cli.get_arg_dict(args) != cli.get_arg_dict(default_args)

def get_object_choice_parser():
    parser = cli.ObjectParser(
        cli.Arg("seed",         "se", type=int, default=0),
        cli.Arg("num_epochs",   "ne", type=int, default=10),
        cli.ObjectChoice(
            "optimiser",
            cli.ObjectArg(
                Adam,
                cli.Arg(
                    "beta",
                    "be",
                    default=[0.9, 0.999],
                    type=float,
                    nargs=2,
                ),
                cli.Arg("weight_decay", "wd", default=0.1,  type=float),
                cli.Arg("lr",           "lr", default=1e-3, type=float),
                name="adam",
                tag="ad",
            ),
            cli.ObjectArg(
                Sgd,
                tag="sg",
            ),
            shared_args=[
                cli.Arg("lr",           "lr", default=1e-2, type=float),
                cli.Arg("weight_decay", "wd", default=0,    type=float),
            ],
            default="adam",
            tag="op",
            init_requires=["params"],
        ),
    )
    return parser

def test_object_choice():
    printer = util.Printer("test_object_choice", OUTPUT_DIR)

    parser = get_object_choice_parser()
    parser.print_help(printer.get_file())
    printer.hline()

    args = parser.parse_args([])
    optimiser = cli.init_object(args, "optimiser", params=[1, 2, 3])
    printer(cli.get_args_summary(args))
    util.save_json(
        cli.get_arg_dict(args),
        "test_object_choice_adam",
        OUTPUT_DIR,
    )
    assert isinstance(optimiser, Adam)
    assert optimiser.lr == 1e-3
    assert optimiser.beta == [0.9, 0.999]
    assert (
        cli.get_args_summary(args)
        == "ne10op.ad.be0.9,0.999op.ad.lr0.001op.ad.wd0.1opadamse0"
    )

    args = parser.parse_args(["--optimiser=adam"])
    optimiser = cli.init_object(args, "optimiser", params=[1, 2, 3])
    assert isinstance(optimiser, Adam)

    args = parser.parse_args(["--optimiser.adam.lr=1e-2"])
    optimiser = cli.init_object(args, "optimiser", params=[1, 2, 3])
    assert isinstance(optimiser, Adam)
    assert optimiser.lr == 1e-2

    args = parser.parse_args(["--optimiser.lr=1e-2"])
    optimiser = cli.init_object(args, "optimiser", params=[1, 2, 3])
    assert isinstance(optimiser, Adam)
    assert optimiser.lr == 1e-3

    args = parser.parse_args(["--optimiser.adam.beta", "0.7", "0.8"])
    optimiser = cli.init_object(args, "optimiser", params=[1, 2, 3])
    assert isinstance(optimiser, Adam)
    assert optimiser.beta == [0.7, 0.8]

    with pytest.raises(SystemExit):
        parser.parse_args(["--optimiser.beta", "0.7", "0.8"])

    args = parser.parse_args(["--optimiser=Sgd"])
    optimiser = cli.init_object(args, "optimiser", params=[1, 2, 3])
    printer(cli.get_args_summary(args))
    util.save_json(
        cli.get_arg_dict(args),
        "test_object_choice_sgd",
        OUTPUT_DIR,
    )
    assert isinstance(optimiser, Sgd)
    assert optimiser.lr == 1e-2
    assert (
        cli.get_args_summary(args)
        == "ne10op.lr0.01op.wd0opSgdse0"
    )

    args = parser.parse_args(["--optimiser=Sgd", "--optimiser.lr=1e-3"])
    optimiser = cli.init_object(args, "optimiser", params=[1, 2, 3])
    assert isinstance(optimiser, Sgd)
    assert optimiser.lr == 1e-3

    args = parser.parse_args(["--optimiser=Sgd", "--optimiser.adam.lr=1e-2"])
    optimiser = cli.init_object(args, "optimiser", params=[1, 2, 3])
    assert isinstance(optimiser, Sgd)
    assert optimiser.lr == 1e-2

    with pytest.raises(SystemExit):
        parser.parse_args(["--optimiser=Adam"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--optimiser=not_an_optimiser"])

def test_object_choice_init():
    class A:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class B:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    parser = cli.ObjectParser(
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
    args = parser.parse_args(["--top=A"])
    a = cli.init_object(args, "top", y=50)
    assert isinstance(a, A)
    assert a.kwargs == {"x": 20, "y": 50, "z": 10}

    args = parser.parse_args(["--top=A"])
    with pytest.raises(ValueError):
        a = cli.init_object(args, "top")

    args = parser.parse_args(["--top=A", "--top.A.z=70"])
    a = cli.init_object(args, "top", y=50, x=60)
    assert isinstance(a, A)
    assert a.kwargs == {"x": 60, "y": 50, "z": 70}

    args = parser.parse_args(["--top=B"])
    b = cli.init_object(args, "top")
    assert isinstance(b, B)
    assert b.kwargs == {"x": 40, "y": 30}

def get_nested_object_choice_parser():
    parser = cli.ObjectParser(
        cli.Arg("seed",         "se", type=int, default=0),
        cli.Arg("num_epochs",   "ne", type=int, default=10),
        cli.ObjectChoice(
            "model",
            cli.ObjectArg(Mlp,      tag="mp"),
            cli.ObjectArg(
                DeepSet,
                tag="ds",
                init_const_kwargs={"output_dim": 27},
            ),
            shared_args=[
                cli.ObjectChoice(
                    "encoder",
                    cli.ObjectArg(
                        Mlp,
                        init_const_kwargs={"output_dim": 15},
                        tag="mp",
                    ),
                    cli.ObjectArg(
                        DeepSet,
                        tag="ds",
                        init_requires=["hidden_dim"],
                    ),
                    shared_args=[
                        cli.Arg("hidden_dim", "hd", type=int, default=100),
                        cli.Arg(
                            "num_hidden_layers",
                            "nl",
                            type=int,
                            default=3,
                        ),
                    ],
                    default="DeepSet",
                    tag="en",
                    init_const_kwargs={"output_dim": 10},
                ),
                cli.Arg("hidden_dim",           "hd", type=int, default=100),
                cli.Arg("num_hidden_layers",    "nl", type=int, default=3),
            ],
            init_requires=["output_dim"],
            tag="mo",
        ),
    )
    return parser

def test_nested_object_choice_parser():
    printer = util.Printer("test_nested_object_choice_parser", OUTPUT_DIR)

    parser = get_nested_object_choice_parser()
    parser.print_help(printer.get_file())
    printer.hline()

    with pytest.raises(SystemExit):
        args = parser.parse_args([])

    args = parser.parse_args(["--model=Mlp"])
    util.save_json(
        cli.get_arg_dict(args),
        "test_nested_object_choice_parser",
        OUTPUT_DIR,
    )
    printer(cli.get_args_summary(args))
    assert (
        cli.get_args_summary(args)
        == "mo.en.hd100mo.en.nl3mo.enDeepSetmo.hd100mo.nl3moMlpne10se0"
    )
    with pytest.raises(ValueError):
        model = cli.init_object(args, "model")

    model = cli.init_object(args, "model", output_dim=23)
    assert isinstance(model, Mlp)
    assert isinstance(model.encoder, DeepSet)
    assert model.output_dim == 23
    assert model.encoder.output_dim == 10

    args = parser.parse_args(["--model=DeepSet"])
    printer(cli.get_args_summary(args))
    model = cli.init_object(args, "model")
    assert isinstance(model, DeepSet)
    assert isinstance(model.encoder, DeepSet)
    assert model.output_dim == 27

    args = parser.parse_args(["--model=DeepSet", "--model.encoder=Mlp"])
    printer(cli.get_args_summary(args))
    model = cli.init_object(args, "model")
    assert isinstance(model, DeepSet)
    assert isinstance(model.encoder, Mlp)
    assert model.output_dim == 27
    assert model.encoder.output_dim == 15

    args = parser.parse_args(["--model=DeepSet", "--model.encoder=Mlp"])
    cli.init_object(args, "model.encoder", output_dim=16)
    model = cli.init_object(args, "model")
    assert isinstance(model, DeepSet)
    assert isinstance(model.encoder, Mlp)
    assert model.output_dim == 27
    assert model.encoder.output_dim == 16
    assert model.encoder.hidden_dim == 100

def test_duplicate_names():
    with pytest.raises(ValueError):
        cli.ObjectParser(
            cli.Arg("arg_name", type=int),
            cli.Arg("arg_name", type=float),
        )
    with pytest.raises(ValueError):
        cli.ObjectParser(
            cli.ObjectArg(
                Mlp,
                cli.Arg("arg_name"),
                cli.ObjectArg(
                    Sgd,
                    cli.Arg("lr"),
                    name="arg_name",
                ),
                name="model",
            ),
        )
    with pytest.raises(ValueError):
        cli.ObjectParser(
            cli.ObjectChoice(
                "model",
                cli.ObjectArg(Mlp),
                cli.ObjectArg(Mlp),
            ),
        )
    with pytest.raises(ValueError):
        cli.ObjectParser(
            cli.ObjectChoice(
                "model",
                cli.ObjectArg(Mlp, name="arg_name"),
                shared_args=[
                    cli.Arg(name="arg_name"),
                ],
            ),
        )

def test_cli_verbose():
    printer = util.Printer("test_cli_verbose", OUTPUT_DIR)

    class A:
        def __init__(self, a, b):
            self.data = [a, b]

    parser = cli.ObjectParser(
        cli.ObjectArg(
            A,
            cli.Arg("a", default=3, type=int),
            init_const_kwargs={"b": 4},
        ),
        cli.ObjectArg(Adam),
    )
    args = parser.parse_args([])
    cli.init_object(args, "A")

    args = parser.parse_args([])
    with cli.verbose:
        cli.init_object(args, "A")

    args = parser.parse_args([])
    with cli.verbose(printer):
        cli.init_object(args, "A")

    printer.flush()
    assert util.load_text(printer.get_filename()) == "cli: A(a=3, b=4)\n"

    args = parser.parse_args(["--A.a=34"])
    with cli.verbose(printer):
        cli.init_object(args, "A")

    printer.flush()
    assert "cli: A(a=34, b=4)" in util.load_text(printer.get_filename())

    args = parser.parse_args([])
    with cli.verbose(printer):
        cli.init_object(args, "A", b="abc")
        cli.init_object(args, "Adam", params=[89])

    printer.flush()
    assert "cli: A(a=3, b='abc')"   in util.load_text(printer.get_filename())
    assert "cli: Adam(params=[89])" in util.load_text(printer.get_filename())

class _Optimiser:
    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, vars(self))

class Sgd(_Optimiser):
    def __init__(self, params, lr=1e-3, weight_decay=None):
        self.inner_params   = params
        self.lr             = lr
        self.weight_decay   = weight_decay

class Adam(_Optimiser):
    def __init__(self, params, lr=1e-3, beta=None, weight_decay=None):
        self.inner_params = params
        self.lr             = lr
        self.beta           = beta
        self.weight_decay   = weight_decay

class _Model:
    def __init__(
        self,
        output_dim,
        hidden_dim,
        num_hidden_layers,
        input_dim=None,
        encoder=None,
    ):
        self.input_dim          = input_dim
        self.output_dim         = output_dim
        self.hidden_dim         = hidden_dim
        self.num_hidden_layers  = num_hidden_layers
        self.encoder            = encoder

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, vars(self))

class Mlp(_Model):
    pass

class Transformer(_Model):
    pass

class Cnn(_Model):
    pass

class DeepSet(_Model):
    pass
