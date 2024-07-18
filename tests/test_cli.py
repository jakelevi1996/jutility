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
            abbreviation="op",
            init_requires=["params"],
        ),
    )
    return parser

def test_print_help():
    printer = util.Printer("test_print_help", OUTPUT_DIR)

    parser = get_parser()
    parser.print_help(printer._file)

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

    printer(s, s2, s3, s4, s5, sep="\n")

def test_init_object():
    printer = util.Printer("test_init_object", OUTPUT_DIR)

    parser = get_parser()
    parser.parse_args([])

    with pytest.raises(ValueError):
        optimiser = parser.init_object("Adam")

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
                abbreviation="en",
                init_requires=["input_dim", "output_dim"],
                init_parsed_kwargs={"output_dim": "model.hidden_dim"},
                init_const_kwargs={"input_dim": 4},
            ),
            name="model",
            abbreviation="ml",
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

    overrides = {"hidden_dim": 25, "num_hidden_layers": 3}
    encoder = cli.init_object(args, "model.encoder", **overrides)
    assert isinstance(encoder, DeepSet)
    assert encoder.hidden_dim == 25
    assert encoder.num_hidden_layers == 3

class Adam:
    def __init__(self, params, lr=1e-3, beta=None):
        self.inner_params   = params
        self.lr             = lr
        self.beta           = beta

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

class Mlp(_Model):
    pass

class Transformer(_Model):
    pass

class Cnn(_Model):
    pass

class DeepSet(_Model):
    pass
