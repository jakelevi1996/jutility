import pytest
from jutility import util, cli
import test_utils

OUTPUT_DIR = test_utils.get_output_dir("test_cli")

def get_parser():
    parser = cli.ObjectParser(
        cli.Arg("int",   "in", default=10,            type=int),
        cli.Arg("float",       default=-4.7,          type=float),
        cli.Arg("list",        default=[30, 20, 10],  type=int, nargs="*"),
        cli.Arg("none",        default=None),
        cli.Arg("true",        default=True),
        cli.Arg("false",       default=False),
        cli.Arg("no_abbrev",   default="random", tagged=False),
        cli.ObjectArg(
            Adam,
            cli.Arg("lr",   type=float, default=1e-3),
            cli.Arg("beta", type=float, default=[0.9, 0.999], nargs=2),
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

    assert parser.parse_args([]).get("int") == 10
    assert parser.parse_args([]).get("list") == [30, 20, 10]
    assert parser.parse_args(["--int=-3"]).get("int") == -3
    assert (
        parser.parse_args(["--list", "3", "4", "5"]).get("list")
        == [3, 4, 5]
    )

def test_get_args_summary():
    printer = util.Printer("test_get_args_summary", OUTPUT_DIR)

    parser = get_parser()
    args = parser.parse_args([])
    s = cli.get_args_summary(args)

    assert s == "faFfl-4.7in10l30,20,10nNop.b0.9,0.999op.l0.001tT"

    s2 = cli.get_args_summary(parser.parse_args(["--no_abbrev=123"]))
    assert s2 == s

    s3 = cli.get_args_summary(parser.parse_args(["--no_abbrev=4567"]))
    assert (s3 == s2) and (s3 == s)

    s4 = cli.get_args_summary(parser.parse_args(["--int=20"]))
    assert s4 != s
    assert s4 == s.replace("in10", "in20")

    s5 = cli.get_args_summary(parser.parse_args(["--Adam.lr=3e-3"]))
    assert s5 != s
    assert s5 == s.replace("op.l0.001", "op.l0.003")

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
    assert cli.init_object(args, "list") == args.get("list")
    assert cli.init_object(args, "Adam.lr") == optimiser.lr

    parser.parse_args(["--Adam.lr=3e-3"])
    optimiser = parser.init_object("Adam", params=[1, 2, 3])
    assert isinstance(optimiser, Adam)
    assert optimiser.lr == 3e-3

    parser.parse_args([])
    optimiser = parser.init_object("Adam", params=[20, 30])
    assert isinstance(optimiser, Adam)
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
    printer(args, cli.get_args_summary(args), "-"*100, sep="\n")
    assert repr(args) == (
        "ParsedArgs(model.encoder.hidden_dim=20, "
        "model.encoder.num_hidden_layers=2, model.hidden_dim=100, "
        "model.num_hidden_layers=5, num_epochs=10)"
    )
    assert cli.get_args_summary(args) == (
        "ml.en.hd20ml.en.nh2ml.hd100ml.nh5ne10"
    )
    assert args.get("num_epochs") == 10
    assert args.get("model.hidden_dim") == 100
    assert args.get("model.encoder.num_hidden_layers") == 2
    with pytest.raises(KeyError):
        args.get("model")
    with pytest.raises(KeyError):
        args.get("model.encoder")

    encoder = cli.init_object(args, "model.encoder")
    printer(args, cli.get_args_summary(args), "-"*100, sep="\n")
    assert repr(args) == (
        "ParsedArgs(model.encoder=DeepSet(encoder=None, hidden_dim=20, "
        "input_dim=4, num_hidden_layers=2, output_dim=100), "
        "model.encoder.hidden_dim=20, model.encoder.num_hidden_layers=2, "
        "model.hidden_dim=100, model.num_hidden_layers=5, num_epochs=10)"
    )
    assert cli.get_args_summary(args) == (
        "ml.en.hd20ml.en.nh2ml.hd100ml.nh5ne10"
    )
    assert isinstance(encoder, DeepSet)
    assert args.get("model.encoder") is encoder
    assert encoder.num_hidden_layers == 2
    assert encoder.input_dim == 4
    assert encoder.output_dim == args.get("model.hidden_dim")
    with pytest.raises(KeyError):
        args.get("model")

    with pytest.raises(ValueError):
        model = cli.init_object(args, "model")

    model = cli.init_object(args, "model", output_dim=23)
    printer(args, cli.get_args_summary(args), "-"*100, sep="\n")
    assert repr(args) == (
        "ParsedArgs(model=Mlp(encoder=DeepSet(encoder=None, hidden_dim=20, "
        "input_dim=4, num_hidden_layers=2, output_dim=100), hidden_dim=100, "
        "input_dim=None, num_hidden_layers=5, output_dim=23), "
        "model.encoder=DeepSet(encoder=None, hidden_dim=20, input_dim=4, "
        "num_hidden_layers=2, output_dim=100), model.encoder.hidden_dim=20, "
        "model.encoder.num_hidden_layers=2, model.hidden_dim=100, "
        "model.num_hidden_layers=5, num_epochs=10)"
    )
    assert cli.get_args_summary(args) == (
        "ml.en.hd20ml.en.nh2ml.hd100ml.nh5ne10"
    )
    assert isinstance(model, Mlp)
    assert isinstance(model.encoder, DeepSet)
    assert args.get("model") is model
    assert args.get("model.encoder") is encoder
    assert model.encoder is encoder
    assert model.output_dim == 23
    assert model.num_hidden_layers == 5
    assert model.encoder.hidden_dim == 20

    argv = ["--model.num_hidden_layers=10", "--model.encoder.hidden_dim=25"]
    args = parser.parse_args(argv)
    printer(args, cli.get_args_summary(args), "-"*100, sep="\n")
    assert repr(args) == (
        "ParsedArgs(model.encoder.hidden_dim=25, "
        "model.encoder.num_hidden_layers=2, model.hidden_dim=100, "
        "model.num_hidden_layers=10, num_epochs=10)"
    )
    assert cli.get_args_summary(args) == (
        "ml.en.hd25ml.en.nh2ml.hd100ml.nh10ne10"
    )
    with pytest.raises(KeyError):
        args.get("model")
    with pytest.raises(KeyError):
        args.get("model.encoder")

    model = cli.init_object(args, "model", output_dim=29)
    assert isinstance(model, Mlp)
    assert isinstance(model.encoder, DeepSet)
    assert args.get("model") is model
    assert args.get("model.encoder") is model.encoder
    assert model.output_dim == 29
    assert model.num_hidden_layers == 10
    assert model.encoder.hidden_dim == 25

    arg_dict = cli.get_arg_dict(args)
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

def test_get_update_args():
    printer = util.Printer("test_get_update_args", OUTPUT_DIR)

    parser = get_nested_object_parser()
    args = parser.parse_args(["--model.hidden_dim=99", "--num_epochs=9"])
    arg_dict_pre_init  = cli.get_arg_dict(args)
    model = cli.init_object(args, "model", output_dim=19)
    arg_dict_post_init = cli.get_arg_dict(args)
    printer(model)
    assert isinstance(model, Mlp)
    assert model.hidden_dim == 99
    assert model.output_dim == 19
    assert arg_dict_pre_init == arg_dict_post_init
    assert cli.get_arg_dict(args) == parser.get_arg_dict()
    arg_dict = cli.get_arg_dict(args)
    full_path = util.save_json(arg_dict, "test_get_update_args", OUTPUT_DIR)

    new_parser = get_nested_object_parser()
    new_args = new_parser.parse_args([])
    new_args.update(util.load_json(full_path))
    new_model = new_parser.init_object("model", output_dim=19)
    printer(new_model)
    assert isinstance(new_model, Mlp)
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

    assert isinstance(default_args, cli.ParsedArgs)
    assert default_args.get("model.hidden_dim") == 100
    default_args.update({"model.hidden_dim": 234})
    assert default_args.get("model.hidden_dim") == 234

    assert default_args.get_kwargs("model.hidden_dim, num_epochs") == {
        "model.hidden_dim": 234,
        "num_epochs": 10,
    }
    assert default_args.get_kwargs() == {
        "num_epochs": 10,
    }

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
            ),
            cli.ObjectArg(
                Sgd,
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
        == "ne10opADAMop.be0.9,0.999op.lr0.001op.wd0.1se0"
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
        == "ne10opSGDop.lr0.01op.wd0se0"
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
            cli.ObjectArg(Mlp),
            cli.ObjectArg(
                DeepSet,
                init_const_kwargs={"output_dim": 27},
            ),
            shared_args=[
                cli.ObjectChoice(
                    "encoder",
                    cli.ObjectArg(
                        Mlp,
                        init_const_kwargs={"output_dim": 15},
                    ),
                    cli.ObjectArg(
                        DeepSet,
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
        == "moMLPmo.enDEEPSETmo.en.hd100mo.en.nl3mo.hd100mo.nl3ne10se0"
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

def test_reset_object_cache():
    printer = util.Printer("test_init_object_cache", OUTPUT_DIR)

    class C:
        def __init__(self, x):
            self.x = x

        def __repr__(self):
            return "C(x=%s)" % self.x

    parser = cli.ObjectParser(
        cli.ObjectArg(
            C,
            cli.Arg("x", default=3),
        ),
        cli.Arg("y", default=4),
    )
    cli.verbose.set_printer(printer)
    with cli.verbose:
        args = parser.parse_args([])
        assert args.get("y") == 4
        assert args.get("C.x") == 3
        with pytest.raises(KeyError):
            args.get("C")

        c = cli.init_object(args, "C")
        assert isinstance(c, C)
        assert args.get("C") is c
        assert c.x == 3

        c2 = cli.init_object(args, "C")
        assert c2 is c

        cli.reset_object_cache(args)
        assert args.get("y") == 4
        assert args.get("C.x") == 3
        with pytest.raises(KeyError):
            args.get("C")

        c3 = cli.init_object(args, "C")
        assert isinstance(c3, C)
        assert args.get("C") is c3
        assert c2 is c
        assert c3 is not c
        assert c3 is not c2

def test_reset_object_cache_nested():
    printer = util.Printer("test_reset_object_cache_nested", OUTPUT_DIR)

    parser = get_nested_object_choice_parser()
    parser.print_help(printer.get_file())
    printer.hline()

    cli.verbose.set_printer(printer)
    with cli.verbose:
        argv = [
            "--model=DeepSet",
            "--model.encoder=Mlp",
            "--seed=1234",
        ]
        args = parser.parse_args(argv)
        assert args.get("seed") == 1234
        with pytest.raises(KeyError):
            args.get("model.DeepSet")
        with pytest.raises(KeyError):
            args.get("model.encoder.Mlp")

        model = cli.init_object(args, "model")
        assert isinstance(model, DeepSet)
        assert isinstance(model.encoder, Mlp)
        assert args.get("model") == "DeepSet"
        assert args.get("model.encoder") == "Mlp"
        assert args.get("model.DeepSet") is model
        assert args.get("model.encoder.Mlp") is model.encoder

        model2 = cli.init_object(args, "model")
        assert isinstance(model2, DeepSet)
        assert model is model2
        assert model.encoder is model2.encoder

        cli.reset_object_cache(args)
        assert args.get("seed") == 1234
        with pytest.raises(KeyError):
            args.get("model.DeepSet")
        with pytest.raises(KeyError):
            args.get("model.encoder.Mlp")

        model3 = cli.init_object(args, "model")
        assert isinstance(model3, DeepSet)
        assert isinstance(model3.encoder, Mlp)
        assert args.get("model") == "DeepSet"
        assert args.get("model.encoder") == "Mlp"
        assert args.get("model.DeepSet") is model3
        assert args.get("model.encoder.Mlp") is model3.encoder

        assert model is model2
        assert model3 is not model
        assert model3 is not model2
        assert model.encoder is model2.encoder
        assert model3.encoder is not model.encoder
        assert model3.encoder is not model2.encoder

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

def test_arg_registered_twice():
    arg = cli.Arg("learning_rate", "lr", type=float, default=1e-3)
    with pytest.raises(RuntimeError):
        parser = cli.ObjectParser(
            cli.ObjectArg(
                Adam,
                arg,
                arg,
            ),
        )

    arg = cli.Arg("learning_rate", "lr", type=float, default=1e-3)
    with pytest.raises(RuntimeError):
        parser = cli.ObjectParser(
            arg,
            cli.ObjectArg(
                Adam,
                arg,
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
    cli.verbose.set_printer(printer)
    with cli.verbose:
        cli.init_object(args, "A")

    printer.flush()
    assert util.load_text(printer.get_filename()) == "cli: A(a=3, b=4)\n"

    args = parser.parse_args(["--A.a=34"])
    with cli.verbose:
        cli.init_object(args, "A")

    printer.flush()
    assert "cli: A(a=34, b=4)" in util.load_text(printer.get_filename())

    args = parser.parse_args([])
    with cli.verbose:
        cli.init_object(args, "A", b="abc")
        cli.init_object(args, "Adam", params=[89])

    printer.flush()
    assert "cli: A(a=3, b='abc')"   in util.load_text(printer.get_filename())
    assert "cli: Adam(params=[89])" in util.load_text(printer.get_filename())

    cache_msg = "retrieving \"A\" from cache"
    assert cache_msg not in util.load_text(printer.get_filename())
    with cli.verbose:
        cli.init_object(args, "A", b="abc")

    printer.flush()
    assert cache_msg in util.load_text(printer.get_filename())

def test_object_choice_nested_tags():
    printer = util.Printer("test_object_choice_nested_tags", OUTPUT_DIR)
    cli.verbose.set_printer(printer)

    class C:
        def __init__(self, a, b, s):
            self.a = a
            self.b = b
            self.s = s

    class D:
        def __init__(self, e, f, s):
            self.e = e
            self.f = f
            self.s = s

    class F:
        def __init__(self, g):
            self.g = g

        def __repr__(self):
            return "F(g=%r)" % self.g

    parser = cli.ObjectParser(
        cli.ObjectChoice(
            "model",
            cli.ObjectArg(
                C,
                cli.Arg("a", "a", default=1),
                cli.Arg("b", "b", default=2),
            ),
            cli.ObjectArg(
                D,
                cli.Arg("e", "e", default=3),
                cli.ObjectArg(
                    F,
                    cli.Arg("g", "g", default=4),
                    name="f",
                    tag="f",
                ),
                cli.Arg("s", "s", default=5),
            ),
            shared_args=[cli.Arg("s", "s", default=6)],
            tag="m",
        ),
    )

    args = parser.parse_args(["--model=C"])
    printer(cli.get_arg_dict(args))
    printer(cli.get_args_summary(args))
    with cli.verbose:
        cli.init_object(args, "model")

    assert cli.get_args_summary(args) == "mCm.a1m.b2m.s6"

    args = parser.parse_args(["--model=D"])
    printer(cli.get_arg_dict(args))
    printer(cli.get_args_summary(args))
    with cli.verbose:
        cli.init_object(args, "model")

    assert cli.get_args_summary(args) == "mDm.e3m.f.g4m.s5"

def test_parsedargs_str_repr():
    printer = util.Printer("test_parsedargs_str_repr", OUTPUT_DIR)

    parser = get_nested_object_choice_parser()
    cli.verbose.set_printer(printer)
    with cli.verbose:
        argv = [
            "--model=DeepSet",
            "--model.encoder=Mlp",
            "--seed=1234",
        ]
        args = parser.parse_args(argv)

    printer(args)
    printer(str(args))
    printer(repr(args))
    assert repr(args) == (
        "ParsedArgs(model='DeepSet', model.encoder='Mlp', "
        "model.encoder.hidden_dim=100, model.encoder.num_hidden_layers=3, "
        "model.hidden_dim=100, model.num_hidden_layers=3, num_epochs=10, "
        "seed=1234)"
    )

def test_init_ignores():
    printer = util.Printer("test_init_ignores", OUTPUT_DIR)

    class C:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

        def forward(self):
            return self.x + self.y + self.z

    class D(C):
        def forward(self):
            return self.x + 2 * self.y + 3 * self.z

    class E(C):
        def __init__(self, x):
            self.x = x

        def forward(self):
            return self.x

    parser = cli.ObjectParser(
        cli.ObjectChoice(
            "model",
            cli.ObjectArg(C),
            cli.ObjectArg(D),
            cli.ObjectArg(E, init_ignores=["y", "z"]),
            shared_args=[
                cli.Arg("x", "x", default=1),
                cli.Arg("y", "y", default=2),
            ],
            tag="m",
            init_const_kwargs={"z": 3},
        ),
    )
    args = parser.parse_args(["--model", "C"])
    cli.verbose.set_printer(printer)
    with cli.verbose:
        m = cli.init_object(args, "model")
        assert isinstance(m, C)
        assert m.forward() == 6
        assert cli.get_args_summary(args) == "mCm.x1m.y2"
        assert printer.read() == "cli: C(x=1, y=2, z=3)\n"

    args = parser.parse_args(["--model", "D"])
    with cli.verbose:
        m = cli.init_object(args, "model")
        assert isinstance(m, C)
        assert m.forward() == 14
        assert cli.get_args_summary(args) == "mDm.x1m.y2"
        assert "cli: D(x=1, y=2, z=3)" in printer.read()

    args = parser.parse_args(["--model", "E"])
    with cli.verbose:
        m = cli.init_object(args, "model")
        assert isinstance(m, C)
        assert m.forward() == 1
        assert cli.get_args_summary(args) == "mEm.x1"
        assert "cli: E(x=1)" in printer.read()

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
        arg_str = util.format_dict(vars(self), ", ", "=")
        return "%s(%s)" % (type(self).__name__, arg_str)

class Mlp(_Model):
    pass

class Transformer(_Model):
    pass

class Cnn(_Model):
    pass

class DeepSet(_Model):
    pass

def test_cache_object_choice():
    printer = util.Printer("test_cache_object_choice", OUTPUT_DIR)

    class C:
        def __init__(self, x: int):
            self.x = x

        def __repr__(self):
            return "C(x=%i)" % self.x

    class D:
        def __init__(self, c: C, y: int):
            self.c = c
            self.y = y

        def __repr__(self):
            return "C(c=%r, y=%i)" % (self.c, self.y)

    parser = cli.ObjectParser(
        cli.ObjectArg(C, cli.Arg("x", type=int)),
        cli.ObjectChoice(
            "model",
            cli.ObjectArg(C, cli.Arg("x", type=int)),
            cli.ObjectArg(
                D,
                cli.ObjectArg(C, cli.Arg("x", type=int)),
                cli.Arg("x", type=int),
            ),
        ),
    )
    args = parser.parse_args(
        ["--C.x", "2", "--model", "C", "--model.C.x", "3"],
    )
    printer(args)
    cli.verbose.set_printer(printer)

    with cli.verbose:
        model = cli.init_object(args, "model")

    assert printer.read().count("cli: C(x=3)") == 1
    assert printer.read().count("cli: C(x=2)") == 0
    assert printer.read().count("cli: retrieving \"model.C\" from cache") == 0
    assert printer.read().count("cli: retrieving \"C\" from cache") == 0
    assert isinstance(model, C)
    assert model.x == 3

    with cli.verbose:
        c = cli.init_object(args, "C")

    assert printer.read().count("cli: C(x=3)") == 1
    assert printer.read().count("cli: C(x=2)") == 1
    assert printer.read().count("cli: retrieving \"model.C\" from cache") == 0
    assert printer.read().count("cli: retrieving \"C\" from cache") == 0
    assert isinstance(c, C)
    assert c.x == 2

    with cli.verbose:
        model2 = cli.init_object(args, "model")

    assert printer.read().count("cli: C(x=3)") == 1
    assert printer.read().count("cli: C(x=2)") == 1
    assert printer.read().count("cli: retrieving \"model.C\" from cache") == 1
    assert printer.read().count("cli: retrieving \"C\" from cache") == 0
    assert model2 is model
    assert isinstance(model2, C)
    assert model2.x == 3

    with cli.verbose:
        c2 = cli.init_object(args, "C")

    assert printer.read().count("cli: C(x=3)") == 1
    assert printer.read().count("cli: C(x=2)") == 1
    assert printer.read().count("cli: retrieving \"model.C\" from cache") == 1
    assert printer.read().count("cli: retrieving \"C\" from cache") == 1
    assert c2 is c
    assert isinstance(c2, C)
    assert c2.x == 2

def test_update_args_reset_cache():
    printer = util.Printer("test_update_args_reset_cache", OUTPUT_DIR)

    class C:
        def __init__(self, x: int):
            self.x = x

        def __repr__(self):
            return "C(x=%i)" % self.x

    parser = cli.ObjectParser(
        cli.ObjectArg(C, cli.Arg("x", type=int)),
    )
    args = parser.parse_args(["--C.x", "2"])

    cli.verbose.set_printer(printer)

    with cli.verbose:
        c = cli.init_object(args, "C")

    assert printer.read().count("cli: C(x=2)") == 1
    assert printer.read().count("cli: retrieving \"C\" from cache") == 0
    assert printer.read().count("cli: C(x=3)") == 0

    assert isinstance(c, C)
    assert c.x == 2

    with cli.verbose:
        c2 = cli.init_object(args, "C")

    assert printer.read().count("cli: C(x=2)") == 1
    assert printer.read().count("cli: retrieving \"C\" from cache") == 1
    assert printer.read().count("cli: C(x=3)") == 0

    assert c2 is c
    assert isinstance(c2, C)
    assert c2.x == 2

    args.update({"C.x": 3})

    with cli.verbose:
        c3 = cli.init_object(args, "C")

    assert printer.read().count("cli: C(x=2)") == 1
    assert printer.read().count("cli: retrieving \"C\" from cache") == 1
    assert printer.read().count("cli: C(x=3)") == 1

    assert c3 is not c
    assert c3 is not c2
    assert isinstance(c3, C)
    assert c3.x == 3

    with cli.verbose:
        c4 = cli.init_object(args, "C")

    assert printer.read().count("cli: C(x=2)") == 1
    assert printer.read().count("cli: retrieving \"C\" from cache") == 2
    assert printer.read().count("cli: C(x=3)") == 1

    assert c4 is not c
    assert c4 is not c2
    assert c4 is c3
    assert isinstance(c4, C)
    assert c4.x == 3

def test_nested_verbose():
    printer = util.Printer("test_nested_verbose", OUTPUT_DIR)

    class C:
        def __init__(self, x: int):
            self.x = x

        def __repr__(self):
            return "C(x=%i)" % self.x

    parser = cli.ObjectParser(
        cli.ObjectArg(C, cli.Arg("x", type=int), name="c1"),
        cli.ObjectArg(C, cli.Arg("x", type=int), name="c2"),
        cli.ObjectArg(C, cli.Arg("x", type=int), name="c3"),
        cli.ObjectArg(C, cli.Arg("x", type=int), name="c4"),
    )
    args = parser.parse_args(
        ["--c1.x", "6", "--c2.x", "7", "--c3.x", "8", "--c4.x", "9"],
    )

    cli.verbose.set_printer(printer)

    with cli.verbose:
        c1 = cli.init_object(args, "c1")

        with cli.verbose:
            c2 = cli.init_object(args, "c2")

        c3 = cli.init_object(args, "c3")

    c4 = cli.init_object(args, "c4")

    assert isinstance(c1, C)
    assert isinstance(c2, C)
    assert isinstance(c3, C)
    assert isinstance(c4, C)

    assert c1.x == 6
    assert c2.x == 7
    assert c3.x == 8
    assert c4.x == 9

    cli_output = printer.read()
    assert cli_output.count("cli: C(x=6)") == 1
    assert cli_output.count("cli: C(x=7)") == 1
    assert cli_output.count("cli: C(x=8)") == 1
    assert cli_output.count("cli: C(x=9)") == 0

def test_parsedargs_update_allow_new_keys():
    printer = util.Printer(
        "test_parsedargs_update_allow_new_keys",
        OUTPUT_DIR,
    )

    class C:
        def __init__(self, x: int):
            self.x = x

        def __repr__(self):
            return "C(x=%i)" % self.x

    parser = cli.ObjectParser(
        cli.ObjectArg(C, cli.Arg("x", type=int), name="c1"),
        cli.ObjectArg(C, cli.Arg("x", type=int), name="c2"),
        cli.ObjectArg(C, cli.Arg("x", type=int), name="c3"),
        cli.ObjectArg(C, cli.Arg("x", type=int), name="c4"),
    )
    args = parser.parse_args(
        ["--c1.x", "6", "--c2.x", "7", "--c3.x", "8", "--c4.x", "9"],
    )

    assert repr(args) == "ParsedArgs(c1.x=6, c2.x=7, c3.x=8, c4.x=9)"

    with cli.verbose:
        c1 = cli.init_object(args, "c1")

    assert isinstance(c1, C)
    assert c1.x == 6

    args.update({"c1.x": 7})

    with cli.verbose:
        c1 = cli.init_object(args, "c1")

    assert isinstance(c1, C)
    assert c1.x == 7

    with pytest.raises(ValueError):
        args.update({"c5.x": 7})

    args.update({"c5.x": 7}, allow_new_keys=True)

    assert repr(args) == "ParsedArgs(c1.x=7, c2.x=7, c3.x=8, c4.x=9, c5.x=7)"
    assert args.get("c5.x") == 7

def test_positional_args():
    class A:
        def __init__(self, x: float):
            self.x = x

    class B:
        def __init__(self, y: float):
            self.y = y

    parser = cli.ObjectParser(
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
    assert args.get("a, b, c") == [5, 3, 4]
    assert args.get_kwargs() == {"a": 5, "b": 3, "c": 4}

    args = parser.parse_args(["5", "6"])
    assert args.get("a, b, c") == [5, 6, 4]
    assert args.get_kwargs() == {"a": 5, "b": 6, "c": 4}

    with pytest.raises(SystemExit):
        args = parser.parse_args(["5", "6", "7"])

    args = parser.parse_args(["5", "6", "--c", "7"])
    assert args.get("a, b, c") == [5, 6, 7]
    assert args.get_kwargs() == {"a": 5, "b": 6, "c": 7}

def test_auto_tag_edge_cases():
    parser = cli.ObjectParser(
        cli.Arg("ab_c", default=123),
        cli.Arg("a_bc", default=456),
    )
    args = parser.parse_args([])
    assert cli.get_args_summary(args) == "a456ab123"

    parser = cli.ObjectParser(
        cli.Arg("abc", default=123),
        cli.Arg("ABC", default=456),
    )
    args = parser.parse_args([])
    assert cli.get_args_summary(args) == "abc123abc456"

    parser = cli.ObjectParser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
    )
    args = parser.parse_args([])
    assert cli.get_args_summary(args) == "abc123abcd456"

    parser = cli.ObjectParser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
        cli.Arg("ABC",  default=789),
    )
    args = parser.parse_args([])
    assert cli.get_args_summary(args) == "abc123abc789abcd456"

    parser = cli.ObjectParser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
        cli.Arg("ABC",  default=789, tag="abc"),
    )
    args = parser.parse_args([])
    assert cli.get_args_summary(args) == "abc123abc789abcd456"

    parser = cli.ObjectParser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
        cli.Arg("ABC",  default=789, tag="abc_"),
    )
    args = parser.parse_args([])
    assert cli.get_args_summary(args) == "abc123abc789abcd456"

    parser = cli.ObjectParser(
        cli.Arg("abc",  default=123),
        cli.Arg("abcd", default=456),
        cli.Arg("ABC",  default=789, tag="a_b_c"),
    )
    args = parser.parse_args([])
    assert cli.get_args_summary(args) == "abc789abc123abcd456"
