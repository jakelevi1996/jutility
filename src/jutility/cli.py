import argparse
import json
from jutility import util

def get_arg_dict(args: "ParsedArgs"):
    return args.get_parser().get_arg_dict()

def get_args_summary(args: "ParsedArgs", replaces=None):
    return args.get_parser().get_args_summary(replaces)

class _ArgParent:
    def _init_arg_list(self, arg_list: list["Arg"]):
        self._arg_list = arg_list

    def register_names(
        self,
        arg_dict: dict[str, "Arg"],
        prefix: str,
    ) -> dict[str, "Arg"]:
        for arg in self._arg_list:
            arg.set_full_name(prefix + arg.name)
            if arg.full_name in arg_dict:
                raise ValueError(
                    "Found duplicates %s and %s"
                    % (arg, arg_dict[arg.full_name])
                )

            arg_dict[arg.full_name] = arg
            arg.register_names(arg_dict, arg.full_name + ".")

        return arg_dict

    def register_tags(
        self,
        tag_dict: dict[str, str],
        prefix: str,
    ) -> dict[str, str]:
        for arg in self._arg_list:
            if (arg.tagged and (arg.tag is not None)):
                tag_dict[arg.full_name] = self._make_tag(prefix, arg.tag)

        default_tag_dict = {
            arg.full_name: self._make_tag(prefix, arg.name)
            for arg in self._arg_list
            if (arg.tagged and (arg.tag is None))
        }
        prefix_dict = util.get_unique_prefixes(
            input_list=default_tag_dict.values(),
            forbidden=set(tag_dict.values()),
            min_len=(len(prefix) + 1),
        )
        for full_name, full_tag in default_tag_dict.items():
            tag_dict[full_name] = prefix_dict[full_tag]

        for arg in self._arg_list:
            if arg.tagged:
                arg_prefix = (
                    prefix
                    if self._hide_tag(arg)
                    else tag_dict[arg.full_name] + "."
                )
                arg.register_tags(tag_dict, arg_prefix)

        return tag_dict

    def _make_tag(self, prefix: str, suffix: str) -> str:
        return prefix + suffix.lower().replace("_", "")

    def _hide_tag(self, arg: "Arg") -> bool:
        return False

class Arg(_ArgParent):
    def __init__(
        self,
        name: str,
        tag: (str | None)=None,
        tagged=True,
        is_kwarg=True,
        **argparse_kwargs,
    ):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument
        """
        self.kwargs = argparse_kwargs
        self._is_kwarg = is_kwarg
        self._init_arg_list([])
        self._init_name(name)
        self._init_tag(tag, tagged)
        self._init_value()
        self._init_help()

    def _init_name(self, name: str | None):
        self.name = name
        self.full_name = None

    def _init_tag(self, tag: str | None, tagged: bool):
        self.tag = tag
        self.tagged = tagged

    def _init_value(self):
        self.value = None

    def _init_help(self):
        if ((len(self.kwargs) > 0) and ("help" not in self.kwargs)):
            self.kwargs["help"] = util.format_dict(self.kwargs)

    def set_full_name(self, full_name: str):
        if self.full_name is not None:
            raise RuntimeError("%s is already registered" % self)

        self.full_name = full_name

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--" + self.full_name, **self.kwargs)

    def init_object(self):
        return self.value

    def reset_object_cache(self):
        return

    def get_arg_dict_keys(self, parsed_args_dict) -> list[str]:
        return [self.full_name]

    def is_kwarg(self) -> bool:
        return self._is_kwarg

    def __repr__(self):
        return util.format_type(
            type(self),
            name=self.name,
            full_name=self.full_name,
            value=self.value,
        )

class PositionalArg(Arg):
    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(self.full_name, **self.kwargs)

class BooleanArg(Arg):
    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--" + self.full_name,
            action=argparse.BooleanOptionalAction,
            **self.kwargs,
        )

    def _init_help(self):
        if "default" in self.kwargs:
            self.kwargs.setdefault("help", "")
        else:
            if "help" in self.kwargs:
                self.kwargs["help"] += " (default: None)"
            else:
                self.kwargs["help"] = "(default: None)"

class JsonArg(Arg):
    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--" + self.full_name,
            type=json.loads,
            **self.kwargs,
        )

    def _init_help(self):
        if ((len(self.kwargs) > 0) and ("help" not in self.kwargs)):
            self.kwargs["help"] = util.format_dict(self.kwargs)
        if "help" in self.kwargs:
            self.kwargs["help"] += " (format: JSON string)"
        else:
            self.kwargs["help"] = "Format: JSON string"

class ObjectArg(Arg):
    def __init__(
        self,
        object_type: type,
        *args: Arg,
        name: (str | None)=None,
        tag:  (str | None)=None,
        tagged=True,
        init_requires: (list[str] | None)=None,
        init_ignores:  (list[str] | None)=None,
        init_const_kwargs:  (dict | None)=None,
    ):
        if name is None:
            name = object_type.__name__

        self.object_type = object_type
        self._init_arg_list(args)
        self._init_name(name)
        self._init_tag(tag, tagged)
        self._init_value()
        self._init_object_options(
            init_requires,
            init_ignores,
            init_const_kwargs,
        )

    def _init_object_options(
        self,
        init_requires: (list[str] | None),
        init_ignores:  (list[str] | None),
        init_const_kwargs:  (dict | None),
    ):
        if init_requires is None:
            init_requires = []
        if init_ignores is None:
            init_ignores = []
        if init_const_kwargs is None:
            init_const_kwargs = dict()

        self.init_requires = init_requires
        self.init_ignores  = init_ignores
        self.init_const_kwargs  = init_const_kwargs

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        for arg in self._arg_list:
            arg.add_argparse_arguments(parser)

    def get_protected_args(self) -> set[str]:
        protected_arg_list = [
            *[arg.name for arg in self._arg_list],
            *self.init_ignores,
            *self.init_const_kwargs.keys(),
        ]
        return set(protected_arg_list)

    def check_missing(self, kwarg_names: set[str]):
        missing_keys = set(self.init_requires) - kwarg_names
        if len(missing_keys) > 0:
            raise ValueError(
                "Please provide values for the following keys %s for \"%s\""
                % (sorted(missing_keys), self.full_name)
            )

    def init_object(self, **extra_kwargs):
        if self.value is not None:
            if verbose:
                verbose.display_retrieve(self.full_name)

            return self.value

        kwargs = {
            arg.name: arg.init_object()
            for arg in self._arg_list
        }
        kwargs.update(self.init_const_kwargs)
        kwargs.update(extra_kwargs)
        self.check_missing(set(kwargs.keys()))

        if verbose:
            verbose.display_init(self.object_type, kwargs)

        self.value = self.object_type(**kwargs)
        return self.value

    def reset_object_cache(self):
        self._init_value()

    def get_arg_dict_keys(self, parsed_args_dict) -> list[str]:
        return [
            name
            for arg in self._arg_list
            for name in arg.get_arg_dict_keys(parsed_args_dict)
        ]

    def is_kwarg(self) -> bool:
        return False

class ObjectChoice(ObjectArg):
    def __init__(
        self,
        name: str,
        *choices: ObjectArg,
        shared_args: (list[Arg] | None)=None,
        default: (str | None)=None,
        tag:     (str | None)=None,
        tagged=True,
        init_requires: (list[str] | None)=None,
        init_ignores:  (list[str] | None)=None,
        init_const_kwargs:  (dict | None)=None,
    ):
        if shared_args is None:
            shared_args = []

        self.shared_args = shared_args
        self.default = default
        self._init_name(name)
        self._init_tag(tag, tagged)
        self._init_value()
        self._init_object_options(
            init_requires,
            init_ignores,
            init_const_kwargs,
        )

        self._init_arg_list(tuple(choices) + tuple(shared_args))
        self.choice_dict = {arg.name: arg for arg in choices}
        if (default is not None) and (default not in self.choice_dict):
            valid_names = [arg.name for arg in choices]
            raise ValueError(
                "%s(\"%s\") received `default=\"%s\"`, please choose from %s"
                % (type(self).__name__, name, default, valid_names)
            )

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--" + self.full_name,
            choices=list(self.choice_dict.keys()),
            default=self.default,
            required=(True if (self.default is None) else False),
        )
        for arg in self._arg_list:
            arg.add_argparse_arguments(parser)

    def init_object(self, **extra_kwargs):
        chosen_arg = self.choice_dict[self.value]
        protected = chosen_arg.get_protected_args()
        kwargs = {
            arg.name: arg.init_object()
            for arg in self.shared_args
            if arg.name not in protected
        }
        for k, v in self.init_const_kwargs.items():
            if k not in protected:
                kwargs[k] = v

        kwargs.update(extra_kwargs)
        self.check_missing(set(kwargs.keys()) | protected)
        return chosen_arg.init_object(**kwargs)

    def reset_object_cache(self):
        return

    def get_arg_dict_keys(self, parsed_args_dict) -> list[str]:
        chosen_arg = self.choice_dict[parsed_args_dict[self.full_name]]
        protected = chosen_arg.get_protected_args()
        return [
            self.full_name,
            *chosen_arg.get_arg_dict_keys(parsed_args_dict),
            *[
                name
                for arg in self.shared_args
                if  arg.name not in protected
                for name in arg.get_arg_dict_keys(parsed_args_dict)
            ],
        ]

    def _hide_tag(self, arg: "Arg") -> bool:
        return (arg.name in self.choice_dict)

class UnknownArg(Arg):
    def __init__(self, value):
        self._init_arg_list([])
        self._init_name(None)
        self._init_tag(None, False)
        self.value = value

    def init_object(self, **extra_kwargs):
        raise ValueError("Can only initialise `Arg`s defined in `Parser`")

class Parser(_ArgParent):
    def __init__(
        self,
        *args: Arg,
        **parser_kwargs,
    ):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser
        """
        self._parser_kwargs = parser_kwargs
        self._init_arg_list(args)
        self._arg_dict = self.register_names(dict(), "")

    def _get_argparse_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(**self._parser_kwargs)
        for arg in self._arg_list:
            arg.add_argparse_arguments(parser)

        return parser

    def print_help(self, file=None):
        self._get_argparse_parser().print_help(file)

    def parse_args(self, *args, **kwargs) -> "ParsedArgs":
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args
        """
        parser = self._get_argparse_parser()
        argparse_namespace = parser.parse_args(*args, **kwargs)
        argparse_dict = vars(argparse_namespace)
        for arg_name in argparse_dict:
            arg_value = argparse_dict[arg_name]
            self._arg_dict[arg_name].value = arg_value

        return ParsedArgs(self._arg_list, self._arg_dict)

    def __repr__(self):
        return util.format_type(
            type(self),
            *self._arg_list,
            **self._parser_kwargs,
        )

class ParsedArgs(_ArgParent):
    def __init__(
        self,
        arg_list: list[Arg],
        arg_dict: dict[str, Arg],
    ):
        self._init_arg_list(arg_list)
        self._arg_dict = arg_dict

    def get(self, arg_name: str):
        return self._arg_dict[arg_name].value

    def get_kwargs(self) -> dict:
        return {
            arg.name: arg.value
            for arg in self._arg_list
            if arg.is_kwarg()
        }

    def update(self, arg_dict: dict, allow_new_keys=False):
        for name, value in arg_dict.items():
            if name in self._arg_dict:
                self._arg_dict[name].value = value
            elif allow_new_keys:
                self._arg_dict[name] = UnknownArg(value)
            else:
                new_keys = set(arg_dict.keys()) - set(self._arg_dict.keys())
                raise ValueError(
                    "Received extra keys %s. Either remove these keys, or "
                    "call `ParsedArgs.update(..., allow_new_keys=True)`."
                    % sorted(new_keys)
                )

        self.reset_object_cache()

    def init_object(self, full_name: str, **extra_kwargs):
        return self._arg_dict[full_name].init_object(extra_kwargs)

    def reset_object_cache(self):
        for arg in self._arg_dict.values():
            arg.reset_object_cache()

    def get_arg_dict(self):
        return {
            key: self._parsed_args_dict[key]
            for arg in self._arg_list
            for key in arg.get_arg_dict_keys(self._parsed_args_dict)
        }

    def get_args_summary(self, replaces=None):
        return util.abbreviate_dictionary(
            input_dict=self.get_arg_dict(),
            key_abbreviations=self.register_tags(dict(), ""),
            replaces=replaces,
        )

    def __repr__(self):
        return util.format_type(
            type(self),
            **self.get_arg_dict(),
        )

class _Verbose:
    def __init__(self):
        self._verbosity = 0
        self.set_printer(util.Printer())

    def display_init(self, object_type, kwargs: dict):
        arg_str = ", ".join("%s=%r" % (k, v) for k, v in kwargs.items())
        self._printer("cli: %s(%s)" % (object_type.__name__, arg_str))

    def display_retrieve(self, full_name):
        self._printer("cli: retrieving \"%s\" from cache" % full_name)

    def set_printer(self, printer: util.Printer):
        self._printer = printer

    def __enter__(self):
        self._verbosity += 1

    def __exit__(self, *args):
        self._verbosity -= 1

    def __bool__(self):
        return self._verbosity > 0

verbose = _Verbose()
