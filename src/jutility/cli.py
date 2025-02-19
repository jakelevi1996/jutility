import argparse
import json
from jutility import util

def init_object(args: "ParsedArgs", full_name, **extra_kwargs):
    return args.get_parser().init_object(full_name, **extra_kwargs)

def reset_object_cache(args: "ParsedArgs"):
    return args.get_parser().reset_object_cache()

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
            if arg.full_name is not None:
                raise RuntimeError("%s is already registered" % arg)

            arg.full_name = prefix + arg.name
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

    def _hide_tag(self, arg: "Arg"):
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
        self._init_arg_list([])
        self._is_kwarg = is_kwarg
        self.kwargs = argparse_kwargs
        self.init_name(name)
        self.init_tag(tag, tagged)
        self.init_help()

    def init_name(self, name: str):
        self.name = name
        self.full_name = None

    def init_tag(self, tag: str | None, tagged: bool):
        self.tag = tag
        self.tagged = tagged

    def init_help(self):
        if ((len(self.kwargs) > 0) and ("help" not in self.kwargs)):
            self.kwargs["help"] = util.format_dict(self.kwargs)

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--" + self.full_name, **self.kwargs)

    def init_object(self, parsed_args_dict):
        return parsed_args_dict[self.full_name]

    def get_arg_dict_keys(self, parsed_args_dict):
        return [self.full_name]

    def is_kwarg(self):
        return self._is_kwarg

    def __repr__(self):
        return util.format_type(
            type(self),
            name=self.name,
            full_name=self.full_name,
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

    def init_help(self):
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

    def init_help(self):
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
        name: str=None,
        tag: (str | None)=None,
        tagged=True,
        init_requires: list[str]=None,
        init_parsed_kwargs: dict=None,
        init_const_kwargs: dict=None,
        init_ignores: list[str]=None,
    ):
        if name is None:
            name = object_type.__name__

        self.object_type = object_type
        self._init_arg_list(args)
        self.init_name(name)
        self.init_tag(tag, tagged)
        self.set_init_attributes(
            init_requires,
            init_parsed_kwargs,
            init_const_kwargs,
            init_ignores,
        )

    def set_init_attributes(
        self,
        init_requires,
        init_parsed_kwargs,
        init_const_kwargs,
        init_ignores,
    ):
        if init_requires is None:
            init_requires = []
        if init_parsed_kwargs is None:
            init_parsed_kwargs = dict()
        if init_const_kwargs is None:
            init_const_kwargs = dict()
        if init_ignores is None:
            init_ignores = []

        self.init_requires      = init_requires
        self.init_parsed_kwargs = init_parsed_kwargs
        self.init_const_kwargs  = init_const_kwargs
        self.init_ignores  = init_ignores

    def get_protected_args(self):
        return (
            set(arg.name for arg in self._arg_list)
            | set(self.init_parsed_kwargs.keys())
            | set(self.init_const_kwargs.keys())
            | set(self.init_ignores)
        )

    def update_kwargs(
        self,
        kwargs: dict,
        parsed_args_dict: dict,
        extra_kwargs: dict,
        protected: set=None,
    ):
        if protected is None:
            protected = set()
        for k, v in self.init_parsed_kwargs.items():
            if k not in protected:
                kwargs[k] = parsed_args_dict[v]
        for k, v in self.init_const_kwargs.items():
            if k not in protected:
                kwargs[k] = v
        for k, v in extra_kwargs.items():
            kwargs[k] = v

        missing_keys = set(self.init_requires) - (set(kwargs) | protected)
        if len(missing_keys) > 0:
            raise ValueError(
                "Please provide values for the following keys %s for \"%s\""
                % (sorted(missing_keys), self.full_name)
            )

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        for arg in self._arg_list:
            arg.add_argparse_arguments(parser)

    def init_object(self, parsed_args_dict, **extra_kwargs):
        if self.full_name in parsed_args_dict:
            if verbose:
                verbose.display_retrieve(self.full_name)

            return parsed_args_dict[self.full_name]

        kwargs = {
            arg.name: arg.init_object(parsed_args_dict)
            for arg in self._arg_list
        }
        self.update_kwargs(kwargs, parsed_args_dict, extra_kwargs)

        if verbose:
            verbose.display_init(self.object_type, kwargs)

        object_value = self.object_type(**kwargs)
        parsed_args_dict[self.full_name] = object_value
        return object_value

    def get_arg_dict_keys(self, parsed_args_dict):
        return [
            name
            for arg in self._arg_list
            for name in arg.get_arg_dict_keys(parsed_args_dict)
        ]

    def is_kwarg(self):
        return False

class ObjectChoice(ObjectArg):
    def __init__(
        self,
        name: str,
        *choices: ObjectArg,
        shared_args: list[Arg]=None,
        default: str=None,
        tag: (str | None)=None,
        tagged=True,
        init_requires: list[str]=None,
        init_parsed_kwargs: dict=None,
        init_const_kwargs: dict=None,
        init_ignores: list[str]=None,
    ):
        if shared_args is None:
            shared_args = []

        self.shared_args = shared_args
        self.default = default
        self.init_name(name)
        self.init_tag(tag, tagged)
        self.set_init_attributes(
            init_requires,
            init_parsed_kwargs,
            init_const_kwargs,
            init_ignores,
        )

        self._init_arg_list(tuple(choices) + tuple(shared_args))
        self.choice_dict = {arg.name: arg for arg in choices}
        if (default is not None) and (default not in self.choice_dict):
            valid_names = [arg.name for arg in choices]
            raise ValueError(
                "%s(\"%s\") received `default=\"%s\"`, please choose from %s"
                % (type(self).__name__, name, default, valid_names)
            )

    def _hide_tag(self, arg: Arg):
        return (arg.name in self.choice_dict)

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--" + self.full_name,
            choices=list(self.choice_dict.keys()),
            default=self.default,
            required=(True if (self.default is None) else False),
        )
        for arg in self._arg_list:
            arg.add_argparse_arguments(parser)

    def init_object(self, parsed_args_dict, **extra_kwargs):
        chosen_arg = self.choice_dict[parsed_args_dict[self.full_name]]
        protected = chosen_arg.get_protected_args()
        kwargs = {
            arg.name: arg.init_object(parsed_args_dict)
            for arg in self.shared_args
            if arg.name not in protected
        }
        self.update_kwargs(kwargs, parsed_args_dict, extra_kwargs, protected)
        return chosen_arg.init_object(parsed_args_dict, **kwargs)

    def get_arg_dict_keys(self, parsed_args_dict):
        chosen_arg = self.choice_dict[parsed_args_dict[self.full_name]]
        protected = chosen_arg.get_protected_args()
        return (
            [self.full_name]
            + chosen_arg.get_arg_dict_keys(parsed_args_dict)
            + [
                name
                for arg in self.shared_args
                if  arg.name not in protected
                for name in arg.get_arg_dict_keys(parsed_args_dict)
            ]
        )

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
        self._parsed_args_dict: dict = None

    def _get_argparse_parser(self):
        parser = argparse.ArgumentParser(**self._parser_kwargs)
        for arg in self._arg_list:
            arg.add_argparse_arguments(parser)

        return parser

    def _check_parsed(self):
        if self._parsed_args_dict is None:
            raise RuntimeError(
                "Must call `parse_args` before calling this method"
            )

    def print_help(self, file=None):
        self._get_argparse_parser().print_help(file)

    def parse_args(self, *args, **kwargs):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args
        """
        parser = self._get_argparse_parser()
        argparse_args = parser.parse_args(*args, **kwargs)
        self._parsed_args_dict = vars(argparse_args)
        self._initial_args_cache = set(self._parsed_args_dict.keys())
        return ParsedArgs(self, self._parsed_args_dict)

    def init_object(self, full_name, **extra_kwargs):
        self._check_parsed()
        if full_name not in self._arg_dict:
            raise ValueError(
                "\"%s\" not in %s"
                % (full_name, sorted(self._arg_dict.keys()))
            )

        object_arg = self._arg_dict[full_name]
        return object_arg.init_object(self._parsed_args_dict, **extra_kwargs)

    def get_arg_dict(self):
        self._check_parsed()
        return {
            key: self._parsed_args_dict[key]
            for arg in self._arg_list
            for key in arg.get_arg_dict_keys(self._parsed_args_dict)
        }

    def get_args_summary(self, replaces=None):
        self._check_parsed()
        return util.abbreviate_dictionary(
            input_dict=self.get_arg_dict(),
            key_abbreviations=self.register_tags(dict(), ""),
            replaces=replaces,
        )

    def get_kwarg_names(self):
        return [arg.name for arg in self._arg_list if arg.is_kwarg()]

    def reset_object_cache(self):
        self._check_parsed()
        current_args_cache = set(self._parsed_args_dict.keys())
        for k in (current_args_cache - self._initial_args_cache):
            self._parsed_args_dict.pop(k)

    def __repr__(self):
        description = ",\n".join(repr(arg) for arg in self._arg_dict.values())
        return "%s(\n%s,\n)" % (type(self).__name__, util.indent(description))

class ParsedArgs:
    def __init__(self, parser: Parser, arg_dict: dict):
        self._parser = parser
        self._arg_dict = arg_dict

    def get_parser(self):
        return self._parser

    def get(self, arg_name: str):
        return self._arg_dict[arg_name]

    def get_kwargs(self):
        kwarg_names = self._parser.get_kwarg_names()
        return {k: self._arg_dict[k] for k in kwarg_names}

    def update(self, arg_dict: dict, allow_new_keys=False):
        if not allow_new_keys:
            new_keys = set(arg_dict.keys()) - set(self._arg_dict.keys())
            if len(new_keys) > 0:
                raise ValueError(
                    "Received extra keys %s. Either remove these keys, or "
                    "call `ParsedArgs.update` with `allow_new_keys=True`."
                    % sorted(new_keys)
                )

        self._parser.reset_object_cache()
        self._arg_dict.update(arg_dict)

    def __repr__(self):
        arg_str = util.format_dict(self._arg_dict)
        return "%s(%s)" % (type(self).__name__, arg_str)

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
