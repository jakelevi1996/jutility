import argparse
import json
from jutility import util

class _ArgParent:
    def _init_arg_parent(
        self,
        arg_list: list["Arg"],
        name: str,
        kwargs: (dict | None)=None,
        value=None,
    ):
        if kwargs is None:
            kwargs = dict()

        self._init_arg_list(arg_list)
        self._init_name(name)
        self._init_value(value)
        self._init_kwargs(kwargs)

    def _init_arg_list(self, arg_list: list["Arg"]):
        self._arg_list = arg_list

    def _init_name(self, name: str):
        self.name = name
        self.full_name = None

    def _init_value(self, value=None):
        self.value = value

    def _init_kwargs(self, kwargs: dict):
        self.kwargs = kwargs

    def get_value_dict(self) -> dict:
        return self.register_values(dict(), False)

    def get_summary(self, replaces=None) -> str:
        return util.abbreviate_dictionary(
            input_dict=self.register_values(dict(), True),
            key_abbreviations=self.register_tags(dict(), ""),
            replaces=replaces,
        )

    def get_kwargs(self) -> dict:
        return {
            arg.name: arg.value
            for arg in self._arg_list
            if arg.is_kwarg()
        }

    def register_names(
        self,
        arg_dict: dict[str, "Arg"],
        prefix: str,
    ) -> dict[str, "Arg"]:
        for arg in self._arg_list:
            arg.set_full_name(prefix + arg.name)
            arg.store_name(arg_dict)
            arg.register_names(arg_dict, arg.full_name + ".")

        return arg_dict

    def register_tags(
        self,
        tag_dict: dict[str, str],
        prefix: str,
    ) -> dict[str, str]:
        tagged_args = [arg for arg in self._get_active_args() if arg.tagged]
        for arg in tagged_args:
            if arg.tag is not None:
                tag_dict[arg.full_name] = self._make_tag(prefix, arg.tag)

        default_tag_dict = {
            arg.full_name: self._make_tag(prefix, arg.name)
            for arg in tagged_args
            if arg.tag is None
        }
        prefix_dict = util.get_unique_prefixes(
            input_list=default_tag_dict.values(),
            forbidden=set(tag_dict.values()),
            min_len=(len(prefix) + 1),
        )
        for full_name, full_tag in default_tag_dict.items():
            tag_dict[full_name] = prefix_dict[full_tag]

        for arg in tagged_args:
            if self._hide_tag(arg):
                arg.register_tags(tag_dict, prefix)
            else:
                arg.register_tags(tag_dict, tag_dict[arg.full_name] + ".")

        return tag_dict

    def register_values(self, value_dict: dict, summarise: bool) -> dict:
        for arg in self._get_active_args():
            arg.store_value(value_dict, summarise)
            arg.register_values(value_dict, summarise)

        return value_dict

    def _make_tag(self, prefix: str, suffix: str) -> str:
        return prefix + suffix.lower().replace("_", "")

    def _hide_tag(self, arg: "Arg") -> bool:
        return False

    def _get_active_args(self) -> list["Arg"]:
        return self._arg_list

    def __repr__(self):
        return util.format_type(
            type(self),
            name=self.name,
            full_name=self.full_name,
            value=self.value,
        )

class _ArgRoot(_ArgParent):
    def _init_arg_dict(self):
        self._arg_dict = self.register_names(dict(), "")

    def _init_sub_commands(self, sub_commands: "SubCommandGroup"):
        if sub_commands is None:
            sub_commands = _NoSubCommandGroup()

        self._sub_commands = sub_commands

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
        self._is_kwarg = is_kwarg
        self._init_arg_parent([], name, argparse_kwargs)
        self._init_tag(tag, tagged)
        self._init_help()

    def _init_tag(self, tag: str | None, tagged: bool):
        self.tag = tag
        self.tagged = tagged

    def _init_help(self):
        if ((len(self.kwargs) > 0) and ("help" not in self.kwargs)):
            self.kwargs["help"] = util.format_dict(self.kwargs)

    def set_full_name(self, full_name: str):
        if self.full_name is not None:
            raise RuntimeError("%s is already registered" % self)

        self.full_name = full_name

    def store_name(self, arg_dict: dict[str, "Arg"]):
        if self.full_name in arg_dict:
            raise ValueError(
                "Found duplicates %s and %s"
                % (self, arg_dict[self.full_name])
            )

        arg_dict[self.full_name] = self

    def store_value(self, value_dict: dict, summarise: bool):
        value_dict[self.full_name] = self.value

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--" + self.full_name, **self.kwargs)

    def init_object(self):
        return self.value

    def reset_object_cache(self):
        return

    def is_kwarg(self) -> bool:
        return self._is_kwarg

class PositionalArg(Arg):
    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(self.full_name, **self.kwargs)

class NoTagArg(Arg):
    def _init_tag(self, tag, tagged):
        self.tag = None
        self.tagged = False

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
        tagged:   bool=True,
        is_group: bool=False,
        init_requires: (list[str] | None)=None,
        init_ignores:  (list[str] | None)=None,
        init_const_kwargs:  (dict | None)=None,
    ):
        if name is None:
            name = object_type.__name__

        self.object_type = object_type
        self._init_arg_parent(list(args), name)
        self._init_tag(tag, tagged)
        self._init_object_options(
            is_group,
            init_requires,
            init_ignores,
            init_const_kwargs,
        )

    def _init_object_options(
        self,
        is_group: bool,
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

        self.is_group = is_group
        self.init_requires = init_requires
        self.init_ignores  = init_ignores
        self.init_const_kwargs  = init_const_kwargs

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        if self.is_group:
            parser = parser.add_argument_group(self.name)

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

    def store_value(self, value_dict: dict, summarise: bool):
        return

    def is_kwarg(self) -> bool:
        return False

class ObjectChoice(ObjectArg):
    def __init__(
        self,
        name: str,
        *choices: ObjectArg,
        default: (str | None)=None,
        tag:     (str | None)=None,
        tagged:   bool=True,
        is_group: bool=False,
        shared_args:   (list[Arg] | None)=None,
        init_requires: (list[str] | None)=None,
        init_ignores:  (list[str] | None)=None,
        init_const_kwargs:  (dict | None)=None,
    ):
        if shared_args is None:
            shared_args = []

        self.shared_args = shared_args
        self.default = default
        arg_list = list(choices) + shared_args
        self._init_arg_parent(arg_list, name)
        self._init_tag(tag, tagged)
        self._init_object_options(
            is_group,
            init_requires,
            init_ignores,
            init_const_kwargs,
        )

        self.choice_dict = {arg.name: arg for arg in choices}
        if (default is not None) and (default not in self.choice_dict):
            raise ValueError(
                "%s received `default=\"%s\"`, please choose from %s"
                % (self, default, sorted(arg.name for arg in choices))
            )

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        if self.is_group:
            parser = parser.add_argument_group(self.name)

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

    def store_value(self, value_dict: dict, summarise: bool):
        if summarise:
            choices_clean = {
                s: s.upper().replace("_", "")
                for s in self.choice_dict.keys()
            }
            tags = util.get_unique_prefixes(choices_clean.values())
            value_clean = choices_clean[self.value]
            value_dict[self.full_name] = tags[value_clean]
        else:
            value_dict[self.full_name] = self.value

    def _hide_tag(self, arg: "Arg") -> bool:
        return (arg.name in self.choice_dict)

    def _get_active_args(self) -> list["Arg"]:
        chosen_arg = self.choice_dict[self.value]
        protected = chosen_arg.get_protected_args()
        return [
            chosen_arg,
            *[
                arg
                for arg in self.shared_args
                if  arg.name not in protected
            ],
        ]

class _UnknownArg(Arg):
    def __init__(self, value):
        self._init_arg_parent([], None, value=value)
        self._init_tag(None, False)

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        return

    def is_kwarg(self):
        return False

class SubCommandGroup(_ArgParent):
    def __init__(
        self,
        *commands: "SubCommand",
        name: str="command",
        required: bool=True,
        **subparser_kwargs,
    ):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers
        """
        self._init_arg_parent([], name, subparser_kwargs)
        self._commands = commands
        self._required = required

        self._command_dict = {c.name: c for c in commands}

    def register_sub_commands(self, prefix: str):
        self.full_name = prefix + self.name
        for c in self._commands:
            c.register_sub_commands(self.full_name + ".")

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        subparser = parser.add_subparsers(
            title=self.name,
            dest=self.full_name,
            required=self._required,
            **self.kwargs,
        )
        for command in self._commands:
            parser = subparser.add_parser(command.name, **command.kwargs)
            command.add_argparse_arguments(parser)

    def parse_args(
        self,
        arg_dict: dict[str, Arg],
        argparse_value_dict: dict,
    ) -> dict[str, Arg]:
        self.value = argparse_value_dict.pop(self.full_name)
        self.get_command().parse_args(arg_dict, argparse_value_dict)

    def get_command(self) -> "SubCommand":
        return self._command_dict[self.value]

class SubCommand(_ArgRoot):
    def __init__(
        self,
        name: str,
        *args: Arg,
        sub_commands: (SubCommandGroup | None)=None,
        **parser_kwargs,
    ):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser
        """
        self._init_arg_parent(list(args), name, parser_kwargs)
        self._init_arg_dict()
        self._init_sub_commands(sub_commands)

    def register_sub_commands(self, prefix: str):
        self.full_name = prefix + self.name
        self._sub_commands.register_sub_commands(self.full_name + ".")

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        for arg in self._arg_list:
            arg.add_argparse_arguments(parser)

        self._sub_commands.add_argparse_arguments(parser)

    def parse_args(
        self,
        arg_dict: dict[str, Arg],
        argparse_value_dict: dict,
    ) -> dict[str, Arg]:
        arg_dict.update(self._arg_dict)
        self._sub_commands.parse_args(arg_dict, argparse_value_dict)

    def get_command(self) -> "SubCommand":
        return self._sub_commands.get_command()

    def run(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        return util.format_type(type(self), self.name, *self._arg_list)

class _NoSubCommandGroup(SubCommandGroup):
    def __init__(self):
        self._init_arg_parent([], None)

    def register_sub_commands(self, prefix: str):
        return

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        return

    def parse_args(
        self,
        arg_dict: dict[str, Arg],
        argparse_value_dict: dict,
    ) -> dict[str, Arg]:
        return

    def get_command(self) -> SubCommand:
        return

class Parser(_ArgRoot):
    def __init__(
        self,
        *args: Arg,
        sub_commands: (SubCommandGroup | None)=None,
        **parser_kwargs,
    ):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser
        """
        self._parser_kwargs = parser_kwargs
        self._init_arg_list(list(args))
        self._init_arg_dict()
        self._init_sub_commands(sub_commands)
        self._sub_commands.register_sub_commands("")

    def _get_argparse_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(**self._parser_kwargs)
        for arg in self._arg_list:
            arg.add_argparse_arguments(parser)

        self._sub_commands.add_argparse_arguments(parser)
        return parser

    def parse_args(self, *args, **kwargs) -> "ParsedArgs":
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args
        """
        parser = self._get_argparse_parser()
        argparse_namespace = parser.parse_args(*args, **kwargs)
        argparse_value_dict = vars(argparse_namespace)

        arg_dict = self._arg_dict.copy()
        self._sub_commands.parse_args(arg_dict, argparse_value_dict)

        for arg_name in argparse_value_dict:
            arg_dict[arg_name].value = argparse_value_dict[arg_name]

        return ParsedArgs(self._arg_list, arg_dict)

    def get_command(self) -> SubCommand:
        return self._sub_commands.get_command()

    def help(self) -> str:
        return self._get_argparse_parser().format_help()

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
        self.reset_object_cache()

    def get_arg(self, arg_name: str) -> Arg:
        return self._arg_dict[arg_name]

    def get_value(self, arg_name: str):
        return self._arg_dict[arg_name].value

    def update(self, value_dict: dict, allow_new_keys=False):
        for name, value in value_dict.items():
            if name in self._arg_dict:
                self._arg_dict[name].value = value
            elif allow_new_keys:
                new_arg = _UnknownArg(value)
                new_arg.set_full_name(name)
                self._arg_dict[name] = new_arg
                self._arg_list.append( new_arg)
            else:
                new_keys = set(value_dict.keys()) - set(self._arg_dict.keys())
                raise ValueError(
                    "Received extra keys %s. Either remove these keys, or "
                    "call `ParsedArgs.update(..., allow_new_keys=True)`."
                    % sorted(new_keys)
                )

        self.reset_object_cache()

    def init_object(self, full_name: str, **extra_kwargs):
        return self._arg_dict[full_name].init_object(**extra_kwargs)

    def reset_object_cache(self):
        for arg in self._arg_dict.values():
            arg.reset_object_cache()

    def __repr__(self):
        return util.format_type(
            type(self),
            **self.get_value_dict(),
        )

class _Verbose:
    def __init__(self):
        self._verbosity = 0
        self.set_printer(util.Printer())

    def display_init(self, object_type: type, kwargs: dict):
        self._printer("cli: %s" % util.format_type(object_type, **kwargs))

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
