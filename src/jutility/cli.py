import argparse
import json
from jutility import util

class _ArgParent:
    def _init_arg_parent(
        self,
        arg_list:   list["Arg"],
        kwargs:     dict,
    ):
        self._arg_list  = arg_list
        self._kwargs    = kwargs

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

    def register_values(self, value_dict: dict, summarise: bool) -> dict:
        for arg in self._get_active_args():
            arg.store_value(value_dict, summarise)
            arg.register_values(value_dict, summarise)

        return value_dict

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
            arg: self._make_tag(prefix, arg.name)
            for arg in tagged_args
            if arg.tag is None
        }
        prefix_dict = util.get_unique_prefixes(
            input_list=default_tag_dict.values(),
            forbidden=set(tag_dict.values()),
            min_len=(len(prefix) + 1),
        )
        for arg, full_tag in default_tag_dict.items():
            if not self._hide_tag(arg):
                tag_dict[arg.full_name] = prefix_dict[full_tag]

        for arg in tagged_args:
            if self._hide_tag(arg):
                arg.register_tags(tag_dict, prefix)
            else:
                arg.register_tags(tag_dict, tag_dict[arg.full_name])

        return tag_dict

    def _make_tag(self, prefix: str, suffix: str) -> str:
        return prefix + suffix.lower().replace("_", "")

    def _hide_tag(self, arg: "Arg") -> bool:
        return False

    def _get_active_args(self) -> list["Arg"]:
        return self._arg_list

    def __repr__(self):
        return util.format_type(type(self))

class Arg(_ArgParent):
    def __init__(
        self,
        name:       str,
        tag:        (str | None)=None,
        tagged:     bool=True,
        is_kwarg:   bool=True,
        **argparse_kwargs,
    ):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument
        """
        self._init_arg_parent([], argparse_kwargs)
        self._init_arg(name, tag, tagged)
        self._is_kwarg = is_kwarg
        if "action" not in self._kwargs:
            self._kwargs.setdefault("metavar", self.name[0].upper())

    def _init_arg(
        self,
        name:   str,
        tag:    str | None,
        tagged: bool,
    ):
        self.name       = name
        self.full_name  = None
        self._init_value()
        self._init_tag(tag, tagged)
        self._init_help()

    def _init_value(self, value=None):
        self.value = value

    def _init_tag(self, tag: str | None, tagged: bool):
        self.tag    = tag
        self.tagged = tagged

    def _init_help(self):
        if ((len(self._kwargs) > 0) and ("help" not in self._kwargs)):
            self._kwargs["help"] = util.format_dict(self._kwargs)

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
        parser.add_argument("--" + self.full_name, **self._kwargs)

    def set_default_choice(self, choice: str | None) -> None:
        raise ValueError(
            "`set_default_choice` only valid for instances of `ObjectChoice`"
        )

    def init_object(self):
        return self.value

    def get_type(self) -> type:
        return type(self.value)

    def get_value_summary(self) -> str:
        return repr(self.value)

    def reset_object_cache(self):
        return

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
        parser.add_argument(self.full_name, **self._kwargs)

class NoTagArg(Arg):
    def _init_tag(self, tag: str | None, tagged: bool):
        self.tag = None
        self.tagged = False

class BooleanArg(Arg):
    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--" + self.full_name,
            action=argparse.BooleanOptionalAction,
            **self._kwargs,
        )

    def _init_help(self):
        if "default" in self._kwargs:
            self._kwargs.setdefault("help", "")
        else:
            if "help" in self._kwargs:
                self._kwargs["help"] += " (default: None)"
            else:
                self._kwargs["help"] = "(default: None)"

class JsonArg(Arg):
    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--" + self.full_name,
            type=json.loads,
            **self._kwargs,
        )

    def _init_help(self):
        if ((len(self._kwargs) > 0) and ("help" not in self._kwargs)):
            self._kwargs["help"] = util.format_dict(self._kwargs)
        if "help" in self._kwargs:
            self._kwargs["help"] += " (format: JSON string)"
        else:
            self._kwargs["help"] = "Format: JSON string"

class ObjectArg(Arg):
    def __init__(
        self,
        object_type:        type,
        *args:              Arg,
        name:               (str | None)=None,
        tag:                (str | None)=None,
        tagged:             bool=True,
        is_group:           bool=False,
        init_requires:      (list[str] | None)=None,
        init_ignores:       (list[str] | None)=None,
        init_const_kwargs:  (dict | None)=None,
    ):
        if name is None:
            name = object_type.__name__

        self.object_type = object_type
        self._init_arg_parent(list(args), dict())
        self._init_arg(name, tag, tagged)
        self._init_object_arg(
            is_group,
            init_requires,
            init_ignores,
            init_const_kwargs,
        )

    def _init_object_arg(
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
            parser = parser.add_argument_group(self.full_name)

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

    def get_type(self) -> type:
        return self.object_type

    def reset_object_cache(self):
        self.value = None

    def store_value(self, value_dict: dict, summarise: bool):
        return

    def is_kwarg(self) -> bool:
        return False

class ObjectChoice(ObjectArg):
    def __init__(
        self,
        name:               str,
        *choices:           ObjectArg,
        default:            (str | None)=None,
        tag:                (str | None)=None,
        tagged:             bool=True,
        is_group:           bool=False,
        shared_args:        (list[Arg] | None)=None,
        init_requires:      (list[str] | None)=None,
        init_ignores:       (list[str] | None)=None,
        init_const_kwargs:  (dict | None)=None,
        required:           (bool | None)=None,
    ):
        if shared_args is None:
            shared_args = []

        self.shared_args = shared_args
        self._init_arg_parent(list(choices) + shared_args, dict())
        self._init_arg(name, tag, tagged)
        self._init_object_arg(
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
        if required is None:
            required = (True if (default is None) else False)

        self.default    = default
        self.required   = required

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        input_parser = parser
        if self.is_group:
            parser = input_parser.add_argument_group(self.full_name)

        parser.add_argument(
            "--" + self.full_name,
            choices=list(self.choice_dict.keys()),
            default=self.default,
            help="default=%s, required=%s" % (self.default, self.required),
        )
        for arg in self.shared_args:
            arg.add_argparse_arguments(parser)

        for arg in self.choice_dict.values():
            if self.is_group:
                parser = input_parser.add_argument_group(arg.full_name)

            arg.add_argparse_arguments(parser)

    def set_default_choice(self, choice: str | None) -> None:
        if choice is None:
            return
        if choice not in self.choice_dict:
            raise ValueError(
                "%r not in %r"
                % (choice, sorted(self.choice_dict.keys()))
            )
        if self.value is None:
            self.value = choice

    def init_object(self, **extra_kwargs):
        chosen_arg = self.get_choice()
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

    def get_choice(self) -> ObjectArg:
        if self.value is None:
            raise ValueError("Please specify --%s" % self.full_name)

        return self.choice_dict[self.value]

    def get_type(self) -> type:
        return self.get_choice().get_type()

    def reset_object_cache(self):
        return

    def get_value_summary(self) -> str:
        chosen_arg = self.get_choice()
        clean = {
            s: s.upper().replace("_", "")
            for s in self.choice_dict.keys()
        }
        tags = {
            clean[arg.name]: arg.tag
            for arg in self.choice_dict.values()
            if arg.tag is not None
        }
        prefix = util.get_unique_prefixes(
            [s for s in clean.values() if s not in tags],
            forbidden=set(tags.values())
        )
        prefix.update(tags)
        return prefix[clean[chosen_arg.name]]

    def store_value(self, value_dict: dict, summarise: bool):
        if summarise:
            value_dict[self.full_name] = self.get_value_summary()
        else:
            value_dict[self.full_name] = self.value

    def _hide_tag(self, arg: "Arg") -> bool:
        return (arg.name in self.choice_dict)

    def _get_active_args(self) -> list["Arg"]:
        chosen_arg = self.get_choice()
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
    def __init__(self, name: str, value):
        self._init_arg_parent([], dict())
        self._init_arg(name, None, False)
        self._init_value(value)
        self.set_full_name(name)

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        return

    def is_kwarg(self):
        return False

class SubCommandGroup:
    def __init__(
        self,
        *commands:  "SubCommand",
        name:       str="command",
        required:   bool=True,
        **subparser_kwargs,
    ):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers
        """
        self.name       = name
        self.value      = None
        self._commands  = commands
        self._required  = required
        self._kwargs    = subparser_kwargs

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
            **self._kwargs,
        )
        for command in self._commands:
            parser = subparser.add_parser(**command.get_subparser_kwargs())
            command.add_argparse_arguments(parser)

    def parse_args(
        self,
        arg_list: list[str, Arg],
        arg_dict: dict[str, Arg],
        argparse_value_dict: dict,
    ) -> dict[str, Arg]:
        self.value = argparse_value_dict.pop(self.full_name)
        self.get_command().parse_args(arg_list, arg_dict, argparse_value_dict)

    def get_command(self) -> "SubCommand | None":
        return self._command_dict[self.value]

class _NoSubCommandGroup(SubCommandGroup):
    def __init__(self):
        self.name   = None
        self.value  = None

    def register_sub_commands(self, prefix: str):
        return

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        return

    def parse_args(
        self,
        arg_list: list[str, Arg],
        arg_dict: dict[str, Arg],
        argparse_value_dict: dict,
    ) -> dict[str, Arg]:
        return

    def get_command(self) -> "SubCommand | None":
        return None

class _SubCommandParent(_ArgParent):
    def _init_subcommand_parent(self, sub_commands: "SubCommandGroup | None"):
        if sub_commands is None:
            sub_commands = _NoSubCommandGroup()

        self._sub_commands  = sub_commands
        self._arg_dict      = self.register_names(dict(), "")

class SubCommand(_SubCommandParent):
    def __init__(
        self,
        name:           str,
        *args:          Arg,
        sub_commands:   (SubCommandGroup | None)=None,
        **parser_kwargs,
    ):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser
        """
        self.name = name
        self._init_arg_parent(list(args), parser_kwargs)
        self._init_subcommand_parent(sub_commands)

    def register_sub_commands(self, prefix: str):
        self.full_name = prefix + self.name
        self._sub_commands.register_sub_commands(self.full_name + ".")

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        for arg in self._arg_list:
            arg.add_argparse_arguments(parser)

        self._sub_commands.add_argparse_arguments(parser)

    def parse_args(
        self,
        arg_list: list[str, Arg],
        arg_dict: dict[str, Arg],
        argparse_value_dict: dict,
    ) -> dict[str, Arg]:
        arg_list.extend(self._arg_list)
        arg_dict.update(self._arg_dict)
        self._sub_commands.parse_args(arg_list, arg_dict, argparse_value_dict)

    def get_command(self) -> "SubCommand | None":
        return self._sub_commands.get_command()

    def get_subparser_kwargs(self) -> dict:
        subparser_kwargs = {"name": self.name}
        subparser_kwargs.update(self._kwargs)
        return subparser_kwargs

    def run(self, args: "ParsedArgs"):
        return self.get_command().run(args)

    def __repr__(self):
        return util.format_type(type(self), self.name, *self._arg_list)

class Parser(_SubCommandParent):
    def __init__(
        self,
        *args:          Arg,
        sub_commands:   (SubCommandGroup | None)=None,
        **parser_kwargs,
    ):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser
        """
        self._init_arg_parent(list(args), parser_kwargs)
        self._init_subcommand_parent(sub_commands)
        self._sub_commands.register_sub_commands("")

    def _get_argparse_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(**self._kwargs)
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

        arg_list = self._arg_list.copy()
        arg_dict = self._arg_dict.copy()
        self._sub_commands.parse_args(arg_list, arg_dict, argparse_value_dict)

        for arg_name in argparse_value_dict:
            arg_dict[arg_name].value = argparse_value_dict[arg_name]

        return ParsedArgs(arg_list, arg_dict, self._sub_commands)

    def help(self) -> str:
        return self._get_argparse_parser().format_help()

    def __repr__(self):
        return util.format_type(
            type(self),
            *self._arg_list,
            **self._kwargs,
        )

class ParsedArgs(_ArgParent):
    def __init__(
        self,
        arg_list:       list[Arg],
        arg_dict:       dict[str, Arg],
        sub_commands:   SubCommandGroup | _NoSubCommandGroup,
    ):
        self._init_arg_parent(arg_list, dict())
        self._arg_dict = arg_dict
        self._sub_commands = sub_commands
        self.reset_object_cache()

    def get_arg(self, arg_name: str) -> Arg:
        return self._arg_dict[arg_name]

    def get_value(self, arg_name: str):
        return self._arg_dict[arg_name].value

    def get_command(self) -> "SubCommand | None":
        return self._sub_commands.get_command()

    def update(self, value_dict: dict, allow_new_keys=False):
        for name, value in value_dict.items():
            if name in self._arg_dict:
                self._arg_dict[name].value = value
            elif allow_new_keys:
                new_arg = _UnknownArg(name, value)
                self._arg_dict[name] = new_arg
                self._arg_list.append(new_arg)
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

    def get_type(self, full_name: str) -> type:
        return self._arg_dict[full_name].get_type()

    def reset_object_cache(self):
        for arg in self._arg_dict.values():
            arg.reset_object_cache()

class _Verbose:
    def __init__(self):
        self._verbosity = 0
        self.set_printer(util.Printer())

    def display_init(self, object_type: type, kwargs: dict):
        self._printer("cli: %s" % util.format_type(object_type, **kwargs))

    def display_retrieve(self, full_name: str):
        self._printer("cli: `%s` retrieved from cache" % full_name)

    def set_printer(self, printer: util.Printer):
        self._printer = printer

    def __enter__(self):
        self._verbosity += 1

    def __exit__(self, *args):
        self._verbosity -= 1

    def __bool__(self):
        return self._verbosity > 0

verbose = _Verbose()
