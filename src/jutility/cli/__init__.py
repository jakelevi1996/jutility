import argparse
import json
from jutility import util
from jutility.cli.arg import (
    Arg,
    PositionalArg,
    NoTagArg,
    BooleanArg,
    JsonArg,
)
from jutility.cli.parent import _ArgParent
from jutility.cli.group import ArgGroup
from jutility.cli.verbose import verbose
from jutility.cli.object_arg import ObjectArg
from jutility.cli.object_choice import ObjectChoice
from jutility.cli.unknown import _UnknownArg

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
        if len(self._command_dict) < len(commands):
            raise ValueError(
                "Received duplicate command names:\n%s"
                % "\n".join("- %r" % c for c in commands)
            )

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

    @classmethod
    def run(
        cls,
        args: "ParsedArgs",
        **kwargs,
    ):
        return

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
