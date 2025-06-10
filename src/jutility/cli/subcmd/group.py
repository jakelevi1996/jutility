import argparse
from jutility.cli.arg import Arg
import jutility.cli.subcmd.subcommand as base

class SubCommandGroup:
    def __init__(
        self,
        *commands:  "base.SubCommand",
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

    def get_command(self) -> "base.SubCommand | None":
        return self._command_dict[self.value]
