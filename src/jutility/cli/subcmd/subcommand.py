import argparse
from jutility import util
from jutility.cli.arg import Arg
from jutility.cli.root import ArgRoot
import jutility.cli.subcmd.group as group

class SubCommand(ArgRoot):
    def __init__(
        self,
        name:           str,
        *args:          Arg,
        sub_commands:   (group.SubCommandGroup | None)=None,
        **subparser_kwargs,
    ):
        """
        See [`argparse.ArgumentParser`](
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser
        ) and [`ArgumentParser.add_subparsers`](
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_subparsers
        )
        """
        self.name = name
        self._init_arg_parent(list(args), subparser_kwargs)
        self._init_subcommand_parent(sub_commands)

    def register_sub_commands(self, prefix: str):
        self.full_name = prefix + self.name
        self._sub_commands.register_sub_commands(self.full_name + ".")

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        for arg in self._arg_list:
            arg.add_argparse_arguments(parser)

        self._sub_commands.add_argparse_arguments(parser)

    def parse_args(self, argparse_value_dict: dict):
        for arg in self._arg_list:
            arg.parse_values(argparse_value_dict)

        self._sub_commands.parse_args(argparse_value_dict)

    def get_subparser_kwargs(self) -> dict:
        subparser_kwargs = {"name": self.name}
        subparser_kwargs.update(self._kwargs)
        return subparser_kwargs

    def run(self, **kwargs):
        return

    def __repr__(self):
        return util.format_type(type(self), self.name, *self._arg_list)
