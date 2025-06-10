import argparse
from jutility import util
from jutility.cli.arg import Arg
import jutility.cli.subcmd.parent as parent
import jutility.cli.subcmd.group as group
import jutility.cli.parsed_args as parsed_args

class SubCommand(parent._SubCommandParent):
    def __init__(
        self,
        name:           str,
        *args:          Arg,
        sub_commands:   (group.SubCommandGroup | None)=None,
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
        args: "parsed_args.ParsedArgs",
        **kwargs,
    ):
        return

    def __repr__(self):
        return util.format_type(type(self), self.name, *self._arg_list)
