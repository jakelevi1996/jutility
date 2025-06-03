import argparse
from jutility import util
from jutility.cli.arg import Arg
from jutility.cli.parsed_args import ParsedArgs
from jutility.cli.subcmd.group import SubCommandGroup
from jutility.cli.subcmd.parent import _SubCommandParent

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
