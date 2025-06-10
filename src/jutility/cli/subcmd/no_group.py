import argparse
from jutility.cli.arg import Arg
import jutility.cli.subcmd.group as group
import jutility.cli.subcmd.subcommand as base

class _NoSubCommandGroup(group.SubCommandGroup):
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

    def get_command(self) -> "base.SubCommand | None":
        return None
