import argparse
from jutility.cli.arg import Arg

class _UnknownArg(Arg):
    def __init__(self, name: str, value):
        self._init_arg_parent([], dict())
        self._init_arg(name, None, False, False)
        self._init_value(value)
        self.set_full_name(name)

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        return
