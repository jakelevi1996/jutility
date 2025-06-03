import argparse
from jutility.cli.arg import Arg

class ArgGroup(Arg):
    def __init__(
        self,
        name:   str,
        *args:  Arg,
        tag:    (str | None)=None,
        tagged: bool=True,
        **kwargs,
    ):
        """
        See [`argparse.ArgumentParser.add_argument_group`](
        https://docs.python.org/3/library/argparse.html#argument-groups )
        """
        self._init_arg_parent(list(args), kwargs)
        self._init_arg(name, tag, tagged, False)

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser = parser.add_argument_group(self.full_name, **self._kwargs)
        for arg in self._arg_list:
            arg.add_argparse_arguments(parser)

    def store_value(self, value_dict: dict, summarise: bool):
        return
