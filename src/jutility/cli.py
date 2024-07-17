import argparse
from jutility import util

class Arg:
    def __init__(
        self,
        name,
        abbreviation=None,
        **argparse_kwargs,
    ):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument
        """
        self.name               = name
        self.abbreviation       = abbreviation
        self.argparse_kwargs    = argparse_kwargs

    def get_name(self):
        start_ind = 0
        while self.name[start_ind] == "-":
            start_ind += 1

        return self.name[start_ind:]

    def add_argparse_argument(self, parser: argparse.ArgumentParser):
        parser.add_argument(self.name, **self.argparse_kwargs)

class ObjectParser:
    def __init__(
        self,
        *args: Arg,
        **parser_kwargs,
    ):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser
        """
        self.arg_list = args
        self._parsed_args = None
        self._parser = argparse.ArgumentParser(*args, **parser_kwargs)

    def parse_args(self, *args, **kwargs):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args
        """
        for arg in self.arg_list:
            arg.add_argparse_argument(self._parser)

        args = self._parser.parse_args(*args, **kwargs)
        args.object_parser = self
        self._parsed_args = args
        return args

    def get_args_summary(self, replaces=None):
        if self._parsed_args is None:
            raise RuntimeError(
                "Must call `parse_args` before `get_args_summary`"
            )

        return util.abbreviate_dictionary(
            vars(self._parsed_args),
            key_abbreviations={
                arg.get_name(): arg.abbreviation
                for arg in self.arg_list
                if arg.abbreviation is not None
            },
            replaces=replaces,
        )
