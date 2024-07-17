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
        self.tag                = abbreviation
        self.argparse_kwargs    = argparse_kwargs
        self.set_full_names()

    def set_full_names(self, name_prefix=None, tag_prefix=None):
        if name_prefix is None:
            self.full_name = self.name
        else:
            self.full_name = "%s.%s" % (name_prefix, self.name)
        if tag_prefix is None:
            self.full_tag = self.tag
        else:
            self.full_tag = "%s.%s" % (tag_prefix, self.tag)

    def add_argparse_arguments(
        self,
        parser: argparse.ArgumentParser,
        prefix: str=None,
    ):
        parser.add_argument("--" + self.full_name, **self.argparse_kwargs)


class ObjectArg(Arg):
    def __init__(
        self,
        object_type,
        *args: Arg,
        abbreviation=None,
        init_requires: list[str]=None,
    ):
        self.object_type    = object_type
        self.name           = object_type.__name__
        self.tag            = abbreviation
        self.args           = args
        self.init_requires  = init_requires
        self.set_full_names()

    def set_full_names(self, name_prefix=None, tag_prefix=None):
        super().set_full_names(name_prefix, tag_prefix)
        for arg in self.args:
            arg.set_full_names(self.full_name, self.full_tag)

    def add_argparse_arguments(
        self,
        parser: argparse.ArgumentParser,
        prefix: str=None,
    ):
        if prefix is None:
            prefix = self.name
        else:
            prefix = "%s.%s" % (prefix, self.name)

        for arg in self.args:
            arg.add_argparse_arguments(parser, prefix)

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
        self._parser = argparse.ArgumentParser(**parser_kwargs)

    def parse_args(self, *args, **kwargs):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args
        """
        for arg in self.arg_list:
            arg.add_argparse_arguments(self._parser)

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
                arg.name: arg.abbreviation
                for arg in self.arg_list
                if arg.abbreviation is not None
            },
            replaces=replaces,
        )
