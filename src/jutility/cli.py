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

    def set_full_names(self, name_prefix="", tag_prefix=""):
        self.full_name = join_non_empty(".", [name_prefix, self.name])
        self.full_tag  = join_non_empty(".", [tag_prefix , self.tag ])

    def add_argparse_arguments(
        self,
        parser: argparse.ArgumentParser,
        prefix: str=None,
    ):
        parser.add_argument("--" + self.full_name, **self.argparse_kwargs)

    def set_key_abbreviations(self, abbreviation_dict, prefix=None):
        if self.tag is not None:
            abbreviation_dict[self.full_name] = self.full_tag

    def __repr__(self):
        return (
            "%s(full_name=\"%s\", full_tag=\"%s\")"
            % (type(self).__name__, self.full_name, self.full_tag)
        )

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

    def set_full_names(self, name_prefix="", tag_prefix=""):
        self.full_name = join_non_empty(".", [name_prefix, self.name])
        self.full_tag  = join_non_empty(".", [tag_prefix , self.tag ])
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

    def set_key_abbreviations(self, abbreviation_dict, prefix=None):
        if self.tag is not None:
            for arg in self.args:
                arg.set_key_abbreviations(abbreviation_dict)

    def __repr__(self):
        return "%s(args=%s)" % (type(self).__name__, self.args)

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

        key_abbreviations = dict()
        for arg in self.arg_list:
            arg.set_key_abbreviations(key_abbreviations)

        return util.abbreviate_dictionary(
            vars(self._parsed_args),
            key_abbreviations=key_abbreviations,
            replaces=replaces,
        )

    def __repr__(self):
        arg_list_str = "\n".join("    %s," % arg for arg in self.arg_list)
        return "%s(\n%s\n)" % (type(self).__name__, arg_list_str)

def join_non_empty(sep: str, input_list):
    return sep.join(s for s in input_list if s is not None and len(s) > 0)
