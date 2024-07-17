import argparse
import textwrap
from jutility import util

def get_args_summary(args, replaces=None):
    object_parser: ObjectParser = args.object_parser
    return object_parser.get_args_summary(replaces)

def init_object(args, full_name, **extra_kwargs):
    object_parser: ObjectParser = args.object_parser
    return object_parser.init_object(full_name, **extra_kwargs)

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

    def register_names(self, arg_dict, parent=None):
        if parent is None:
            self.full_name = self.name
            self.full_tag = self.tag
        else:
            self.full_name = "%s.%s" % (parent.full_name, self.name)
            if (parent.full_tag is not None) and (self.tag is not None):
                self.full_tag = "%s.%s" % (parent.full_tag, self.tag)
            else:
                self.full_tag = None

        arg_dict[self.full_name] = self

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--" + self.full_name, **self.argparse_kwargs)

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
        name=None,
        abbreviation=None,
        init_requires: list[str]=None,
    ):
        if name is None:
            name = object_type.__name__
        if init_requires is None:
            init_requires = []

        self.object_type    = object_type
        self.name           = name
        self.tag            = abbreviation
        self.args           = args
        self.init_requires  = init_requires

    def register_names(self, arg_dict, parent=None):
        super().register_names(arg_dict, parent)
        for arg in self.args:
            arg.register_names(arg_dict, self)

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        for arg in self.args:
            arg.add_argparse_arguments(parser)

    def init_object(self, parsed_kwargs, extra_kwargs):
        input_keys = set(parsed_kwargs) | set(extra_kwargs)
        missing_keys = set(self.init_requires) - input_keys
        if len(missing_keys) > 0:
            raise ValueError(
                "Please provide values for the following keys: %s"
                % sorted(missing_keys)
            )

        return self.object_type(**parsed_kwargs, **extra_kwargs)

    def __repr__(self):
        description = ",\n".join(
            [   "full_name=\"%s\"" % self.full_name]
            + [ "full_tag=\"%s\""  % self.full_tag]
            + [ repr(arg) for arg in self.args]
        )
        return "%s(\n%s,\n)" % (type(self).__name__, indent(description))

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
        self._parser_kwargs = parser_kwargs
        self._parsed_args = None
        self._arg_dict: dict[str, Arg] = dict()
        for arg in self.arg_list:
            arg.register_names(self._arg_dict)

    def _get_argparse_parser(self):
        parser = argparse.ArgumentParser(**self._parser_kwargs)
        for arg in self.arg_list:
            arg.add_argparse_arguments(parser)

        return parser

    def print_help(self, file=None):
        self._get_argparse_parser().print_help(file)

    def parse_args(self, *args, **kwargs):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args
        """
        parser = self._get_argparse_parser()
        args = parser.parse_args(*args, **kwargs)
        args.object_parser = self
        self._parsed_args = args
        return args

    def check_parsed(self):
        if self._parsed_args is None:
            raise RuntimeError(
                "Must call `parse_args` before calling this method"
            )

    def get_args_summary(self, replaces=None):
        self.check_parsed()
        key_abbreviations = {
            name: arg.full_tag
            for name, arg in self._arg_dict.items()
            if arg.full_tag is not None
        }

        return util.abbreviate_dictionary(
            vars(self._parsed_args),
            key_abbreviations=key_abbreviations,
            replaces=replaces,
        )

    def init_object(self, full_name, **extra_kwargs):
        if full_name not in self._arg_dict:
            raise ValueError("\"%s\" not in %s" % (full_name, self._arg_dict))

        object_arg: ObjectArg = self._arg_dict[full_name]
        util.check_type(object_arg, ObjectArg)

        self.check_parsed()
        arg_dict = {arg.full_name: arg for arg in object_arg.args}
        relevant_kwargs = {
            v.name: vars(self._parsed_args)[k]
            if k in vars(self._parsed_args)
            else self.init_object(k)
            for k, v in arg_dict.items()
        }
        return object_arg.init_object(relevant_kwargs, extra_kwargs)

    def __repr__(self):
        description = ",\n".join(repr(arg) for arg in self.arg_list)
        return "%s(\n%s,\n)" % (type(self).__name__, indent(description))

def join_non_empty(sep: str, input_list):
    return sep.join(s for s in input_list if s is not None and len(s) > 0)

def indent(input_str, num_spaces=4):
    return textwrap.indent(input_str, " " * num_spaces)
