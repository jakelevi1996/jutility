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
        self.args: list[Arg]    = []

    def register_names(self, arg_dict, parent: "Arg"=None):
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

        for arg in self.args:
            arg.register_names(arg_dict, self)

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--" + self.full_name, **self.argparse_kwargs)

    def init_object(self, *args):
        raise TypeError(
            "Cannot call `init_object` on an instance of `%s`"
            % type(self).__name__
        )

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
        name: str=None,
        abbreviation: str=None,
        init_requires: list[str]=None,
        init_parsed_kwargs: dict[str, str]=None,
        init_const_kwargs:  dict[str, str]=None,
    ):
        if name is None:
            name = object_type.__name__

        self.object_type = object_type
        self.name        = name
        self.tag         = abbreviation
        self.args        = args
        self.set_init_attributes(
            init_requires,
            init_parsed_kwargs,
            init_const_kwargs,
        )

    def set_init_attributes(
        self,
        init_requires,
        init_parsed_kwargs,
        init_const_kwargs,
    ):
        if init_requires is None:
            init_requires = []
        if init_parsed_kwargs is None:
            init_parsed_kwargs = dict()
        if init_const_kwargs is None:
            init_const_kwargs = dict()

        self.init_requires      = init_requires
        self.init_parsed_kwargs = init_parsed_kwargs
        self.init_const_kwargs  = init_const_kwargs

    def update_kwargs(
        self,
        kwargs: dict,
        parsed_args_dict: dict,
        extra_kwargs: dict,
    ):
        for k, v in self.init_parsed_kwargs.items():
            kwargs[k] = parsed_args_dict[v]
        for k, v in self.init_const_kwargs.items():
            kwargs[k] = v
        for k, v in extra_kwargs.items():
            kwargs[k] = v

        missing_keys = set(self.init_requires) - set(kwargs)
        if len(missing_keys) > 0:
            raise ValueError(
                "Please provide values for the following keys %s for \"%s\""
                % (sorted(missing_keys), self.full_name)
            )

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        for arg in self.args:
            arg.add_argparse_arguments(parser)

    def init_object(
        self,
        parser: "ObjectParser",
        parsed_args_dict: dict,
        extra_kwargs: dict,
    ):
        for arg in self.args:
            if arg.full_name not in parsed_args_dict:
                parser.init_object(arg.full_name)

        kwargs = {
            arg.name: parsed_args_dict[arg.full_name]
            for arg in self.args
        }
        self.update_kwargs(kwargs, parsed_args_dict, extra_kwargs)
        return self.object_type(**kwargs)

    def __repr__(self):
        description = ",\n".join(
            ["full_name=\"%s\"" % self.full_name]
            + ["full_tag=\"%s\"" % self.full_tag]
            + [repr(arg) for arg in self.args]
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
        self._arg_list = args
        self._parser_kwargs = parser_kwargs
        self._parsed_args_dict = None
        self._arg_dict: dict[str, Arg] = dict()
        for arg in self._arg_list:
            arg.register_names(self._arg_dict)

    def _get_argparse_parser(self):
        parser = argparse.ArgumentParser(**self._parser_kwargs)
        for arg in self._arg_list:
            arg.add_argparse_arguments(parser)

        return parser

    def _check_parsed(self):
        if self._parsed_args_dict is None:
            raise RuntimeError(
                "Must call `parse_args` before calling this method"
            )

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
        self._parsed_args_dict = vars(args)
        return args

    def get_args_summary(self, replaces=None):
        self._check_parsed()
        key_abbreviations = {
            arg.full_name: arg.full_tag
            for arg in self._arg_dict.values()
            if arg.full_tag is not None
            and not isinstance(arg, ObjectArg)
        }
        return util.abbreviate_dictionary(
            self._parsed_args_dict,
            key_abbreviations=key_abbreviations,
            replaces=replaces,
        )

    def init_object(self, full_name, **extra_kwargs):
        self._check_parsed()
        if full_name not in self._arg_dict:
            raise ValueError(
                "\"%s\" not in %s"
                % (full_name, sorted(self._arg_dict.keys()))
            )

        object_arg: ObjectArg = self._arg_dict[full_name]
        object_value = object_arg.init_object(
            self,
            self._parsed_args_dict,
            extra_kwargs,
        )
        self._parsed_args_dict[object_arg.full_name] = object_value
        return object_value

    def __repr__(self):
        description = ",\n".join(repr(arg) for arg in self._arg_list)
        return "%s(\n%s,\n)" % (type(self).__name__, indent(description))

def indent(input_str, num_spaces=4):
    return textwrap.indent(input_str, " " * num_spaces)
