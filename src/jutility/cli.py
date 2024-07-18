import argparse
import textwrap
from jutility import util

def get_arg_dict(args):
    object_parser: ObjectParser = args.object_parser
    return object_parser.get_arg_dict()

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

    def get_arg_dict_keys(self, parsed_args_dict):
        return [self.full_name]

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
        object_value = self.object_type(**kwargs)
        parsed_args_dict[self.full_name] = object_value
        return object_value

    def get_arg_dict_keys(self, parsed_args_dict):
        return [
            name
            for arg in self.args
            for name in arg.get_arg_dict_keys(parsed_args_dict)
        ]

class ObjectChoice(ObjectArg):
    def __init__(
        self,
        name,
        *choices: ObjectArg,
        shared_args: list[Arg]=None,
        default=None,
        abbreviation: str=None,
        init_requires: list[str]=None,
        init_parsed_kwargs: dict[str, str]=None,
        init_const_kwargs:  dict[str, str]=None,
    ):
        if shared_args is None:
            shared_args = []

        self.name           = name
        self.choices        = choices
        self.shared_args    = shared_args
        self.default        = default
        self.tag            = abbreviation
        self.set_init_attributes(
            init_requires,
            init_parsed_kwargs,
            init_const_kwargs,
        )

        self.args = tuple(choices) + tuple(shared_args)
        self.choice_dict = {arg.name: arg for arg in choices}
        if (default is not None) and (default not in self.choice_dict):
            valid_names = [arg.name for arg in choices]
            raise ValueError(
                "%s(\"%s\") received `default=\"%s\"`, please choose from %s"
                % (type(self).__name__, name, default, valid_names)
            )

    def _get_relevant_shared_args(self, object_arg):
        protected_args = (
            set(arg.name for arg in object_arg.args)
            | set(object_arg.init_parsed_kwargs.keys())
            | set(object_arg.init_const_kwargs.keys())
        )
        return [
            arg for arg in self.shared_args
            if arg.name not in protected_args
        ]

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--" + self.full_name,
            choices=[arg.name for arg in self.choices],
            default=self.default,
            required=(True if (self.default is None) else False),
        )
        for arg in self.args:
            arg.add_argparse_arguments(parser)

    def init_object(
        self,
        parser: "ObjectParser",
        parsed_args_dict: dict,
        extra_kwargs: dict,
    ):
        object_name = parsed_args_dict[self.full_name]
        object_arg = self.choice_dict[object_name]
        relevant_shared_args = self._get_relevant_shared_args(object_arg)
        for arg in relevant_shared_args:
            if arg.full_name not in parsed_args_dict:
                parser.init_object(arg.full_name)

        kwargs = {
            arg.name: parsed_args_dict[arg.full_name]
            for arg in relevant_shared_args
        }
        self.update_kwargs(kwargs, parsed_args_dict, extra_kwargs)
        object_value  = parser.init_object(object_arg.full_name, **kwargs)
        return object_value

    def get_arg_dict_keys(self, parsed_args_dict):
        object_name = parsed_args_dict[self.full_name]
        object_arg = self.choice_dict[object_name]
        return (
            [self.full_name]
            + object_arg.get_arg_dict_keys(parsed_args_dict)
            + [
                name
                for arg in self._get_relevant_shared_args(object_arg)
                for name in arg.get_arg_dict_keys(parsed_args_dict)
            ]
        )

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
        return object_value

    def get_arg_dict(self):
        self._check_parsed()
        arg_dict_keys = [
            key
            for arg in self._arg_list
            for key in arg.get_arg_dict_keys(self._parsed_args_dict)
        ]
        return {
            key: self._parsed_args_dict[key]
            for key in arg_dict_keys
        }

    def get_args_summary(self, replaces=None):
        self._check_parsed()
        arg_names = self.get_arg_dict().keys()
        arg_list = [self._arg_dict[arg_name] for arg_name in arg_names]
        key_abbreviations = {
            arg.full_name: arg.full_tag
            for arg in arg_list
        }
        return util.abbreviate_dictionary(
            self._parsed_args_dict,
            key_abbreviations=key_abbreviations,
            replaces=replaces,
        )

    def __repr__(self):
        description = ",\n".join(repr(arg) for arg in self._arg_dict.values())
        return "%s(\n%s,\n)" % (type(self).__name__, indent(description))

def indent(input_str, num_spaces=4):
    return textwrap.indent(input_str, " " * num_spaces)
