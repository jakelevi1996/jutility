import argparse
from jutility import util

def init_object(args: "Namespace", full_name, **extra_kwargs):
    return args.get_parser().init_object(full_name, **extra_kwargs)

def reset_object_cache(args: "Namespace"):
    return args.get_parser().reset_object_cache()

def get_arg_dict(args: "Namespace"):
    return args.get_parser().get_arg_dict()

def get_args_summary(args: "Namespace", replaces=None):
    return args.get_parser().get_args_summary(replaces)

class Arg:
    def __init__(
        self,
        name: str,
        tag: str=None,
        **argparse_kwargs,
    ):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument
        """
        self.argparse_kwargs = argparse_kwargs
        self.args: list[Arg] = []
        self.init_names(name, tag)

    def init_names(self, name: str, tag: str):
        self.name = name
        self.tag  = tag
        self.full_name: str = None
        self.full_tag:  str = None

    def register_names(self, arg_dict, parent: "Arg"=None):
        if self.full_name is not None:
            raise RuntimeError("%s is already registered" % self)

        if parent is None:
            self.full_name = self.name
            self.full_tag = self.tag
        else:
            self.full_name = "%s.%s" % (parent.full_name, self.name)
            if parent.full_tag is not None:
                if parent.hide_tag(self):
                    self.full_tag = parent.full_tag
                elif self.tag is not None:
                    self.full_tag = "%s.%s" % (parent.full_tag, self.tag)

        if self.full_name in arg_dict:
            raise ValueError(
                "Found duplicates %s and %s"
                % (self, arg_dict[self.full_name])
            )

        arg_dict[self.full_name] = self

        for arg in self.args:
            arg.register_names(arg_dict, self)

    def hide_tag(self, arg: "Arg"):
        return False

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--" + self.full_name, **self.argparse_kwargs)

    def init_object(self, parsed_args_dict):
        return parsed_args_dict[self.full_name]

    def get_arg_dict_keys(self, parsed_args_dict):
        return [self.full_name]

    def __repr__(self):
        return (
            "%s(name=\"%s\", full_name=\"%s\", full_tag=\"%s\")"
            % (type(self).__name__, self.name, self.full_name, self.full_tag)
        )

class ObjectArg(Arg):
    def __init__(
        self,
        object_type: type,
        *args: Arg,
        name: str=None,
        tag: str=None,
        init_requires: list[str]=None,
        init_parsed_kwargs: dict=None,
        init_const_kwargs: dict=None,
        init_ignores: list[str]=None,
    ):
        if name is None:
            name = object_type.__name__

        self.object_type = object_type
        self.args        = args
        self.init_names(name, tag)
        self.set_init_attributes(
            init_requires,
            init_parsed_kwargs,
            init_const_kwargs,
            init_ignores,
        )

    def set_init_attributes(
        self,
        init_requires,
        init_parsed_kwargs,
        init_const_kwargs,
        init_ignores,
    ):
        if init_requires is None:
            init_requires = []
        if init_parsed_kwargs is None:
            init_parsed_kwargs = dict()
        if init_const_kwargs is None:
            init_const_kwargs = dict()
        if init_ignores is None:
            init_ignores = []

        self.init_requires      = init_requires
        self.init_parsed_kwargs = init_parsed_kwargs
        self.init_const_kwargs  = init_const_kwargs
        self.init_ignores  = init_ignores

    def get_protected_args(self):
        return (
            set(arg.name for arg in self.args)
            | set(self.init_parsed_kwargs.keys())
            | set(self.init_const_kwargs.keys())
            | set(self.init_ignores)
        )

    def update_kwargs(
        self,
        kwargs: dict,
        parsed_args_dict: dict,
        extra_kwargs: dict,
        protected: set=None,
    ):
        if protected is None:
            protected = set()
        for k, v in self.init_parsed_kwargs.items():
            if k not in protected:
                kwargs[k] = parsed_args_dict[v]
        for k, v in self.init_const_kwargs.items():
            if k not in protected:
                kwargs[k] = v
        for k, v in extra_kwargs.items():
            kwargs[k] = v

        missing_keys = set(self.init_requires) - (set(kwargs) | protected)
        if len(missing_keys) > 0:
            raise ValueError(
                "Please provide values for the following keys %s for \"%s\""
                % (sorted(missing_keys), self.full_name)
            )

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        for arg in self.args:
            arg.add_argparse_arguments(parser)

    def init_object(self, parsed_args_dict, **extra_kwargs):
        if self.full_name in parsed_args_dict:
            if verbose.is_verbose:
                verbose.display_retrieve(self.full_name)

            return parsed_args_dict[self.full_name]

        kwargs = {
            arg.name: arg.init_object(parsed_args_dict)
            for arg in self.args
        }
        self.update_kwargs(kwargs, parsed_args_dict, extra_kwargs)

        if verbose.is_verbose:
            verbose.display_init(self.object_type, kwargs)

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
        name: str,
        *choices: ObjectArg,
        shared_args: list[Arg]=None,
        default: str=None,
        tag: str=None,
        init_requires: list[str]=None,
        init_parsed_kwargs: dict=None,
        init_const_kwargs: dict=None,
        init_ignores: list[str]=None,
    ):
        if shared_args is None:
            shared_args = []

        self.shared_args    = shared_args
        self.default        = default
        self.init_names(name, tag)
        self.set_init_attributes(
            init_requires,
            init_parsed_kwargs,
            init_const_kwargs,
            init_ignores,
        )

        self.args = tuple(choices) + tuple(shared_args)
        self.choice_dict = {arg.name: arg for arg in choices}
        if (default is not None) and (default not in self.choice_dict):
            valid_names = [arg.name for arg in choices]
            raise ValueError(
                "%s(\"%s\") received `default=\"%s\"`, please choose from %s"
                % (type(self).__name__, name, default, valid_names)
            )

    def hide_tag(self, arg: Arg):
        return (arg.name in self.choice_dict)

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--" + self.full_name,
            choices=list(self.choice_dict.keys()),
            default=self.default,
            required=(True if (self.default is None) else False),
        )
        for arg in self.args:
            arg.add_argparse_arguments(parser)

    def init_object(self, parsed_args_dict, **extra_kwargs):
        chosen_arg = self.choice_dict[parsed_args_dict[self.full_name]]
        protected = chosen_arg.get_protected_args()
        kwargs = {
            arg.name: arg.init_object(parsed_args_dict)
            for arg in self.shared_args
            if arg.name not in protected
        }
        self.update_kwargs(kwargs, parsed_args_dict, extra_kwargs, protected)
        return chosen_arg.init_object(parsed_args_dict, **kwargs)

    def get_arg_dict_keys(self, parsed_args_dict):
        chosen_arg = self.choice_dict[parsed_args_dict[self.full_name]]
        protected = chosen_arg.get_protected_args()
        return (
            [self.full_name]
            + chosen_arg.get_arg_dict_keys(parsed_args_dict)
            + [
                name
                for arg in self.shared_args
                if  arg.name not in protected
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
        self._parser_kwargs = parser_kwargs
        self._arg_list: list[Arg] = []
        self._arg_dict: dict[str, Arg] = dict()
        self._parsed_args_dict: dict = None
        self.add_arguments(*args)

    def add_arguments(self, *args: Arg):
        for arg in args:
            self._arg_list.append(arg)
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
        kwargs.setdefault("namespace", Namespace())
        args = parser.parse_args(*args, **kwargs)
        assert isinstance(args, Namespace)
        args.set_parser(self)
        self._parsed_args_dict = vars(args)
        self._initial_args_cache = set(self._parsed_args_dict.keys())
        return args

    def init_object(self, full_name, **extra_kwargs):
        self._check_parsed()
        if full_name not in self._arg_dict:
            raise ValueError(
                "\"%s\" not in %s"
                % (full_name, sorted(self._arg_dict.keys()))
            )

        object_arg = self._arg_dict[full_name]
        return object_arg.init_object(self._parsed_args_dict, **extra_kwargs)

    def get_arg_dict(self):
        self._check_parsed()
        return {
            key: self._parsed_args_dict[key]
            for arg in self._arg_list
            for key in arg.get_arg_dict_keys(self._parsed_args_dict)
        }

    def get_args_summary(self, replaces=None):
        self._check_parsed()
        arg_list = [
            self._arg_dict[arg_name]
            for arg_name in self.get_arg_dict().keys()
        ]
        return util.abbreviate_dictionary(
            self._parsed_args_dict,
            key_abbreviations={
                arg.full_name: arg.full_tag
                for arg in arg_list
                if arg.full_tag is not None
            },
            replaces=replaces,
        )

    def reset_object_cache(self):
        self._check_parsed()
        current_args_cache = set(self._parsed_args_dict.keys())
        for k in (current_args_cache - self._initial_args_cache):
            self._parsed_args_dict.pop(k)

    def __repr__(self):
        description = ",\n".join(repr(arg) for arg in self._arg_dict.values())
        return "%s(\n%s,\n)" % (type(self).__name__, util.indent(description))

class Namespace(argparse.Namespace):
    def __init__(self):
        self._parser = None

    def set_parser(self, parser: ObjectParser):
        if self._parser is not None:
            raise RuntimeError("This Namespace already has a parser")

        self._parser = parser

    def get_parser(self):
        return self._parser

    def get(self, key):
        return vars(self).get(key)

    def update(self, arg_dict):
        self._parser.reset_object_cache()
        return vars(self).update(arg_dict)

    def __str__(self):
        return " ".join(
            "--%s %r" % (k, v)
            for k, v in sorted(self._parser.get_arg_dict().items())
        )

    def __repr__(self):
        arg_str = ", ".join(
            "%s=%r" % (k, v)
            for k, v in sorted(self._parser.get_arg_dict().items())
        )
        return "%s(%s)" % (type(self).__name__, arg_str)

class _Verbose:
    def __init__(self):
        self.is_verbose = False
        self._printer = util.Printer()

    def display_init(self, object_type, kwargs: dict):
        arg_str = ", ".join("%s=%r" % (k, v) for k, v in kwargs.items())
        self._printer("cli: %s(%s)" % (object_type.__name__, arg_str))

    def display_retrieve(self, full_name):
        self._printer("cli: retrieving \"%s\" from cache" % full_name)

    def set_printer(self, printer: util.Printer):
        self._printer = printer

    def __enter__(self):
        self.is_verbose = True

    def __exit__(self, *args):
        self.is_verbose = False

verbose = _Verbose()
