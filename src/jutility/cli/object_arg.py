import argparse
from jutility import util
from jutility.cli.arg import Arg

class ObjectArg(Arg):
    def __init__(
        self,
        object_type:        type,
        *args:              Arg,
        name:               (str | None)=None,
        tag:                (str | None)=None,
        tagged:             bool=True,
        is_group:           bool=False,
        is_kwarg:           bool=False,
    ):
        if name is None:
            name = object_type.__name__

        self.object_type = object_type
        self.is_group = is_group
        self._init_arg_parent(list(args), dict())
        self._init_arg(name, tag, tagged, is_kwarg)

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        if self.is_group:
            parser = parser.add_argument_group(self.full_name)

        for arg in self._arg_list:
            arg.add_argparse_arguments(parser)

    def _has_parsed_value(self) -> bool:
        return False

    def init_object(self, printer: (util.Printer | None), **extra_kwargs):
        if self.value is not None:
            if printer is not None:
                printer("cli: `%s` retrieved from cache" % self.full_name)

            return self.value

        kwargs = {
            arg.name: arg.init_object(printer)
            for arg in self._arg_list
        }
        kwargs.update(extra_kwargs)

        if printer is not None:
            printer("cli: %s" % util.format_type(self.object_type, **kwargs))

        self.value = self.object_type(**kwargs)
        return self.value

    def get_type(self) -> type:
        return self.object_type

    def reset_object_cache(self):
        self.value = None
