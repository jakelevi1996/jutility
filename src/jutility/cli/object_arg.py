import argparse
from jutility.cli.arg import Arg
from jutility.cli.verbose import verbose

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
        init_requires:      (list[str] | None)=None,
        init_ignores:       (list[str] | None)=None,
        init_const_kwargs:  (dict | None)=None,
    ):
        if name is None:
            name = object_type.__name__

        self.object_type = object_type
        self._init_arg_parent(list(args), dict())
        self._init_arg(name, tag, tagged, is_kwarg)
        self._init_object_arg(
            is_group,
            init_requires,
            init_ignores,
            init_const_kwargs,
        )

    def _init_object_arg(
        self,
        is_group: bool,
        init_requires: (list[str] | None),
        init_ignores:  (list[str] | None),
        init_const_kwargs:  (dict | None),
    ):
        if init_requires is None:
            init_requires = []
        if init_ignores is None:
            init_ignores = []
        if init_const_kwargs is None:
            init_const_kwargs = dict()

        self.is_group = is_group
        self.init_requires = init_requires
        self.init_ignores  = init_ignores
        self.init_const_kwargs  = init_const_kwargs

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        if self.is_group:
            parser = parser.add_argument_group(self.full_name)

        for arg in self._arg_list:
            arg.add_argparse_arguments(parser)

    def get_protected_args(self) -> set[str]:
        protected_arg_list = [
            *[arg.name for arg in self._arg_list],
            *self.init_ignores,
            *self.init_const_kwargs.keys(),
        ]
        return set(protected_arg_list)

    def check_missing(self, kwarg_names: set[str]):
        missing_keys = set(self.init_requires) - kwarg_names
        if len(missing_keys) > 0:
            raise ValueError(
                "Please provide values for the following keys %s for \"%s\""
                % (sorted(missing_keys), self.full_name)
            )

    def init_object(self, **extra_kwargs):
        if self.value is not None:
            if verbose:
                verbose.display_retrieve(self.full_name)

            return self.value

        kwargs = {
            arg.name: arg.init_object()
            for arg in self._arg_list
        }
        kwargs.update(self.init_const_kwargs)
        kwargs.update(extra_kwargs)
        self.check_missing(set(kwargs.keys()))

        if verbose:
            verbose.display_init(self.object_type, kwargs)

        self.value = self.object_type(**kwargs)
        return self.value

    def get_type(self) -> type:
        return self.object_type

    def reset_object_cache(self):
        self.value = None

    def store_value(self, value_dict: dict, summarise: bool):
        return
