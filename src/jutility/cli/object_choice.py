import argparse
from jutility import util
from jutility.cli.arg import Arg
from jutility.cli.object_arg import ObjectArg

class ObjectChoice(ObjectArg):
    def __init__(
        self,
        name:               str,
        *choices:           ObjectArg,
        default:            (str | None)=None,
        tag:                (str | None)=None,
        tagged:             bool=True,
        is_group:           bool=False,
        is_kwarg:           bool=False,
        shared_args:        (list[Arg] | None)=None,
        init_requires:      (list[str] | None)=None,
        init_ignores:       (list[str] | None)=None,
        init_const_kwargs:  (dict | None)=None,
        required:           (bool | None)=None,
    ):
        if shared_args is None:
            shared_args = []

        self.shared_args = shared_args
        self._init_arg_parent(list(choices) + shared_args, dict())
        self._init_arg(name, tag, tagged, is_kwarg)
        self._init_object_arg(
            is_group,
            init_requires,
            init_ignores,
            init_const_kwargs,
        )

        self.choice_dict = {arg.name: arg for arg in choices}
        if (default is not None) and (default not in self.choice_dict):
            raise ValueError(
                "%s received `default=\"%s\"`, please choose from %s"
                % (self, default, sorted(arg.name for arg in choices))
            )
        if required is None:
            required = (True if (default is None) else False)

        self.default    = default
        self.required   = required

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        input_parser = parser
        if self.is_group:
            parser = input_parser.add_argument_group(self.full_name)

        parser.add_argument(
            "--" + self.full_name,
            choices=list(self.choice_dict.keys()),
            default=self.default,
            help="default=%s, required=%s" % (self.default, self.required),
        )
        for arg in self.shared_args:
            arg.add_argparse_arguments(parser)

        for arg in self.choice_dict.values():
            if self.is_group:
                parser = input_parser.add_argument_group(arg.full_name)

            arg.add_argparse_arguments(parser)

    def set_default_choice(self, choice: type | None) -> None:
        if choice is None:
            return
        if choice.__name__ not in self.choice_dict:
            raise ValueError(
                "%r not in %r"
                % (choice.__name__, sorted(self.choice_dict.keys()))
            )
        if self.value is None:
            self.value = choice.__name__

    def init_object(self, **extra_kwargs):
        chosen_arg = self.get_choice()
        protected = chosen_arg.get_protected_args()
        kwargs = {
            arg.name: arg.init_object()
            for arg in self.shared_args
            if arg.name not in protected
        }
        for k, v in self.init_const_kwargs.items():
            if k not in protected:
                kwargs[k] = v

        kwargs.update(extra_kwargs)
        self.check_missing(set(kwargs.keys()) | protected)
        return chosen_arg.init_object(**kwargs)

    def get_choice(self) -> ObjectArg:
        if self.value is None:
            raise ValueError(
                "Please specify --%s %s"
                % (self.full_name, sorted(self.choice_dict.keys()))
            )

        return self.choice_dict[self.value]

    def get_type(self) -> type:
        return self.get_choice().get_type()

    def reset_object_cache(self):
        return

    def get_value_summary(self) -> str:
        chosen_arg = self.get_choice()
        clean = {
            s: s.upper().replace("_", "")
            for s in self.choice_dict.keys()
        }
        tags = {
            clean[arg.name]: arg.tag
            for arg in self.choice_dict.values()
            if arg.tag is not None
        }
        prefix = util.get_unique_prefixes(
            [s for s in clean.values() if s not in tags],
            forbidden=set(tags.values())
        )
        prefix.update(tags)
        return prefix[clean[chosen_arg.name]]

    def store_value(self, value_dict: dict, summarise: bool):
        if summarise:
            value_dict[self.full_name] = self.get_value_summary()
        else:
            value_dict[self.full_name] = self.value

    def _hide_tag(self, arg: "Arg") -> bool:
        return (arg.name in self.choice_dict)

    def _get_active_args(self) -> list["Arg"]:
        chosen_arg = self.get_choice()
        protected = chosen_arg.get_protected_args()
        return [
            chosen_arg,
            *[
                arg
                for arg in self.shared_args
                if  arg.name not in protected
            ],
        ]
