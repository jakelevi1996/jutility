import argparse
import json
from jutility import util
import jutility.cli.parent as parent

class Arg(parent._ArgParent):
    def __init__(
        self,
        name:       str,
        tag:        (str | None)=None,
        tagged:     bool=True,
        is_kwarg:   bool=True,
        **argparse_kwargs,
    ):
        """
        See
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.add_argument
        """
        self._init_arg_parent([], argparse_kwargs)
        self._init_arg(name, tag, tagged, is_kwarg)
        if "action" not in self._kwargs:
            self._kwargs.setdefault("metavar", self.name[0].upper())

    def _init_arg(
        self,
        name:       str,
        tag:        str | None,
        tagged:     bool,
        is_kwarg:   bool,
    ):
        self.name       = name
        self.full_name  = None
        self.is_kwarg   = is_kwarg
        self._init_value()
        self._init_tag(tag, tagged)
        self._init_help()

    def _init_value(self, value=None):
        self.value = value

    def _init_tag(self, tag: str | None, tagged: bool):
        self.tag    = tag
        self.tagged = tagged

    def _init_help(self):
        if ((len(self._kwargs) > 0) and ("help" not in self._kwargs)):
            self._kwargs["help"] = util.format_dict(self._kwargs)

    def set_full_name(self, full_name: str):
        if self.full_name is not None:
            raise RuntimeError("%s is already registered" % self)

        self.full_name = full_name

    def store_name(self, arg_dict: dict[str, "Arg"]):
        if self.full_name in arg_dict:
            raise ValueError(
                "Found duplicates %s and %s"
                % (self, arg_dict[self.full_name])
            )

        arg_dict[self.full_name] = self

    def store_value(self, value_dict: dict, summarise: bool):
        value_dict[self.full_name] = self.value

    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument("--" + self.full_name, **self._kwargs)

    def set_default_choice(self, choice: type | None) -> None:
        raise ValueError(
            "`set_default_choice` only valid for instances of "
            "`ObjectChoice`, not %r"
            % self
        )

    def init_object(self):
        return self.value

    def get_type(self) -> type:
        return type(self.value)

    def get_value_summary(self) -> str:
        return repr(self.value)

    def reset_object_cache(self):
        return

    def __repr__(self):
        return util.format_type(
            type(self),
            name=self.name,
            full_name=self.full_name,
            value=self.value,
        )

class PositionalArg(Arg):
    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(self.full_name, **self._kwargs)

class NoTagArg(Arg):
    def _init_tag(self, tag: str | None, tagged: bool):
        self.tag = None
        self.tagged = False

class BooleanArg(Arg):
    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--" + self.full_name,
            action=argparse.BooleanOptionalAction,
            **self._kwargs,
        )

    def _init_help(self):
        if "default" in self._kwargs:
            self._kwargs.setdefault("help", "")
        else:
            if "help" in self._kwargs:
                self._kwargs["help"] += " (default: None)"
            else:
                self._kwargs["help"] = "(default: None)"

class JsonArg(Arg):
    def add_argparse_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--" + self.full_name,
            type=json.loads,
            **self._kwargs,
        )

    def _init_help(self):
        if ((len(self._kwargs) > 0) and ("help" not in self._kwargs)):
            self._kwargs["help"] = util.format_dict(self._kwargs)
        if "help" in self._kwargs:
            self._kwargs["help"] += " (format: JSON string)"
        else:
            self._kwargs["help"] = "Format: JSON string"
