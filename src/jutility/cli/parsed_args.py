from jutility.cli.arg import Arg
from jutility.cli.parent import _ArgParent
from jutility.cli.unknown import _UnknownArg
from jutility.cli.subcmd.subcommand import SubCommand
from jutility.cli.subcmd.no_group import _NoSubCommandGroup
from jutility.cli.subcmd.group import SubCommandGroup

class ParsedArgs(_ArgParent):
    def __init__(
        self,
        arg_list:       list[Arg],
        arg_dict:       dict[str, Arg],
        sub_commands:   SubCommandGroup | _NoSubCommandGroup,
    ):
        self._init_arg_parent(arg_list, dict())
        self._arg_dict = arg_dict
        self._sub_commands = sub_commands
        self.reset_object_cache()

    def get_arg(self, arg_name: str) -> Arg:
        return self._arg_dict[arg_name]

    def get_value(self, arg_name: str):
        return self._arg_dict[arg_name].value

    def get_command(self) -> "SubCommand | None":
        return self._sub_commands.get_command()

    def update(self, value_dict: dict, allow_new_keys=False):
        for name, value in value_dict.items():
            if name in self._arg_dict:
                self._arg_dict[name].value = value
            elif allow_new_keys:
                new_arg = _UnknownArg(name, value)
                self._arg_dict[name] = new_arg
                self._arg_list.append(new_arg)
            else:
                new_keys = set(value_dict.keys()) - set(self._arg_dict.keys())
                raise ValueError(
                    "Received extra keys %s. Either remove these keys, or "
                    "call `ParsedArgs.update(..., allow_new_keys=True)`."
                    % sorted(new_keys)
                )

        self.reset_object_cache()

    def init_object(self, full_name: str, **extra_kwargs):
        return self._arg_dict[full_name].init_object(**extra_kwargs)

    def get_type(self, full_name: str) -> type:
        return self._arg_dict[full_name].get_type()

    def reset_object_cache(self):
        for arg in self._arg_dict.values():
            arg.reset_object_cache()
