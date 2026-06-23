from jutility.cli.arg import Arg
from jutility.cli.root import ArgRoot
from jutility.cli.subcmd.group import SubCommandGroup

class ParsedArgs(ArgRoot):
    def __init__(
        self,
        arg_list:       list[Arg],
        arg_dict:       dict[str, Arg],
        sub_commands:   SubCommandGroup,
    ):
        self._init_arg_parent(arg_list, dict())
        self._init_arg_root(arg_dict, sub_commands)
        self.reset_object_cache()
