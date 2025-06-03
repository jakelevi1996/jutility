from jutility.cli.parent import _ArgParent
import jutility.cli.subcmd.group as group
import jutility.cli.subcmd.no_group as no_group

class _SubCommandParent(_ArgParent):
    def _init_subcommand_parent(
        self,
        sub_commands: "group.SubCommandGroup | None",
    ):
        if sub_commands is None:
            sub_commands = no_group._NoSubCommandGroup()

        self._sub_commands  = sub_commands
        self._arg_dict      = self.register_names(dict(), "")
