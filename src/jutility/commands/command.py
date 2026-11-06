from jutility import cli, util

class Command(cli.SubCommand):
    def run(self, **kwargs):
        raise NotImplementedError()

    @classmethod
    def get_cli_args(cls) -> list[cli.Arg]:
        raise NotImplementedError()

    @classmethod
    def get_subcommands(cls) -> list[type["Command"]]:
        return []

    @classmethod
    def init_command(cls) -> "Command":
        return cls(
            cls.get_name(),
            *cls.get_cli_args(),
            sub_commands=cli.SubCommandGroup(
                *[c.init_command() for c in cls.get_subcommands()],
            ),
        )

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

    def __repr__(self) -> str:
        return self.get_name()
