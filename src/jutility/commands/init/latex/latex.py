from jutility import cli, util
from jutility.commands.command import Command
from jutility.commands.init.latex.doc import Doc

class Latex(Command):
    def run(
        self,
    ):
        command = self.get_command()
        with util.Timer(repr(command)):
            command.run(**command.get_kwargs())

    @classmethod
    def get_cli_args(cls) -> list[cli.Arg]:
        return []

    @classmethod
    def get_subcommands(cls) -> list[type[Command]]:
        return [
            Doc,
        ]
