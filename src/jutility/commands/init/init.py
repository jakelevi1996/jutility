from jutility.commands.command import Command
from jutility.commands.parent import Parent
from jutility.commands.init.repo import Repo
from jutility.commands.init.latex.latex import Latex

class Init(Parent):
    @classmethod
    def get_subcommands(cls) -> list[type[Command]]:
        return [
            Repo,
            Latex,
        ]
