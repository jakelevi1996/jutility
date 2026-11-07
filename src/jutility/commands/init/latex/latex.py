from jutility.commands.command import Command
from jutility.commands.parent import Parent
from jutility.commands.init.latex.doc import Doc

class Latex(Parent):
    @classmethod
    def get_subcommands(cls) -> list[type[Command]]:
        return [
            Doc,
        ]
