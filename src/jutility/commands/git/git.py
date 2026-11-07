from jutility.commands.command import Command
from jutility.commands.parent import Parent
from jutility.commands.git.add_readme_images import AddReadmeImages
from jutility.commands.git.revert_copy import RevertCopy

class Git(Parent):
    @classmethod
    def get_subcommands(cls) -> list[type[Command]]:
        return [
            AddReadmeImages,
            RevertCopy,
        ]
