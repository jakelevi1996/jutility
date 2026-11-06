from jutility import cli, util
from jutility.commands.command import Command

class ImageDiff(Command):
    def run(
        self,
        full_path_1: str,
        full_path_2: str,
        output_name: str,
    ):
        util.save_image_diff(full_path_1, full_path_2, output_name)

    @classmethod
    def get_cli_args(cls) -> list[cli.Arg]:
        return [
            cli.PositionalArg("full_path_1", type=str),
            cli.PositionalArg("full_path_2", type=str),
            cli.PositionalArg("output_name", type=str),
        ]
