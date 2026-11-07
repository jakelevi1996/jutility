from jutility import cli, util
from jutility.commands.command import Command
from jutility.commands.init.init import Init
from jutility.commands.git.git import Git
from jutility.commands.image_diff import ImageDiff

class EntryPoint:
    @classmethod
    def get_commands(cls) -> list[type[Command]]:
        return [
            Init,
            Git,
            ImageDiff,
        ]

    @classmethod
    def get_parser(cls):
        return cli.Parser(
            sub_commands=cli.SubCommandGroup(
                *[
                    command_type.init_command()
                    for command_type in cls.get_commands()
                ],
            ),
        )

    @classmethod
    def run(cls, *parser_args, **parser_kwargs):
        parser = cls.get_parser()
        args = parser.parse_args(*parser_args, **parser_kwargs)
        command = args.get_command()
        with util.Timer(repr(command)):
            command.run(**command.get_kwargs())
