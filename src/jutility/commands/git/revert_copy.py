from jutility import cli, util
from jutility.commands.command import Command

class RevertCopy(Command):
    def run(
        self,
        input_path: str,
    ):
        temp_path   = util.add_filename_suffix(input_path, "_temp")
        revert_path = util.add_filename_suffix(input_path, "_revert")
        util.run_subprocess(["cp", input_path, temp_path])
        util.run_subprocess("git checkout --".split() + [input_path])
        util.run_subprocess(["mv", input_path, revert_path])
        util.run_subprocess(["mv", temp_path, input_path])
        util.run_subprocess(["code", input_path])
        util.run_subprocess(["code", revert_path])

    @classmethod
    def get_cli_args(cls) -> list[cli.Arg]:
        return [
            cli.PositionalArg("input_path", metavar="input_path", help=""),
        ]
