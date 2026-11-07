from jutility import cli, util
from jutility.commands.command import Command

class AddReadmeImages(Command):
    def run(
        self,
    ):
        readme_str = util.load_text("README.md")
        img_paths = util.extract_all_substrings(readme_str, "![](", ")")
        img_paths = [p for p in img_paths if ("https://" not in p)]
        git_add_cmd = "git add -f".split() + img_paths
        util.run_subprocess(git_add_cmd)

    @classmethod
    def get_cli_args(cls) -> list[cli.Arg]:
        return []
