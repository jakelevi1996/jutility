import os
import urllib.request
import http.client
from jutility import cli
from jutility.commands.command import Command
from jutility.commands.init.from_dict import init_from_dict

class VsCode(Command):
    def run(
        self,
        force: bool,
    ):
        base_url = (
            "https://raw.githubusercontent.com/jakelevi1996/notes/refs/heads"
            "/main/topics/vscode/"
        )
        base_path = os.path.expanduser("~/.config/Code/User/")
        tails = [
            "snippets/python.json",
            "snippets/markdown.json",
            "snippets/latex.json",
            "keybindings.json",
            "settings.json",
        ]
        path_contents_dict = dict()

        for t in tails:
            url = base_url + t
            full_path = base_path + t

            print("Downloading \"%s\"" % url)
            with urllib.request.urlopen(url) as f:
                assert isinstance(f, http.client.HTTPResponse)
                path_contents_dict[full_path] = f.read().decode()

        init_from_dict(path_contents_dict, force)

    @classmethod
    def get_cli_args(cls) -> list[cli.Arg]:
        return [
            cli.Arg("force", action="store_true"),
        ]
