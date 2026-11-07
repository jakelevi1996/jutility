from jutility import cli, util
from jutility.commands.command import Command
from jutility.commands.init.from_dict import init_from_dict

class Repo(Command):
    def run(
        self,
    ):
        path_contents_dict = {
            ".temp.py":     TEMP_PY,
            ".temp.txt":    EMPTY,
            ".temp.sh":     EMPTY,
            "README.md":    EMPTY,
            ".gitignore":   GITIGNORE,
        }
        init_from_dict(path_contents_dict)
        util.run_subprocess("git add -f .gitignore".split())

    @classmethod
    def get_cli_args(cls) -> list[cli.Arg]:
        return []

EMPTY = ""

TEMP_PY = """from jutility import plotting, util, cli

util.numpy_set_print_options()
printer = util.Printer(
    # ".temp",
    # ".",
    # print_to_console=False,
)

###############################################################################

import torch
import juml

juml.util.torch_set_print_options(
    # threshold=int(1e9),
)
torch.manual_seed(0)

print_tensor = juml.util.TensorPrinter(printer)

###############################################################################

"""

GITIGNORE = """.*

*.txt
*.png
*.json
*.sh
*.md
*.pkl
*.gif
*.pyc
*.egg-info

__pycache__

dist/*

*.pdf
*.aux
*.fdb_latexmk
*.fls
*.log
*.gz
*.bbl
*.blg
*.out
*.synctex(busy)
*.cut
*.brf
*.toc
"""
