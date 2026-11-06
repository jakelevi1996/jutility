import os
from jutility import cli, util
from jutility.commands.command import Command

class Repo(Command):
    def run(
        self,
    ):
        name_contents_dict = {
            ".temp.py":     TEMP_PY,
            ".temp.txt":    EMPTY,
            ".temp.sh":     EMPTY,
            "README.md":    EMPTY,
            ".gitignore":   GITIGNORE,
        }

        for name, content in name_contents_dict.items():
            if not os.path.isfile(name):
                util.save_text(content, name, dir_name=".", file_ext=None)

        util.run_subprocess("git add -f .gitignore".split())
        util.run_subprocess(["code"] + list(name_contents_dict.keys()))

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
