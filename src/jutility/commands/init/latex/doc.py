import os
from jutility import cli, util
from jutility.commands.command import Command

class Doc(Command):
    def run(
        self,
        dir_name: str,
    ):
        main_tex_path = os.path.join(dir_name, "main.tex")
        references_path = os.path.join(dir_name, "references.bib")
        util.save_text(MAIN_TEX, "main", dir_name, "tex", end="")
        util.save_text("", "references", dir_name, "bib", end="")
        util.run_subprocess(["code", main_tex_path, references_path])

    @classmethod
    def get_cli_args(cls) -> list[cli.Arg]:
        return [
            cli.PositionalArg("dir_name", type=str),
        ]

MAIN_TEX = """\\documentclass[12pt]{article}

\\usepackage[backref=page]{hyperref}
\\usepackage{geometry}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{subcaption}
\\usepackage{multirow}
\\usepackage{parskip}
\\usepackage{xcolor}
\\usepackage{graphicx}
\\usepackage{breakcites}
\\usepackage{tikz}

\\geometry{a4paper, margin=1in}
\\graphicspath{{../..}}
\\usetikzlibrary{arrows.meta}
\\hbadness 10000
\\vbadness 10000
\\hfuzz 100pt
\\vfuzz 100pt
\\righthyphenmin 100
\\emergencystretch 1em
\\allowdisplaybreaks

\\begin{document}

\\begin{center}
    \\LARGE
    \\setlength{\\baselineskip}{1.4em}
    \\textbf{Document Title \\\\ That Spans Multiple Lines} \\\\
    \\vspace{12pt}
    \\normalsize
    Jake Levi $\\cdot$ \\today
\\end{center}

\\begin{abstract}
Very abstract.
\\end{abstract}

\\section{Introduction}

...

\\bibliographystyle{apalike}
\\bibliography{references}

\\appendix

\\section{Additional results}

\\end{document}
"""
