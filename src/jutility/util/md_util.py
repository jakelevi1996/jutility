import os
from jutility.util.print_util import Printer
from jutility.util.save_load import get_full_path
from jutility.util.sys_util import get_argv_str, get_program_command

class MarkdownPrinter(Printer):
    def __init__(
        self,
        name:               str,
        dir_name:           (str | None)=None,
        display_path:       bool=True,
        print_to_console:   bool=False,
    ):
        full_path = get_full_path(
            filename=name,
            dir_name=dir_name,
            file_ext="md",
            verbose=display_path,
        )
        self._file = open(full_path, "w")
        self.set_print_to_console(print_to_console)
        self._count = 1

    def title(self, name: str, end: str="\n"):
        self(("# %s" % name), end=end)

    def heading(self, name: str, level: int=2, end: str="\n"):
        self(("\n%s %s" % ("#" * level, name)), end=end)

    def paragraph(self, input_str: str, end: str="\n"):
        self(("\n%s" % input_str), end=end)

    def image(self, rel_path: str, name: str="", end: str="\n"):
        self("\n![%s](%s)" % (name, rel_path), end=end)

    def file_link(self, rel_path: str, name: str, end: str="\n"):
        self("\n[%s](%s)" % (name, rel_path), end=end)

    def code_block(self, *lines: str, ext: str="", end: str="\n"):
        self("\n```%s\n%s\n```" % (ext, "\n".join(lines)), end=end)

    def git_add(self, *paths: str):
        commands = "\n".join("git add -f %s" % p for p in paths)
        self("\n## `git add`\n\n```\n\n%s\n\n```" % commands)

    def readme_include(self, link_name: str, *paths: str):
        rm_path     = os.path.relpath("README.md", self.get_dir_name())
        self_path   = os.path.relpath(self.get_filename())
        links       = "".join("\n\n![](%s)" % p for p in paths)
        template    = (
            "\n## [`README.md`](%s) include\n\n```md\n\n[%s](%s)%s\n\n```"
        )
        self(template % (rm_path, link_name, self_path, links))

    def show_command(self, command_name: str, include_python: bool=False):
        self.heading("`%s` command" % command_name)
        cmd_str = get_program_command() if include_python else get_argv_str()
        self.code_block(cmd_str)

    def contents(self, *headings: str):
        self.heading("Contents")
        repeats = dict()
        for h in headings:
            level_str, _, name = h.partition("# ")
            indent = "".ljust(2 * len(level_str))
            link_str = "".join(
                c
                for c in name.lower().replace(" ", "-")
                if (c.isalnum() or c in "_-")
            )
            if link_str in repeats:
                repeats[link_str] += 1
                link_str += ("-%i" % repeats[link_str])
            else:
                repeats[link_str] = 0

            self("%s- [%s](#%s)" % (indent, name, link_str))

    def rel_path(self, input_path: str) -> str:
        return os.path.relpath(input_path, self.get_dir_name())

    @classmethod
    def make_link(cls, rel_path: str, name: str) -> str:
        return "[%s](%s)" % (name, rel_path)

    @classmethod
    def code(cls, input_str: str) -> str:
        return "`%s`" % input_str
