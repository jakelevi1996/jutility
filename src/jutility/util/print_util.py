import os
import datetime
from jutility.util.save_load import get_full_path, load_text

HLINE_LEN = 79

class Printer:
    def __init__(
        self,
        filename=None,
        dir_name=None,
        file_ext="txt",
        display_path=True,
        print_to_console=True,
    ):
        if filename is not None:
            full_path = get_full_path(
                filename,
                dir_name,
                file_ext,
                verbose=display_path,
            )
            self._file = open(full_path, "w")
        else:
            self._file = None

        self.set_print_to_console(print_to_console)
        self._count = 1

    def __call__(self, *args, **kwargs):
        if self._print_to_console:
            print(*args, **kwargs)
        if self._file is not None:
            print(*args, **kwargs, file=self._file)

    def set_print_to_console(self, print_to_console: bool):
        self._print_to_console = print_to_console

    def timestamp(self):
        self(datetime.datetime.now())

    def hline(self, line_char="-", line_len=HLINE_LEN):
        self(line_char * line_len)

    def heading(self, heading_str: str, fill_char="-", line_len=HLINE_LEN):
        numbered_heading = (" (%i) %s " % (self._count, heading_str))
        self("\n%s\n" % numbered_heading.center(line_len, fill_char))
        self._count += 1

    def get_file(self):
        return self._file

    def get_filename(self) -> (str | None):
        if self._file is not None:
            return self._file.name

    def get_dir_name(self) -> (str | None):
        if self._file is not None:
            return os.path.dirname(self._file.name)

    def flush(self):
        if self._file is not None:
            self._file.flush()

    def read(self) -> (str | None):
        if self._file is not None:
            self.flush()
            return load_text(self.get_filename())

    def close(self):
        if self._file is not None:
            self._file.close()

def hline(line_char="-", line_len=HLINE_LEN):
    print(line_char * line_len)

def print_hline(*args, **kwargs):
    print(*args, **kwargs)
    hline()

class ColumnFormatter:
    def __init__(
        self,
        *column_formats:    str,
        sep:                str=" | ",
        default_format:     str="%s",
        printer:            (Printer | None)=None,
    ):
        if printer is None:
            printer = Printer()

        self._format_dict = {i: f for i, f in enumerate(column_formats)}
        self._sep = sep
        self._default_format = default_format
        self._printer = printer

    def format(self, *args):
        return self._sep.join(
            self._format_dict.get(i, self._default_format) % a
            for i, a in enumerate(args)
        )

    def print(self, *args, **kwargs):
        self._printer(self.format(*args), **kwargs)
