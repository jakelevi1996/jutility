import os
import sys
import pickle
import json
import textwrap
import traceback
import datetime
import time
import numpy as np
import PIL.Image
from jutility import units

CURRENT_DIR = os.path.abspath(os.getcwd())
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

HLINE_LEN = 79

class CallbackContext:
    def __init__(
        self,
        enter_callback=None,
        exit_callback=None,
        enter_return=None,
        suppress_exceptions=False,
    ):
        self._enter_callback        = enter_callback
        self._exit_callback         = exit_callback
        self._enter_return          = enter_return
        self._suppress_exceptions   = suppress_exceptions

    def __enter__(self):
        if self._enter_callback is not None:
            self._enter_callback()

        return self._enter_return

    def __exit__(self, *args):
        if self._exit_callback is not None:
            self._exit_callback()

        if self._suppress_exceptions:
            return True

class ExceptionContext:
    def __init__(self, suppress_exceptions=True, printer=None):
        if printer is None:
            printer = Printer()
        self._suppress_exceptions = suppress_exceptions
        self._print = printer

    def __enter__(self):
        return

    def __exit__(self, *args):
        if args[0] is not None:
            self._print("%s: An exception occured:" % datetime.datetime.now())
            self._print("".join(traceback.format_exception(*args)))
            if self._suppress_exceptions:
                self._print("Suppressing exception and continuing...")
                return True

class StoreDictContext:
    def __init__(self, input_dict, *keys):
        self._input_dict = input_dict
        self._keys = keys
        self._stored_dict = dict()

    def __enter__(self):
        for k in self._keys:
            if k in self._input_dict:
                self._stored_dict[k] = self._input_dict.pop(k)

    def __exit__(self, *args):
        for k in self._keys:
            if k in self._stored_dict:
                self._input_dict[k] = self._stored_dict.pop(k)

        assert len(self._stored_dict) == 0

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

        self._print_to_console = print_to_console
        self._count = 1

    def __call__(self, *args, **kwargs):
        if self._print_to_console:
            print(*args, **kwargs)
        if self._file is not None:
            print(*args, **kwargs, file=self._file)

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

    def get_filename(self):
        if self._file is not None:
            return self._file.name

    def flush(self):
        if self._file is not None:
            self._file.flush()

    def read(self):
        if self._file is not None:
            self.flush()
            return load_text(self.get_filename())

    def close(self):
        if self._file is not None:
            self._file.close()

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
        self._print_to_console = print_to_console
        self._count = 1

    def title(self, name: str, end: str="\n\n"):
        self(("# %s" % name), end=end)

    def heading(self, name: str, end: str="\n\n"):
        self(("\n## %s" % name), end=end)

    def image(self, rel_path: str, name: str=""):
        self("\n![%s](%s)" % (name, rel_path))

    def file_link(self, rel_path: str, name: str):
        self("\n[%s](%s)" % (name, rel_path))

    def code_block(self, *lines: str, ext: str=""):
        self("\n```%s\n%s\n```" % (ext, "\n".join(lines)))

    @classmethod
    def make_link(cls, rel_path: str, name: str) -> str:
        return "[%s](%s)" % (name, rel_path)

    @classmethod
    def code(cls, input_str: str) -> str:
        return "`%s`" % input_str

def hline(line_char="-", line_len=HLINE_LEN):
    print(line_char * line_len)

def print_hline(*args, **kwargs):
    print(*args, **kwargs)
    hline()

class ColumnFormatter:
    def __init__(
        self,
        *column_formats: str,
        sep: str=" | ",
        default_format: str="%s",
        printer: Printer=None,
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

class Seeder:
    def __init__(self):
        self._used_seeds = set()

    def get_seed(self, *args):
        seed = sum(i * ord(c) for i, c in enumerate(str(args), start=1))
        while seed in self._used_seeds:
            seed += 1

        self._used_seeds.add(seed)
        return seed

    def get_rng(self, *args):
        seed = self.get_seed(*args)
        rng = np.random.default_rng(seed)
        return rng

class Timer:
    def __init__(
        self,
        name=None,
        printer=None,
        verbose=True,
        hline=False,
    ):
        if printer is None:
            printer = Printer()

        self._printer   = printer
        self._verbose   = verbose
        self._hline     = hline
        self.set_name(name)
        self.reset()

    def reset(self):
        self._t0 = time.perf_counter()

    def set_time(self, num_seconds: float):
        self._t0 = time.perf_counter() - num_seconds

    def set_name(self, name):
        self._name_str = (" for `%s`" % name) if (name is not None) else ""

    def get_time_taken(self) -> float:
        t1 = time.perf_counter()
        return t1 - self._t0

    def format_time(self, concise: bool=False) -> str:
        return time_format(self.get_time_taken(), concise)

    def __enter__(self):
        if self._hline:
            self._printer.hline()
            self._printer("Starting timer%s..." % self._name_str)

        self.reset()
        return self

    def __exit__(self, *args):
        self.time_taken = self.get_time_taken()
        if self._verbose:
            t_str = time_format(self.time_taken)
            self._printer("Time taken%s = %s" % (self._name_str, t_str))
        if self._hline:
            self._printer.hline()

class Counter:
    def __init__(self, init_count=0):
        self._init_count = init_count
        self.reset()

    def __call__(self):
        count = self._count
        self._count += 1
        return count

    def get_value(self):
        return self._count

    def reset(self):
        self._count = self._init_count

class _Interval:
    def ready(self):
        raise NotImplementedError()

    def reset(self):
        return

    def full_reset(self):
        return

class Always(_Interval):
    def ready(self):
        return True

class Never(_Interval):
    def ready(self):
        return False

class CountInterval(_Interval):
    def __init__(self, max_count):
        self._max_count = max_count
        self.full_reset()

    def ready(self):
        self._count += 1
        return self._count >= self._max_count

    def reset(self):
        self._count = 0

    def full_reset(self):
        self._count = self._max_count

class TimeInterval(_Interval):
    def __init__(self, num_seconds):
        self._num_seconds_interval = num_seconds
        self._num_seconds_limit = 0
        self._timer = Timer()

    def ready(self):
        return self._timer.get_time_taken() >= self._num_seconds_limit

    def reset(self):
        t = self._timer.get_time_taken()
        while self._num_seconds_limit < t:
            self._num_seconds_limit += self._num_seconds_interval

class Column:
    def __init__(
        self,
        name:           str,
        value_format:   str="s",
        width:          (int | None)=None,
        title:          (str | None)=None,
        silent:         bool=False,
    ):
        if title is None:
            title = name.capitalize().replace("_", " ")
        if width is None:
            width = max(len(title), 10)

        self.name       = name
        self.title      = title.ljust(abs(width))
        self._blank     = "".ljust(abs(width))
        self._format    = "%" + str(width) + value_format
        self._data_list = []
        self._silent    = silent

    def update(self, data, level: int):
        self._data_list.append(data)

    def format_item(self, row_ind: int) -> str:
        data = self._data_list[row_ind]
        if (data is None) or self._silent:
            return self._blank
        else:
            return self._format % data

    def get_item(self, row_ind: int):
        return self._data_list[row_ind]

    def set_callback(
        self,
        callback,
        level:      int=0,
        interval:   (_Interval | None)=None,
    ):
        raise NotImplementedError()

    def reset_callback(self):
        self._callback = None

    def __iter__(self):
        return iter(self._data_list)

    def __len__(self):
        return len(self._data_list)

    def __repr__(self):
        return format_type(type(self), name=self.name, len=len(self))

class CallbackColumn(Column):
    def set_callback(
        self,
        callback,
        level:      int=0,
        interval:   (_Interval | None)=None,
    ):
        self._callback = callback
        self._callback_level = level
        if interval is None:
            interval = Always()

        self._callback_interval = interval
        return self

    def update(self, data, level: int):
        if level >= self._callback_level:
            if self._callback_interval.ready():
                data = self._callback()
                self._callback_interval.reset()

        self._data_list.append(data)

class TimeColumn(Column):
    def __init__(self, name: str="t", width: int=-11):
        self.name       = name
        self.title      = "Time".ljust(abs(width))
        self._format    = "%" + str(width) + "s"
        self._data_list = []
        self._timer     = Timer()

    def update(self, data, level: int):
        self._data_list.append(self._timer.get_time_taken())

    def format_item(self, row_ind: int) -> str:
        t = self._data_list[row_ind]
        t_str = time_format(t, concise=True)
        return self._format % t_str

    def get_timer(self):
        return self._timer

class CountColumn(Column):
    def __init__(self, name: str="c", width: int=-5):
        self.name       = name
        self.title      = "Count".ljust(abs(width))
        self._format    = "%" + str(width) + "i"
        self._data_list = []
        self._count     = 0

    def update(self, data, level: int):
        self._data_list.append(self._count)
        self._count += 1

    def format_item(self, row_ind: int) -> str:
        count = self._data_list[row_ind]
        return self._format % count

class Table:
    def __init__(
        self,
        *columns:       Column,
        printer:        (Printer | None)=None,
        print_interval: (_Interval | None)=None,
        print_level:    int=0,
    ):
        if print_interval is None:
            print_interval = Always()
        if printer is None:
            printer = Printer()

        self._column_list = list(columns)
        self._column_dict = {
            column.name: column
            for column in self._column_list
        }
        if len(self._column_list) > len(self._column_dict):
            raise ValueError("%r has duplicate column names" % self)

        self.set_printer(printer)
        self._print_interval = print_interval
        self._print_level = print_level
        self._num_updates = 0
        if len(columns) > 0:
            self._printer(self.format_header())

    @classmethod
    def key_value(
        cls,
        printer:        (Printer | None)=None,
        total_width:    int=HLINE_LEN,
    ):
        col_width = (total_width - len(" | ")) // 2
        return cls(
            Column("k", "s", -col_width, "Key"),
            Column("v", "s", -col_width, "Value"),
            printer=printer,
        )

    @classmethod
    def from_dict(
        cls,
        input_dict:     dict,
        printer:        (Printer | None)=None,
        total_width:    int=HLINE_LEN,
    ):
        table = cls.key_value(printer, total_width)
        for k, v in input_dict.items():
            table.update(k=k, v=v)

        return table

    def add_column(self, column: Column):
        if column.name in self._column_dict:
            raise ValueError(
                "Column %s with name %s already in table"
                % (self._column_dict[column.name], column.name)
            )
        self._column_list.append(column)
        self._column_dict[column.name] = column

    def get_column(self, column_name: str) -> Column:
        return self._column_dict[column_name]

    def set_printer(self, printer: (Printer | None)):
        self._printer = printer

    def update(self, level: int=0, **kwargs):
        for name, column in self._column_dict.items():
            column.update(kwargs.get(name), level)

        if level > self._print_level:
            self.print_last(level)
            self._print_interval.full_reset()

        if level == self._print_level:
            if self._print_interval.ready():
                self.print_last(level)
                self._print_interval.reset()

        self._num_updates += 1

    def format_header(self) -> str:
        title_list = [column.title for column in self._column_list]
        title_str = " | ".join(title_list)
        hline_str = " | ".join("-" * len(t) for t in title_list)
        return "\n".join([title_str, hline_str])

    def format_row(self, row_ind) -> str:
        value_list = [
            column.format_item(row_ind)
            for column in self._column_list
        ]
        return " | ".join(value_list)

    def print_last(self, level=0):
        if level >= self._print_level:
            self._printer(self.format_row(-1))

    def get_data(self, column_name: str) -> list:
        return [
            x
            for x in self._column_dict[column_name]
            if x is not None
        ]

    def save_pickle(self, filename, dir_name=None) -> str:
        self.set_printer(None)
        for column in self._column_list:
            column.reset_callback()

        return save_pickle(self, filename, dir_name)

    def to_json(self, name_list: (list[str] | None)=None) -> list[dict]:
        if name_list is None:
            name_list = [column.name for column in self._column_list]

        return [
            {
                n: self._column_dict[n].get_item(i)
                for n in name_list
            }
            for i in range(len(self))
        ]

    def to_latex(self, col_fmt: (str | None)=None) -> str:
        if col_fmt is None:
            col_fmt = " ".join(["c"] * len(self._column_list))

        hl          = "\\hline"
        endl        = " \\\\"
        header_list = [column.title for column in self._column_list]
        rows_list   = [self.format_row(i) for i in range(len(self))]
        parts       = [
            "\\begin{tabular}{%s}" % col_fmt,
            hl,
            " & ".join(header_list) + endl,
            hl,
            (endl + "\n").join(rows_list).replace(" | ", " & ") + endl,
            hl,
            "\\end{tabular}",
        ]
        return "\n".join(parts)

    def __len__(self):
        return self._num_updates

    def __str__(self):
        header_str  = self.format_header()
        row_list    = [self.format_row(i) for i in range(self._num_updates)]
        return "\n".join([header_str] + row_list)

    def __repr__(self):
        return format_type(type(self), *self._column_list)

class FunctionList:
    def __init__(self, *functions):
        self._function_list = []
        self.add_functions(*functions)

    def add_functions(self, *functions):
        for f in functions:
            self._function_list.append(f)

    def call_all(self, return_results=True):
        result_list = [f() for f in self._function_list]
        return_val = (result_list if return_results else None)
        return return_val

def remove_duplicate_substring(s, sub_str):
    duplicates = sub_str * 2
    while duplicates in s:
        s = s.replace(duplicates, sub_str)

    return s

def clean_string(s, allowed_non_alnum_chars="-_.,", replacement="_"):
    s_clean = "".join(
        c if (c.isalnum() or c in allowed_non_alnum_chars) else replacement
        for c in str(s)
    )
    s_clean = remove_duplicate_substring(s_clean, replacement)
    return s_clean

def trim_string(s, max_len, suffix="_..._"):
    if len(s) > max_len:
        trim_len = max(max_len - len(suffix), 0)
        s = s[:trim_len] + suffix

    return s

def wrap_string(s, max_len=80, wrap_len=60):
    if len(s) > max_len:
        s = textwrap.fill(s, width=wrap_len, break_long_words=False)
    return s

def indent(input_str, num_spaces=4):
    return textwrap.indent(input_str, " " * num_spaces)

def time_format(num_seconds: float, concise=False):
    if concise:
        return units.time_concise.format(num_seconds)
    else:
        return units.time_verbose.format(num_seconds)

def timestamp(s, suffix=False):
    now = datetime.datetime.now()
    s = ("%s %s" % (s, now)) if suffix else ("%s %s" % (now, s))
    return s

def get_full_path(
    filename,
    dir_name=None,
    file_ext=None,
    loading=False,
    verbose=True,
):
    if dir_name is None:
        dir_name = RESULTS_DIR
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    filename = clean_string(filename)
    filename = trim_string(filename, 240 - len(os.path.abspath(dir_name)))

    if file_ext is not None:
        filename = "%s.%s" % (filename, file_ext)

    full_path = os.path.join(dir_name, filename)

    if verbose:
        action_str = "Loading from" if loading else "Saving in"
        print("%s \"%s\"" % (action_str, full_path))

    return full_path

def save_text(s, filename, dir_name=None, file_ext="txt", verbose=True):
    full_path = get_full_path(filename, dir_name, file_ext, verbose=verbose)
    with open(full_path, "w") as f:
        print(s, file=f)

    return full_path

def load_text(full_path):
    with open(full_path, "r") as f:
        s = f.read()

    return s

def save_pickle(data, filename, dir_name=None, verbose=True, **kwargs):
    full_path = get_full_path(filename, dir_name, "pkl", verbose=verbose)
    with open(full_path, "wb") as f:
        pickle.dump(data, f, **kwargs)

    return full_path

def load_pickle(full_path):
    with open(full_path, "rb") as f:
        data = pickle.load(f)

    return data

def save_json(data, filename, dir_name=None, verbose=True, **kwargs):
    full_path = get_full_path(filename, dir_name, "json", verbose=verbose)
    kwargs.setdefault("indent", 4)
    kwargs.setdefault(
        "default",
        (lambda x: x.tolist() if isinstance(x, np.ndarray) else None),
    )
    with open(full_path, "w") as f:
        json.dump(data, f, **kwargs)

    return full_path

def load_json(full_path):
    with open(full_path, "r") as f:
        data = json.load(f)

    return data

def save_image(
    image_uint8: np.ndarray,
    filename: str,
    dir_name: str=None,
    verbose: bool=True,
):
    if image_uint8.dtype != np.uint8:
        im_ge0 = image_uint8 - np.min(image_uint8)
        im_255 = im_ge0 * (255 / np.max(im_ge0))
        image_uint8 = np.uint8(im_255)

    shape = image_uint8.shape
    mode_dict_1 = {(3, 3): "RGB", (3, 4): "RGBA"}
    mode_dict_2 = {2: "L"}
    mode = mode_dict_1.get(
        (len(shape), shape[-1]),
        mode_dict_2.get(len(shape)),
    )

    pil_image = PIL.Image.fromarray(image_uint8, mode=mode)
    full_path = get_full_path(filename, dir_name, "png", verbose=verbose)
    pil_image.save(full_path)
    return full_path

def save_image_diff(
    full_path_1:    str,
    full_path_2:    str,
    output_name:    str="diff",
    dir_name:       (str | None)=None,
    normalise:      bool=True,
):
    if dir_name is None:
        dir_name = os.path.dirname(full_path_1)

    x_pil = PIL.Image.open(full_path_1)
    y_pil = PIL.Image.open(full_path_2)
    print("Input sizes = %s and %s" % (x_pil.size, y_pil.size))
    if y_pil.size != x_pil.size:
        print("Resizing %s to %s" % (y_pil.size, x_pil.size))
        y_pil = y_pil.resize(x_pil.size)

    x = np.array(x_pil, dtype=np.float64)
    y = np.array(y_pil, dtype=np.float64)
    z = np.uint8(np.abs(x - y))
    print("Min image difference = %s" % z.min())
    print("Max image difference = %s" % z.max())
    if normalise and (z.max() > 0):
        z = np.float64(z)
        z *= 255 / z.max()
        z = np.uint8(z)
    if (len(z.shape) == 3) and (z.shape[-1] == 4):
        z[:, :, 3] = 255

    return save_image(z, output_name, dir_name, verbose=True)

def load_image(full_path) -> np.ndarray:
    image_uint8 = np.array(PIL.Image.open(full_path))
    return image_uint8

def get_argv_str() -> str:
    return " ".join(sys.argv)

def get_program_command() -> str:
    return " ".join([sys.executable, get_argv_str()])

def extract_substring(s, prefix, suffix, offset=None, strip=True):
    s = str(s)
    start_ind   = s.index(prefix, offset) + len(prefix)
    end_ind     = s.index(suffix, start_ind)
    s_substring = s[start_ind:end_ind]
    if strip:
        s_substring = s_substring.strip()

    return s_substring

def format_dict(
    input_dict: dict,
    item_fmt="%s=%r",
    item_sep=", ",
    prefix="",
    suffix="",
    key_order=None,
) -> str:
    if key_order is None:
        key_order = sorted(input_dict.keys())

    items_str = item_sep.join(
        item_fmt % (k, input_dict[k])
        for k in key_order
    )
    return prefix + items_str + suffix

def format_type(
    input_type: type,
    *args,
    item_fmt="%s=%r",
    key_order=None,
    **kwargs,
):
    prefix = input_type.__name__ + "("
    if len(args) > 0:
        prefix += ", ".join(repr(a) for a in args)
        if len(kwargs) > 0:
            prefix += ", "

    return format_dict(
        input_dict=kwargs,
        item_fmt=item_fmt,
        prefix=prefix,
        suffix=")",
        key_order=key_order,
    )

def abbreviate_dictionary(
    input_dict: dict,
    key_abbreviations: dict[str, str],
    replaces: (dict[str, str] | None)=None,
):
    if replaces is None:
        replaces = {
            "_":        "",
            "FALSE":    "F",
            "TRUE":     "T",
            "NONE":     "N",
        }

    sorted_keys = sorted(
        sorted(set(input_dict.keys()) & set(key_abbreviations.keys())),
        key=lambda k: key_abbreviations[k],
    )
    pairs_list = [
        (key_abbreviations[k].lower() + str(input_dict[k]).upper())
        for k in sorted_keys
    ]
    s_clean = clean_string("".join(pairs_list))
    for k, v in replaces.items():
        s_clean = s_clean.replace(k, v)

    return s_clean

def merge_strings(input_list: list[str], clean=True):
    output_str = ""
    while len(input_list) > 0:
        next_char_set = set(s[0] for s in input_list)
        if len(next_char_set) == 1:
            [c] = next_char_set
            output_str += c
            input_list = [s[1:] for s in input_list]
        else:
            remaining_chars = set(c for s in input_list for c in s)
            valid_next_chars = [
                c for c in remaining_chars
                if all((c in s) for s in input_list)
            ]
            if len(valid_next_chars) > 0:
                max_ind_dict = {
                    c: max(s.index(c) for s in input_list)
                    for c in valid_next_chars
                }
                next_char = min(max_ind_dict, key=lambda c: max_ind_dict[c])
                prefix_dict = {s: s.index(next_char) for s in input_list}
            else:
                prefix_dict = {s: len(s) for s in input_list}

            prefix_list = [s[:n] for s, n in prefix_dict.items()]
            input_list  = [s[n:] for s, n in prefix_dict.items()]
            output_str += str(sorted(set(prefix_list)))

        input_list = [s for s in input_list if len(s) > 0]

    if clean:
        output_str = clean_string(output_str).replace("_", "")

    return output_str

def get_unique_prefixes(
    input_list: list[str],
    forbidden: (set[str] | None)=None,
    min_len: int=1,
) -> dict[str, str]:
    if len(input_list) == 0:
        return dict()
    if forbidden is None:
        forbidden = set()

    remaining = set(input_list)
    prefix_dict = dict()
    max_len = max(len(s) for s in remaining)

    for i in range(min_len, max_len):
        partial_dict = {s: s[:i] for s in remaining}
        partial_list = list(partial_dict.values())
        new_prefixes = {
            s: p
            for s, p in partial_dict.items()
            if ((partial_list.count(p) == 1) and (p not in forbidden))
        }
        prefix_dict.update(new_prefixes)
        remaining -= set(new_prefixes.keys())
        if len(remaining) == 0:
            break

    prefix_dict.update({s: s for s in remaining})
    return prefix_dict

def is_numeric(x):
    return any(isinstance(x, t) for t in [int, float, np.number])

def numpy_set_print_options(
    precision=3,
    linewidth=10000,
    suppress=True,
    threshold=10000,
):
    np.set_printoptions(
        precision=precision,
        linewidth=linewidth,
        suppress=suppress,
        threshold=threshold,
    )

def log_range(start, stop, num=50, unique_integers=False) -> np.ndarray:
    if unique_integers:
        max_num = int(np.ceil(abs(stop - start) + 1))
        for num_float in range(num, max_num + 1):
            x = log_range(start, stop, num_float, unique_integers=False)
            x = np.unique(np.int64(np.round(x)))
            if len(x) >= num:
                return x

        x = np.linspace(start, stop, max_num)
        x = np.unique(np.int64(np.round(x)))
        return x

    return np.exp(np.linspace(np.log(start), np.log(stop), num))

def check_type(instance, expected_type, name=None):
    if not isinstance(instance, expected_type):
        name_str = ("`%s` = " % name) if (name is not None) else ""
        exp_type_name = expected_type.__name__
        inst_type_name = type(instance).__name__
        error_msg = (
            "Expected %sinstance of `%s`, but received %sinstance of `%s`"
            % (name_str, exp_type_name, name_str, inst_type_name)
        )
        raise TypeError(error_msg)

def check_equal(value, expected_value, name=None):
    if value != expected_value:
        name_str = ("%s == " % name) if (name is not None) else ""
        error_msg = (
            "Expected `%s%s`, but received `%s%s`"
            % (name_str, expected_value, name_str, value)
        )
        raise RuntimeError(error_msg)

def circular_iterator(input_iter):
    while True:
        for i in input_iter:
            yield i

def progress(
    input_iter,
    prefix="",
    printer=None,
    print_interval=None,
    bar_length=25,
    end="",
):
    if printer is None:
        printer = Printer()
    if print_interval is None:
        print_interval = TimeInterval(1)

    total_len = len(input_iter)
    cf = ColumnFormatter(
        "\r%s%%%ii/%i" % (prefix, len(str(total_len)), total_len),
        "%5.1f %%",
        "t+ %11s",
        "t- %11s",
        "ETA %19s",
        printer=printer,
    )
    timer = Timer()
    for i, element in enumerate(input_iter, start=1):
        yield element

        if print_interval.ready() or (i == total_len):
            fraction = i / total_len
            percent  = 100 * fraction
            t_taken  = timer.get_time_taken()
            t_total  = total_len * (t_taken / i)
            t_remain = t_total - t_taken
            t_now    = datetime.datetime.now()
            t_delta  = datetime.timedelta(seconds=t_remain)
            str_elements = [
                i,
                percent,
                units.time_concise.format(t_taken),
                units.time_concise.format(t_remain),
                (t_now + t_delta).replace(microsecond=0),
            ]
            if bar_length is not None:
                num_done_chars = int(fraction * bar_length)
                num_remain_chars = bar_length - num_done_chars
                bar = ("*" * num_done_chars) + ("-" * num_remain_chars)
                str_elements.append(bar)

            cf.print(*str_elements, end=end)
            print_interval.reset()
        if (i == total_len) and (end == ""):
            printer()
