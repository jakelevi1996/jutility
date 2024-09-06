"""
MIT License

Copyright (c) 2022 JAKE LEVI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

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

CURRENT_DIR = os.path.abspath(os.getcwd())
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

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

    def __call__(self, *args, **kwargs):
        if self._print_to_console:
            print(*args, **kwargs)
        if self._file is not None:
            print(*args, **kwargs, file=self._file)

    def timestamp(self):
        self(datetime.datetime.now())

    def hline(self, line_char="-", line_len=100):
        self(line_char * line_len)

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
        verbose_enter=False,
        verbose_exit=True,
        hline=False,
    ):
        if printer is None:
            printer = Printer()
        self._print         = printer
        self._verbose_enter = verbose_enter
        self._verbose_exit  = verbose_exit
        self._hline         = hline
        self.set_name(name)
        self.reset()

    def reset(self):
        self._t0 = time.perf_counter()

    def set_name(self, name):
        self._name_str = (" for `%s`" % name) if (name is not None) else ""

    def get_time_taken(self):
        t1 = time.perf_counter()
        return t1 - self._t0

    def __enter__(self):
        if self._hline:
            self._print.hline()
        if self._verbose_enter or self._hline:
            self._print("Starting timer%s..." % self._name_str)

        self.reset()
        return self

    def __exit__(self, *args):
        self.time_taken = self.get_time_taken()
        if self._verbose_exit or self._hline:
            t_str = time_format(self.time_taken)
            self._print("Time taken%s = %s" % (self._name_str, t_str))
        if self._hline:
            self._print.hline()

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
        name: str,
        value_format=None,
        width=None,
        title=None,
        silent=False,
    ):
        if value_format is None:
            value_format = "s"
        if title is None:
            title = name.capitalize().replace("_", " ")
        if width is None:
            width = len(title)

        self.name = name
        self.title = title.ljust(abs(width))
        self._blank = "".ljust(abs(width))
        self._format = "%%%i%s" % (width, value_format)
        self._data_list = []
        self._silent = silent

    def update(self, data, level):
        self._data_list.append(data)

    def format_item(self, row_ind):
        data = self._data_list[row_ind]
        if (data is None) or self._silent:
            return self._blank
        else:
            return self._format % data

    def get_data(self):
        return self._data_list

    def reset_callback(self):
        self._callback = None

    def __repr__(self):
        return "%s(name=\"%s\")" % (type(self).__name__, self.name)

class CallbackColumn(Column):
    def set_callback(self, callback, level=0, interval=None):
        self._callback = callback
        self._callback_level = level
        if interval is None:
            interval = Always()

        self._callback_interval = interval
        return self

    def update(self, data, level):
        if level >= self._callback_level:
            if self._callback_interval.ready():
                data = self._callback()
                self._callback_interval.reset()

        self._data_list.append(data)

class TimeColumn(Column):
    def __init__(self, name="t", width=11):
        self.name = name
        self.title = "Time".ljust(abs(width))
        self._format = "%%%is" % width
        self._data_list = []
        self._timer = Timer()

    def update(self, data, level):
        self._data_list.append(self._timer.get_time_taken())

    def format_item(self, row_ind):
        t = self._data_list[row_ind]
        t_str = time_format(t, concise=True)
        return self._format % t_str

class CountColumn(Column):
    def __init__(self, name="c", width=5):
        self.name = name
        self.title = "Count".ljust(abs(width))
        self._format = "%%%ii" % width
        self._data_list = []
        self._count = 0

    def update(self, data, level):
        self._data_list.append(self._count)
        self._count += 1

    def format_item(self, row_ind):
        count = self._data_list[row_ind]
        return self._format % count

class Table:
    def __init__(
        self,
        *columns,
        print_interval=None,
        print_level=0,
        printer=None,
    ):
        if print_interval is None:
            print_interval = Always()
        if printer is None:
            printer = Printer()

        self._column_list: list[Column] = []
        self._column_dict: dict[str, Column] = dict()
        for column in columns:
            self.add_column(column)

        self._print_interval = print_interval
        self._print_level = print_level
        self._print = printer
        self._num_updates = 0
        if len(columns) > 0:
            self._print(self.format_header())

    def add_column(self, column: Column):
        if column.name in self._column_dict:
            raise ValueError(
                "Column %s with name %s already in table"
                % (self._column_dict[column.name], column.name)
            )
        self._column_list.append(column)
        self._column_dict[column.name] = column

    def update(self, level=0, **kwargs):
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

    def format_header(self):
        title_list = [column.title for column in self._column_list]
        title_str = " | ".join(title_list)
        hline_str = " | ".join("-" * len(t) for t in title_list)
        header_str = "\n".join([title_str, hline_str])
        return header_str

    def format_row(self, row_ind):
        value_list = [
            column.format_item(row_ind) for column in self._column_list
        ]
        row_str = " | ".join(value_list)
        return row_str

    def print_last(self, level=0):
        if level >= self._print_level:
            self._print(self.format_row(-1))

    def get_data(self, *names, filter_none=True, unpack_single=True):
        data_table = [
            self._column_dict[name].get_data()
            for name in names
        ]
        if filter_none:
            valid_row_inds = [
                i for i, data_row in enumerate(zip(*data_table))
                if all(data is not None for data in data_row)
            ]
            data_table = [
                [data_list[i] for i in valid_row_inds]
                for data_list in data_table
            ]
        if (len(names) == 1) and unpack_single:
            [data_table] = data_table
        return data_table

    def save_pickle(self, filename, dir_name=None):
        self._print = None
        for column in self._column_list:
            column.reset_callback()

        return save_pickle(self, filename, dir_name)

    def save_json(self, filename, dir_name=None, name_list=None):
        if name_list is None:
            name_list = [column.name for column in self._column_list]
        data_dict = {
            name: self._column_dict[name].get_data()
            for name in name_list
        }
        data_list = [
            {name: data[i] for name, data in data_dict.items()}
            for i in range(len(self))
        ]
        return save_json(data_list, filename, dir_name)

    def load_json(self, full_path):
        old_print_interval = self._print_interval
        self._print_interval = Never()
        data_list = load_json(full_path)
        name_list = set(name for row in data_list for name in row.keys())
        for name in name_list:
            self.add_column(Column(name))
        for row in data_list:
            self.update(**row)

        self._print_interval = old_print_interval
        return self

    def __len__(self):
        return self._num_updates

    def __str__(self):
        header_str = self.format_header()
        row_list = [self.format_row(i) for i in range(self._num_updates)]
        table_str = "\n".join([header_str] + row_list)
        return table_str

    def __repr__(self):
        return "%s(columns=%s)" % (type(self).__name__, self._column_list)

    def latex(self):
        raise NotImplementedError()

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

def time_format(t, concise=False):
    if concise:
        h_str = "h"
        m_str = "m"
        s_str = "s"
    else:
        h_str = " hours"
        m_str = " minutes"
        s_str = " seconds"

    if t < 60:
        return "%.4f%s" % (t, s_str)
    m, s = divmod(t, 60)
    if m < 60:
        return "%i%s %5.2f%s" % (m, m_str, s, s_str)
    h, m = divmod(m, 60)
    return "%i%s %2i%s %2i%s" % (h, h_str, m, m_str, s, s_str)

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
    mode="L",
    rgba=False,
):
    if rgba:
        mode = "RGBA"
    pil_image = PIL.Image.fromarray(image_uint8, mode=mode)
    full_path = get_full_path(filename, dir_name, "png", verbose=verbose)
    pil_image.save(full_path)
    return full_path

def save_image_diff(
    x_uint8: np.ndarray,
    y_uint8: np.ndarray,
    filename: str,
    dir_name: str=None,
    verbose: bool=True,
    mode="L",
    rgba=False,
):
    z = np.abs(x_uint8.astype(float) - y_uint8.astype(float))
    z_uint8 = z.astype(np.uint8)
    if rgba:
        mode = "RGBA"
    if mode == "RGBA":
        z_uint8[:, :, 3] = 255

    return save_image(z_uint8, filename, dir_name, verbose, mode)

def load_image(full_path) -> np.ndarray:
    image_uint8 = np.array(PIL.Image.open(full_path))
    return image_uint8

def get_program_command():
    return " ".join([sys.executable] + sys.argv)

def extract_substring(s, prefix, suffix, offset=None, strip=True):
    s = str(s)
    start_ind   = s.index(prefix, offset) + len(prefix)
    end_ind     = s.index(suffix, start_ind)
    s_substring = s[start_ind:end_ind]
    if strip:
        s_substring = s_substring.strip()

    return s_substring

def abbreviate_dictionary(
    input_dict: dict,
    key_abbreviations: dict[str, str],
    replaces: dict[str, str]=None,
):
    replaces_defaults = {
        "_":        "",
        "FALSE":    "F",
        "TRUE":     "T",
        "NONE":     "N",
    }
    if replaces is None:
        replaces = dict()
    for k, v in replaces_defaults.items():
        replaces.setdefault(k, v)

    s_sorted = sorted(
        "%s: %s" % (key_abbreviations[k].lower(), str(v).upper())
        for k, v in input_dict.items()
        if k in key_abbreviations
    )
    s_clean = clean_string("".join(s_sorted))
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

def log_range(start, stop, num=50, unique_integers=False, min_num=None):
    x = np.exp(np.linspace(np.log(start), np.log(stop), num))
    if unique_integers:
        x = np.unique(np.int64(np.round(x)))
        max_num = x.max() - x.min() + 1
        if min_num is None:
            min_num = num
        if (len(x) < min_num) and (len(x) < max_num):
            x = log_range(start, stop, num + 1, unique_integers, min_num)

    return x

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
    i_str_len = len(str(total_len))
    timer = Timer(printer=printer)
    for i, element in enumerate(input_iter, start=1):
        if print_interval.ready() or (i == total_len):
            i_str = str(i).rjust(i_str_len)
            fraction = i / total_len
            percent  = 100 * fraction
            t_taken  = timer.get_time_taken()
            t_total  = total_len * (t_taken / i)
            t_remain = t_total - t_taken
            str_elements = [
                "\r%s%s/%i"   % (prefix, i_str, total_len),
                "%5.1f %%"  % percent,
                "time taken = %10s"     % time_format(t_taken,  True),
                "time remaining = %10s" % time_format(t_remain, True),
            ]
            if bar_length is not None:
                num_done_chars = int(fraction * bar_length)
                num_remain_chars = bar_length - num_done_chars
                bar = ("*" * num_done_chars) + ("-" * num_remain_chars)
                str_elements.append(bar)

            printer(" | ".join(str_elements), end=end)
            print_interval.reset()
        if (i == total_len) and (end == ""):
            printer()

        yield element
