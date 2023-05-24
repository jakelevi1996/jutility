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
import pickle
import json
import textwrap
import traceback
import datetime
import time
import numpy as np

CURRENT_DIR = os.path.abspath(os.getcwd())
RESULTS_DIR = os.path.join(CURRENT_DIR, "Results")

class Result:
    def __init__(self, filename, dir_name=None, data=None):
        self._filename = filename
        self._dir_name = dir_name
        self._data = data

    def get_data(self):
        return self._data

    def get_context(self, save=True, suppress_exceptions=False):
        return ResultSavingContext(self, save, suppress_exceptions)

    def save(self):
        save_pickle(self._data, self._filename, self._dir_name)

    def load(self):
        self._data = load_pickle(self._filename, self._dir_name)
        return self._data

class ResultSavingContext:
    def __init__(self, result, save, suppress_exceptions):
        self._result = result
        self._save = save
        self._suppress_exceptions = suppress_exceptions

    def __enter__(self):
        return self._result

    def __exit__(self, *args):
        if self._save:
            self._result.save()
        if self._suppress_exceptions:
            return True

class ExceptionContext:
    def __init__(self, suppress_exceptions=True, printer=None):
        self._suppress_exceptions = suppress_exceptions
        if printer is None:
            printer = Printer()
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

class Printer:
    def __init__(
        self,
        filename=None,
        dir_name=None,
        print_to_console=True,
    ):
        if filename is not None:
            full_path = get_full_path(filename, dir_name, for_saving=True)
            self._file = open(full_path, "w")
        else:
            self._file = None

        self._print_to_console = print_to_console

    def __call__(self, *args, **kwargs):
        self.print(*args, **kwargs)

    def print(self, *args, **kwargs):
        if self._print_to_console:
            print(*args, **kwargs)
        if self._file is not None:
            print(*args, **kwargs, file=self._file)

    def close(self):
        if self._file is not None:
            self._file.close()

class Seeder:
    def __init__(self):
        self._used_seeds = set()

    def get_seed(self, *args):
        seed = sum((i + 1) * ord(c) for i, c in enumerate(str(args)))
        while seed in self._used_seeds:
            seed += 1

        self._used_seeds.add(seed)
        return seed

    def get_rng(self, *args):
        seed = self.get_seed(*args)
        rng = np.random.default_rng(seed)
        return rng

class Timer:
    def __init__(self, name=None, printer=None):
        self._name = name
        if printer is None:
            printer = Printer()
        self._print = printer
        self._t0 = time.perf_counter()

    def time_taken(self):
        t1 = time.perf_counter()
        return t1 - self._t0

    def print_time_taken(self):
        t = self.time_taken()
        t_str = time_format(t)
        if self._name is None:
            prefix = "Time taken"
        else:
            prefix = "Time taken for %s" % self._name

        self._print("%s = %s" % (prefix, t_str))

    def __enter__(self):
        return

    def __exit__(self, *args):
        self.print_time_taken()

class Counter:
    def __init__(self, init_count=0):
        self._count = init_count

    def __call__(self):
        count = self._count
        self._count += 1
        return count

class Column:
    def __init__(self, name, value_format=None, title=None, width=None):
        self.name = name
        if title is None:
            title = name.capitalize().replace("_", " ")
        if width is None:
            width = len(title)
        self._width = max(width, len(title))
        self.title = title.ljust(self._width)
        if value_format is None:
            value_format = "s"
        self._format = "%%%i%s" % (self._width, value_format)
        self._data_list = []
        self._callback = None

    def get_last(self):
        if self._data_list[-1] is not None:
            return self._format % self._data_list[-1]
        else:
            return "".rjust(self._width)

    def update(self, data, level):
        if (self._callback is not None) and (level >= self._callback_level):
            data = self._callback()
        self._data_list.append(data)

    def get_data(self):
        return self._data_list

    def set_callback(self, callback, level=0):
        self._callback = callback
        self._callback_level = level
        return self

class SilentColumn(Column):
    def get_last(self):
        return "".rjust(self._width)

class TimeColumn(Column):
    def __init__(self, name, width=11):
        self.name = name
        self.title = "Time".ljust(width)
        self._width = width
        self._data_list = []
        self._timer = Timer()

    def update(self, data, level):
        self._data_list.append(self._timer.time_taken())

    def get_last(self):
        t = self._data_list[-1]
        t_str = time_format(t)
        return t_str.rjust(self._width)

class Table:
    def __init__(
        self,
        *columns,
        print_every=1,
        print_every_level=0,
        printer=None,
    ):
        self._column_list = []
        self._column_dict = dict()
        for column in columns:
            self.add_column(column)
        self._print_every = print_every
        self._print_every_level = print_every_level
        if printer is None:
            printer = Printer()
        self._print = printer
        self._num_updates = 0
        self._print_counter = print_every
        if len(columns) > 0:
            self.print_headings()

    def add_column(self, column):
        if column.name in self._column_dict:
            raise ValueError(
                "Column with name %s already in table"
                % column.name
            )
        self._column_list.append(column)
        self._column_dict[column.name] = column

    def update(self, level=0, **kwargs):
        for name, column in self._column_dict.items():
            column.update(kwargs.get(name), level)

        if level > self._print_every_level:
            self.print_last()
            self._print_counter = self._print_every

        if level == self._print_every_level:
            if self._print_counter >= self._print_every:
                self.print_last()
                self._print_counter = 0
            self._print_counter += 1

        self._num_updates += 1

    def print_headings(self):
        title_list = [column.title for column in self._column_list]
        self._print(" | ".join(title_list))
        self._print(" | ".join("-" * len(t) for t in title_list))

    def print_last(self):
        value_list = [column.get_last() for column in self._column_list]
        self._print(" | ".join(value_list))

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

    def save_json(self, filename, dir_name=None, name_list=None):
        full_path = get_full_path(
            filename,
            dir_name,
            for_saving=True,
            file_ext="json",
        )
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
        save_json(data_list, filename, dir_name)

    def load_json(self, filename, dir_name=None):
        data_list = load_json(filename, dir_name)
        name_list = set(name for row in data_list for name in row.keys())
        for name in name_list:
            self.add_column(Column(name))
        for row in data_list:
            self._print_counter = 0
            self.update(**row)

        self._print_counter = self._print_every
        return self

    def __len__(self):
        return self._num_updates

    def latex(self):
        raise NotImplementedError()

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

def trim_string(s, max_len, suffix="(...)"):
    if len(s) > max_len:
        trim_len = max(max_len - len(suffix), 0)
        s = s[:trim_len] + suffix

    return s

def wrap_string(s, max_len=80, wrap_len=60):
    if len(s) > max_len:
        s = textwrap.fill(s, width=wrap_len, break_long_words=False)
    return s

def time_format(t):
    if t < 60:
        t_str = "%7.4fs" % t
    else:
        m, s = divmod(t, 60)
        if m < 60:
            t_str = "%2im %5.2fs" % (m, s)
        else:
            h, m = divmod(m, 60)
            t_str = "%2ih %2im %2is" % (h, m, s)

    return t_str

def get_full_path(filename, dir_name=None, for_saving=False, file_ext=None):
    if dir_name is None:
        dir_name = RESULTS_DIR

    if for_saving:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        filename = clean_string(filename)
        filename = trim_string(filename, 240 - len(os.path.abspath(dir_name)))

    if file_ext is not None:
        filename = "%s.%s" % (filename, file_ext)

    full_path = os.path.join(dir_name, filename)
    return full_path

def save_pickle(data, filename, dir_name=None, verbose=True):
    full_path = get_full_path(
        filename,
        dir_name,
        for_saving=True,
        file_ext="pkl",
    )
    if verbose:
        print("Saving pickle in \"%s\"" % full_path)
    with open(full_path, "wb") as f:
        pickle.dump(data, f)

def load_pickle(filename, dir_name=None, verbose=False):
    full_path = get_full_path(filename, dir_name)
    if verbose:
        print("Loading pickle from \"%s\"" % full_path)
    with open(full_path, "rb") as f:
        data = pickle.load(f)

    return data

def save_json(data, filename, dir_name=None, verbose=True):
    full_path = get_full_path(
        filename,
        dir_name,
        for_saving=True,
        file_ext="json",
    )
    if verbose:
        print("Saving JSON in \"%s\"" % full_path)
    with open(full_path, "w") as f:
        json.dump(data, f)

def load_json(filename, dir_name=None, verbose=False):
    full_path = get_full_path(filename, dir_name)
    if verbose:
        print("Loading JSON from \"%s\"" % full_path)
    with open(full_path, "r") as f:
        data = json.load(f)

    return data

def is_numeric(x):
    return any(isinstance(x, t) for t in [int, float, np.number])

def numpy_set_print_options():
    np.set_printoptions(
        precision=3,
        linewidth=10000,
        suppress=True,
        threshold=10000,
    )
