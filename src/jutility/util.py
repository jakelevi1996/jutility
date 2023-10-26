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
                display_path,
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

    def get_filename(self):
        if self._file is not None:
            return self._file.name

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
        if printer is None:
            printer = Printer()
        self._name = name
        self._print = printer
        self.reset()

    def reset(self):
        self._t0 = time.perf_counter()

    def time_taken(self):
        t1 = time.perf_counter()
        return t1 - self._t0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.time_measured = self.time_taken()
        t_str = time_format(self.time_measured)
        has_name = (self._name is not None)
        name_str = ("for %s " % self._name) if has_name else ""
        self._print("Time taken %s= %s" % (name_str, t_str))

class Counter:
    def __init__(self, init_count=0):
        self._count = init_count

    def __call__(self):
        count = self._count
        self._count += 1
        return count

class _Interval:
    def __init__(self):
        self._total_count = 0

    def ready(self):
        self._total_count += 1
        raise NotImplementedError()

    def reset(self):
        return

    def init(self):
        return

    def get_total_count(self):
        return self._total_count

class Always(_Interval):
    def ready(self):
        self._total_count += 1
        return True

class Never(_Interval):
    def ready(self):
        self._total_count += 1
        return False

class CountInterval(_Interval):
    def __init__(self, max_count):
        self._total_count = 0
        self._max_count = max_count
        self.init()

    def ready(self):
        self._total_count += 1
        self._count += 1
        return self._count >= self._max_count

    def reset(self):
        self._count = 0

    def init(self):
        self._count = self._max_count

class TimeInterval(_Interval):
    def __init__(self, num_seconds):
        self._total_count = 0
        self._num_seconds_interval = num_seconds
        self._num_seconds_limit = 0
        self._timer = Timer()

    def ready(self):
        self._total_count += 1
        return self._timer.time_taken() >= self._num_seconds_limit

    def reset(self):
        t = self._timer.time_taken()
        while self._num_seconds_limit < t:
            self._num_seconds_limit += self._num_seconds_interval

class Column:
    def __init__(self, name, value_format=None, title=None, width=None):
        if value_format is None:
            value_format = "s"
        if title is None:
            title = name.capitalize().replace("_", " ")
        if width is None:
            width = len(title)
        self.name = name
        self._width = max(width, len(title))
        self.title = title.ljust(self._width)
        self._format = "%%%i%s" % (self._width, value_format)
        self._data_list = []
        self._callback = None

    def format_item(self, row_ind):
        if self._data_list[row_ind] is not None:
            return self._format % self._data_list[row_ind]
        else:
            return "".rjust(self._width)

    def update(self, data, level):
        if (self._callback is not None) and (level >= self._callback_level):
            if self._callback_interval.ready():
                data = self._callback()
                self._callback_interval.reset()
        self._data_list.append(data)

    def get_data(self):
        return self._data_list

    def set_callback(self, callback, level=0, interval=None):
        self._callback = callback
        self._callback_level = level
        if interval is None:
            interval = Always()
        self._callback_interval = interval
        return self

class SilentColumn(Column):
    def format_item(self, row_ind):
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

    def format_item(self, row_ind):
        t = self._data_list[row_ind]
        t_str = time_format(t, concise=True)
        return t_str.rjust(self._width)

class CountColumn(Column):
    def __init__(self, name, width=5):
        self.name = name
        self.title = "Count".ljust(width)
        self._width = width
        self._data_list = []
        self._count = 0

    def update(self, data, level):
        self._data_list.append(self._count)
        self._count += 1

    def format_item(self, row_ind):
        count = self._data_list[row_ind]
        return str(count).rjust(self._width)

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

        self._column_list = []
        self._column_dict = dict()
        for column in columns:
            self.add_column(column)

        self._print_interval = print_interval
        self._print_level = print_level
        self._print = printer
        self._num_updates = 0
        if len(columns) > 0:
            self._print(self.format_header())

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

        if level > self._print_level:
            self.print_last()
            self._print_interval.init()

        if level == self._print_level:
            if self._print_interval.ready():
                self.print_last()
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

    def print_last(self):
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
        save_json(data_list, filename, dir_name)

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

    def latex(self):
        raise NotImplementedError()

def confidence_bounds(
    data_list,
    n_sigma=1,
    split_dim=None,
    downsample_ratio=1,
):
    if split_dim is not None:
        data_list = np.array(data_list)
        num_split = int(data_list.shape[split_dim] / downsample_ratio)
        split_len = num_split * downsample_ratio
        new_data_list = np.split(data_list[:split_len], num_split, split_dim)
        if len(data_list[split_len:]) > 0:
            new_data_list.append(data_list[split_len:])
        data_list = new_data_list
    mean = np.array([np.mean(x) for x in data_list])
    std  = np.array([np.std( x) for x in data_list])
    ucb = mean + (n_sigma * std)
    lcb = mean - (n_sigma * std)
    return mean, ucb, lcb

class NoisyData:
    def __init__(self, log_space_data=False):
        self._results_list_dict = dict()
        self._log_space_data = log_space_data

    def update(self, x, y):
        if self._log_space_data:
            y = np.log(y)
        if x in self._results_list_dict:
            self._results_list_dict[x].append(y)
        else:
            self._results_list_dict[x] = [y]

    def get_all_data(self):
        all_results_pairs = [
            [x, y]
            for x, result_list in self._results_list_dict.items()
            for y in result_list
        ]
        all_x, all_y = zip(*all_results_pairs)
        if self._log_space_data:
            all_y = np.exp(all_y)

        return all_x, all_y

    def get_statistics(self, n_sigma):
        x = sorted(
            x_i for x_i in self._results_list_dict.keys()
            if len(self._results_list_dict[x_i]) > 0
        )
        results_list_list = [self._results_list_dict[x_i] for x_i in x]
        mean, ucb, lcb = confidence_bounds(results_list_list, n_sigma)
        if self._log_space_data:
            mean, ucb, lcb = np.exp([mean, ucb, lcb])

        return x, mean, ucb, lcb

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
    verbose=True,
    for_saving=True,
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
        action_str = "Saving in" if for_saving else "Loading from"
        print("%s \"%s\"" % (action_str, full_path))

    return full_path

def save_pickle(data, filename, dir_name=None, verbose=True):
    full_path = get_full_path(filename, dir_name, "pkl", verbose)
    with open(full_path, "wb") as f:
        pickle.dump(data, f)

    return full_path

def load_pickle(full_path):
    with open(full_path, "rb") as f:
        data = pickle.load(f)

    return data

def save_json(data, filename, dir_name=None, verbose=True):
    full_path = get_full_path(filename, dir_name, "json", verbose)
    with open(full_path, "w") as f:
        json.dump(data, f)

    return full_path

def load_json(full_path):
    with open(full_path, "r") as f:
        data = json.load(f)

    return data

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

def log_range(x_lo, x_hi, num_x=50):
    log_x_lo, log_x_hi = np.log([x_lo, x_hi])
    return np.exp(np.linspace(log_x_lo, log_x_hi, num_x))

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
