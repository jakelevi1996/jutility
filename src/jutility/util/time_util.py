import time
import datetime
from jutility.util.units import time_verbose, time_concise
from jutility.util.print_util import Printer

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
        self._t_list    = []
        self.set_name(name)
        self.reset()

    def set_name(self, name: str):
        self._name_str = (" for `%s`" % name) if (name is not None) else ""

    def reset(self):
        self._t0 = time.perf_counter()

    def set_time(self, num_seconds: float):
        self._t0 = time.perf_counter() - num_seconds

    def get_time_taken(self) -> float:
        t1 = time.perf_counter()
        return t1 - self._t0

    def get_last(self) -> float:
        return self._t_list[-1]

    def format_time(self, concise: bool=False) -> str:
        return time_format(self.get_time_taken(), concise)

    def __enter__(self):
        if self._hline:
            self._printer.hline()
            self._printer("Starting timer%s..." % self._name_str)

        self.reset()
        return self

    def __exit__(self, *args):
        t = self.get_time_taken()
        self._t_list.append(t)
        if self._verbose:
            t_str = time_format(t)
            self._printer("Time taken%s = %s" % (self._name_str, t_str))
        if self._hline:
            self._printer.hline()

    def __iter__(self):
        return iter(self._t_list)

    def __len__(self):
        return len(self._t_list)

def time_format(num_seconds: float, concise=False):
    if concise:
        return time_concise.format(num_seconds)
    else:
        return time_verbose.format(num_seconds)

def timestamp(s, suffix=False):
    now = datetime.datetime.now()
    s = ("%s %s" % (s, now)) if suffix else ("%s %s" % (now, s))
    return s
