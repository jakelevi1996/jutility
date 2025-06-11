import datetime
from jutility.util.units import time_concise
from jutility.util.print_util import Printer, ColumnFormatter
from jutility.util.time_util import Timer
from jutility.util.interval import _Interval, TimeInterval

def circular_iterator(input_iter):
    while True:
        for i in input_iter:
            yield i

def progress(
    input_iter,
    prefix:         str="",
    printer:        (Printer | None)=None,
    print_interval: (_Interval | None)=None,
    bar_length:     (int | None)=25,
    end:            str="",
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
                time_concise.format(t_taken),
                time_concise.format(t_remain),
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

class Counter:
    def __init__(self, init_count: int=0):
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
