import datetime

class UnitsFormatter:
    def __init__(
        self,
        names: list[str],
        num_divisions: list[int],
        base_precisions: list[int],
        widths: list[int],
    ):
        self._num_divisions = num_divisions
        self._names = names
        self._format_list = []
        for i in range(len(names)):
            p = base_precisions[min(i, len(base_precisions) - 1)]
            w = (widths[0] + p + 1) if (p > 0) else widths[0]
            i_format_parts = ["%%%i.%if%s" % (w, p, names[0])]
            for j in range(i):
                w = widths[min(i, len(widths) - 1)]
                i_format_parts.append("%%%ii%s" % (w, names[j + 1]))

            self._format_list.append(" ".join(reversed(i_format_parts)))

        v = 1
        values = [1]
        for d in num_divisions:
            v *= d
            values.append(v)

        self._base_units = {
            names[i]: values[i]
            for i in range(len(names))
        }

    def format(self, num_base_units: float):
        parts = []
        num_units = num_base_units
        for n in self._num_divisions:
            if num_units >= n:
                num_units, remainder = divmod(num_units, n)
                parts.append(remainder)
            else:
                break

        parts.append(num_units)
        format_str = self._format_list[len(parts) - 1]
        return format_str % tuple(reversed(parts))

    def parse(self, input_str: str):
        matches = {
            input_str.index(name): name
            for name in self._names
            if name in input_str
        }
        num_base_units = 0
        start = 0
        for i in sorted(matches.keys()):
            name = matches[i]
            num_units = float(input_str[start:i])
            num_base_units += num_units * self._base_units[name]
            start = i + len(name)

        return num_base_units

    def diff(self, x: str, y: str):
        return self.format(abs(self.parse(x) - self.parse(y)))

    def sum(self, a: str, b: str, b_scale=1.0):
        return self.format(self.parse(a) + b_scale * self.parse(b))

class TimeFormatter(UnitsFormatter):
    def future_time(self, time_delta_str: str, base_units="seconds"):
        kwargs = {base_units: self.parse(time_delta_str)}
        delta = datetime.timedelta(**kwargs)
        now = datetime.datetime.now()
        return (now + delta).replace(microsecond=0)

class SinglePartFormatter(UnitsFormatter):
    def __init__(
        self,
        names: list[str],
        num_divisions: list[int],
        precision: int,
    ):
        ...

    def format(self, num_base_units: float):
        ...

time_verbose = TimeFormatter(
    names=[" seconds", " minutes", " hours", " days"],
    num_divisions=[60, 60, 24],
    base_precisions=[4, 2, 0],
    widths=[2],
)
time_concise = TimeFormatter(
    names=["s", "m", "h", "d"],
    num_divisions=[60, 60, 24],
    base_precisions=[4, 2, 0],
    widths=[2],
)
metric = SinglePartFormatter(
    names=["k", "m", "b", "t"],
    num_divisions=[1000],
    precision=1,
)
file_size = SinglePartFormatter(
    names=[" bytes", " kb", " mb", " gb", " tb", " pb"],
    num_divisions=[1024],
    precision=1,
)

def time_format(num_seconds: float, concise=False):
    if concise:
        return time_concise.format(num_seconds)
    else:
        return time_verbose.format(num_seconds)
