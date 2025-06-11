import datetime

class Unit:
    def get_format_parts(self, format_parts: list[str]) -> list[str]:
        raise NotImplementedError()

    def set_name(self, name: str):
        self.name = name

    def set_num_base_units(self, num_base_units: int):
        self.num_base_units = num_base_units

    def set_format_str(self, display_child_units: bool):
        self.display_child_units = display_child_units
        self.format_str = " ".join(self.get_format_parts([]))

    def format(self, part_units: list[float], last_part: float) -> str:
        if self.display_child_units:
            part_units.append(last_part)
            return self.format_str % tuple(reversed(part_units))
        else:
            return self.format_str % last_part

    def __repr__(self):
        type_name = type(self).__name__
        return "%s(\"%s\", %i, \"%s\")" % (
            type_name,
            self.name,
            self.num_base_units,
            self.format_str,
        )

class BaseUnit(Unit):
    def __init__(
        self,
        name: str,
        width: int,
        precisions: list[int],
    ):
        self.width = width
        self.precisions = precisions
        self.set_name(name)
        self.set_num_base_units(1)

    def get_format_parts(self, format_parts: list[str]) -> list[str]:
        n = len(format_parts)
        i = min(n, len(self.precisions) - 1)
        p = self.precisions[i]
        d = 1 if (p > 0) else 0
        w = (self.width + d + p) if (n > 0) else 0
        self_format = "%%%i.%if%s" % (w, p, self.name)
        return format_parts + [self_format]

class CompoundUnit(Unit):
    def __init__(
        self,
        name: str,
        width: int,
        precision: int,
        num_child_units: int,
    ):
        self.width = width
        self.precision = precision
        self.num_child_units = num_child_units
        self.set_name(name)

    def set_child_unit(self, child_unit: Unit):
        self.child_unit = child_unit
        self.set_num_base_units(
            self.num_child_units *
            self.child_unit.num_base_units
        )

    def get_format_parts(self, format_parts: list[str]) -> list[str]:
        n = len(format_parts)
        d = 1 if (self.precision > 0) else 0
        w = (self.width + d + self.precision) if (n > 0) else 0
        self_format = "%%%i.%if%s" % (w, self.precision, self.name)
        if self.display_child_units:
            format_parts.append(self_format)
            return self.child_unit.get_format_parts(format_parts)
        else:
            return [self_format]

class UnitsFormatter:
    def __init__(
        self,
        base_unit: BaseUnit,
        *compound_units: CompoundUnit,
        display_child_units=True,
    ):
        self._base_unit = base_unit
        self._compound_units = compound_units
        self._all_units = [base_unit] + list(compound_units)
        self._display_child_units = display_child_units

        for parent, child in zip(compound_units, self._all_units):
            parent.set_child_unit(child)
        for unit in self._all_units:
            unit.set_format_str(display_child_units)

    def format(self, x: float):
        part_units = []
        for unit in self._compound_units:
            if x < unit.num_child_units:
                return unit.child_unit.format(part_units, x)

            if self._display_child_units:
                x, child_units = divmod(x, unit.num_child_units)
                part_units.append(child_units)
            else:
                x /= unit.num_child_units

        return self._compound_units[-1].format(part_units, x)

    def parse(self, input_str: str):
        matches = {
            input_str.index(unit.name): unit
            for unit in self._all_units
            if unit.name in input_str
        }
        num_base_units = 0
        start = 0
        for i in sorted(matches.keys()):
            unit = matches[i]
            num_units = float(input_str[start:i])
            num_base_units += num_units * unit.num_base_units
            start = i + len(unit.name)

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

time_verbose = TimeFormatter(
    BaseUnit(" seconds", 2, [4, 2, 0]),
    CompoundUnit(" minutes",    2, 0, 60),
    CompoundUnit(" hours",      2, 0, 60),
    CompoundUnit(" days",       2, 0, 24),
)
time_concise = TimeFormatter(
    BaseUnit("s", 2, [4, 2, 0]),
    CompoundUnit("m", 2, 0, 60),
    CompoundUnit("h", 2, 0, 60),
    CompoundUnit("d", 2, 0, 24),
)
metric = UnitsFormatter(
    BaseUnit("", 0, [0]),
    CompoundUnit("k", 0, 1, 1000),
    CompoundUnit("m", 0, 1, 1000),
    CompoundUnit("b", 0, 1, 1000),
    CompoundUnit("t", 0, 1, 1000),
    display_child_units=False,
)
file_size = UnitsFormatter(
    BaseUnit(" bytes", 0, [0]),
    CompoundUnit(" kb", 0, 1, 1024),
    CompoundUnit(" mb", 0, 1, 1024),
    CompoundUnit(" gb", 0, 1, 1024),
    CompoundUnit(" tb", 0, 1, 1024),
    CompoundUnit(" pb", 0, 1, 1024),
    display_child_units=False,
)
