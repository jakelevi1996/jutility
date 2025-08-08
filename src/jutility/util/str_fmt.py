class StringFormatter:
    def format(self, x) -> str:
        raise NotImplementedError

class NoFormat(StringFormatter):
    def format(self, x) -> str:
        return x

class PercentFormatter(StringFormatter):
    def __init__(self, fmt_str: str):
        self._fmt_str = fmt_str

    def format(self, x) -> str:
        return self._fmt_str % x

class FloatFormatter(PercentFormatter):
    def __init__(
        self,
        precision:  (int | None)=None,
        width:      (int | None)=None,
    ):
        p = ("." + str(precision)) if (precision is not None) else ""
        w = str(width) if (width is not None) else ""
        self._fmt_str = "%" + w + p + "f"
