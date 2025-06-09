from jutility.util.interval import _Interval, Always
from jutility.util.time_util import Timer, time_format
from jutility.util.dict_util import format_type

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
