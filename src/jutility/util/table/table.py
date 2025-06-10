from jutility.util.table.column import Column
from jutility.util.interval import _Interval, Always
from jutility.util.print_util import Printer, HLINE_LEN
from jutility.util.save_load import save_pickle
from jutility.util.dict_util import format_type

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
            self.print_last(level, False)
            self._print_interval.full_reset()

        if level == self._print_level:
            if self._print_interval.ready():
                self.print_last(level, False)
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

    def print_last(self, level=0, flush=True):
        if level >= self._print_level:
            self._printer(self.format_row(-1))
            if flush:
                self._printer.flush()

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
