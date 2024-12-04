from jutility import units, util
import test_utils

OUTPUT_DIR = test_utils.get_output_dir("test_units")

def test_time_format():
    printer = util.Printer("test_time_format", OUTPUT_DIR)

    column_formats = ["%7i", "%18.7f", "%17s", "%40s"]
    cf = util.ColumnFormatter(*column_formats, printer=printer)
    t_list = [
        unit.num_base_units
        for unit in units.time_verbose._all_units
    ]
    for i, t in enumerate(t_list):
        cf.print(i, t, util.time_format(t, True), util.time_format(t))

    assert t_list == [1, 60, 60*60, 60*60*24]
    assert [util.time_format(t) for t in t_list] == [
        "1.0000 seconds",
        "1 minutes  0.00 seconds",
        "1 hours  0 minutes  0 seconds",
        "1 days  0 hours  0 minutes  0 seconds",
    ]
    assert [util.time_format(t, concise=True) for t in t_list] == [
        "1.0000s",
        "1m  0.00s",
        "1h  0m  0s",
        "1d  0h  0m  0s",
    ]

    t_list = [10 ** i for i in range(-6, 9)]
    for i, t in enumerate(t_list):
        cf.print(i, t, util.time_format(t, True), util.time_format(t))

    assert [util.time_format(t) for t in t_list] == [
        "0.0000 seconds",
        "0.0000 seconds",
        "0.0001 seconds",
        "0.0010 seconds",
        "0.0100 seconds",
        "0.1000 seconds",
        "1.0000 seconds",
        "10.0000 seconds",
        "1 minutes 40.00 seconds",
        "16 minutes 40.00 seconds",
        "2 hours 46 minutes 40 seconds",
        "1 days  3 hours 46 minutes 40 seconds",
        "11 days 13 hours 46 minutes 40 seconds",
        "115 days 17 hours 46 minutes 40 seconds",
        "1157 days  9 hours 46 minutes 40 seconds",
    ]
    assert [util.time_format(t, concise=True) for t in t_list] == [
        "0.0000s",
        "0.0000s",
        "0.0001s",
        "0.0010s",
        "0.0100s",
        "0.1000s",
        "1.0000s",
        "10.0000s",
        "1m 40.00s",
        "16m 40.00s",
        "2h 46m 40s",
        "1d  3h 46m 40s",
        "11d 13h 46m 40s",
        "115d 17h 46m 40s",
        "1157d  9h 46m 40s",
    ]

    column_formats = ["%7.3f", "%4i", "%4i", "%4i", "%17s", "%40s"]
    cf = util.ColumnFormatter(*column_formats, printer=printer)
    smhd_list = [
        (0,   45, 0, 0),
        (0.1, 45, 0, 0),
        (2,   45, 0, 0),
        (30,  45, 0, 0),

        (0,   0,  23, 0),
        (0.1, 0,  23, 0),
        (2,   0,  23, 0),
        (30,  0,  23, 0),
        (0,   2,  23, 0),
        (0,   30, 23, 0),

        (0,   0,  0,  1234),
        (0.1, 0,  0,  1234),
        (2,   0,  0,  1234),
        (30,  0,  0,  1234),
        (0,   2,  0,  1234),
        (0,   30, 0,  1234),
        (0,   0,  2,  1234),
        (0,   0,  13, 1234),
    ]
    t_list = [s + 60*(m + 60*(h + 24*d)) for s, m, h, d in smhd_list]
    for (s, m, h, d), t in zip(smhd_list, t_list):
        cf.print(s, m, h, d, util.time_format(t, True), util.time_format(t))

    assert [util.time_format(t) for t in t_list] == [
        "45 minutes  0.00 seconds",
        "45 minutes  0.10 seconds",
        "45 minutes  2.00 seconds",
        "45 minutes 30.00 seconds",
        "23 hours  0 minutes  0 seconds",
        "23 hours  0 minutes  0 seconds",
        "23 hours  0 minutes  2 seconds",
        "23 hours  0 minutes 30 seconds",
        "23 hours  2 minutes  0 seconds",
        "23 hours 30 minutes  0 seconds",
        "1234 days  0 hours  0 minutes  0 seconds",
        "1234 days  0 hours  0 minutes  0 seconds",
        "1234 days  0 hours  0 minutes  2 seconds",
        "1234 days  0 hours  0 minutes 30 seconds",
        "1234 days  0 hours  2 minutes  0 seconds",
        "1234 days  0 hours 30 minutes  0 seconds",
        "1234 days  2 hours  0 minutes  0 seconds",
        "1234 days 13 hours  0 minutes  0 seconds",
    ]
    assert [util.time_format(t, concise=True) for t in t_list] == [
        "45m  0.00s",
        "45m  0.10s",
        "45m  2.00s",
        "45m 30.00s",
        "23h  0m  0s",
        "23h  0m  0s",
        "23h  0m  2s",
        "23h  0m 30s",
        "23h  2m  0s",
        "23h 30m  0s",
        "1234d  0h  0m  0s",
        "1234d  0h  0m  0s",
        "1234d  0h  0m  2s",
        "1234d  0h  0m 30s",
        "1234d  0h  2m  0s",
        "1234d  0h 30m  0s",
        "1234d  2h  0m  0s",
        "1234d 13h  0m  0s",
    ]

def test_parse():
    printer = util.Printer("test_parse", OUTPUT_DIR)

    cf = util.ColumnFormatter("%14.4f", "%17s", "%14.4f", printer=printer)
    t_list = [
        *[0, 60, 61, 60*60, 60*60+1, 60*61, 12*60*60+34*60+56],
        *[10 ** i for i in range(-4, 9)],
    ]
    for t in t_list:
        s = units.time_concise.format(t)
        p = units.time_concise.parse(s)
        cf.print(t, s, p)
        assert t == p

    st_list = [
        ("  12.3  s  45  h  6  m  ", 12.3 + 45*60*60 + 6*60),
        ("1157d  9h 46m 40s", 100000000.0),
    ]
    for s, t in st_list:
        assert units.time_concise.parse(s) == t

def test_future_time():
    printer = util.Printer("test_future_time", OUTPUT_DIR)

    t_complete  = "6h  1m 10s"
    t_current   = "4h 51m 24s"
    n_complete  = 3
    n_total     = 5

    n = n_total - (n_complete + 1)
    current_remaining = units.time_concise.diff(t_complete, t_current)
    total_remaining = units.time_concise.sum(current_remaining, t_complete, n)
    finish_time = units.time_concise.future_time(total_remaining)

    cf = util.ColumnFormatter("%-19s", sep=" = ", printer=printer)
    cf.print("Time left (current)", current_remaining)
    cf.print("Time left (total)",   total_remaining)
    cf.print("Estimated finish",    finish_time)

    assert current_remaining == "1h  9m 46s"
    assert total_remaining   == "7h 10m 56s"

def test_format_list():
    def get_format_list(formatter: units.UnitsFormatter):
        return [
            unit.format_str
            for unit in formatter._all_units
        ]

    assert get_format_list(units.time_concise) == [
        '%0.4fs',
        '%0.0fm %5.2fs',
        '%0.0fh %2.0fm %2.0fs',
        '%0.0fd %2.0fh %2.0fm %2.0fs',
    ]
    assert get_format_list(units.time_verbose) == [
        '%0.4f seconds',
        '%0.0f minutes %5.2f seconds',
        '%0.0f hours %2.0f minutes %2.0f seconds',
        '%0.0f days %2.0f hours %2.0f minutes %2.0f seconds',
    ]
    assert get_format_list(units.metric) == [
        '%0.0f',
        '%0.1fk',
        '%0.1fm',
        '%0.1fb',
        '%0.1ft',
    ]
    assert get_format_list(units.file_size) == [
        '%0.0f bytes',
        '%0.1f kb',
        '%0.1f mb',
        '%0.1f gb',
        '%0.1f tb',
        '%0.1f pb',
    ]

    time_wide_format = units.TimeFormatter(
        units.BaseUnit("s", 2, [4, 2, 0]),
        units.CompoundUnit("m", 3, 0, 60),
        units.CompoundUnit("h", 4, 0, 60),
        units.CompoundUnit("d", 5, 0, 24),
    )
    assert get_format_list(time_wide_format) == [
        '%0.4fs',
        '%0.0fm %5.2fs',
        '%0.0fh %3.0fm %2.0fs',
        '%0.0fd %4.0fh %3.0fm %2.0fs',
    ]

def test_metric():
    printer = util.Printer("test_metric", OUTPUT_DIR)

    cf = util.ColumnFormatter("%25.4f", "%8s", printer=printer)
    x_list = [1.23 * (10 ** i) for i in range(-1, 17)]
    for x in x_list:
        cf.print(x, units.metric.format(x))

    assert [units.metric.format(x) for x in x_list] == [
        "0",
        "1",
        "12",
        "123",
        "1.2k",
        "12.3k",
        "123.0k",
        "1.2m",
        "12.3m",
        "123.0m",
        "1.2b",
        "12.3b",
        "123.0b",
        "1.2t",
        "12.3t",
        "123.0t",
        "1230.0t",
        "12300.0t",
    ]

def test_file_size():
    printer = util.Printer("test_file_size", OUTPUT_DIR)

    cf = util.ColumnFormatter("%25.2f", "%11s", printer=printer)
    x_list = [2 ** i for i in range(-1, 70, 3)] + [1e7/17]
    for x in x_list:
        cf.print(x, units.file_size.format(x))

    assert [units.file_size.format(x) for x in x_list] == [
        "0 bytes",
        "4 bytes",
        "32 bytes",
        "256 bytes",
        "2.0 kb",
        "16.0 kb",
        "128.0 kb",
        "1.0 mb",
        "8.0 mb",
        "64.0 mb",
        "512.0 mb",
        "4.0 gb",
        "32.0 gb",
        "256.0 gb",
        "2.0 tb",
        "16.0 tb",
        "128.0 tb",
        "1.0 pb",
        "8.0 pb",
        "64.0 pb",
        "512.0 pb",
        "4096.0 pb",
        "32768.0 pb",
        "262144.0 pb",
        "574.4 kb",
    ]
