from jutility import units, util
import test_utils

OUTPUT_DIR = test_utils.get_output_dir("test_units")

def test_time_format():
    printer = util.Printer("test_time_format", OUTPUT_DIR)

    column_formats = ["%7i", "%18.7f", "%17s", "%40s"]
    cf = util.ColumnFormatter(*column_formats, printer=printer)
    t_list = [10 ** i for i in range(-6, 9)]
    for i, t in enumerate(t_list):
        cf.print(i, t, units.time_format(t, True), units.time_format(t))

    assert [units.time_format(t) for t in t_list] == [
        " 0.0000 seconds",
        " 0.0000 seconds",
        " 0.0001 seconds",
        " 0.0010 seconds",
        " 0.0100 seconds",
        " 0.1000 seconds",
        " 1.0000 seconds",
        "10.0000 seconds",
        " 1 minutes 40.00 seconds",
        "16 minutes 40.00 seconds",
        " 2 hours 46 minutes 40 seconds",
        " 1 days  3 hours 46 minutes 40 seconds",
        "11 days 13 hours 46 minutes 40 seconds",
        "115 days 17 hours 46 minutes 40 seconds",
        "1157 days  9 hours 46 minutes 40 seconds",
    ]
    assert [units.time_format(t, concise=True) for t in t_list] == [
        " 0.0000s",
        " 0.0000s",
        " 0.0001s",
        " 0.0010s",
        " 0.0100s",
        " 0.1000s",
        " 1.0000s",
        "10.0000s",
        " 1m 40.00s",
        "16m 40.00s",
        " 2h 46m 40s",
        " 1d  3h 46m 40s",
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
        cf.print(s, m, h, d, units.time_format(t, True), units.time_format(t))

    assert [units.time_format(t) for t in t_list] == [
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
    assert [units.time_format(t, concise=True) for t in t_list] == [
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

    assert current_remaining == " 1h  9m 46s"
    assert total_remaining   == " 7h 10m 56s"
