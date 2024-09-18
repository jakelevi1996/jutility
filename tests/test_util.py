import os
import time
import numpy as np
import pytest
from jutility import util, plotting
import test_utils

OUTPUT_DIR = test_utils.get_output_dir("test_util")

def test_counter():
    count = util.Counter()
    num_counts = 10
    count_list = [count() for _ in range(num_counts)]
    assert count_list == list(range(num_counts))

def test_table():
    printer = util.Printer("test_table", OUTPUT_DIR)
    table = util.Table(
        util.Column("epoch", width=8),
        util.Column("train_loss", ".5f"),
        util.Column("test_loss", ".5f"),
        util.TimeColumn("t"),
        print_interval=util.CountInterval(100),
        printer=printer,
    )
    num_updates = 2000
    for i in range(num_updates):
        train_loss = pow(2, -0.001 * i)
        test_loss = np.sqrt(2) * train_loss
        table.update(
            epoch=i,
            train_loss=train_loss,
            test_loss=test_loss,
        )
    table.print_last()

    assert len(table) == num_updates
    printer("len(table) == %s" % len(table))

    assert table.get_data("epoch") == list(range(num_updates))
    assert all(isinstance(t, float) for t in table.get_data("t"))
    assert len(set(table.get_data("t"))) == num_updates

def test_left_justified_columns():
    printer = util.Printer("test_left_justified_columns", OUTPUT_DIR)
    table = util.Table(
        util.CountColumn("c", width=-11),
        util.TimeColumn("t", width=-11),
        util.Column("epoch", width=-10),
        util.Column("train_loss", ".5f"),
        util.Column("test_loss", ".5f", width=-12),
        util.CallbackColumn("left",  width=-12).set_callback(lambda: 42),
        util.CallbackColumn("right", width=+12).set_callback(lambda: 42),
        print_interval=util.CountInterval(100),
        printer=printer,
    )
    num_updates = 800
    for i in range(num_updates):
        train_loss = pow(2, -0.001 * i)
        test_loss = np.sqrt(2) * train_loss
        table.update(
            epoch=i,
            train_loss=train_loss,
            test_loss=test_loss,
        )
    table.print_last()

def test_column_width():
    printer = util.Printer("test_column_width", OUTPUT_DIR)
    table = util.Table(
        util.Column("title_longer_than_width", width=3),
        util.Column("width_longer_than_title", width=30),
        printer=printer,
    )
    table.update(title_longer_than_width=100, width_longer_than_title=200)

def test_get_data():
    printer = util.Printer("test_get_data", OUTPUT_DIR)
    table = util.Table(
        util.Column("a", width=5),
        util.Column("b", width=5),
        util.Column("c", width=5),
        util.Column("d", silent=True),
        printer=printer,
    )
    num_updates = [3, 4, 5]
    a_counter = util.Counter()
    b_counter = util.Counter()
    c_counter = util.Counter()
    for i in range(num_updates[0]):
        table.update(a=a_counter(), d=np.arange(i))
    for _ in range(num_updates[1]):
        table.update(a=a_counter(), b=b_counter())
    for _ in range(num_updates[2]):
        table.update(b=b_counter(), c=c_counter())

    assert len(table) == sum(num_updates)
    assert len(table.get_data("a")) == num_updates[0] + num_updates[1]
    assert len(table.get_data("b")) == num_updates[1] + num_updates[2]
    assert len(table.get_data("c")) == num_updates[2]
    assert len(table.get_data("d")) == num_updates[0]
    for name in ["a", "b", "c", "d"]:
        assert len(table.get_data(name, filter_none=False)) == len(table)

    a, d = table.get_data("a", "d")
    assert len(a) == num_updates[0]
    assert len(d) == num_updates[0]
    assert all(isinstance(i, int) for i in a)
    assert all(isinstance(i, np.ndarray) for i in d)
    a, b = table.get_data("a", "b")
    assert len(a) == num_updates[1]
    assert len(b) == num_updates[1]
    b, c = table.get_data("b", "c")
    assert len(b) == num_updates[2]
    assert len(c) == num_updates[2]
    a, c = table.get_data("a", "c")
    assert len(a) == 0
    assert len(c) == 0
    a, b, c = table.get_data("a", "b", "c")
    assert len(a) == 0
    assert len(b) == 0
    assert len(c) == 0
    a, c = table.get_data("a", "c", filter_none=False)
    assert len(a) == len(table)
    assert len(c) == len(table)

@pytest.mark.parametrize("print_level", [0, 1, 2])
def test_table_callback_level(print_level):
    timer = util.Timer()
    printer = util.Printer(
        "test_table_callback_level, print level %i" % print_level,
        OUTPUT_DIR,
    )
    table = util.Table(
        util.Column("epoch", width=8),
        util.CallbackColumn("c0", ".3f", width=10).set_callback(
            lambda: 10 * timer.get_time_taken(),
            level=0,
        ),
        util.CallbackColumn("c1", ".3f", width=10).set_callback(
            lambda: 100 * timer.get_time_taken(),
            level=1,
        ),
        util.CallbackColumn("c2", ".3f", width=10).set_callback(
            lambda: 1000 * timer.get_time_taken(),
            level=2,
        ),
        printer=printer,
        print_level=print_level,
    )
    count = 0
    for i in range(5):
        table.update(level=2, epoch=count)
        for j in range(5):
            table.update(level=1, epoch=count)
            for k in range(5):
                table.update(level=0, epoch=count)
                count += 1

    assert len(table) == 5*5*5 + 5*5 + 5
    assert len(table.get_data("epoch")) == len(table)

    c2_epoch_array, c2_array = table.get_data("epoch", "c2")
    assert len(c2_array) == 5
    assert len(c2_epoch_array) == 5
    assert np.all(c2_epoch_array == 25 * np.arange(5))

def test_table_json():
    printer = util.Printer("test_table_json", OUTPUT_DIR)
    rng = util.Seeder().get_rng("test_table_json")
    table = util.Table(
        util.Column("x", "i",   width=8),
        util.Column("y", ".3f", width=8),
        util.CallbackColumn("z", "s",   width=40).set_callback(
            lambda: rng.normal(size=5),
            level=1,
        ),
        util.TimeColumn("t"),
        printer=printer,
    )
    for i in range(20):
        table.update(x=i)
        if i % 5 == 0:
            table.update(x=i, y=rng.normal(), level=1)

    table.save_json("test_table_json_all", OUTPUT_DIR)
    table.save_json("test_table_json_z",   OUTPUT_DIR, ["z"])
    table.save_json("test_table_json_xyt", OUTPUT_DIR, ["x", "y", "t"])

    full_path = util.get_full_path("test_table_json_xyt", OUTPUT_DIR, "json")
    loaded_table = util.Table(printer=printer)
    loaded_table.load_json(full_path)

    assert len(loaded_table.get_data("x")) == len(loaded_table.get_data("t"))
    assert len(loaded_table.get_data("x")) > len(loaded_table.get_data("y"))
    assert len(loaded_table.get_data("t")) > len(loaded_table.get_data("y"))
    assert loaded_table.get_data("x") == table.get_data("x")
    assert loaded_table.get_data("y") == table.get_data("y")
    assert loaded_table.get_data("t") == table.get_data("t")
    assert loaded_table.get_data("x", "y") == table.get_data("x", "y")
    printer(loaded_table.get_data("x", "y"))
    printer(loaded_table.get_data("x", "t"))
    printer(loaded_table.get_data("x", "y", "t"))

def test_silent_column():
    printer = util.Printer("test_silent_column", OUTPUT_DIR)
    w = np.linspace(2, 3, 10)
    table = util.Table(
        util.Column("epoch", width=8),
        util.Column("batch", width=8),
        util.CallbackColumn("mean_w", ".3f").set_callback(
            lambda: np.mean(w),
        ),
        util.Column("std_w", ".3f"),
        util.CallbackColumn("w", silent=True).set_callback(
            lambda: w.copy(),
        ),
        util.CallbackColumn("w^2", silent=True).set_callback(
            lambda: np.square(w),
            level=1,
        ),
        util.TimeColumn("t"),
        printer=printer,
    )
    num_epochs = 20
    batches_per_epoch = 5
    for epoch in range(num_epochs):
        w *= 1.1
        table.update(level=1, epoch=epoch, std_w=np.std(w))
        for batch in range(batches_per_epoch):
            w *= 1.01
            table.update(batch=batch)

    w_data_list = table.get_data("w")
    w_sq_data_list = table.get_data("w^2")
    assert len(table) == (1 + batches_per_epoch) * num_epochs
    assert len(w_data_list) == len(table)
    assert len(w_sq_data_list) == num_epochs
    for data_list in [w_data_list, w_sq_data_list]:
        assert all(isinstance(array, np.ndarray) for array in data_list)
        assert all(array.shape == w.shape for array in data_list)
        tuple_list = [tuple(array) for array in data_list]
        assert len(set(tuple_list)) == len(tuple_list)

def test_trim_string():
    printer = util.Printer("test_trim_string", OUTPUT_DIR)
    printer(util.trim_string("12345678", max_len=10))
    printer(util.trim_string("12345678", max_len=8))
    printer(util.trim_string("12345678", max_len=7))
    printer(util.trim_string("12345678", max_len=3))
    printer(util.trim_string("12345678", max_len=0))
    printer(util.trim_string("12345678", max_len=-3))

    assert len(util.trim_string("12345678", max_len=10)) == 8
    assert len(util.trim_string("12345678", max_len=7 )) == 7

def test_exception_context():
    printer = util.Printer(
        filename="test_exception_context",
        dir_name=OUTPUT_DIR,
    )
    printer("About to enter ExceptionContext...")
    with util.ExceptionContext(suppress_exceptions=True, printer=printer):
        printer("In context, about to raise ValueError...")
        raise ValueError("Error message")

    printer("ExceptionContext has exited, now back in test_exception_context")

    with pytest.raises(ValueError):
        with util.ExceptionContext(
            suppress_exceptions=False,
            printer=printer,
        ):
            raise ValueError()

def test_printer():
    printer = util.Printer(
        filename="test_printer",
        dir_name=OUTPUT_DIR,
        print_to_console=False,
    )
    printer("Testing __call__ method")
    printer("Testing timestamp method")
    printer.timestamp()
    printer("Testing get_filename method")
    printer(printer.get_filename())
    printer("Testing close method")
    printer.close()

    with pytest.raises(ValueError):
        printer("Checking close method worked")

    assert os.path.isfile(os.path.join(OUTPUT_DIR, "test_printer.txt"))

def test_printer_read():
    printer = util.Printer()
    assert printer.read() is None
    printer = util.Printer(filename="test_printer_read", dir_name=OUTPUT_DIR)
    assert printer.read() == ""

    printer = util.Printer(filename="test_printer_read", dir_name=OUTPUT_DIR)

    printer(123)
    printer("Hello", "world")
    printer("x", "y", "z", sep="\n---\n", end="\nend")
    printer()

    assert printer.read() == "123\nHello world\nx\n---\ny\n---\nz\nend\n"

def test_seeder():
    assert util.Seeder().get_seed("123") == util.Seeder().get_seed("123")
    assert util.Seeder().get_seed("123") != util.Seeder().get_seed("321")
    assert util.Seeder().get_seed(1, 2, 3) == util.Seeder().get_seed(1, 2, 3)
    assert util.Seeder().get_seed(1, 2, 3) != util.Seeder().get_seed(3, 2, 1)

    seeder = util.Seeder()
    assert seeder.get_seed("123") != seeder.get_seed("123")

    seed_list = [
        seeder.get_seed(3, "string", seeder),
        seeder.get_seed(3, "string", seeder),
        seeder.get_seed(3, "string", seeder),
        seeder.get_seed(3, "string", seeder),
        seeder.get_seed(123),
        seeder.get_seed(123),
        seeder.get_seed(123),
        seeder.get_seed(321),
    ]
    num_seeds = len(seed_list)
    num_unique_seeds = len(set(seed_list))
    assert num_unique_seeds == num_seeds

    printer = util.Printer("test_seeder", OUTPUT_DIR)
    printer("seed_list = %s" % seed_list)

    x1 = util.Seeder().get_rng("test_seeder").normal(size=10)
    x2 = util.Seeder().get_rng("test_seeder").normal(size=10)
    assert x1.size == 10
    assert x2.size == 10
    assert np.all(x1 == x2)

    x3 = util.Seeder().get_rng("test_seeder", 2).normal(size=10)
    assert x3.size == 10
    assert not np.all(x1 == x3)
    assert not np.all(x2 == x3)

    seeder2 = util.Seeder()
    x4 = seeder2.get_rng("test_seeder").normal(size=10)
    x5 = seeder2.get_rng("test_seeder").normal(size=10)
    assert x4.size == 10
    assert x5.size == 10
    assert not np.all(x4 == x5)

def test_is_numeric():
    assert util.is_numeric(3)
    assert util.is_numeric(3.3)
    assert util.is_numeric(np.double(4))
    assert util.is_numeric(np.int32(4))
    assert util.is_numeric(np.int64(4))
    assert util.is_numeric(np.uint(4.5))
    assert util.is_numeric(np.linspace(0, 1)[20])
    assert util.is_numeric(np.linspace(0, 1, dtype=float)[20])
    assert util.is_numeric(np.linspace(0, 100, dtype=int)[20])
    assert util.is_numeric(np.linspace(0, 100, dtype=np.uint)[20])
    assert not util.is_numeric(complex(3.3, 4.2))
    assert not util.is_numeric(np.linspace(0, 1))
    assert not util.is_numeric("frog")
    assert not util.is_numeric(util)
    assert not util.is_numeric(util.is_numeric)

@pytest.mark.parametrize("raise_error", [True, False])
def test_callbackcontext(raise_error):
    data = [1, 2, 3]
    exit_callback = lambda: util.save_pickle(data, "result_data", OUTPUT_DIR)

    with util.CallbackContext(
        exit_callback=exit_callback,
        suppress_exceptions=True,
    ):
        data[2] *= 4
        if raise_error:
            raise ValueError()

    full_path = util.get_full_path("result_data", OUTPUT_DIR, "pkl")
    loaded_data = util.load_pickle(full_path)
    assert loaded_data == [1, 2, 12]

    os.remove(full_path)

def test_time_format():
    printer = util.Printer("test_time_format", dir_name=OUTPUT_DIR)
    rng = util.Seeder().get_rng("test_time_format")
    for concise in [True, False]:
        for num_hours in [0, 5, 15]:
            for num_mins in [0, 5, 15]:
                for num_secs in [0, 5, 15]:
                    t = num_secs + 60 * (num_mins + 60 * num_hours)
                    printer(
                        util.time_format(t, concise).rjust(30),
                        util.time_format(t + rng.normal(), concise).rjust(30),
                        sep=" | ",
                    )

def test_timer_context():
    printer = util.Printer("test_timer", dir_name=OUTPUT_DIR)
    sleep_interval = 0.01

    with util.Timer("sleep", printer) as t:
        time.sleep(sleep_interval)

    assert t.time_taken >= sleep_interval

    with util.Timer(printer=printer) as t:
        time.sleep(sleep_interval)

    assert t.time_taken >= sleep_interval

    with util.Timer("sleep 2", printer, verbose_enter=True) as t:
        time.sleep(sleep_interval)

    assert t.time_taken >= sleep_interval

    with util.Timer("hline", printer, hline=True) as t:
        printer("In Timer context")
        time.sleep(sleep_interval)

    assert t.time_taken >= sleep_interval

    with util.Timer("name 1", printer) as t:
        time.sleep(sleep_interval)
        t.set_name("name 2")

def test_intervals():
    printer = util.Printer("test_intervals", dir_name=OUTPUT_DIR)

    num_updates = 50
    count_interval = 5
    time_interval = 0.1
    sleep_interval = 0.01

    a = util.Always()
    n = util.Never()
    c = util.CountInterval(count_interval)
    t = util.TimeInterval(time_interval)

    table = util.Table(
        util.CountColumn("i"),
        util.TimeColumn("tc"),
        util.Column("c", width=5),
        util.Column("t", width=5),
        util.Column("a", width=5),
        util.Column("n", width=5),
        printer=printer,
    )
    for i in range(num_updates):
        c_ready = c.ready()
        t_ready = t.ready()
        table.update(a=a.ready(), n=n.ready(), c=c_ready, t=t_ready)
        if c_ready:
            c.reset()
        if t_ready:
            t.reset()
        time.sleep(sleep_interval)

    t_data = table.get_data("tc")
    t_total = t_data[-1] - t_data[0]
    num_ready = lambda s: table.get_data(s).count(True)

    assert num_ready("a") == num_updates
    assert num_ready("n") == 0
    assert num_ready("c") == int(num_updates / count_interval)
    assert num_ready("t") >= int(t_total / time_interval)
    assert num_ready("t") <= int(t_total / time_interval) + 1

def test_callback_interval():
    printer = util.Printer("test_callback_interval", dir_name=OUTPUT_DIR)
    rng = util.Seeder().get_rng("test_callback_interval")

    table = util.Table(
        util.CountColumn("c"),
        util.TimeColumn("t"),
        util.CallbackColumn("x1", ".3f", width=10).set_callback(
            lambda: rng.normal(loc=10),
        ),
        util.CallbackColumn("x2", ".3f", width=10).set_callback(
            lambda: rng.normal(loc=20),
            interval=util.CountInterval(5),
        ),
        util.CallbackColumn("x3", ".3f", width=10).set_callback(
            lambda: rng.normal(loc=30),
            interval=util.CountInterval(10),
        ),
        printer=printer,
    )

    for _ in range(100):
        table.update()

    assert len(table.get_data("x1")) == 100
    assert len(table.get_data("x2")) == 20
    assert len(table.get_data("x3")) == 10

def test_log_range():
    printer = util.Printer("test_log_range", dir_name=OUTPUT_DIR)

    n = 27
    x = util.log_range(0.1, 10, n)
    y = util.log_range(3, 2789, n)
    z = np.linspace(3, 2789, n)

    printer(x, y, z, sep="\n")

    lines = [
        plotting.Line(x, y, c="b", marker="o"),
        plotting.Line(x, z, c="r", marker="o"),
    ]
    mp = plotting.MultiPlot(
        plotting.Subplot(*lines),
        plotting.Subplot(
            *lines,
            axis_properties=plotting.AxisProperties(log_x=True),
        ),
        plotting.Subplot(
            *lines,
            axis_properties=plotting.AxisProperties(log_y=True),
        ),
        plotting.Subplot(
            *lines,
            axis_properties=plotting.AxisProperties(
                log_x=True,
                log_y=True,
            ),
        ),
    )
    mp.save("test_log_range", dir_name=OUTPUT_DIR)

    assert len(util.log_range(10, 100, 10, True)) == 10
    assert len(util.log_range(10, 100, 50, True)) == 50
    assert len(util.log_range(10, 100, 50, True, 0)) == 45
    assert len(util.log_range(10, 12, 10, True)) == 3
    assert len(util.log_range(10.1, 12.2, 10, True)) == 3
    assert len(util.log_range(10.2, 12.1, 10, True)) == 3
    assert len(util.log_range(10.2, 12.9, 10, True)) == 4
    assert util.log_range(10, 12, 10, True).dtype == np.int64
    assert util.log_range(10, 12, 10, False).dtype == np.float64

def test_check_type():
    printer = util.Printer("test_check_type", dir_name=OUTPUT_DIR)

    i = 7
    f = 1.3
    s = "Hello, world"

    util.check_type(i, int)
    util.check_type(f, float)
    util.check_type(s, str)
    util.check_type(i, int,     "i")
    util.check_type(f, float,   "f")
    util.check_type(s, str,     "s")

    for instance, t, name in [
        [i, float,  "i"],
        [i, str,    "i"],
        [f, int,    "f"],
        [f, str,    "f"],
        [s, int,    "s"],
        [s, float,  "s"],
    ]:
        with pytest.raises(TypeError):
            with util.ExceptionContext(False, printer):
                util.check_type(instance, t)

        with pytest.raises(TypeError):
            with util.ExceptionContext(False, printer):
                util.check_type(instance, t, name)

def test_check_equal():
    printer = util.Printer("test_check_equal", dir_name=OUTPUT_DIR)

    i = 7
    f1 = 7.0
    f2 = 7.1
    s = "Hello, world"

    util.check_equal(i, i)
    util.check_equal(i, f1)
    util.check_equal(f1, f1)
    util.check_equal(f2, f2)
    util.check_equal(s, s)

    for value, expected_value, name in [
        [f2,    i,      "i"],
        [s,     i,      "i"],
        [i,     f2,     "f2"],
        [s,     f2,     "f2"],
        [i,     s,      "s"],
        [f2,    s,      "s"],
    ]:
        with pytest.raises(RuntimeError):
            with util.ExceptionContext(False, printer):
                util.check_equal(value, expected_value)

        with pytest.raises(RuntimeError):
            with util.ExceptionContext(False, printer):
                util.check_equal(value, expected_value, name)

def test_function_list():
    fl = util.FunctionList()
    c1 = util.Counter()
    c2 = util.Counter()

    fl.add_functions(
        lambda: c1(),
        lambda: c2(),
        lambda: c2(),
    )

    assert c1.get_value() == 0
    assert c2.get_value() == 0

    results = fl.call_all()

    assert c1.get_value() == 1
    assert c2.get_value() == 2
    assert results == [0, 0, 1]

    results = fl.call_all(return_results=False)

    assert c1.get_value() == 2
    assert c2.get_value() == 4
    assert results == None

def test_extract_substring():
    substr = "123123"

    s = "hello $123123& hello"
    assert util.extract_substring(s, "$", "&") == substr

    s = "hello & $123123& hello $ & $ & $"
    assert util.extract_substring(s, "$", "&") == substr

    with pytest.raises(ValueError):
        s = "hello &123123$ hello"
        util.extract_substring(s, "$", "&")

    s = "hello $123123$ hello"
    assert util.extract_substring(s, "$", "$") == substr

def test_save_load_text():
    rng = util.Seeder().get_rng("test_save_load_text")
    x = rng.normal(size=[8, 7])
    util.numpy_set_print_options(precision=8)

    full_path = util.save_text(x, "test_save_load_text", OUTPUT_DIR)

    s = util.load_text(full_path)

    assert s.rstrip() == str(x)

def test_abbreviate_dictionary():
    d = {
        "num_epochs":   10,
        "hidden_dim":   [30, 20, 10],
        "lr":           1e-3,
        "log_std":      -4,
        "model_name":   None,
        "top_down":     True,
        "bottom_up":    False,
        "model_type":   "VeryLongModelName",
    }
    key_abbreviations = {
        "num_epochs":   "ne",
        "hidden_dim":   "hd",
        "lr":           "lr",
        "log_std":      "ls",
        "top_down":     "td",
        "bottom_up":    "bu",
        "model_type":   "mt",
    }
    replaces = {"VERYLONGMODELNAME": "VLMN"}

    s = util.abbreviate_dictionary(d, key_abbreviations, replaces)
    assert s == "buFhd30,20,10lr0.001ls-4mtVLMNne10tdT"

def test_table_str():
    rng = util.Seeder().get_rng("test_table_str")

    table = util.Table(
        util.CountColumn("c"),
        util.TimeColumn("t"),
        util.Column("x", ".5f", 10),
        util.Column("u", ".5f", 10),
    )
    for i in range(20):
        if i % 2 == 0:
            table.update(x=rng.normal(), u=rng.uniform())
        else:
            table.update(x=rng.normal())

    util.save_text(table, "test_table_str", OUTPUT_DIR)

def test_table_save_pickle():
    test_name = "test_table_save_pickle"
    printer = util.Printer(test_name, OUTPUT_DIR)
    table = util.Table(
        util.CallbackColumn("c", "s", 10).set_callback(
            lambda: 42,
        ),
        util.Column("d", "s", 10),
        printer=printer,
    )
    for i in range(3):
        table.update(d=3*i+4)

    util.save_text(table, test_name, OUTPUT_DIR)
    with pytest.raises(AttributeError):
        util.save_pickle(table, test_name, OUTPUT_DIR)

    full_path = table.save_pickle(test_name, OUTPUT_DIR)

    loaded_table = util.load_pickle(full_path)

    assert isinstance(loaded_table, util.Table)
    assert loaded_table is not table
    assert loaded_table.get_data("c", "d") == table.get_data("c", "d")
    assert str(loaded_table) == str(table)

    printer.hline()
    printer(str(loaded_table))

def test_progress():
    printer = util.Printer("test_progress", OUTPUT_DIR)

    for i in util.progress(
        range(23),
        printer=printer,
        print_interval=util.Always(),
    ):
        pass

def test_store_dict_context():
    printer = util.Printer("test_store_dict_context", OUTPUT_DIR)

    def g(**kwargs):
        return {k: v+1 for k, v in kwargs.items()}

    def f(d, secrets):
        with util.StoreDictContext(d, *secrets):
            return g(**d)

    d = {"x_%i" % i: 10*i*i for i in range(10)}
    secrets = ["secret_1", "secret_2"]
    for s in secrets:
        d[s] = 42

    len_d = len(d)
    assert len(d) == len_d
    printer(d)

    f_output = f(d, secrets)

    assert len(d) == len_d
    assert len(f_output) == len_d - len(secrets)
    for s in secrets:
        assert s in d
        assert s not in f_output
    for k in f_output:
        assert f_output[k] == d[k] + 1

    printer(f_output)
    printer(d)

def test_save_json():
    data = {
        "x": 23,
        "y": 4.6,
        "z": np.zeros([5, 5]),
    }
    with pytest.raises(TypeError):
        util.save_json(data, "test_save_json", OUTPUT_DIR, default=None)

    util.save_json(data, "test_save_json", OUTPUT_DIR)
    util.save_json(data, "test_save_json_no_indent", OUTPUT_DIR, indent=None)

def test_repr_table_columns():
    printer = util.Printer("test_repr_table_columns", OUTPUT_DIR)
    rng = util.Seeder().get_rng("test_repr_table_columns")

    columns = [
        util.CountColumn("c", -5),
        util.TimeColumn("t"),
        util.Column("x", "s", 10),
        util.Column("y", "s", 10),
        util.CallbackColumn("cb", "s", 10).set_callback(
            lambda: rng.integers(1000)
        ),
    ]
    table = util.Table(*columns)

    for _ in range(10):
        table.update(
            x=rng.integers(0, 10),
            y=rng.integers(0, 100),
        )

    printer(table, repr(table), *columns, sep="\n\n")

def test_negative_column_width():
    printer = util.Printer("test_negative_column_width", OUTPUT_DIR)

    table = util.Table(
        util.Column("a", "s", -10),
        util.Column("b", "s", -10),
        util.Column("c", "s", 10),
        util.Column("d", "s", 10),
        printer=printer,
    )
    table.update()
    table.update(a=1, b="frog")
    table.update(c=2, d="dog")
    table.update()

def test_repeated_column_names():
    with pytest.raises(ValueError):
        table = util.Table(
            util.Column("a"),
            util.Column("a"),
        )

def test_circular_iterator():
    printer = util.Printer("test_circular_iterator", OUTPUT_DIR)
    rng = util.Seeder().get_rng("test_circular_iterator")

    x = rng.integers(0, 100, 7)
    x_iter = util.circular_iterator(x)

    for i in range(50):
        printer(i, next(x_iter))

def test_merge_strings():
    printer = util.Printer("test_merge_strings", OUTPUT_DIR)

    input_lists = [
        [
            "abc1def9gh",
            "abc22def88gh",
            "abc333def777gh",
        ],
        [
            "abc1def9gh",
            "abc22def88gh22ij",
            "abc333def777gh1ijkl",
        ],
        [
            "abc1def9gh!",
            "abc22def88gh22ij",
            "abc333def777gh1ijkl",
        ],
        [
            "abc1def9ghi",
            "abc22def88gh22ij",
            "abc333def777gh1ijkl",
        ],
        [
            "Xabc1",
            "YYabc22",
            "ZZZabc333",
        ],
        [
            "abcdef",
            "abc1def",
            "abc22def",
            "abc3d33def",
        ],
        [
            "abcdef",
            "abc22def",
        ],
        ["abcdef(g=h)"] * 5,
        [
            "abcdef",
            "abc1def",
            "abc22def",
            "abc3d33def",
            "abc3d4ef",
            "abc3d4ef!",
        ],
    ]
    expected_outputs = [
        "abc['1', '22', '333']def['777', '88', '9']gh",
        "abc['1', '22', '333']def['777', '88', '9']gh['1', '22']ijkl",
        "abc['1', '22', '333']def['777', '88', '9']gh['!', '1ijkl', '22ij']",
        "abc['1', '22', '333']def['777', '88', '9']gh['', '1', '22']ijkl",
        "['X', 'YY', 'ZZZ']abc['1', '22', '333']",
        "abc['', '1', '22', '3']d['', '33d']ef",
        "abc['', '22']def",
        "abcdef(g=h)",
        "abc['', '1', '22', '3']d['', '33d', '4']ef!",
    ]

    assert len(input_lists) == len(expected_outputs)

    for i, o in zip(input_lists, expected_outputs):
        assert util.merge_strings(i, clean=False) == o

        printer(*i, o, util.merge_strings(i), sep="\n")
        printer.hline()

@pytest.mark.parametrize("print_level", [0, 1])
def test_force_print_table(print_level):
    test_name = "test_force_print_table, print_level = %i" % print_level
    printer = util.Printer(test_name, OUTPUT_DIR)
    rng = util.Seeder().get_rng(test_name)

    table = util.Table(
        util.Column("x", width=-5),
        util.Column("y", width=-5),
        util.Column("z", width=-5),
        printer=printer,
        print_level=print_level,
    )

    for _ in range(10):
        table.update(
            x=rng.integers(10),
            y=rng.integers(10),
            z=rng.integers(10),
        )

    def check_num_lines(nl0, nl1):
        num_lines = len(printer.read().split("\n"))
        if print_level == 0:
            assert num_lines == nl0
        if print_level == 1:
            assert num_lines == nl1

    check_num_lines(13, 3)
    table.print_last()
    check_num_lines(14, 3)
    table.print_last(level=1)
    check_num_lines(15, 4)
    table.print_last(level=2)
    check_num_lines(16, 5)
    table.print_last(level=0)
    check_num_lines(17, 5)
    table.print_last(level=-1)
    check_num_lines(17, 5)

def test_save_image_load_image():
    test_name = "test_save_image_load_image"
    rng = util.Seeder().get_rng(test_name)

    shape = (100, 150)
    x = rng.integers(0, 256, shape).astype(np.uint8)
    util.save_image(x, test_name, OUTPUT_DIR)
    full_path = util.get_full_path(test_name, OUTPUT_DIR, "png")
    y = util.load_image(full_path)

    assert isinstance(x, np.ndarray)
    assert x.shape == shape
    assert x.dtype == np.uint8
    assert x.dtype != np.float32

    assert isinstance(y, np.ndarray)
    assert y.shape == shape
    assert y.dtype == np.uint8
    assert y.dtype != np.float32

    assert np.all(y == x)

def test_save_image_load_image_rgba():
    test_name = "test_save_image_load_image_rgba"
    rng = util.Seeder().get_rng(test_name)

    shape = (100, 150, 4)
    x = rng.integers(0, 256, shape).astype(np.uint8)
    with pytest.raises(ValueError):
        util.save_image(x, test_name, OUTPUT_DIR)

    util.save_image(x, test_name, OUTPUT_DIR, rgba=True)
    full_path = util.get_full_path(test_name, OUTPUT_DIR, "png")
    y = util.load_image(full_path)

    assert isinstance(x, np.ndarray)
    assert x.shape == shape
    assert x.dtype == np.uint8
    assert x.dtype != np.float32

    assert isinstance(y, np.ndarray)
    assert y.shape == shape
    assert y.dtype == np.uint8
    assert y.dtype != np.float32

    assert np.all(y == x)

def test_save_image_diff():
    rng = util.Seeder().get_rng("test_save_image_diff")
    mp1 = plotting.plot(
        plotting.Polygon(rng.uniform(0, 1, 3), rng.uniform(0, 1, 3), fc="b"),
        xlim=[0, 1],
        ylim=[0, 1],
        grid=False,
        plot_name="test_save_image_diff_input_1",
        dir_name=OUTPUT_DIR,
    )
    mp2 = plotting.plot(
        plotting.Polygon(rng.uniform(0, 1, 5), rng.uniform(0, 1, 5), fc="r"),
        xlim=[0, 1],
        ylim=[0, 1],
        grid=False,
        plot_name="test_save_image_diff_input_2",
        dir_name=OUTPUT_DIR,
    )
    im1 = util.load_image(mp1.full_path)
    im2 = util.load_image(mp2.full_path)
    util.save_image_diff(
        im1,
        im2,
        "test_save_image_diff rgba=True",
        OUTPUT_DIR,
        rgba=True,
    )
    util.save_image_diff(
        im1.mean(axis=2),
        im2.mean(axis=2),
        "test_save_image_diff rgba=False",
        OUTPUT_DIR,
    )
