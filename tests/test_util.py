import os
import time
import numpy as np
import pytest
from jutility import util, plotting
import tests.util

OUTPUT_DIR = tests.util.get_output_dir("test_util")

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
        util.SilentColumn("d"),
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
        util.Column("c0", ".3f", width=10).set_callback(
            lambda: 10 * timer.time_taken(),
            level=0,
        ),
        util.Column("c1", ".3f", width=10).set_callback(
            lambda: 100 * timer.time_taken(),
            level=1,
        ),
        util.Column("c2", ".3f", width=10).set_callback(
            lambda: 1000 * timer.time_taken(),
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
        util.Column("z", "s",   width=40).set_callback(
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

    with pytest.raises(TypeError):
        table.save_json("test_table_json", OUTPUT_DIR)
        table.save_json("test_table_json", OUTPUT_DIR, ["z"])

    table.save_json("test_table_json", OUTPUT_DIR, ["x", "y", "t"])

    full_path = util.get_full_path("test_table_json", OUTPUT_DIR, "json")
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
        util.Column("mean_w", ".3f").set_callback(
            lambda: np.mean(w),
        ),
        util.Column("std_w", ".3f"),
        util.SilentColumn("w").set_callback(
            lambda: w.copy(),
        ),
        util.SilentColumn("w^2").set_callback(
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
    sleep_interval = 0.1

    with util.Timer("sleep", printer) as t:
        time.sleep(sleep_interval)

    assert t.time_measured >= sleep_interval

    with util.Timer(printer=printer) as t:
        time.sleep(sleep_interval)

    assert t.time_measured >= sleep_interval

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
        util.Column("x1", ".3f", width=10).set_callback(
            lambda: rng.normal(loc=10),
        ),
        util.Column("x2", ".3f", width=10).set_callback(
            lambda: rng.normal(loc=20),
            interval=util.CountInterval(5),
        ),
        util.Column("x3", ".3f", width=10).set_callback(
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
            axis_properties=plotting.AxisProperties(log_xscale=True),
        ),
        plotting.Subplot(
            *lines,
            axis_properties=plotting.AxisProperties(log_yscale=True),
        ),
        plotting.Subplot(
            *lines,
            axis_properties=plotting.AxisProperties(
                log_xscale=True,
                log_yscale=True,
            ),
        ),
    )
    mp.save("test_log_range", dir_name=OUTPUT_DIR)

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

def test_confidence_bounds():
    rng = util.Seeder().get_rng("test_confidence_bounds")
    printer = util.Printer("test_confidence_bounds", dir_name=OUTPUT_DIR)
    m = 19
    n = 7
    y1 = np.arange(m).reshape(m, 1) + 100
    y2 = np.arange(n).reshape(1, n) + 20
    y = y1 + y2 + rng.normal(size=[m, n])
    printer(y)
    subplots = []
    for split_dim in [0, 1]:
        mean, ucb, lcb = util.confidence_bounds(y, split_dim=split_dim)
        x = np.arange(mean.size)
        sp = plotting.Subplot(
            plotting.Line(x, mean, c="b", marker="o"),
            plotting.FillBetween(x, ucb, lcb, c="b", alpha=0.2),
        )
        subplots.append(sp)

    mp = plotting.MultiPlot(*subplots)
    mp.save("test_confidence_bounds", OUTPUT_DIR)

@pytest.mark.parametrize("data_len", [23, 25])
def test_confidence_bounds_downsample(data_len):
    test_name = "test_confidence_bounds_downsample_%s" % data_len
    printer = util.Printer(test_name, dir_name=OUTPUT_DIR)

    x = list(range(data_len))

    subplots = []
    for ds in [1, 3, 5]:
        mean, ucb, lcb = util.confidence_bounds(
            x,
            split_dim=0,
            downsample_ratio=ds,
        )
        printer(mean, ucb, lcb, sep="\n", end="\n"+"*"*50+"\n")
        sp = plotting.Subplot(
            plotting.Line(x[::ds], mean, c="b", marker="o"),
            plotting.FillBetween(x[::ds], ucb, lcb, c="b", alpha=0.3),
            axis_properties=plotting.AxisProperties(title="ds = %s" % ds)
        )
        subplots.append(sp)

    mp = plotting.MultiPlot(
        *subplots,
        figure_properties=plotting.FigureProperties(
            num_rows=1,
            title=test_name,
            constrained_layout=True,
        ),
    )
    mp.save(test_name, OUTPUT_DIR)

@pytest.mark.parametrize("plot_all_data", [True, False])
def test_noisy_data(plot_all_data):
    rng = util.Seeder().get_rng("test_noisy_data", plot_all_data)
    noisy_data_blue = util.NoisyData()
    noisy_data_red  = util.NoisyData()
    x_list = np.linspace(0, 1)

    for x in x_list:
        for num_repeats in range(rng.integers(10)):
            noisy_data_blue.update(x, x + 0.04 * rng.normal())
        for num_repeats in range(rng.integers(10)):
            noisy_data_red.update(x, 0.3 + (0.3 * x) + (0.04 * rng.normal()))

    mp = plotting.plot(
        *plotting.get_noisy_data_lines(
            noisy_data_blue,
            colour="b",
            name="Blue data",
            plot_all_data=plot_all_data,
        ),
        *plotting.get_noisy_data_lines(
            noisy_data_red,
            colour="r",
            name="Red data",
            plot_all_data=plot_all_data,
        ),
        plot_name="test_noisy_data, plot_all_data = %s" % plot_all_data,
        dir_name=OUTPUT_DIR,
        axis_properties=plotting.AxisProperties("x", "y", ylim=[-0.2, 1.2]),
        legend=True,
    )
    assert os.path.isfile(mp.filename)

def test_noisy_log_data():
    rng = util.Seeder().get_rng("test_noisy_log_data")
    noisy_log_data = util.NoisyData(log_space_data=True)
    noisy_data = util.NoisyData()
    x_list = util.log_range(0.01, 10, 50)
    for x in x_list:
        for repeat in range(rng.integers(10)):
            y = x * np.exp(rng.normal())
            noisy_log_data.update(x, y)
            noisy_data.update(x, y)

    mp = plotting.MultiPlot(
        plotting.Subplot(
            *plotting.get_noisy_data_lines(noisy_log_data),
            axis_properties=plotting.AxisProperties(
                title="noisy_log_data, log_space_data=True",
                log_xscale=True,
                log_yscale=True,
            ),
        ),
        plotting.Subplot(
            *plotting.get_noisy_data_lines(noisy_data),
            axis_properties=plotting.AxisProperties(
                title="noisy_data, log_space_data=False",
                log_xscale=True,
                log_yscale=True,
            ),
        ),
    )
    mp.save("test_noisy_log_data", OUTPUT_DIR)
