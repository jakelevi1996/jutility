import os
import numpy as np
import pytest
from jutility import util
import tests.util

OUTPUT_DIR = tests.util.get_output_dir("test_util")

def test_counter():
    count = util.Counter()
    num_counts = 10
    count_list = [count() for _ in range(num_counts)]
    assert count_list == list(range(num_counts))

def test_table():
    printer = util.Printer("test_table.txt", OUTPUT_DIR)
    table = util.Table(
        util.Column("epoch", width=8),
        util.Column("train_loss", ".5f"),
        util.Column("test_loss", ".5f"),
        util.TimeColumn("t"),
        print_every=100,
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
    printer = util.Printer("test_column_width.txt", OUTPUT_DIR)
    table = util.Table(
        util.Column("title_longer_than_width", width=3),
        util.Column("width_longer_than_title", width=30),
        printer=printer,
    )
    table.update(title_longer_than_width=100, width_longer_than_title=200)

def test_get_data():
    printer = util.Printer("test_get_data.txt", OUTPUT_DIR)
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

@pytest.mark.parametrize("print_every_level", [0, 1, 2])
def test_table_callback_level(print_every_level):
    timer = util.Timer()
    printer = util.Printer(
        "test_table_callback_level, print level %i.txt" % print_every_level,
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
        print_every_level=print_every_level,
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
    printer = util.Printer("test_table_json.txt", OUTPUT_DIR)
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

    loaded_table = util.Table(printer=printer)
    loaded_table.load_json("test_table_json.json", OUTPUT_DIR)

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
    printer = util.Printer("test_silent_column.txt", OUTPUT_DIR)
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
    printer = util.Printer("test_trim_string.txt", OUTPUT_DIR)
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
        filename="test_exception_context.txt",
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
        filename="test_printer.txt",
        dir_name=OUTPUT_DIR,
        print_to_console=False,
    )
    printer.print("Testing print method")
    printer("Testing __call__ method")
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

    printer = util.Printer("test_seeder.txt", OUTPUT_DIR)
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

def test_result():
    data = [1, 2, 3]
    result = util.Result("result_data", dir_name=OUTPUT_DIR, data=data)
    with result.get_context(save=True, suppress_exceptions=True):
        data[2] *= 4
        raise ValueError()

    loaded_data = util.Result("result_data.pkl", dir_name=OUTPUT_DIR).load()
    assert loaded_data == [1, 2, 12]

def test_time_format():
    printer = util.Printer("test_time_format.txt", dir_name=OUTPUT_DIR)
    rng = util.Seeder().get_rng("test_time_format")
    for num_hours in [0, 5, 15]:
        for num_mins in [0, 5, 15]:
            for num_secs in [0, 5, 15]:
                t = num_secs + 60 * (num_mins + 60 * num_hours)
                printer(
                    util.time_format(t).rjust(11),
                    util.time_format(t + rng.normal()).rjust(11),
                    sep=" | ",
                )
