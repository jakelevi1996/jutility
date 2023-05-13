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
